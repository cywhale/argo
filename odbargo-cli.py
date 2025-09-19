"""ODB Argo CLI: downloader + view plugin bridge + slash REPL.

Implements:
- WebSocket server for download/view workflows (single port).
- Subprocess bridge to `odbargo_view` plugin via NDJSON.
- Slash-style CLI commands mapped 1:1 to WS messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shlex
import ssl
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set

try:  # Optional readline support for interactive editing
    import readline  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    try:
        import pyreadline3 as readline  # type: ignore
    except ImportError:  # pragma: no cover - readline unavailable
        readline = None  # type: ignore

import websockets

from argo_cli import (
    HELP_ENTRIES,
    ParsedCommand,
    exit_code_for_error,
    parse_slash_command,
    split_commands,
)


def _append_chunk(path: Path, data: bytes) -> None:
    with path.open("ab") as fh:
        fh.write(data)


# ---------------------------------------------------------------------------
# Download helpers (legacy functionality)
# ---------------------------------------------------------------------------


async def download_with_retry(wmo_list: List[int], output_path: str, retries: int = 3, delay: int = 10, force_insecure: bool = False) -> bool:
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(download_argo_data, wmo_list, output_path, insecure=force_insecure)
        except Exception as exc:
            print(f"⚠️ Attempt {attempt + 1} failed: {exc}")
            if "SSL error" in str(exc) and attempt == 0:
                force_insecure = True
            if attempt < retries - 1:
                print(f"⏳ Retrying in {delay} seconds…")
                await asyncio.sleep(delay)
            else:
                raise
    return False


def download_argo_data(wmo_list: List[int], output_path: str, insecure: bool = False) -> bool:
    import urllib.request

    base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats-synthetic-BGC.nc"
    if len(wmo_list) == 1:
        constraint = f'&platform_number=%22{wmo_list[0]}%22'
    else:
        wmo_pattern = "%7C".join(str(wmo) for wmo in wmo_list)
        constraint = f'&platform_number=~%22{wmo_pattern}%22'
    full_url = f"{base_url}?{constraint}"
    print(f"Downloading from: {full_url} {'[INSECURE]' if insecure else ''}")
    try:
        request = urllib.request.Request(full_url)
        request.add_header("User-Agent", "odbargo-cli/1.0")
        request.add_header("Connection", "keep-alive")
        request.add_header("Keep-Alive", "timeout=120, max=1000")
        ssl_ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=300, context=ssl_ctx) as response:
            if response.getcode() != 200:
                raise Exception(f"HTTP {response.getcode()}: {response.reason}")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) < 100:
                error_content = response.read().decode("utf-8")
                raise Exception(f"Server returned error: {error_content}")
            with open(output_path, "wb") as fh:
                fh.write(response.read())
        if Path(output_path).stat().st_size < 100:
            raise Exception("Downloaded file is too small, likely an error")
        return True
    except Exception as exc:  # pragma: no cover - network heavy
        raise Exception(f"Download failed: {exc}")


# ---------------------------------------------------------------------------
# Plugin bridge
# ---------------------------------------------------------------------------


class PluginMessageError(Exception):
    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload.get("message", "Plugin error"))
        self.payload = payload


class PluginUnavailableError(RuntimeError):
    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint


class PluginClient:
    def __init__(self, mode: str = "auto", binary_override: Optional[str] = None) -> None:
        self.mode = mode
        self._explicit_binary = Path(binary_override).expanduser() if binary_override else None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._msg_counter = 0
        self._launch_cmd: Optional[List[str]] = None
        self._startup_error: Optional[str] = None

    async def ensure_started(self) -> None:
        if self.mode == "none":
            raise PluginUnavailableError("View plugin disabled (--plugin none)")
        if self._startup_error:
            raise PluginUnavailableError(self._startup_error)
        if self._process and self._process.returncode is None:
            return
        if self._launch_cmd is None:
            self._launch_cmd = self._resolve_launch_command()
        if not self._launch_cmd:
            self._startup_error = "View plugin not available; run odbargo-view separately or pass --plugin view/--plugin-binary"
            raise PluginUnavailableError(self._startup_error)
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._launch_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
        except FileNotFoundError as exc:
            self._startup_error = f"Unable to launch view plugin: {exc}"
            raise PluginUnavailableError(self._startup_error) from exc
        assert self._process.stdout
        hello_line = await self._process.stdout.readline()
        if not hello_line:
            self._startup_error = "View plugin exited before handshake"
            raise PluginUnavailableError(self._startup_error)
        try:
            hello = json.loads(hello_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._startup_error = f"Invalid view plugin handshake: {hello_line!r}"
            raise PluginUnavailableError(self._startup_error) from exc
        if hello.get("type") != "plugin.hello_ok":
            self._startup_error = f"Unexpected view plugin handshake: {hello}"
            raise PluginUnavailableError(self._startup_error)

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except asyncio.TimeoutError:  # pragma: no cover - defensive
                self._process.kill()
        self._process = None

    async def open_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._request("open_dataset", payload, mode="simple")
        return response

    async def list_vars(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("list_vars", payload, mode="simple")

    async def preview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("preview", payload, mode="simple")

    async def close_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("close_dataset", payload, mode="simple")

    async def plot(
        self,
        payload: Dict[str, Any],
        on_header: Callable[[Dict[str, Any]], Awaitable[None]],
        on_binary: Callable[[bytes], Awaitable[None]],
    ) -> None:
        await self._request("plot", payload, mode="plot", on_header=on_header, on_binary=on_binary)

    async def export(
        self,
        payload: Dict[str, Any],
        on_start: Callable[[Dict[str, Any]], Awaitable[None]],
        on_chunk: Callable[[bytes], Awaitable[None]],
        on_end: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        await self._request(
            "export",
            payload,
            mode="export",
            on_start=on_start,
            on_binary=on_chunk,
            on_end=on_end,
        )

    async def _request(
        self,
        op: str,
        payload: Dict[str, Any],
        *,
        mode: str,
        on_header: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_binary: Optional[Callable[[bytes], Awaitable[None]]] = None,
        on_start: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_end: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            await self.ensure_started()
            assert self._process and self._process.stdin and self._process.stdout
            self._msg_counter += 1
            msg_id = f"m{self._msg_counter}"
            message = dict(payload)
            message.update({"op": op, "msgId": msg_id})
            encoded = (json.dumps(message) + "\n").encode("utf-8")
            self._process.stdin.write(encoded)
            await self._process.stdin.drain()

            while True:
                try:
                    line = await self._process.stdout.readline()
                except asyncio.LimitOverrunError:
                    raise PluginMessageError({
                        "code": "INTERNAL_ERROR",
                        "message": "Plugin response exceeded pipe size limit",
                        "hint": "Reduce preview size or filter the dataset",
                    })
                if not line:
                    raise RuntimeError("Plugin terminated unexpectedly")
                response = json.loads(line.decode("utf-8"))
                if response.get("msgId") and response["msgId"] != msg_id:
                    # Sequential contract ensures this should not happen, but guard anyway.
                    continue
                if response.get("op") == "error":
                    raise PluginMessageError(response)
                if mode == "simple":
                    return response
                if mode == "plot":
                    if response.get("op") != "plot_blob":
                        continue
                    if on_header:
                        await on_header(response)
                    size = int(response.get("size", 0))
                    data = await self._process.stdout.readexactly(size)
                    if on_binary:
                        await on_binary(data)
                    return response
                if mode == "export":
                    op_name = response.get("op")
                    if op_name == "file_start":
                        if on_start:
                            await on_start(response)
                        continue
                    if op_name == "file_chunk":
                        chunk_size = int(response.get("size", 0))
                        data = await self._process.stdout.readexactly(chunk_size)
                        if on_binary:
                            await on_binary(data)
                        continue
                    if op_name == "file_end":
                        if on_end:
                            await on_end(response)
                        return response
        raise RuntimeError("Unhandled plugin mode")

    def _resolve_launch_command(self) -> Optional[List[str]]:
        candidates: List[List[str]] = []

        def _make_path(path_str: str) -> Optional[List[str]]:
            path = Path(path_str).expanduser()
            if path.exists() and path.is_file():
                return [str(path)]
            return None

        if self._explicit_binary:
            if self._explicit_binary.exists():
                candidates.append([str(self._explicit_binary)])
            else:
                self._startup_error = f"Configured view binary not found: {self._explicit_binary}"
                return None
        env_binary = os.environ.get("ODBARGO_VIEW_BINARY")
        if env_binary:
            from_env = _make_path(env_binary) or shlex.split(env_binary)
            candidates.append(from_env)
        if getattr(sys, "frozen", False):  # PyInstaller sibling
            exe_path = Path(sys.executable)
            sibling = exe_path.with_name("odbargo-view")
            sibling_exe = sibling.with_suffix(".exe")
            for candidate in (sibling, sibling_exe):
                if candidate.exists():
                    candidates.append([str(candidate)])
                    break
        else:
            if self.mode == "view":
                candidates.append([sys.executable or "python3", "-m", "odbargo_view"])

        for cmd in candidates:
            if not cmd:
                continue
            executable = cmd[0]
            if Path(executable).exists() or os.path.sep not in executable:
                return cmd
        return None

    @property
    def availability_error(self) -> Optional[str]:
        return self._startup_error

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("XARRAY_DISABLE_PLUGIN_AUTOLOADING", "1")
        return env


# ---------------------------------------------------------------------------
# Slash command helpers (shared with tests)
# ---------------------------------------------------------------------------


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_DATASET = 3
EXIT_FILTER = 4
EXIT_OUTPUT = 5
EXIT_PLUGIN = 10


def parse_numeric_list(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    return items


# ---------------------------------------------------------------------------
# CLI application
# ---------------------------------------------------------------------------


class ArgoCLIApp:
    def __init__(self, port: int, insecure: bool, plugin_mode: str, plugin_binary: Optional[str]) -> None:
        self.port = port
        self.insecure = insecure
        self.plugin = PluginClient(mode=plugin_mode, binary_override=plugin_binary)
        self.frontend_connected = asyncio.Event()
        self._last_dataset_key: Optional[str] = None
        self._job_counter = 0
        self._ws_server: Optional[asyncio.AbstractServer] = None
        self._readline = readline if 'readline' in globals() else None
        self._interactive = False
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        if self._readline:
            self._configure_readline()

    # ----------------------------- WS layer -----------------------------
    async def run_server(self) -> None:
        self._ws_server = await websockets.serve(self._ws_handler, "localhost", self.port)
        print(f"[Argo-CLI] WebSocket listening on ws://localhost:{self.port}")

    async def _ws_handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "code": "BAD_JSON", "message": "Invalid JSON"}))
                    continue
                await self._dispatch_ws_message(websocket, data)
        except websockets.ConnectionClosed:  # pragma: no cover - runtime
            return

    async def _dispatch_ws_message(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")
        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
            self.frontend_connected.set()
            return
        if msg_type == "hello":
            response = {
                "type": "hello_ok",
                "capabilities": {"download": True, "view": True, "binaryFrames": True},
                "wsProtocolVersion": "1.0",
            }
            await websocket.send(json.dumps(response))
            return
        if msg_type == "start_job":
            await self._handle_download_request(websocket, data)
            return
        if msg_type and msg_type.startswith("view."):
            await self._handle_view_request(websocket, data)
            return
        await websocket.send(json.dumps({"type": "error", "code": "UNKNOWN_MSG", "message": f"Unsupported type {msg_type}"}))

    async def _handle_download_request(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        job_id = data.get("jobId") or f"job-{int(time.time())}"
        wmo_list = data.get("wmoList", [])
        job_number = data.get("jobNumber")
        asyncio.create_task(self._process_download(websocket, job_id, wmo_list, job_number))

    async def _process_download(self, websocket: websockets.WebSocketServerProtocol, job_id: str, wmo_list: List[int], job_number: Optional[int]) -> None:
        try:
            out_dir = tempfile.mkdtemp(prefix="argo_")
            file_path = Path(out_dir) / f"argo_{'_'.join(map(str, wmo_list))}.nc"
            await websocket.send(json.dumps({
                "type": "job_status",
                "jobId": job_id,
                "jobNumber": job_number,
                "status": "running",
                "message": f"⏳ Downloading data for {wmo_list}",
            }))
            await download_with_retry(wmo_list, str(file_path), force_insecure=self.insecure)
            await websocket.send(json.dumps({
                "type": "job_status",
                "jobId": job_id,
                "jobNumber": job_number,
                "status": "success",
                "resultPath": str(file_path),
                "message": f"✅ Download complete: {file_path}",
            }))
        except Exception as exc:
            await websocket.send(json.dumps({
                "type": "job_status",
                "jobId": job_id,
                "jobNumber": job_number,
                "status": "failed",
                "message": f"❌ {exc}",
            }))

    async def _handle_view_request(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        request_id = data.get("requestId")
        dataset_key = data.get("datasetKey")
        try:
            if data["type"] == "view.open_dataset":
                response = await self.plugin.open_dataset({
                    "path": data.get("path"),
                    "datasetKey": dataset_key,
                    "enginePreference": data.get("enginePreference"),
                })
                self._last_dataset_key = dataset_key
                payload = {
                    "type": "view.dataset_opened",
                    "requestId": request_id,
                    "datasetKey": dataset_key,
                    "summary": response.get("summary", {}),
                }
                await websocket.send(json.dumps(payload))
                return
            if data["type"] == "view.list_vars":
                response = await self.plugin.list_vars({"datasetKey": dataset_key})
                payload = {
                    "type": "view.vars",
                    "requestId": request_id,
                    "datasetKey": dataset_key,
                    "coords": response.get("coords", []),
                    "vars": response.get("vars", []),
                }
                await websocket.send(json.dumps(payload))
                return
            if data["type"] == "view.preview":
                response = await self.plugin.preview({k: v for k, v in data.items() if k not in {"type", "requestId"}})
                payload = {
                    "type": "view.preview_result",
                    "requestId": request_id,
                    "datasetKey": dataset_key,
                    "columns": response.get("columns", []),
                    "rows": response.get("rows", []),
                    "limitHit": response.get("limitHit", False),
                    "nextCursor": response.get("nextCursor"),
                    "subsetKey": response.get("subsetKey"),
                }
                subset_key = response.get("subsetKey")
                if subset_key:
                    self._last_dataset_key = subset_key
                else:
                    self._last_dataset_key = dataset_key
                await websocket.send(json.dumps(payload))
                return
            if data["type"] == "view.close_dataset":
                await self.plugin.close_dataset({"datasetKey": dataset_key})
                payload = {
                    "type": "view.dataset_closed",
                    "requestId": request_id,
                    "datasetKey": dataset_key,
                }
                await websocket.send(json.dumps(payload))
                if self._last_dataset_key == dataset_key:
                    self._last_dataset_key = None
                return
            if data["type"] == "view.plot":
                async def on_header(message: Dict[str, Any]) -> None:
                    header = {
                        "type": "plot_blob",
                        "requestId": request_id,
                        "contentType": message.get("contentType", "image/png"),
                        "size": message.get("size"),
                    }
                    await websocket.send(json.dumps(header))

                async def on_binary(chunk: bytes) -> None:
                    await websocket.send(chunk)

                await self.plugin.plot({k: v for k, v in data.items() if k not in {"type", "requestId"}}, on_header, on_binary)
                return
            if data["type"] == "view.export":
                async def on_start(message: Dict[str, Any]) -> None:
                    header = {
                        "type": "file_start",
                        "requestId": request_id,
                        "contentType": message.get("contentType", "text/csv"),
                        "filename": message.get("filename"),
                    }
                    await websocket.send(json.dumps(header))

                async def on_chunk(chunk: bytes) -> None:
                    await websocket.send(chunk)

                async def on_end(message: Dict[str, Any]) -> None:
                    footer = {
                        "type": "file_end",
                        "requestId": request_id,
                        "sha256": message.get("sha256"),
                        "size": message.get("size"),
                    }
                    await websocket.send(json.dumps(footer))

                await self.plugin.export({k: v for k, v in data.items() if k not in {"type", "requestId"}}, on_start, on_chunk, on_end)
                return
            raise ValueError(f"Unsupported view command: {data['type']}")
        except PluginMessageError as exc:
            error_payload = exc.payload
            await websocket.send(json.dumps({
                "type": "error",
                "requestId": request_id,
                "code": error_payload.get("code", "PLUGIN_ERROR"),
                "message": error_payload.get("message", "Plugin error"),
                "hint": error_payload.get("hint"),
                "details": error_payload.get("details"),
            }))
        except PluginUnavailableError as exc:
            await websocket.send(json.dumps({
                "type": "error",
                "requestId": request_id,
                "code": "PLUGIN_UNAVAILABLE",
                "message": str(exc),
            }))
        except Exception as exc:
            await websocket.send(json.dumps({
                "type": "error",
                "requestId": request_id,
                "code": "INTERNAL_ERROR",
                "message": str(exc),
            }))

    # ----------------------------- CLI layer -----------------------------
    async def run_repl(self) -> None:
        self._interactive = True
        print("🟡 CLI interactive mode. Type '/view ...' commands or comma-separated WMO list. Type 'exit' to quit.")
        while True:
            try:
                line = await asyncio.to_thread(input, "argo> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if not line.strip():
                continue
            if self._readline:
                try:
                    self._readline.add_history(line)
                except Exception:  # pragma: no cover - readline quirk
                    pass
            if line.strip().lower() in {"exit", "quit", ":q"}:
                break
            commands = split_commands(line)
            for command in commands:
                command = command.strip()
                if not command:
                    continue
                if command.startswith("/"):
                    exit_code = await self.execute_slash_command(command)
                    if exit_code != EXIT_SUCCESS:
                        print(f"(exit code {exit_code})")
                    continue
                await self._run_download_from_cli(command)
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        self._interactive = False

    async def _run_download_from_cli(self, line: str) -> None:
        digits = [c for c in line if c.isdigit() or c == ","]
        if not digits:
            print("⚠️  No valid WMO IDs found. Use comma-separated numbers or slash commands.")
            return
        try:
            wmo_list = parse_numeric_list("".join(digits))
        except ValueError:
            print("⚠️  Invalid WMO ID list")
            return
        self._job_counter += 1
        job_id = f"cli-{self._job_counter}"
        try:
            out_dir = tempfile.mkdtemp(prefix="argo_")
            file_path = Path(out_dir) / f"argo_{'_'.join(map(str, wmo_list))}.nc"
            print(f"▶️  Downloading {wmo_list} …")
            await download_with_retry(wmo_list, str(file_path), force_insecure=self.insecure)
            print(f"✅ Saved to: {file_path}")
        except Exception as exc:
            print(f"❌ {exc}")

    async def execute_slash_command(self, command: str) -> int:
        try:
            parsed = parse_slash_command(command, self._last_dataset_key)
        except Exception as exc:
            print(f"❌ {exc}")
            return EXIT_USAGE
        if parsed.echo_json:
            payload = dict(parsed.request_payload)
            payload["type"] = parsed.request_type
            print(json.dumps(payload, indent=2))
        try:
            if parsed.request_type == "view.open_dataset":
                response = await self.plugin.open_dataset(parsed.request_payload)
                self._last_dataset_key = parsed.request_payload["datasetKey"]
                render_dataset_summary(response.get("summary", {}))
                return EXIT_SUCCESS
            if parsed.request_type == "view.list_vars":
                response = await self.plugin.list_vars(parsed.request_payload)
                render_variable_summary(response)
                return EXIT_SUCCESS
            if parsed.request_type == "view.help":
                render_view_help(parsed.request_payload.get("topic"))
                return EXIT_SUCCESS
            if parsed.request_type == "view.preview":
                response = await self.plugin.preview(parsed.request_payload)
                render_preview(response)
                subset_key = response.get("subsetKey")
                if subset_key:
                    self._last_dataset_key = subset_key
                else:
                    self._last_dataset_key = parsed.request_payload["datasetKey"]
                return EXIT_SUCCESS
            if parsed.request_type == "view.close_dataset":
                await self.plugin.close_dataset(parsed.request_payload)
                if self._last_dataset_key == parsed.request_payload["datasetKey"]:
                    self._last_dataset_key = None
                print(f"🧹 Closed dataset {parsed.request_payload['datasetKey']}")
                return EXIT_SUCCESS
            if parsed.request_type == "view.plot":
                show_window = parsed.out_path is None
                if not show_window:
                    parsed.out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    print("🖼️  Plot will open in a window (use --out to save instead).")
                loop = asyncio.get_running_loop()
                async def on_header(message: Dict[str, Any]) -> None:
                    size = message.get("size")
                    ctype = message.get("contentType")
                    print(f"🖼️  Plot ready ({ctype}, {size} bytes)")
                async def on_binary(data: bytes) -> None:
                    if parsed.out_path:
                        parsed.out_path.write_bytes(data)
                        print(f"✅ Plot saved to {parsed.out_path}")
                    elif show_window:
                        print(f"🖼️  Displaying plot window …")
                        display_plot_window(data, loop)
                    else:
                        print(f"📡 Plot bytes available ({len(data)} bytes); use --out to save locally.")
                await self.plugin.plot(parsed.request_payload, on_header, on_binary)
                return EXIT_SUCCESS
            if parsed.request_type == "view.export":
                if self._interactive:
                    print("🚚 Export running in background …")
                    self._start_background_task(self._export_dataset(parsed, background=True), f"export {parsed.request_payload.get('datasetKey')}")
                    return EXIT_SUCCESS
                await self._export_dataset(parsed, background=False)
                return EXIT_SUCCESS
        except PluginMessageError as exc:
            handle_cli_plugin_error(exc.payload)
            return exit_code_for_error(exc.payload.get("code"))
        except PluginUnavailableError as exc:
            print(f"ℹ️  {exc}")
            if exc.hint:
                print(f"   Hint: {exc.hint}")
            return EXIT_PLUGIN
        except Exception as exc:
            print(f"❌ {exc}")
            return EXIT_PLUGIN
        return EXIT_USAGE

    async def run_single_commands(self, commands: Iterable[str]) -> int:
        exit_code = EXIT_SUCCESS
        for command in commands:
            code = await self.execute_slash_command(command)
            if code != EXIT_SUCCESS:
                exit_code = code
        return exit_code

    def _start_background_task(self, coro: Awaitable[None], label: str) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(t)
            try:
                t.result()
            except PluginMessageError as exc:
                handle_cli_plugin_error(exc.payload)
            except Exception as exc:
                print(f"❌ Background {label} failed: {exc}")

        task.add_done_callback(_done)

    async def _export_dataset(self, parsed: ParsedCommand, *, background: bool) -> None:
        out_path = parsed.out_path
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()
        bytes_written = 0

        async def on_start(message: Dict[str, Any]) -> None:
            filename = out_path or message.get("filename")
            print(f"📁 Export starting → {filename}")

        async def on_chunk(data: bytes) -> None:
            nonlocal bytes_written
            bytes_written += len(data)
            if out_path:
                await asyncio.to_thread(_append_chunk, out_path, data)

        async def on_end(message: Dict[str, Any]) -> None:
            if out_path:
                print(f"✅ Export complete ({bytes_written} bytes) → {out_path}")
            else:
                print(f"📡 Export ready: {bytes_written} bytes streamed via WS. sha256={message.get('sha256')}")

        try:
            await self.plugin.export(parsed.request_payload, on_start, on_chunk, on_end)
        except PluginMessageError as exc:
            if background:
                handle_cli_plugin_error(exc.payload)
            else:
                raise
        except Exception as exc:
            if background:
                print(f"❌ {exc}")
            else:
                raise

    async def wait_background_tasks(self) -> None:
        if not self._background_tasks:
            return
        await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

    def _configure_readline(self) -> None:
        try:
            self._readline.parse_and_bind("set editing-mode emacs")
            self._readline.parse_and_bind("tab: complete")
        except Exception:  # pragma: no cover - readline/pyreadline differences
            pass


def handle_cli_plugin_error(error: Dict[str, Any]) -> None:
    message = error.get("message", "Plugin error")
    hint = error.get("hint")
    code = error.get("code", "PLUGIN_ERROR")
    print(f"❌ {code}: {message}")
    if hint:
        print(f"   Hint: {hint}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_dataset_summary(summary: Dict[str, Any]) -> None:
    dims = summary.get("dims", {})
    coords = summary.get("coords", [])
    vars_meta = summary.get("vars", [])
    print("📂 Dataset opened:")
    print("   Dims   : " + ", ".join(f"{k}={v}" for k, v in dims.items()))
    print(f"   Coords : {len(coords)} | Vars: {len(vars_meta)}")


def render_variable_summary(response: Dict[str, Any]) -> None:
    coords = response.get("coords", [])
    vars_meta = response.get("vars", [])
    print("🧭 Coordinates:")
    for coord in coords[:10]:
        print(f"   - {coord['name']} ({coord['dtype']}, size={coord['size']})")
    if len(coords) > 10:
        print(f"   … ({len(coords) - 10} more)")
    print("🧪 Variables:")
    for var in vars_meta[:10]:
        shape = "×".join(map(str, var.get("shape", [])))
        print(f"   - {var['name']} ({var['dtype']}, shape={shape})")
    if len(vars_meta) > 10:
        print(f"   … ({len(vars_meta) - 10} more)")


def render_view_help(topic: Optional[str]) -> None:
    if topic:
        key = topic.lower()
        entry = HELP_ENTRIES.get(key)
        if entry:
            print(entry)
        else:
            print(f"❓ Unknown /view command '{topic}'.")
            print("   Available: " + ", ".join(sorted(HELP_ENTRIES.keys())))
        return
    print("Available /view commands:")
    for key in sorted(HELP_ENTRIES.keys()):
        print(f"  - {HELP_ENTRIES[key]}")


def display_plot_window(png_bytes: bytes, loop: Optional[asyncio.AbstractEventLoop] = None, *, filename_hint: str = "plot.png") -> None:
    """
    Display a PNG without importing matplotlib:
      1) write to a unique temp file
      2) best-effort open with the OS default image viewer
      3) fall back to the default web browser, else just print the path

    The 'loop' arg is kept for API compatibility but not used here.
    """
    import platform
    import subprocess
    import tempfile
    import webbrowser
    from pathlib import Path

    # 1) Save to a temp file (prefix/suffix help users recognize the file)
    suffix = filename_hint if filename_hint.endswith(".png") else (filename_hint + ".png")
    fd, temp_path_str = tempfile.mkstemp(prefix="odbargo_", suffix="_" + suffix)
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(png_bytes)
    except Exception as exc:
        print(f"⚠️  Failed to write plot to temp file: {exc}")
        return

    # 2) Try to open with the OS default image viewer
    opened = False
    try:
        sysname = platform.system()
        if sysname == "Windows":
            # 'start' is a shell built-in; use cmd /c
            subprocess.run(["cmd", "/c", "start", "", str(temp_path)], check=False)
            opened = True
        elif sysname == "Darwin":
            subprocess.run(["open", str(temp_path)], check=False)
            opened = True
        else:
            # Linux/BSD: xdg-open is the common launcher; harmless if it no-ops on headless
            subprocess.run(["xdg-open", str(temp_path)], check=False)
            opened = True
    except Exception:
        opened = False

    if opened:
        print(f"🖼️  Opened with system viewer → {temp_path}")
        return

    # 3) Fallback: try default browser
    try:
        webbrowser.open("file://" + str(temp_path))
        print(f"🖼️  Opened in default browser → {temp_path}")
        return
    except Exception:
        pass

    # 4) Last resort: tell the user where the file is
    print(f"🖼️  Plot saved to: {temp_path}")
    print("    (Could not auto-open; open the file manually or use --out <file.png>)")


def render_preview(response: Dict[str, Any], limit_cli: int = 20) -> None:
    columns = response.get("columns", [])
    rows = response.get("rows", [])
    limit_hit = response.get("limitHit", False)
    next_cursor = response.get("nextCursor")
    subset_key = response.get("subsetKey")
    print("🔎 Preview:")
    if not columns:
        print("   (no columns)")
        return
    header = " | ".join(columns)
    print("   " + header)
    print("   " + "-" * len(header))
    for row in rows[:limit_cli]:
        printable = []
        for value in row:
            if value is None:
                value = "—"
            elif isinstance(value, float):
                if math.isnan(value):
                    value = "—"
                else:
                    value = f"{value:.3f}" if abs(value) < 1000 else f"{value:.3e}"
            printable.append(str(value))
        print("   " + " | ".join(printable))
    if len(rows) > limit_cli:
        print(f"   … ({len(rows) - limit_cli} more rows in response slice)")
    if limit_hit:
        print("   (additional rows available; use --cursor " + str(next_cursor) + ")")
    if subset_key:
        print(f"   Subset saved as '{subset_key}'")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ODB Argo CLI with view support")
    parser.add_argument("command", nargs="*", help="Optional slash command(s) to run")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve WebSocket")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification for downloads")
    parser.add_argument(
        "--plugin",
        choices=["auto", "view", "none"],
        default="auto",
        help="View plugin mode: auto (attempt if available), view (force), none (disable)",
    )
    parser.add_argument(
        "--plugin-binary",
        help="Path to an odbargo-view executable (overrides auto detection)",
    )
    args = parser.parse_args()

    commands: List[str] = []
    if args.command and args.command[0].startswith("/"):
        raw = " ".join(args.command)
        commands = split_commands(raw)

    app = ArgoCLIApp(args.port, args.insecure, args.plugin, args.plugin_binary)

    async def runner() -> int:
        if commands:
            code = await app.run_single_commands(commands)
            await app.plugin.close()
            return code
        await app.run_server()
        try:
            await app.run_repl()
        finally:
            await app.wait_background_tasks()
            if app._ws_server:
                app._ws_server.close()
                await app._ws_server.wait_closed()
            await app.plugin.close()
        return EXIT_SUCCESS

    try:
        exit_code = asyncio.run(runner())
    except KeyboardInterrupt:
        exit_code = EXIT_SUCCESS
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
