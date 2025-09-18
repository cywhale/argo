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
import ssl
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import websockets

from argo_cli import (
    HELP_ENTRIES,
    ParsedCommand,
    exit_code_for_error,
    parse_slash_command,
    split_commands,
)
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


class PluginClient:
    def __init__(self) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._msg_counter = 0

    async def ensure_started(self) -> None:
        if self._process and self._process.returncode is None:
            return
        python_executable = sys.executable or "python3"
        self._process = await asyncio.create_subprocess_exec(
            python_executable,
            "-m",
            "odbargo_view",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert self._process.stdout
        hello_line = await self._process.stdout.readline()
        if not hello_line:
            raise RuntimeError("Plugin exited prematurely with no handshake")
        try:
            hello = json.loads(hello_line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Plugin handshake invalid: {hello_line!r}") from exc
        if hello.get("type") != "plugin.hello_ok":
            raise RuntimeError(f"Unexpected plugin handshake: {hello}")

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except asyncio.TimeoutError:  # pragma: no cover - defensive
                self._process.kill()

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
    def __init__(self, port: int, insecure: bool) -> None:
        self.port = port
        self.insecure = insecure
        self.plugin = PluginClient()
        self.frontend_connected = asyncio.Event()
        self._last_dataset_key: Optional[str] = None
        self._job_counter = 0
        self._ws_server: Optional[asyncio.AbstractServer] = None

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
        except Exception as exc:
            await websocket.send(json.dumps({
                "type": "error",
                "requestId": request_id,
                "code": "INTERNAL_ERROR",
                "message": str(exc),
            }))

    # ----------------------------- CLI layer -----------------------------
    async def run_repl(self) -> None:
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
                else:
                    await self._run_download_from_cli(command)

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
                if parsed.out_path is None:
                    print("ℹ️  No --out specified; binary plot will be emitted via WebSocket if connected.")
                else:
                    parsed.out_path.parent.mkdir(parents=True, exist_ok=True)
                async def on_header(message: Dict[str, Any]) -> None:
                    size = message.get("size")
                    ctype = message.get("contentType")
                    print(f"🖼️  Plot ready ({ctype}, {size} bytes)")
                async def on_binary(data: bytes) -> None:
                    if parsed.out_path:
                        parsed.out_path.write_bytes(data)
                        print(f"✅ Plot saved to {parsed.out_path}")
                    else:
                        print(f"📡 Plot bytes available ({len(data)} bytes); use --out to save locally.")
                await self.plugin.plot(parsed.request_payload, on_header, on_binary)
                return EXIT_SUCCESS
            if parsed.request_type == "view.export":
                if parsed.out_path is None:
                    print("ℹ️  No --out specified; CSV bytes will stream over WebSocket if connected.")
                else:
                    parsed.out_path.parent.mkdir(parents=True, exist_ok=True)
                    if parsed.out_path.exists():
                        parsed.out_path.unlink()
                bytes_written = 0
                async def on_start(message: Dict[str, Any]) -> None:
                    filename = parsed.out_path or message.get("filename")
                    print(f"📁 Export starting → {filename}")
                async def on_chunk(data: bytes) -> None:
                    nonlocal bytes_written
                    bytes_written += len(data)
                    if parsed.out_path:
                        with parsed.out_path.open("ab") as fh:
                            fh.write(data)
                async def on_end(message: Dict[str, Any]) -> None:
                    if parsed.out_path:
                        print(f"✅ Export complete ({bytes_written} bytes) → {parsed.out_path}")
                    else:
                        print(f"📡 Export ready: {bytes_written} bytes streamed via WS. sha256={message.get('sha256')}")
                await self.plugin.export(parsed.request_payload, on_start, on_chunk, on_end)
                return EXIT_SUCCESS
        except PluginMessageError as exc:
            handle_cli_plugin_error(exc.payload)
            return exit_code_for_error(exc.payload.get("code"))
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
    args = parser.parse_args()

    commands: List[str] = []
    if args.command and args.command[0].startswith("/"):
        raw = " ".join(args.command)
        commands = split_commands(raw)

    app = ArgoCLIApp(args.port, args.insecure)

    async def runner() -> int:
        if commands:
            code = await app.run_single_commands(commands)
            await app.plugin.close()
            return code
        await app.run_server()
        try:
            await app.run_repl()
        finally:
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
