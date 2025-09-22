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
import shlex
import ssl
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set
import shutil, subprocess, importlib.util

try:  # Optional readline support for interactive editing
    import readline  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    try:
        import pyreadline3 as readline  # type: ignore
    except ImportError:  # pragma: no cover - readline unavailable
        readline = None  # type: ignore

import websockets
import os
ODBARGO_DEBUG = os.getenv("ODBARGO_DEBUG", "0") not in ("", "0", "false", "False")

from cli.slash import (
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

# --- Plugin peer (WS) registration state ---
PLUGIN_AUTH_TOKEN = os.environ.get("ODBARGO_PLUGIN_TOKEN", "odbargoplot")  # optional shared secret
_plugin_ws: Optional[websockets.WebSocketServerProtocol] = None

def _ws_is_open(conn) -> bool:
    """
    websockets (server side) uses a ServerConnection that does NOT expose `.open`.
    Use `.closed` instead; when False, the connection is open.
    """
    if conn is None:
        return False
    try:
        closed = getattr(conn, "closed", None)
        if closed is None:
            # Be permissive: if attribute doesn't exist, assume open.
            return True
        if isinstance(closed, bool):
            return not closed
        # Some versions expose a close state object; treat truthy as 'closed'
        return not bool(closed)
    except Exception:
        return False

def plugin_ws_available() -> bool:
    return _ws_is_open(_plugin_ws)

async def plugin_ws_send_json(obj: dict) -> None:
    # forward control frames to the plugin WS peer
    if not plugin_ws_available():
        raise RuntimeError("Plugin WS not available")
    await _plugin_ws.send(json.dumps(obj))

async def plugin_ws_send_binary(data: bytes) -> None:
    if not plugin_ws_available():
        raise RuntimeError("Plugin WS not available")
    await _plugin_ws.send(data)

def _adjacent_view_binary() -> str | None:
    """Return path to an adjacent odbargo-view binary, searching script dir, parent, dist, and PATH."""
    # script dir where the entry was invoked
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    parent = os.path.abspath(os.path.join(here, ".."))
    exe = "odbargo-view" + (".exe" if os.name == "nt" else "")
    candidates = [
        os.path.join(here, exe),
        os.path.join(here, "dist", exe),
        os.path.join(parent, exe),
        os.path.join(parent, "dist", exe),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("odbargo-view")

async def ensure_viewer(self) -> bool:
    """
    Try to ensure the viewer is running and has registered via WS.
    Returns True if available (already or after spawn), else False.
    """
    # already registered?
    if getattr(self, "plugin_ws_available", None) and self.plugin_ws_available():
        return True

    # respect --plugin none
    if getattr(self, "plugin_mode", "auto") == "none":
        if ODBARGO_DEBUG:
            print("[View] ensure_viewer: disabled by --plugin none")
        return False

    # avoid spawning multiple times
    if getattr(self, "_viewer_spawn_inflight", False):
        # just wait for registration below
        pass
    else:
        self._viewer_spawn_inflight = True

        # prefer explicit --plugin-binary, else adjacent, else python -m
        bin_path = getattr(self, "plugin_binary", None) or _adjacent_view_binary()

        # figure out WS URL from attrs (port may be named ws_port or port)
        ws_port = getattr(self, "ws_port", None) or getattr(self, "port", None)
        ws_url = f"ws://127.0.0.1:{ws_port}"
        env = dict(os.environ)
        env["ODBARGO_CLI_WS"] = ws_url
        token = os.environ.get("ODBARGO_PLUGIN_TOKEN", "")
        if token:
            env["ODBARGO_PLUGIN_TOKEN"] = token

        try:
            if bin_path:
                if ODBARGO_DEBUG:
                    print(f"[View] ensure_viewer: spawning binary {bin_path} with ODBARGO_CLI_WS={ws_url}")
                subprocess.Popen([bin_path], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                if importlib.util.find_spec("odbargo_view") is None:
                    if ODBARGO_DEBUG:
                        print("[View] ensure_viewer: odbargo_view module not found")
                    self._viewer_spawn_inflight = False
                    return False
                if ODBARGO_DEBUG:
                    print(f"[View] ensure_viewer: launching module 'python -m odbargo_view' with ODBARGO_CLI_WS={ws_url}")
                subprocess.Popen([sys.executable, "-m", "odbargo_view"], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            if ODBARGO_DEBUG:
                print(f"[View] ensure_viewer: spawn failed: {exc}")
            self._viewer_spawn_inflight = False
            return False

    # wait for registration (heavy deps can take time on first import)
    # allow override via env, default ~8s
    timeout_s = float(os.environ.get("ODBARGO_VIEW_STARTUP_TIMEOUT", "8.0"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if self.plugin_ws_available():
            self._viewer_spawn_inflight = False
            if ODBARGO_DEBUG:
                print("[View] ensure_viewer: plugin registered")
            return True
        await asyncio.sleep(0.1)

    self._viewer_spawn_inflight = False
    if ODBARGO_DEBUG:
        print("[View] ensure_viewer: timeout waiting for plugin registration")
    return False

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
            # If the plugin WS isn’t up, try to launch/register it now
            if not plugin_ws_available():
                ok = await ensure_viewer(self)
                if not ok:
                    self._startup_error = "View plugin not available; run odbargo-view separately or pass --plugin view/--plugin-binary"
                    print(f"ℹ️ {self._startup_error}")
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
class _LocalClientSink:
    """
    Minimal in-process stand-in for a WebSocket client.
    Used by slash commands to reuse _handle_view_request() and the plugin bridges.

    .send(...) is awaited by server code and we capture:
      - JSON messages (strings) into self.json_messages
      - Binary frames (bytes) into self.binary (for plot/export)
    .wait_done() completes when we get a terminal JSON (dataset_opened/vars/
    preview_result/dataset_closed/file_end) or after a single plot binary.
    """
    def __init__(self, request_id: str, expect_binary: bool = False) -> None:
        self.request_id = request_id
        self.expect_binary = expect_binary
        self.json_messages: list[dict] = []
        self.binary: bytearray | None = bytearray() if expect_binary else None
        self._done = asyncio.Event()
        self._saw_plot_header = False

    async def send(self, payload) -> None:
        # Server code calls this with either str(JSON) or bytes
        if isinstance(payload, (bytes, bytearray)):
            if self.binary is not None:
                self.binary += bytes(payload)
            # For plots we expect exactly one binary frame → done
            self._done.set()
            return

        if isinstance(payload, str):
            try:
                obj = json.loads(payload)
            except Exception:
                return
            self.json_messages.append(obj)

            t = obj.get("type")
            rid = obj.get("requestId")

            # terminal error → unblock immediately
            if t == "error" and rid == self.request_id:
                self._done.set()
                return

            if rid != self.request_id:
                return

            # Terminal messages
            if t in ("view.dataset_opened", "view.vars", "view.preview_result", "view.dataset_closed"):
                self._done.set()
            elif t == "plot_blob":
                self._saw_plot_header = True
            elif t == "file_end":
                self._done.set()

    async def wait_done(self, timeout: float = 30.0) -> None:
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

class ArgoCLIApp:
    def __init__(self, port: int, insecure: bool, plugin_mode: str, plugin_binary: Optional[str]) -> None:
        self.port = port
        self.insecure = insecure
        self.plugin = PluginClient(mode=plugin_mode, binary_override=plugin_binary)
        self.frontend_connected = asyncio.Event()
        self._last_dataset_key: Optional[str] = None
        self._preview_state: Dict[str, List[str]] = {}  # datasetKey -> last displayed preview columns
        self._job_counter = 0
        self._ws_server: Optional[asyncio.AbstractServer] = None
        self._readline = readline if 'readline' in globals() else None
        self._interactive = False
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        # --- WS plugin self-registration state ---
        self._plugin_ws: Optional[websockets.WebSocketServerProtocol] = None
        self._plugin_caps: Dict[str, Any] = {}
        self._plugin_pending: Dict[str, Dict[str, Any]] = {}  # msgId -> {"client_ws":..., "kind":..., "expect_binary": bool}
        self._plugin_reader_task: Optional[asyncio.Task[Any]] = None
        self._plugin_token = os.environ.get("ODBARGO_PLUGIN_TOKEN", "")        
        if self._readline:
            self._configure_readline()

    # ----------------------------- WS layer -----------------------------
    def _ws_is_open(self, conn) -> bool:
        """
        websockets ≥12 uses ServerConnection on the server side, which does not have `.open`.
        Use `.closed` instead (False means open).
        """
        if conn is None:
            return False
        # Some versions expose .closed as a bool, others as asyncio.Future-like.
        try:
            closed = getattr(conn, "closed", None)
            if closed is None:
                return True  # no 'closed' attribute → assume open
            if isinstance(closed, bool):
                return not closed
            # If it's an awaitable/future, 'done()' indicates it is closed.
            return not closed
        except Exception:
            return False
    
    def plugin_ws_available(self) -> bool:
        """Return True if the WS-registered viewer is connected."""
        return self._ws_is_open(self._plugin_ws)

    def _maybe_reprompt(self) -> None:
        # Repaint the REPL prompt immediately (no newline) if we’re in interactive mode.
        try:
            if getattr(self, "_interactive", False):
                sys.stdout.write("argo> ")
                sys.stdout.flush()
        except Exception:
            pass    

    async def run_server(self) -> None:
        self._ws_server = await websockets.serve(self._ws_handler, "localhost", self.port)
        print(f"[Argo-CLI] WebSocket listening on ws://localhost:{self.port}")

    async def _ws_handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Per-connection handler. For normal frontend/clients, we parse JSON and dispatch.
        For a WS-registered plugin, we **handover** this connection to _plugin_reader(),
        and DO NOT read from it here anymore (avoids concurrent recv()).
        """
        try:
            while True:
                raw = await websocket.recv()

                # Frontend shouldn't send binary; plugin will, but after registration
                if isinstance(raw, (bytes, bytearray)):
                    # Ignore stray binary from non-plugin connections
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "code": "BAD_JSON", "message": "Invalid JSON"}))
                    continue

                # --- Plugin self-registration happens here (handover after ack) ---
                if data.get("type") == "plugin.register":
                    token = data.get("token", "")
                    if self._plugin_token and token != self._plugin_token:
                        await websocket.send(json.dumps({
                            "type": "plugin.register_err",
                            "code": "UNAUTHORIZED",
                            "message": "Invalid plugin token",
                        }))
                        print("[View] WS-plugin register: rejected (bad token)")
                        continue

                    # mark this socket as the plugin peer
                    self._plugin_ws = websocket
                    self._plugin_caps = data.get("capabilities", {}) or {}
                    await websocket.send(json.dumps({
                        "type": "plugin.register_ok",
                        "pluginProtocolVersion": data.get("pluginProtocolVersion", "1.0"),
                    }))
                    print("[View] WS-plugin registered; caps =", self._plugin_caps)
                    self._maybe_reprompt()

                    # handover: only _plugin_reader() reads this socket from now on
                    try:
                        await self._plugin_reader(websocket)
                    finally:
                        if self._plugin_ws is websocket:
                            self._plugin_ws = None
                            self._plugin_caps = {}
                            self._plugin_pending.clear()
                            print("[View] WS-plugin disconnected; cleared state")
                            self._maybe_reprompt()
                    return  # end handler for this connection

                # normal client message
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

        # --- 💡 Plugin self-registration over the same WS server ---
        if msg_type == "plugin.register":
            token = data.get("token", "")
            if self._plugin_token and token != self._plugin_token:
                await websocket.send(json.dumps({
                    "type": "plugin.register_err",
                    "code": "UNAUTHORIZED",
                    "message": "Invalid plugin token",
                }))
                return
            # Mark this connection as the plugin peer
            self._plugin_ws = websocket
            self._plugin_caps = data.get("capabilities", {})
            # Launch a reader task dedicated to this plugin connection
            if self._plugin_reader_task is None or self._plugin_reader_task.done():
                self._plugin_reader_task = asyncio.create_task(self._plugin_reader(websocket))
            await websocket.send(json.dumps({
                "type": "plugin.register_ok",
                "pluginProtocolVersion": data.get("pluginProtocolVersion", "1.0"),
            }))
            return
                
        if msg_type == "start_job":
            await self._handle_download_request(websocket, data)
            return
        if msg_type and msg_type.startswith("view."):
            if not self.plugin_ws_available():
                ok = await ensure_viewer(self)
                if not ok:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "requestId": data.get("requestId"),
                        "code": "PLUGIN_NOT_AVAILABLE",
                        "message": "View plugin not available; install or provide --plugin-binary",
                    }))
                    return
            await self._handle_view_request(websocket, data)
            return
        
        await websocket.send(json.dumps({"type": "error", "code": "UNKNOWN_MSG", "message": f"Unsupported type {msg_type}"}))

    async def _plugin_reader(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Sole reader for the WS-registered plugin socket.
        Routes plugin control frames + following binary frames back to the requesting client.
        """
        try:
            async for raw in websocket:
                # ---------- Binary from plugin → deliver to whichever request expects it ----------
                if isinstance(raw, (bytes, bytearray)):
                    # find the earliest pending that expects binary
                    msg_id = None
                    client_ws = None
                    for k, rec in list(self._plugin_pending.items()):
                        if rec.get("expect_binary"):
                            msg_id = k
                            client_ws = rec.get("client_ws")
                            break

                    if client_ws is not None:
                        # forward the binary verbatim to the waiting client
                        await client_ws.send(raw)

                        # for plots we’re done after a single binary frame
                        if self._plugin_pending.get(msg_id, {}).get("kind") == "plot":
                            self._plugin_pending.pop(msg_id, None)

                        # optional debug breadcrumb
                        # print(f"[View] → forwarded binary ({'plot' if msg_id else 'unknown'})")
                    else:
                        print("[View] plugin sent binary but no pending request expects it")
                    continue

                # ---------- Text JSON from plugin ----------
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    print("[View] plugin sent bad JSON; ignored")
                    continue

                op = msg.get("op")
                msg_id = msg.get("msgId")
                if not op or not msg_id:
                    # ignore non-framed noise
                    continue

                # "debug" messages are useful even if there is no pending waiter
                if op == "debug":
                    if ODBARGO_DEBUG:
                        mtxt = msg.get("message", "")
                        if msg_id:
                            print(f"[View][debug] [{msg_id}] {mtxt}")
                        else:
                            print(f"[View][debug] {mtxt}")
                    # do not require a pending waiter for debug frames
                    continue
                
                rec = self._plugin_pending.get(msg_id)
                client_ws = rec.get("client_ws") if rec else None
                if client_ws is None:
                    # don’t crash when a late message arrives after the waiter was cleared
                    print(f"[View] plugin op {op} for unknown msgId {msg_id}; ignored")
                    continue

                # ---------- Map plugin ops back to WS 'type' ----------
                if op == "open_dataset.ok":
                    await client_ws.send(json.dumps({
                        "type": "view.dataset_opened",
                        "requestId": msg_id,
                        "datasetKey": msg.get("datasetKey"),
                        "summary": msg.get("summary", {}),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← open_dataset.ok routed")

                elif op == "list_vars.ok":
                    await client_ws.send(json.dumps({
                        "type": "view.vars",
                        "requestId": msg_id,
                        "datasetKey": msg.get("datasetKey"),
                        "coords": msg.get("coords", []),
                        "vars": msg.get("vars", []),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← list_vars.ok routed")

                elif op == "preview.ok":
                    await client_ws.send(json.dumps({
                        "type": "view.preview_result",
                        "requestId": msg_id,
                        "datasetKey": msg.get("datasetKey"),
                        "columns": msg.get("columns", []),
                        "rows": msg.get("rows", []),
                        "limitHit": msg.get("limitHit", False),
                        "nextCursor": msg.get("nextCursor"),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← preview.ok routed")

                elif op == "dataset_closed":
                    await client_ws.send(json.dumps({
                        "type": "view.dataset_closed",
                        "requestId": msg_id,
                        "datasetKey": msg.get("datasetKey"),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← dataset_closed routed")
                    self._maybe_reprompt()

                elif op == "plot_blob":
                    # header first; the PNG bytes will arrive next as a binary WS frame
                    if rec is not None:
                        rec["expect_binary"] = True
                    await client_ws.send(json.dumps({
                        "type": "plot_blob",
                        "requestId": msg_id,
                        "contentType": msg.get("contentType", "image/png"),
                        "size": msg.get("size"),
                    }))
                    print("[View] ← plot_blob header routed; awaiting binary.")
                    self._maybe_reprompt()

                elif op == "file_start":
                    # CSV export starts – mark this waiter as expecting binary chunks
                    if rec is not None:
                        rec["expect_binary"] = True
                    await client_ws.send(json.dumps({
                        "type": "file_start",
                        "requestId": msg_id,
                        "contentType": msg.get("contentType", "text/csv"),
                        "filename": msg.get("filename"),
                    }))
                    # optional: print minimal breadcrumb (binary chunks are forwarded in the binary branch)

                elif op == "file_chunk":
                    # control marker only; the actual bytes arrive as a separate binary frame
                    pass

                elif op == "file_end":
                    await client_ws.send(json.dumps({
                        "type": "file_end",
                        "requestId": msg_id,
                        "sha256": msg.get("sha256"),
                        "size": msg.get("size"),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← file_end routed")
                    self._maybe_reprompt()

                elif op == "error":
                    await client_ws.send(json.dumps({
                        "type": "error",
                        "requestId": msg_id,
                        "code": msg.get("code", "PLUGIN_ERROR"),
                        "message": msg.get("message", "Plugin error"),
                        "hint": msg.get("hint"),
                        "details": msg.get("details"),
                    }))
                    self._plugin_pending.pop(msg_id, None)
                    print("[View] ← error routed:", msg.get("code"))
                    self._maybe_reprompt()

        except websockets.ConnectionClosed:  # plugin went away
            pass

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

        # If a WS-registered plugin is available, forward to it.
        # if self._plugin_ws is not None and self._plugin_ws.open:
        if self._ws_is_open(self._plugin_ws):
            t = data.get("type")
            op_map = {
                "view.open_dataset": "open_dataset",
                "view.list_vars": "list_vars",
                "view.preview": "preview",
                "view.close_dataset": "close_dataset",
                "view.plot": "plot",
                "view.export": "export",
            }
            op = op_map.get(t)
            if not op:
                await websocket.send(json.dumps({"type": "error", "requestId": request_id, "code": "UNKNOWN_MSG", "message": f"Unsupported type {t}"}))
                return

            payload = {k: v for k, v in data.items() if k not in {"type", "requestId"}}
            payload.update({"op": op, "msgId": request_id})

            # remember who should get the responses (and whether to expect binary)
            kind = "plot" if t == "view.plot" else ("export" if t == "view.export" else "simple")
            self._plugin_pending[request_id] = {"client_ws": websocket, "kind": kind, "expect_binary": False}

            try:
                await self._plugin_ws.send(json.dumps(payload))
                print(f"[View] → forwarded to WS-plugin: op={op} msgId={request_id}")
            except Exception as exc:
                self._plugin_pending.pop(request_id, None)
                await websocket.send(json.dumps({
                    "type": "error",
                    "requestId": request_id,
                    "code": "PLUGIN_UNAVAILABLE",
                    "message": f"WS plugin send failed: {exc}",
                }))
            return

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

    async def invoke_view_request_from_cli(
        self,
        data: dict,
        *,
        want_binary: bool = False,
        timeout: float = 30.0,
    ) -> tuple[list[dict], bytes | None]:
        """
        Run a view.* request through the normal WS-dispatch pipeline, but keep I/O in-process.
        Returns (json_messages, binary_bytes_or_None).
        """
        request_id = data.get("requestId") or f"cli-{int(time.time()*1000)}"
        data["requestId"] = request_id

        sink = _LocalClientSink(request_id, expect_binary=want_binary)

        # This reuses _handle_view_request(), which already knows how to:
        #   - use the WS-registered plugin if available (in your newer build), OR
        #   - fall back to stdio PluginClient
        await self._handle_view_request(sink, data)

        await sink.wait_done(timeout=timeout)

        bin_bytes = bytes(sink.binary) if sink.binary is not None and len(sink.binary) else None
        return (sink.json_messages, bin_bytes)

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
            # Guard: only try to start the viewer for real view operations (not help)
            is_view = parsed.request_type.startswith("view.")
            is_help = parsed.request_type in ("view.help", "view.usage") or command.strip().lower() in ("/view help", "/view --help")
            if is_view and not is_help:
                if not self.plugin_ws_available():
                    ok = await ensure_viewer(self)
                    if not ok:
                        print("⚠️ View plugin not available; run odbargo-view separately or pass --plugin view/--plugin-binary")
                        return EXIT_PLUGIN
                            
            if parsed.request_type == "view.open_dataset":
                req = dict(parsed.request_payload)
                req["type"] = "view.open_dataset"
                msgs, _ = await self.invoke_view_request_from_cli(req, want_binary=False)
                # find dataset_opened
                opened = next((m for m in msgs if m.get("type") == "view.dataset_opened"), None)
                if not opened:
                    print("⚠️  No response from viewer")
                    return EXIT_PLUGIN
                self._last_dataset_key = opened.get("datasetKey") or parsed.request_payload["datasetKey"]
                render_dataset_summary(opened.get("summary", {}))
                return EXIT_SUCCESS
            if parsed.request_type == "view.list_vars":
                req = dict(parsed.request_payload); req["type"] = "view.list_vars"
                msgs, _ = await self.invoke_view_request_from_cli(req, want_binary=False)
                listing = next((m for m in msgs if m.get("type") == "view.vars"), None)
                if not listing:
                    print("⚠️  No response from viewer")
                    return EXIT_PLUGIN
                # simple pretty print
                coords = listing.get("coords", [])
                vars_  = listing.get("vars", [])
                print("Coords:")
                for c in coords[:50]:
                    nm, dt, sz = c.get("name"), c.get("dtype"), c.get("size")
                    print(f"  - {nm} [{dt}] (size={sz})")
                print("Vars:")
                for v in vars_[:200]:
                    nm, dt, shp = v.get("name"), v.get("dtype"), v.get("shape")
                    print(f"  - {nm} [{dt}] shape={shp}")
                return EXIT_SUCCESS
            if parsed.request_type == "view.help":
                render_view_help(parsed.request_payload.get("topic"))
                return EXIT_SUCCESS
            if parsed.request_type == "view.preview":
                req = dict(parsed.request_payload); req["type"] = "view.preview"
                if parsed.request_payload.get("limit") is not None:
                    req["limit"] = int(parsed.request_payload["limit"])  
                if parsed.request_payload.get("trimDimensions"):
                    req["trim-dims"] = True

                msgs, _ = await self.invoke_view_request_from_cli(req, want_binary=False)
                result = next((m for m in msgs if m.get("type") == "view.preview_result"), None)
                if not result:
                    print("⚠️  No response from viewer")
                    return EXIT_PLUGIN
                render_preview(result)
                subset_key = result.get("subsetKey")
                self._last_dataset_key = subset_key or parsed.request_payload.get("datasetKey")
                return EXIT_SUCCESS
            if parsed.request_type == "view.close_dataset":
                req = dict(parsed.request_payload); req["type"] = "view.close_dataset"
                msgs, _ = await self.invoke_view_request_from_cli(req, want_binary=False)
                closed = next((m for m in msgs if m.get("type") == "view.dataset_closed"), None)
                if self._last_dataset_key == parsed.request_payload.get("datasetKey"):
                    self._last_dataset_key = None
                print(f"🧹 Closed dataset {parsed.request_payload.get('datasetKey')}")
                return EXIT_SUCCESS
            if parsed.request_type == "view.plot":
                req = dict(parsed.request_payload); req["type"] = "view.plot"
                gb = parsed.request_payload.get("groupBy")
                if gb:
                    # ensure list[str]
                    if isinstance(gb, str):
                        gb = [s.strip() for s in gb.split(",") if s.strip()]
                    req["groupBy"] = gb
                if parsed.request_payload.get("agg"):
                    req["agg"] = parsed.request_payload["agg"]                 
                if parsed.request_payload.get("limit") is not None:
                    req["limit"] = int(parsed.request_payload["limit"])

                show_window = parsed.out_path is None
                if not show_window:
                    parsed.out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    print("🖼️  Plot will open in a window (use --out to save instead).")

                msgs, png = await self.invoke_view_request_from_cli(req, want_binary=True)

                # If viewer returned an error, show it and exit immediately (no timeout)
                err = next((m for m in msgs if m.get("type") == "error"), None)
                if err:
                    code = err.get("code", "PLOT_FAIL")
                    msg  = err.get("message", "Plot failed")
                    hint = err.get("hint")
                    print(f"❌ {code}: {msg}" + (f" — {hint}" if hint else ""))
                    return EXIT_OUTPUT

                hdr = next((m for m in msgs if m.get("type") == "plot_blob"), None)
                if hdr:
                    size = hdr.get("size"); ctype = hdr.get("contentType")
                    print(f"🖼️  Plot ready ({ctype}, {size} bytes)")

                if not png:
                    print("⚠️  No plot bytes received")
                    return EXIT_OUTPUT

                if parsed.out_path:
                    parsed.out_path.write_bytes(png)
                    print(f"✅ Plot saved to {parsed.out_path}")
                elif show_window:
                    loop = asyncio.get_running_loop()
                    print(f"🖼️  Displaying plot window …")
                    display_plot_window(png, loop)
                else:
                    print(f"📡 Plot bytes available ({len(png)} bytes); use --out to save locally.")
                return EXIT_SUCCESS

            if parsed.request_type == "view.export":
                # Build a WS-style request and run it through the same bridge used by WS clients
                req = dict(parsed.request_payload)
                req["type"] = "view.export"
                if parsed.request_payload.get("limit") is not None:
                    req["limit"] = int(parsed.request_payload["limit"])                

                # Accumulate all binary frames until file_end
                msgs, data = await self.invoke_view_request_from_cli(req, want_binary=True)

                if not data:
                    print("⚠️  No export data received")
                    return EXIT_OUTPUT

                out_path = parsed.out_path
                if out_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    print(f"💾 CSV saved to {out_path}")
                else:
                    import tempfile
                    from pathlib import Path
                    tmp = Path(tempfile.mkdtemp(prefix="argo_")) / "export.csv"
                    tmp.write_bytes(data)
                    print(f"💾 CSV saved to {tmp}")
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


# if __name__ == "__main__":
#     main()
