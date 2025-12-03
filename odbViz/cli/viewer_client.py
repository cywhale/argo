from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


class PluginUnavailableError(RuntimeError):
    pass


class PluginMessageError(RuntimeError):
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        super().__init__(payload.get("message") or "Plugin error")


class BaseViewer:
    async def open_records(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def request(
        self,
        op: str,
        payload: Dict[str, Any],
        on_binary: Optional[Callable[[bytes], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Generic request for viewer ops other than plot/open_records.
        """
        raise NotImplementedError

    async def plot(
        self,
        payload: Dict[str, Any],
        on_header: Callable[[Dict[str, Any]], Awaitable[None]],
        on_binary: Callable[[bytes], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {}


class AttachedViewer(BaseViewer):
    """
    Viewer that talks to an already-connected odbViz over WebSocket.
    """

    def __init__(self, ws, capabilities: Dict[str, Any]):
        self._ws = ws
        self._capabilities = capabilities or {}
        self._counter = 0
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> Dict[str, Any]:
        return self._capabilities

    async def _send(self, message: Dict[str, Any]) -> None:
        await self._ws.send(json.dumps(message))

    async def open_records(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            self._counter += 1
            msg_id = f"m{self._counter}"
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[msg_id] = fut
            await self._send({"op": "open_records", "msgId": msg_id, **payload})
        return await fut

    async def plot(
        self,
        payload: Dict[str, Any],
        on_header: Callable[[Dict[str, Any]], Awaitable[None]],
        on_binary: Callable[[bytes], Awaitable[None]],
    ) -> None:
        async with self._lock:
            self._counter += 1
            msg_id = f"m{self._counter}"
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[msg_id] = fut
            await self._send({"op": "plot", "msgId": msg_id, **payload})
        # wait for completion via fut; callbacks handled by reader
        await fut
        return

    async def handle_message(self, obj: Dict[str, Any]) -> None:
        op = obj.get("op")
        msg_id = obj.get("msgId")
        if not msg_id:
            return
        fut = self._pending.get(msg_id)
        if not fut:
            return
        if op == "plot_blob":
            # Header + binary handled by _viewer_reader; mark done
            if not fut.done():
                fut.set_result(obj)
            return
        if op.endswith(".ok") or op == "open_records.ok":
            if not fut.done():
                fut.set_result(obj)
        elif op == "error":
            if not fut.done():
                fut.set_exception(PluginMessageError(obj))
        # other ops ignored


# Optional: stdio spawn viewer kept for potential future use
class StdioViewer(BaseViewer):
    def __init__(self, launch_cmd: Optional[list] = None) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._counter = 0
        self._lock = asyncio.Lock()
        self._launch_cmd = launch_cmd
        self._caps: Dict[str, Any] = {}

    @property
    def capabilities(self) -> Dict[str, Any]:
        return self._caps

    async def ensure_started(self) -> None:
        if self._process and self._process.returncode is None:
            return
        cmd = self._launch_cmd or [sys.executable or "python3", "-m", "odbViz.plugin"]
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        assert self._process.stdout
        hello = await self._process.stdout.readline()
        obj = json.loads(hello.decode("utf-8"))
        if obj.get("type") != "plugin.hello_ok":
            raise PluginUnavailableError(f"Unexpected handshake: {obj}")
        self._caps = obj.get("capabilities") or {}

    async def _request(
        self,
        op: str,
        payload: Dict[str, Any],
        on_binary: Optional[Callable[[bytes], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        await self.ensure_started()
        assert self._process and self._process.stdin and self._process.stdout
        async with self._lock:
            self._counter += 1
            msg_id = f"m{self._counter}"
            message = {"op": op, "msgId": msg_id, **payload}
            encoded = (json.dumps(message) + "\n").encode("utf-8")
            self._process.stdin.write(encoded)
            await self._process.stdin.drain()
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise PluginUnavailableError("Viewer exited")
                resp = json.loads(line.decode("utf-8"))
                if resp.get("msgId") != msg_id:
                    continue
                if resp.get("op") == "error":
                    raise PluginMessageError(resp)
                if op == "plot" and resp.get("op") == "plot_blob":
                    size = int(resp.get("size", 0))
                    data = await self._process.stdout.readexactly(size)
                    if on_binary:
                        await on_binary(data)
                    return resp
                return resp

    async def open_records(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("open_records", payload)

    async def plot(
        self,
        payload: Dict[str, Any],
        on_header: Callable[[Dict[str, Any]], Awaitable[None]],
        on_binary: Callable[[bytes], Awaitable[None]],
    ) -> None:
        def _wrap_header(resp: Dict[str, Any]) -> None:
            return None

        await self._request("plot", payload, on_binary=on_binary)
        await on_header({"op": "plot_blob", "msgId": payload.get("msgId")})


@dataclass
class PlotResult:
    header: Dict[str, Any]
    png: bytes


def display_plot_window(png_bytes: bytes, filename_hint: str = "plot.png") -> Optional[str]:
    import platform
    import subprocess
    import tempfile
    import webbrowser

    fd, temp_path = tempfile.mkstemp(prefix="odbchat_", suffix=filename_hint if filename_hint.endswith(".png") else f"_{filename_hint}")
    os.close(fd)
    with open(temp_path, "wb") as fh:
        fh.write(png_bytes)

    sysname = platform.system()
    opened = False
    try:
        if sysname == "Windows":
            os.startfile(temp_path)  # type: ignore[attr-defined]
            opened = True
        elif sysname == "Darwin":
            subprocess.Popen(["open", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            opened = True
        else:
            subprocess.Popen(["xdg-open", temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            opened = True
    except Exception:
        opened = False

    if not opened:
        try:
            webbrowser.open("file://" + temp_path)
        except Exception:
            pass
    return temp_path
