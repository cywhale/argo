import asyncio
import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "odbargo_cli_module",
    Path(__file__).resolve().parents[1] / "odbargo-cli.py",
)
assert _SPEC and _SPEC.loader
odbargo_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(odbargo_cli)  # type: ignore[attr-defined]

PluginClient = odbargo_cli.PluginClient
PluginUnavailableError = odbargo_cli.PluginUnavailableError


def test_plugin_disabled_mode_raises():
    client = PluginClient(mode="none")
    with pytest.raises(PluginUnavailableError):
        asyncio.run(client.ensure_started())
