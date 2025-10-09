import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cli as _cli_package
    sys.modules.setdefault("argo_cli", _cli_package)
except ImportError:
    pass


_SPEC = importlib.util.spec_from_file_location(
    "odbargo_cli_module",
    Path(__file__).resolve().parents[1] / "odbargo-cli.py",
)
if _SPEC is None or _SPEC.loader is None:
    pytest.skip("legacy odbargo-cli.py entrypoint not available", allow_module_level=True)
odbargo_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(odbargo_cli)  # type: ignore[attr-defined]

PluginClient = odbargo_cli.PluginClient
PluginUnavailableError = odbargo_cli.PluginUnavailableError


def test_plugin_disabled_mode_raises():
    client = PluginClient(mode="none")
    with pytest.raises(PluginUnavailableError):
        asyncio.run(client.ensure_started())
