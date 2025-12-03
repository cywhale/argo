import asyncio
import os
from pathlib import Path

import pytest

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from cli.viewer_client import PluginClient, PluginUnavailableError


@pytest.mark.asyncio
async def test_viewer_handshake_with_local_argo(tmp_path):
    """
    Smoke test: launch odbViz via python -m odbViz.plugin (requires argo repo adjacent).
    """
    argo_dir = ROOT.parent / "argo"
    if not argo_dir.exists():
        pytest.skip("argo repo not found; skipping viewer handshake test")

    # Ensure python can import odbViz from the argo repo
    env_path = os.environ.get("PYTHONPATH", "")
    extra = str(argo_dir)
    os.environ["PYTHONPATH"] = f"{extra}:{env_path}" if env_path else extra

    client = PluginClient()
    try:
        await client.ensure_started()
    except PluginUnavailableError:
        pytest.skip("odbViz not available in this environment")

    caps = client.capabilities
    assert caps.get("open_dataset") is True
    assert caps.get("plot") is True

    await client.close()
