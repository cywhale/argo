# import xarray as xr
import argopy
from argopy import DataFetcher
from fastapi import FastAPI, BackgroundTasks, WebSocket #HTTPException, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse #, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import os, tempfile, asyncio, sys #zipfile
from argo_app.src import config
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
from datetime import datetime # timedelta
import uvicorn
import warnings
import asyncio
import argparse
import sys

# Optional MCP mode
try:
    import fastapi_mcp
    from fastapi_mcp import FastApiMCP
except ImportError:
    fastapi_mcp = None
    print("Warning: fastapi-mcp not installed.", file=sys.stderr)

from argo_app.src import config

warnings.filterwarnings("ignore", category=FutureWarning, message="The return type of `Dataset.dims` will be changed")
# from dask.distributed import Client
# client = Client('tcp://localhost:8786')

websocket_connected = False

def generate_custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ODB Argo App",
        version="1.0.0",
        description=(
            "Web API server to query Argo data by argopy and FastAPI.\n"
            + "* Trial/test only/under construction...\n"
        ),
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{config.default_port}"
        }
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    current_dir = os.path.abspath(os.path.dirname("__file__"))
    cache_dir = os.path.join(current_dir, config.cachePath)
    os.makedirs(cache_dir, exist_ok=True)
    argopy.set_options(cachedir=cache_dir)
    if (config.outputPath == ''):
        config.outputPath = tempfile.mkdtemp()
    if not os.path.exists(config.outputPath):
        os.makedirs(config.outputPath)
    print(f"APP start at {datetime.now()}, set cache at: {cache_dir}, output at: {config.outputPath}") 
    yield
    # below code to execute when app is shutting down
    print(f"APP end at {datetime.now()}...")


app = FastAPI(
    lifespan=lifespan, docs_url=None
) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/argo/api/swagger/openapi.json", include_in_schema=False)
async def custom_openapi():
    return JSONResponse(generate_custom_openapi())


@app.get("/argo/api/swagger", include_in_schema=False, operation_id="get_swagger")
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/argo/api/swagger/openapi.json", title=app.title
    )

@app.get("/argo/api/test", tags=["Test"], summary="Test Argo operations", operation_id="get_test")
async def test_argo_connection():
    config.download_status = "Success: message transmitted ok."
    return {"success": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for any message from the client
            data = await websocket.receive_text()
            if data == "subscribe_updates":
                # Monitor for any updates to send back to the client
                while True:
                    # Simulating sending an update
                    if config.download_status:
                        await websocket.send_json({"message":  config.download_status})
                    await asyncio.sleep(1)  # Sleep to simulate waiting for updates

    except WebSocketDisconnect:
        print("WebSocket connection was closed.")
        # Here you can add any cleanup logic if needed

    except Exception as e:
        print(f"An error occurred while websocket connecting: {str(e)}")
        # Handle any other exceptions that may occur

    finally:
        print("WebSocket connection has been properly closed.")

def fetch_and_prepare_nc(wmo_list):
    """Fetching an NC file and saving it locally."""
    try:
        # Assume fetch_nc_data handles downloading and returns the path to the fetched data
        config.ds = DataFetcher(ds='bgc', mode='expert', src='erddap', params='all', parallel=True).float(wmo_list).to_xarray()
        nc_path = os.path.join(config.outputPath, 'argo_data.nc')
        config.ds.to_netcdf(nc_path)
        # profile_path = os.path.join(config.outputPath, 'argo_profiles.nc')
        # config.ds.argo.point2profile().to_netcdf(profile_path)
        # Zip the files. In real application, replace print with a notification mechanism
        # zip_path = os.path.join(config.outputPath, 'argo_data.zip')
        # with zipfile.ZipFile(zip_path, 'w') as z:
            # z.write(nc_path, arcname='argo_data.nc')
            # z.write(profile_path, arcname='argo_profiles.nc')
        # NetCDF and .zip are equally in size, so no need to compress .nc
        os.chmod(config.outputPath, 0o755)
        os.chmod(nc_path, 0o644) #zip_path
        config.download_status = f"Data is ready and available at {nc_path}. Updated: {datetime.now()}"
        print(config.download_status)

    except (ConnectionError, TimeoutError) as e:
        config.download_status = f"Network Error: {str(e)}"
        print(config.download_status)
    except ValueError as e:
        config.download_status = f"Data Error: {str(e)}"
        print(config.download_status)
    except Exception as e:
        config.download_status = f"Unexpected Error: {str(e)}"
        print(config.download_status)

class FloatDownloadRequest(BaseModel):
    wmo_list: List[int] = Field(
        ...,
        description="List of WMO identifiers for Argo floats to download data for.",
        example=[5903377, 5903594]
    )

@app.post("/argo/api/floats/download/", tags=["Argo"], summary="Download Argo floats NetCDF", operation_id="get_data")
async def download_nc_file(background_tasks: BackgroundTasks, float_request: FloatDownloadRequest):
    """
    Initiates a background task to download NetCDF files for specified Argo floats.

    - **wmo_list**: A list of WMO identifiers for Argo floats and specified in POST request body.
    """

    wmo_list = list(set(float_request.wmo_list))
    background_tasks.add_task(fetch_and_prepare_nc, wmo_list)
    msg = f"Download {', '.join(str(wmo) for wmo in wmo_list)} started at {datetime.now()}, you will be notified when it is ready."
    config.download_status = msg
    print("message: ", msg)
    return JSONResponse(content={"message": msg})

@app.get("/argo/api/status", tags=["Argo"], summary="Download status", operation_id="get_status")
async def get_status():
    """
    Get Argo NetCDF file download status.
    """
    return {"status": config.download_status, "timestamp": datetime.now()}

# def run_mcp_server():
#    """Mount MCP server on separate port"""
if fastapi_mcp:
    mcp = FastApiMCP(
            app,
            name="Local Argo MCP server",
            description="MCP server for local Argo data processing",
            base_url=f"http://localhost:{config.mcp_port}",
    )
    mcp.mount()
    print(f"MCP server mounted on port {config.mcp_port} (JSON-RPC)")
else:
    print("fastapi_mcp not installed, MCP support unavailable.")

async def cli_interactive_mode():
    """Fallback CLI interaction mode"""
    print("\n🟡 Entering CLI fallback mode. Type natural language to get Argo data (type 'exit' to quit):")
    while True:
        query = input(">>> ").strip()
        if query.lower() == "exit":
            print("Goodbye.")
            break
        query_wmo = ''.join([c if c.isdigit() or c == ',' else '' for c in query])
        wmo_list = [int(x) for x in query_wmo.split(',') if x.strip().isdigit()]
        if wmo_list:
            print(f"🔍 Downloading Argo data for WMO(s): {wmo_list}\n")
            fetch_and_prepare_nc(wmo_list)
        else:
            print("⚠️  No valid WMO IDs found in input. Please try again.\n")

async def wait_for_enter_to_start_cli(timeout=5):
    """Wait for user to press Enter to start CLI mode"""
    try:
        await asyncio.wait_for(asyncio.to_thread(input, "\nPress Enter to enter CLI fallback mode...\n"), timeout)
        await cli_interactive_mode()
    except asyncio.TimeoutError:
        print("🟢 Frontend not detected. Entering CLI fallback mode.")
        await cli_interactive_mode()

def main():
    try:
        parser = argparse.ArgumentParser(description="ODB Argo App - Run server with optional MCP/CLI fallback.")
        parser.add_argument("port", type=int, nargs='?', default=config.default_port, help=f"Port for HTTP server (default: {config.default_port})")
        args = parser.parse_args()
        if not (1024 < args.port < 65535):
            raise ValueError("Port number must be between 1024 and 65535")
    except (ValueError, IndexError) as e:
        print(f"Invalid or missing port number: {e}")
        sys.exit(1)

    config.default_port = args.port
    config.mcp_port = args.port + 1
    print("Odbargo running on port: ", config.default_port, " and MCP port: ", config.mcp_port)
    # uvicorn.run("argo_app.app:app", host="127.0.0.1", port=port, log_level="info")

    import threading
    threading.Thread(target=lambda: uvicorn.run("argo_app.app:app", host="127.0.0.1", port=config.default_port, log_level="info")).start()

    print("⌛ Waiting for frontend WebSocket connection... (press Enter to start CLI mode manually)")
    try:
        asyncio.run(wait_for_enter_to_start_cli())
    except KeyboardInterrupt:
        print("\n🟥 Interrupted. Exiting.")

if __name__ == "__main__":
    main()
