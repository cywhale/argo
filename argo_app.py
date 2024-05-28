# import xarray as xr
import argopy
from argopy import DataFetcher
from fastapi import FastAPI, Query, BackgroundTasks, WebSocket, TTPException #status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse #, ORJSONResponse
import os, zipfile, tempfile, asyncio #shutil
import src.config as config
from pydantic import BaseModel
from typing import List #Optional, Union
# import requests, json
# from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from datetime import datetime # timedelta
# from dask.distributed import Client
# client = Client('tcp://localhost:8786')


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
    # openapi_schema["servers"] = [
    #    {
    #        "url": "https://localhost:8090"
    #    }
    # ]
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
)  # , default_response_class=ORJSONResponse)


@app.get("/argo/api/swagger/openapi.json", include_in_schema=False)
async def custom_openapi():
    return JSONResponse(generate_custom_openapi())


@app.get("/argo/api/swagger", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/argo/api/swagger/openapi.json", title=app.title
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
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
        zip_path = os.path.join(config.outputPath, 'argo_data.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.write(nc_path, arcname='argo_data.nc')
            # z.write(profile_path, arcname='argo_profiles.nc')
            
        print(f"Data is ready and available at {zip_path}")
        config.download_status = f"Data is ready and available at {zip_path}. Updated: {datetime.now()}"
        print(config.download_status)
   
    except Exception as e:
        config.download_status = f"Error: {str(e)}"
        print(config.download_status)

class FloatDownloadRequest(BaseModel):
    wmo_list: List[int]

@app.post("/api/floats/download/", tags=["Argo"], summary="Download Argo floats NetCDF")
async def download_nc_file(background_tasks: BackgroundTasks, float_request: FloatDownloadRequest):
    """Endpoint to start the background task of downloading NC data."""
    
    background_tasks.add_task(fetch_and_prepare_nc, float_request.wmo_list)
    return JSONResponse(content={"message": f"Download started at {datetime.now()}, you will be notified when it is ready."})
