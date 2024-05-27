# A test version, not work yet
# import xarray as xr
import argopy
from fastapi import FastAPI, Query  # , status, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse  # , ORJSONResponse

# from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from typing import Optional  # , List, Union
# from pydantic import BaseModel
# import requests
# import json
# from datetime import datetime, timedelta
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


# @app.on_event("startup")
# async def startup():
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("APP start...")
    yield
    # below code to execute when app is shutting down
    print("APP end...")


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


# class OutResponse(BaseModel):
#    longitude: float
#    latitude: float
#    grid_lon: float
#    grid_lat: float
#    type: str


@app.get("/argo/api", tags=["Argo"], summary="Query Argo data")
async def get_agro(params: str):
    ds = argopy.DataFetcher().to_xarray()
    # Perform any necessary processing
    processed_data = ds.sel(parameter=params).to_dict()
    return processed_data


@app.get("/argo/api/test", tags=["Test"], summary="Test Argo operations")
async def get_tide_const(
    op: Optional[str] = Query(None, description="Operation code."),
):
    return {"success": "ok"}
