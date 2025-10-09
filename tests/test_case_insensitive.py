import numpy as np
import pytest
import xarray as xr

from cli.odbargo_cli import ArgoCLIApp
from odbargo_view.plugin import DatasetStore, OdbArgoViewPlugin, PluginError


def test_dataset_store_case_insensitive_renames(tmp_path):
    ds = xr.Dataset(
        {
            "TEMP": ("N_POINTS", np.arange(3.0)),
            "PSAL": ("N_POINTS", np.arange(3.0) + 1),
        },
        coords={
            "N_POINTS": np.arange(3),
            "LATITUDE": ("N_POINTS", np.full(3, 10.0)),
            "LONGITUDE": ("N_POINTS", np.full(3, 20.0)),
            "TIME": ("N_POINTS", np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")),
        },
    )
    path = tmp_path / "legacy.nc"
    ds.to_netcdf(path)

    store = DatasetStore()
    opened = store.open_dataset("legacy", str(path), case_insensitive=True)

    assert "latitude" in opened.coords
    assert "longitude" in opened.coords
    assert "temp" in opened.data_vars
    assert store.is_case_insensitive("legacy")


def test_dataset_store_flattened_promotes_coords(tmp_path):
    row = np.arange(5)
    ds = xr.Dataset(
        {
            "longitude": ("row", np.linspace(-30, -20, 5)),
            "latitude": ("row", np.linspace(10, 15, 5)),
            "time": ("row", np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"], dtype="datetime64[ns]")),
            "temp": ("row", np.linspace(5, 9, 5)),
            "pres": ("row", np.linspace(0, 100, 5)),
        },
        coords={"row": row},
    )
    path = tmp_path / "flat.nc"
    ds.to_netcdf(path)

    store = DatasetStore()
    opened = store.open_dataset("flat", str(path), case_insensitive=True)

    for coord in ("longitude", "latitude", "time"):
        assert coord in opened.coords

    bad_ds = xr.Dataset({"temp": ("row", np.arange(3))}, coords={"row": np.arange(3)})
    bad_path = tmp_path / "bad.nc"
    bad_ds.to_netcdf(bad_path)

    with pytest.raises(PluginError):
        store.open_dataset("bad", str(bad_path), case_insensitive=True)


def test_parse_filter_case_insensitive():
    pandas = pytest.importorskip("pandas")
    df = pandas.DataFrame({"temp": [1, 2, 3], "pres": [5, 15, 25], "time": pandas.to_datetime(["2020-01-01"] * 3)})
    plugin = OdbArgoViewPlugin()
    message = {"filter": {"dsl": "PRES > 10"}}
    mask, _ = plugin._parse_filter(message, df, case_insensitive=True)
    assert list(mask) == [False, True, True]


def test_cli_normalise_case_payload():
    app = ArgoCLIApp(port=8765, insecure=False, plugin_mode="none", plugin_binary=None, case_insensitive_vars=True)
    payload = {
        "x": "TEMP",
        "y": "PRES",
        "columns": ["TEMP", "PRES", "TIME"],
        "groupBy": ["LATITUDE:2"],
        "orderBy": [{"col": "TIME", "dir": "desc"}],
        "bins": {"Y": 10, "LON": 2},
    }
    normalised = app._normalise_case_payload(payload)
    assert normalised["x"] == "temp"
    assert normalised["groupBy"] == ["latitude:2"]
    assert normalised["orderBy"][0]["col"] == "time"
    assert normalised["bins"] == {"y": 10, "lon": 2}


def test_store_ensure_case_insensitive(tmp_path):
    ds = xr.Dataset(
        {"TEMP": ("N_POINTS", np.arange(3.0))},
        coords={
            "N_POINTS": np.arange(3),
            "LATITUDE": ("N_POINTS", np.full(3, 10.0)),
            "LONGITUDE": ("N_POINTS", np.full(3, 20.0)),
            "TIME": ("N_POINTS", np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")),
        },
    )
    path = tmp_path / "legacy.nc"
    ds.to_netcdf(path)

    store = DatasetStore()
    opened = store.open_dataset("legacy", str(path), case_insensitive=False)
    assert "TEMP" in opened.data_vars
    store.ensure_case_insensitive("legacy")
    lower = store.get("legacy")
    assert "temp" in lower.data_vars
    assert store.is_case_insensitive("legacy")
