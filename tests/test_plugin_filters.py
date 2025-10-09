import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pandas", reason="pandas is required for plugin filtering tests")

from odbargo_view.plugin import DatasetStore, OdbArgoViewPlugin, PluginError, _build_dataframe


def make_dataset() -> xr.Dataset:
    obs = np.arange(3)
    times = np.array(["2006-01-01", "2009-06-15", "2011-08-01"], dtype="datetime64[ns]")
    longitudes = np.array([118.0, 120.5, 123.0])
    latitudes = np.array([19.5, 21.0, 24.0])
    pres = np.array([5.0, 50.0, 75.0])
    return xr.Dataset(
        data_vars={"PRES": ("obs", pres)},
        coords={
            "obs": obs,
            "TIME": ("obs", times),
            "LONGITUDE": ("obs", longitudes),
            "LATITUDE": ("obs", latitudes),
        },
    )


def test_parse_filter_combines_bbox_and_time():
    plugin = OdbArgoViewPlugin()
    df = _build_dataframe(make_dataset())

    mask, stored_spec = plugin._parse_filter(
        {
            "bbox": [119.0, 20.0, 122.0, 23.0],
            "start": "2007-12-01",
            "end": "2012-01-01",
        },
        df,
    )

    assert mask.sum() == 1
    assert stored_spec is not None
    assert stored_spec.json_spec is not None


def test_parse_filter_invalid_bbox_length():
    plugin = OdbArgoViewPlugin()
    df = _build_dataframe(make_dataset())

    with pytest.raises(PluginError):
        plugin._parse_filter({"bbox": [119.0, 20.0, 122.0]}, df)


def test_resolve_map_columns_defaults():
    plugin = OdbArgoViewPlugin()
    df = _build_dataframe(make_dataset())
    lon_col, lat_col, value_col = plugin._resolve_map_columns(df, {"kind": "map", "y": "PRES"}, None)
    assert lon_col == "LONGITUDE"
    assert lat_col == "LATITUDE"
    assert value_col == "PRES"


def test_build_map_grid_with_regular_coords():
    plugin = OdbArgoViewPlugin()
    ds = xr.Dataset(
        data_vars={
            "DOXY": (("LATITUDE", "LONGITUDE"), np.array([[1.0, 2.0], [3.0, 4.0]])),
        },
        coords={
            "LATITUDE": np.array([-10.0, 0.0]),
            "LONGITUDE": np.array([100.0, 120.0]),
        },
    )
    df = _build_dataframe(ds)
    grid = plugin._build_map_grid(df, "LONGITUDE", "LATITUDE", "DOXY", {"lon": 20.0, "lat": 10.0})
    assert grid is not None
    lon_grid, lat_grid, values = grid
    assert lon_grid.shape == (2, 2)
    assert lat_grid.shape == (2, 2)
    assert values.shape == (2, 2)
    assert pytest.approx(values[0, 0]) == 1.0


def test_apply_order_sorts_dataframe():
    import pandas as pd

    plugin = OdbArgoViewPlugin()
    df = pd.DataFrame({"A": [3, 1, 2], "B": [30, 10, 20]})
    ordered = plugin._apply_order(df, [{"col": "A", "dir": "asc"}])
    assert ordered["A"].tolist() == [1, 2, 3]
    ordered_desc = plugin._apply_order(df, [{"col": "B", "dir": "desc"}])
    assert ordered_desc["B"].tolist() == [30, 20, 10]


def test_json_filter_between_node_is_supported():
    import pandas as pd

    plugin = OdbArgoViewPlugin()
    df = pd.DataFrame({"longitude": [0.0, 5.0, 15.0]})

    mask, _ = plugin._parse_filter(
        {"filter": {"json": {"between": ["longitude", -1.0, 10.0]}}},
        df,
    )

    assert mask.tolist() == [True, True, False]


def test_timeseries_plot_honours_case_insensitive_columns(tmp_path):
    plugin = OdbArgoViewPlugin()

    times = np.array([
        "2003-01-01T00:00:00",
        "2003-01-02T00:00:00",
        "2003-01-03T00:00:00",
        "2003-01-04T00:00:00",
    ], dtype="datetime64[ns]")
    ds = xr.Dataset(
        data_vars={
            "temp": ("row", np.linspace(5.0, 8.0, 4)),
            "pres": ("row", np.array([5.0, 25.0, 50.0, 75.0])),
        },
        coords={
            "row": np.arange(4),
            "time": ("row", times),
            "longitude": ("row", np.linspace(120.0, 121.0, 4)),
            "latitude": ("row", np.linspace(22.0, 24.0, 4)),
        },
    )

    path = tmp_path / "flat_plot.nc"
    ds.to_netcdf(path)

    store = DatasetStore()
    store.open_dataset("flat", str(path), case_insensitive=True)

    plugin.store = store

    png = plugin._render_plot(
        {
            "datasetKey": "flat",
            "kind": "timeseries",
            "x": "TIME",
            "y": "TEMP",
            "groupBy": ["PRES:25"],
            "caseInsensitive": True,
            "filter": {"dsl": "PRES >= 0"},
        }
    )

    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 0


def test_map_plot_case_insensitive_defaults(tmp_path):
    plugin = OdbArgoViewPlugin()

    ds = xr.Dataset(
        data_vars={
            "temp": ("row", np.linspace(5.0, 8.0, 6)),
            "pres": ("row", np.linspace(0.0, 60.0, 6)),
        },
        coords={
            "row": np.arange(6),
            "time": ("row", np.array(["2003-01-01", "2003-01-02", "2003-01-03", "2003-01-04", "2003-01-05", "2003-01-06"], dtype="datetime64[ns]")),
            "longitude": ("row", np.linspace(120.0, 121.0, 6)),
            "latitude": ("row", np.linspace(22.0, 24.5, 6)),
        },
    )

    path = tmp_path / "map_plot.nc"
    ds.to_netcdf(path)

    store = DatasetStore()
    store.open_dataset("map", str(path), case_insensitive=True)
    plugin.store = store

    png = plugin._render_plot(
        {
            "datasetKey": "map",
            "kind": "map",
            "y": "TEMP",
            "filter": {"dsl": "PRES >= 0"},
            "caseInsensitive": True,
        }
    )

    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 0
