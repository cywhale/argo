from pathlib import Path

try:
    from argo_cli import ParsedCommand, exit_code_for_error, parse_slash_command, split_commands
except ImportError:  # fallback to renamed package
    from cli import ParsedCommand, exit_code_for_error, parse_slash_command, split_commands


def test_split_commands_handles_semicolons_and_quotes():
    raw = "/view open data.nc as ds1; /view preview ds1 --filter 'PRES>5'"
    commands = split_commands(raw)
    assert commands == ["/view open data.nc as ds1", "/view preview ds1 --filter 'PRES>5'"]


def test_parse_open_sets_dataset_key_from_path(tmp_path):
    raw = f"/view open {tmp_path / 'sample.nc'}"
    parsed = parse_slash_command(raw, None)
    assert isinstance(parsed, ParsedCommand)
    assert parsed.request_type == "view.open_dataset"
    assert parsed.request_payload["datasetKey"] == "sample"


def test_parse_preview_with_options():
    raw = "/view preview ds1 --cols TIME,PRES --limit 200 --order TIME:asc --filter 'PRES >= 10' --cursor 100"
    parsed = parse_slash_command(raw, None)
    assert parsed.request_type == "view.preview"
    payload = parsed.request_payload
    assert payload["datasetKey"] == "ds1"
    assert payload["columns"] == ["TIME", "PRES"]
    assert payload["limit"] == 200
    assert payload["cursor"] == "100"
    assert payload["orderBy"] == [{"col": "TIME", "dir": "asc"}]
    assert payload["filter"]["dsl"] == "PRES >= 10"


def test_parse_preview_with_spatiotemporal_filters():
    raw = "/view preview ds1 --bbox 119,20,122,23 --start 2007-12-01 --end 2012-01-01"
    parsed = parse_slash_command(raw, None)
    payload = parsed.request_payload
    assert payload["bbox"] == [119.0, 20.0, 122.0, 23.0]
    assert payload["start"] == "2007-12-01"
    assert payload["end"] == "2012-01-01"


def test_parse_preview_with_order_option():
    raw = "/view preview ds1 --order TIME:desc,LONGITUDE"
    parsed = parse_slash_command(raw, None)
    assert parsed.request_payload["orderBy"] == [
        {"col": "TIME", "dir": "desc"},
        {"col": "LONGITUDE", "dir": "asc"},
    ]


def test_parse_preview_with_box_alias():
    raw = "/view preview ds1 --box 100,-20,179.99,25"
    parsed = parse_slash_command(raw, None)
    assert parsed.request_payload["bbox"] == [100.0, -20.0, 179.99, 25.0]


def test_parse_plot_with_output_and_flags(tmp_path):
    raw = (
        "/view plot ds1 timeseries --x TIME --y DOXY "
        "--size 800x600 --dpi 120 --grid --cmap plasma "
        "--legend --legend-loc lower-left --legend-fontsize 10 "
        "--point-size 24 --group-by PRES:25 --agg mean "
        "--order TIME:desc --out plot.png"
    )
    parsed = parse_slash_command(raw, None)
    assert parsed.request_type == "view.plot"
    style = parsed.request_payload.get("style")
    assert style["width"] == 800
    assert style["height"] == 600
    assert style["dpi"] == 120
    assert style["grid"] is True
    assert style["cmap"] == "plasma"
    assert style["legend"] is True
    assert style["legend_loc"] == "lower-left"
    assert style["legend_fontsize"] == "10"
    assert style["pointSize"] == 24.0
    assert parsed.out_path == Path("plot.png")
    assert parsed.request_payload["orderBy"] == [{"col": "TIME", "dir": "desc"}]
    assert parsed.request_payload["groupBy"] == ["PRES:25"]
    assert parsed.request_payload["agg"] == "mean"


def test_parse_plot_profile_with_bins_y():
    raw = "/view plot ds1 profile --x TEMP --y PRES --bins y=10 --agg median"
    parsed = parse_slash_command(raw, None)
    assert parsed.request_type == "view.plot"
    assert parsed.request_payload["bins"] == {"y": 10.0}
    assert parsed.request_payload["agg"] == "median"


def test_parse_bins_rejects_unknown_keys():
    raw = "/view plot ds1 profile --x TEMP --y PRES --bins depth=10"
    try:
        parse_slash_command(raw, None)
    except ValueError as exc:
        assert "Unknown --bins key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown --bins key")


def test_exit_code_for_known_errors():
    assert exit_code_for_error("DATASET_OPEN_FAIL") == 3
    assert exit_code_for_error("FILTER_INVALID") == 4
    assert exit_code_for_error("PLOT_FAIL") == 5
    assert exit_code_for_error("PLUGIN_NOT_AVAILABLE") == 10


def test_parse_help_command_with_topic():
    parsed = parse_slash_command("/view help preview", None)
    assert parsed.request_type == "view.help"
    assert parsed.request_payload["topic"] == "preview"


def test_parse_preview_with_subset_alias():
    raw = "/view preview ds1 as ds2 --cols PRES,DOXY --filter 'PRES BETWEEN 25 AND 100'"
    parsed = parse_slash_command(raw, None)
    assert parsed.request_type == "view.preview"
    payload = parsed.request_payload
    assert payload["datasetKey"] == "ds1"
    assert payload["subsetKey"] == "ds2"
    assert payload["columns"] == ["PRES", "DOXY"]
