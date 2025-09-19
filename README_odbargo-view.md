# ODBArgo View Plugin

## Overview
`odbargo_view` is the spec-driven visualization plugin consumed by `odbargo-cli`. It runs as a subprocess, speaks NDJSON over stdin/stdout, and keeps heavy data dependencies (xarray, pandas, matplotlib) out of the lightweight CLI binary. The plugin loads NetCDF datasets, applies safe filters, streams tabular previews, renders plots, and exports CSV slices requested through the single WebSocket channel exposed by `odbargo-cli`.

## Installation
Install the package alongside the CLI in a virtual environment:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[view]  # ensures pandas/matplotlib extras are present
```
Run the subprocess manually for smoke testing with `python -m odbargo_view`; it waits for NDJSON commands on stdin and emits a handshake message when ready.

## Supported Operations
The plugin implements the contract defined in `specs/odbargo_view_spec_v_0.md`:
- `open_dataset` loads a NetCDF file using `h5netcdf` (fallback to default engine) and returns dataset metadata.
- `list_vars` replays coordinates and variables with their shapes and attributes.
- `preview` builds a pandas DataFrame, applies the validated DSL/JSON filters, and streams a bounded row slice (`<=1000` rows) as JSON.
- `plot` renders `timeseries`, `profile`, or `map` plots via matplotlib and streams PNG bytes after a control header.
- `export` writes CSV data into 256 KiB chunks, sending `file_start`, chunk descriptors, and a `file_end` footer containing the SHA-256 digest.
- `close_dataset` releases dataset handles and frees resources.

Errors surface as `{ "op": "error", ... }` envelopes with canonical codes such as `DATASET_OPEN_FAIL`, `FILTER_INVALID`, or `PLOT_FAIL`. The CLI maps these to human-readable messages and exit codes.

## CLI Integration Snapshot
Within the `/view` slash interface, the commands proxy directly to the plugin:
- `/view open data.nc as ds1`
- `/view preview ds1 --cols TIME,PRES,DOXY`
- `/view preview ds1 as ds2 --cols PRES,DOXY --filter "PRES BETWEEN 25 AND 100"`
- `/view preview ds1 --bbox 119,20,122,23 --start 2007-12-01 --end 2012-01-01`
- `/view plot ds1 timeseries --x TIME --y DOXY --out doxy.png`
- `/view export ds1 csv --filter "PRES BETWEEN 25 AND 100" --out subset.csv`
Use `/view help` for a quick reminder of the available verbs.

### Spatial & Temporal Filters
- `--bbox x0,y0,x1,y1` (or the alias `--box`) constrains rows to latitude/longitude ranges, automatically matching the dataset’s coordinate column names.
- `--start` and `--end` clip observations by the `TIME` coordinate; either bound is optional, both can be combined with existing DSL/JSON filters.
- These options apply to `/view preview`, `/view plot`, and `/view export`, layering on top of any subset filters saved earlier.
- Map plots default to plotting `LONGITUDE` (x-axis) vs `LATITUDE` (y-axis); supply `--y <variable>` to colour the grid by a data variable without having to set axes explicitly.

### Plot Display Without `--out`
When a plot command originates from the slash interface without `--out`, the CLI now opens a non-blocking matplotlib window (`figure.raise_window = False`, `plt.show(block=False)`) so a local session can inspect the rendered image without waiting for the binary payload over WebSocket.

### Map Rendering Defaults
- The plugin reshapes Argo fields onto a lon/lat mesh (via pandas pivot tables) before calling `pcolormesh`, so gridded data display correctly.
- If the grid cannot be formed (e.g., scattered observations), the plot falls back to a coloured scatter map.
- Use `--cmap <name>` to pick a matplotlib colormap (default: `viridis`), and `--grid` to overlay a light lat/lon grid.
- `--order <field[:asc|desc]>` sorts records before previewing or plotting (timeseries/profile), ensuring lines follow the intended progression.

### Interactive REPL conveniences
- The CLI REPL now leverages readline (or pyreadline on Windows) so arrow keys recall history and allow in-place editing of previous commands.
- Long-running exports run as background tasks, so the prompt remains responsive while CSV streaming completes.

## Development Notes
- Preview responses are intentionally capped to keep NDJSON lines small enough for asyncio’s pipe reader limits.
- The filtering DSL is parsed via a custom tokenizer/parser—no `eval` or arbitrary function calls—to guard against untrusted input.
- All binary payloads (plots, CSV chunks) are written to `stdout.buffer`; control messages always end with a newline.
- Keep the protocol version (`PLUGIN_PROTOCOL_VERSION`) in sync with the handshake expected by `odbargo-cli` and update the spec file if changes are introduced.
