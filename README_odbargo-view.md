# ODBArgo View Plugin — README

A lightweight **viewer/analytics companion** for `odbargo-cli`. It keeps heavy deps (xarray/pandas/matplotlib/h5netcdf/h5py) out of the main CLI and speaks the same single‑port WebSocket contract via the CLI bridge. You can also run everything offline from the CLI REPL using `/view ...` slash commands.

---

## Install

```bash
# recommended: venv
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# install the view module with optional pandas
pip install -e .[view]
```

> PyInstaller users: build **two** one‑file binaries and ship together: the lean `odbargo-cli` and the heavy `odbargo-view`.

---

## Run

### A) With the main CLI (preferred)

```bash
# start the CLI server + REPL
python odbargo-cli.py --port 8765   # default 8765

# (optional) force/point to the viewer binary
# python odbargo-cli.py --plugin view --plugin-binary ./dist/odbargo-view
```

The CLI will spawn/bridge the viewer on demand the first time you issue a `/view ...` command. The same WS port is used for platform apps (e.g., APIverse) and the local REPL.

### B) Standalone smoke test

```bash
# developer mode — run the plugin by itself (stdin/stdout NDJSON)
python -m odbargo_view
```

---

## Quickstart (Slash in CLI REPL)

```text
argo> /view open /tmp/argo_data.nc as ds1
argo> /view list_vars ds1
argo> /view preview ds1 --cols TIME,PRES,TEMP \
       --filter "PRES BETWEEN 25 AND 100" --order TIME:asc --limit 500
argo> /view plot ds1 timeseries --x TIME --y TEMP --out temp_ts.png
argo> /view export ds1 csv --cols TIME,LATITUDE,LONGITUDE,PRES,TEMP --out ds1_subset.csv
```

---

## Commands

### `open`

```
/view open <path> [as <datasetKey>]
```

* Engines: prefers **h5netcdf**, falls back to netcdf4/default.

### `list_vars`

```
/view list_vars [<datasetKey>]
```

* Prints coordinates and variables with dtypes/shapes.

### `preview`

```
/view preview <datasetKey> [--cols A,B,...] [--filter <DSL>] [--json-filter <JSON>]
                    [--bbox x0,y0,x1,y1 | --box ...] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                    [--order COL[:asc|desc]] [--cursor N] [--limit N]
                    [--trim-dims]
                    [as <subsetKey>]
```

**Notes**

* Keeps coordinate/dimension columns by default when you narrow `--cols`; use `--trim-dims` to drop them.
* Pagination with `--cursor` preserves the **last displayed** columns unless you pass new `--cols`.
* `--bbox/--start/--end` layer on top of `--filter` and are persisted when saving `as <subsetKey>`.

### `plot`

```
/view plot <datasetKey> <timeseries|profile|map>
           [--x COL] [--y COL]
           [--group-by COL[,COL2...]] [--agg mean|median|max|min|count]
           [--filter <DSL>] [--order COL[:asc|desc]] [--limit N]
           [--size W×H] [--dpi N] [--title "..."] [--grid] [--invert-y]
           [--cmap NAME]  # map/color helpers
           [--out <png>]
```

**Grouping & aggregation**

* `--group-by WMO` → one series per WMO.
* `--group-by PRES:10.0` → numeric binning (10‑dbar bins) then one series per bin.
* `--group-by TIME:1D` → resample along time (single series). Combine with others for multi‑series (e.g., `WMO,TIME:1D`).
* `--agg` applies the reducer per series (`mean` default).

**Display**

* Without `--out`, the CLI shows a non‑blocking local window **if** matplotlib is available; otherwise it saves to a temp file and opens with the OS default viewer.

### `export`

```
/view export <datasetKey> csv [--cols A,B,...] [--filter <DSL>]
                             [--order COL[:asc|desc]] [--limit N]
                             [--out <csv>]
```

* Streams CSV over the same WS as **binary**; the CLI writes chunks to `--out`.

### `close`

```
/view close <datasetKey>
```

---

## Common Options

* **Filtering**: `--filter` (DSL), `--json-filter` (structured), `--bbox/--box`, `--start/--end`.
* **Projection**: `--cols` (keeps coords unless `--trim-dims`).
* **Paging/Order**: `--limit`, `--cursor`, `--order COL[:asc|desc]`.
* **Grouping** (plot): `--group-by`, `--agg`.
* **Output**: `--out <file>` (PNG/CSV); omit to preview.

---

## Environment

* `ODBARGO_VIEW_BINARY` — path to the viewer binary for the CLI to spawn (otherwise autodetect adjacent binary/module).
* `ODBARGO_PLUGIN_TOKEN` — optional shared secret for viewer self‑registering over the CLI’s WS.
* `ODBARGO_DEBUG=1` — enable verbose bridge diagnostics (plugin ⇄ CLI). When off/absent, debug frames are skipped.

---

## Notes

* Heavy deps live **only** in this module; the main `odbargo-cli` stays small.
* One WS port for everything. Images/CSV stream as WS **binary** frames; no base64.
* Safe, closed‑world filter DSL; coordinates allowed even when a subset projection narrows variables.
