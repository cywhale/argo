# ODBArgo-View Spec v0.2.8

**Purpose:** Replace the v0.2.1 base spec and the case-format patch with a single,
current contract for the downloader + viewer workflow. The FastAPI app inside
`argo_app/` is deprecated; the canonical implementation now lives entirely in
`cli/` (FastAPI-free websocket server) and `odbargo_view/` (viewer subprocess or
plugin). This document covers everything the CLI, the viewer, and any front-end
clients must agree on.

> Changes vs v0.2.1: case-insensitive datasets on by default, ERDDAP flattened
> files are supported, the viewer can self-register over WS, preview persists
> subsets, and plotting/export functionality has grown additional switches
> (`groupBy`, `bins`, `params`, richer legends, etc.).

---

## 0. Scope & Goals

**In scope**

* Source of truth for `odbargo-cli` ↔ `odbargo_view` ↔ WS-client behaviour.
* Single-port WebSocket workflow: JSON control frames + binary frames for
  plots/exports.
* Viewer as an independent Python package (`odbargo_view`), invokable either as
  a stdio subprocess (`python -m odbargo_view`) or as a WS peer that connects
  back to the CLI (`ODBARGO_CLI_WS=ws://127.0.0.1:8765 python -m odbargo_view`).
* Core features: open/list/close datasets, bounded previews with pagination and
  subset persistence, filtering (DSL + JSON + bbox/time helpers), plotting
  (timeseries/profile/map), CSV export streaming.

**Out of scope**

* APIverse/preact UI layers and any FastAPI services (they can consume the WS
  contract but are not specified here).
* Argo downloader logic (unchanged besides sharing the WS port).

**Principles**

* **One port** for both download and view traffic.
* **Binary-first**: PNGs/CSVs are streamed as binary WS frames, never base64.
* **Case-insensitive datasets by default** to match ERDDAP flat files, with a
  CLI flag/env to fall back to legacy case-sensitive behaviour.
* **Spec-first**: all fields and message types are versioned; filters rely on a
  bounded, audited DSL rather than `eval`.
* **Chunked work**: preview/exports/page cursors guard memory usage, and long
  operations can be cancelled by closing the WS.

---

## 1. High-Level Architecture

```
+------------------+                  +-------------------------+
| Front-end (WS)   |  JSON + binary   | Main CLI: odbargo-cli   |
|  (UI/Automations)| <--------------> | - Single WS server      |
+------------------+                  | - Slash REPL / jobs     |
                                       | - Plugin bridge         |
                                       +-----------+-------------+
                                                   |
                             NDJSON + binary (stdio)  or  WS self-register
                                                   |
                                       +-----------v-------------+
                                       | Plugin: odbargo_view    |
                                       | - xarray/pandas/h5netcdf|
                                       | - matplotlib export     |
                                       +-------------------------+
```

* CLI owns the websocket server (`ws://localhost:<port>`, default 8765).
* Viewer is spawned on-demand. Preferred path:
  * CLI first tries to connect to a WS-registered plugin (spawned via
    `_adjacent_view_binary` or `python -m odbargo_view` with `ODBARGO_CLI_WS`).
  * If WS registration does not happen within the timeout, CLI falls back to
    the stdio bridge (`PluginClient` running NDJSON over pipes).

---

## 2. Versioning & Handshakes

### 2.1 Protocol versions

* `wsProtocolVersion`: `1.0`
* `pluginProtocolVersion`: `1.0`

### 2.2 WS handshake (Front-end ⇄ Main CLI)

**Request**

```json
{
  "type": "hello",
  "from": "apiverse",
  "wanted": ["download", "view"],
  "wsProtocolVersion": "1.0"
}
```

**Response**

```json
{
  "type": "hello_ok",
  "capabilities": {
    "download": true,
    "view": true,
    "binaryFrames": true
  },
  "wsProtocolVersion": "1.0"
}
```

`ping`/`pong` are also supported to keep idle connections alive.

### 2.3 CLI ⇄ Plugin bootstrap (stdio mode)

* CLI spawns `odbargo_view` as a stdio subprocess (`PluginClient`) on the first
  `/view` command when no WS plugin is registered.
* Plugin must emit one NDJSON line:

```json
{
  "type": "plugin.hello_ok",
  "pluginProtocolVersion": "1.0",
  "capabilities": {
    "open_dataset": true,
    "list_vars": true,
    "preview": true,
    "plot": true,
    "export": true,
    "subset": true
  }
}
```

### 2.4 Plugin self-registration over WS (preferred mode)

* CLI listens for `plugin.register` messages on the same WS port. It enforces an
  optional token (`ODBARGO_PLUGIN_TOKEN`).
* Viewer launched with `ODBARGO_CLI_WS=ws://127.0.0.1:8765` connects back using
  `websockets.connect(...)` and sends:

```json
{
  "type": "plugin.register",
  "pluginProtocolVersion": "1.0",
  "capabilities": {
    "open_dataset": true,
    "list_vars": true,
    "preview": true,
    "plot": true,
    "export": true,
    "subset": true
  },
  "token": "<optional shared secret>"
}
```

* CLI replies with `plugin.register_ok`. After that:
  * Front-end `view.*` messages are forwarded verbatim (with `msgId=requestId`)
    to the plugin WS.
  * Plugin replies with the same NDJSON op names it would use on stdio.
  * Plugin sends PNG/CSV bytes as binary WS frames following the control JSON.

### 2.5 Tokens

* CLI reads `ODBARGO_PLUGIN_TOKEN`; if set, both plugin registration and WS
  bridge requests must supply the same token (`token` field, or `Authorization`
  header for WS clients).

---

## 3. CLI runtime & flags

* `odbargo-cli` arguments:
  * `--port <int>` (default `8765`).
  * `--insecure` (disable SSL verification for ERDDAP downloads).
  * `--plugin {auto,view,none}` and `--plugin-binary <path>` control how the
    viewer is launched.
  * `--case-insensitive-vars / --no-case-insensitive-vars`, default derived from
    `ODBARGO_CASE_INSENSITIVE_VARS` (default `true`). When enabled, slash commands
    auto-lowercase dataset keys (`--x`, `--cols`, `--group-by`, etc.) before
    they are forwarded. The flag also sets `caseInsensitive` on `view.open_dataset`.
* Slash REPL mirrors the WS protocol: `/view open`, `/view preview`, `/view plot`,
  `/view export csv`, `/view close`, `/view help`.
* CLI normalises payloads when case-insensitive mode is on: `columns`,
  `orderBy[].col`, `groupBy` items (`TIME:1D`, `WMO:10`), `bins`, `filter`
  JSON fields, `--bbox` arguments, `subsetKey`, etc.

---

## 4. WebSocket Message Types (Front-end ⇄ CLI)

Every request must have a `requestId` string. Responses echo the same id.

### 4.1 Dataset lifecycle

* **Open dataset**

```json
{
  "type": "view.open_dataset",
  "requestId": "r1",
  "path": "/tmp/argo.nc",
  "datasetKey": "ds1",
  "enginePreference": ["h5netcdf","netcdf4"],
  "caseInsensitive": true   // optional; defaults to CLI flag
}
```

**Response**

```json
{
  "type": "view.dataset_opened",
  "requestId": "r1",
  "datasetKey": "ds1",
  "summary": {
    "dims": {"n_points": 304919},
    "coords": [{"name":"longitude","dtype":"float64","size":304919,"attrs":{...}}, ...],
    "vars": [{"name":"pres","dtype":"float32","shape":[304919],"attrs":{...}}, ...]
  }
}
```

* **List variables/coords**

```json
{"type":"view.list_vars","requestId":"r2","datasetKey":"ds1"}
```

**Response**

```json
{"type":"view.vars","requestId":"r2","datasetKey":"ds1","coords":[...],"vars":[...]}
```

* **Close dataset**

```json
{"type":"view.close_dataset","requestId":"r3","datasetKey":"ds1"}
```

**Response**: `view.dataset_closed`.

### 4.2 Preview (bounded table + pagination)

```json
{
  "type": "view.preview",
  "requestId": "r4",
  "datasetKey": "ds1",
  "subsetKey": "warm_surface",        // optional, registers a subset
  "columns": ["time","pres","doxy"],
  "filter": {"dsl":"pres BETWEEN 25 AND 100"},
  "bbox": [-60,-40,-50,-30],          // or "box":"lon0,lat0,lon1,lat1"
  "start": "2018-01-01",
  "end": "2020-01-01",
  "orderBy": [{"col":"time","dir":"asc"}],
  "cursor": "0",                      // offset for pagination
  "limit": 500,
  "trimDimensions": false
}
```

**Response**

```json
{
  "type":"view.preview_result",
  "requestId":"r4",
  "datasetKey":"ds1",
  "columns":["time","pres","doxy"],
  "rows":[["2019-01-01T00:00:00Z",26.1,210.0], ...],
  "limitHit": true,
  "nextCursor": "500",
  "subsetKey": "warm_surface"
}
```

Notes:

* Preview enforces `MAX_PREVIEW_ROWS=1000` and `MAX_PREVIEW_COLUMNS=16`.
* When `subsetKey` is supplied, CLI remembers it as the most recent dataset key
  so `/view preview warm_surface` works immediately.
* `trimDimensions=true` prevents coordinate vars from being auto-added when the
  user picks a smaller column list.

### 4.3 Subset persistence

* Each subset stores:
  * Filter specs applied when it was created (includes bbox/time helpers).
  * Projected columns (if user explicitly chose them).
  * Case-insensitive flag inherited from the parent dataset.
* Subsequent `view.preview`/`view.plot`/`view.export` requests can reference the
  subset key as `datasetKey`.

### 4.4 Plot (PNG returned as binary)

**Request**

```json
{
  "type": "view.plot",
  "requestId": "r5",
  "datasetKey": "ds1",
  "kind": "timeseries",                      // "timeseries" | "profile" | "map"
  "x": "time",
  "y": "doxy",
  "z": "temp",                               // optional (map)
  "groupBy": ["platform_number", "time:1M"], // column or column:bin-width, or TIME:<freq>
  "agg": "median",                           // mean|min|max|median|sum|count
  "bins": {"lon":0.5,"lat":0.5},             // map binning
  "limit": 2000,
  "orderBy": [{"col":"time","dir":"asc"}],
  "filter": {"json":{"and":[{">=":["pres",25]},{"<=":["pres",100]}]}},
  "style": {
    "width": 900,
    "height": 500,
    "dpi": 120,
    "title": "DOXY (25–100 dbar)",
    "cmap": "viridis",
    "marker": ".",
    "legend": true,
    "legend_loc": "bottom",
    "legend_fontsize": 9,
    "grid": true,
    "pointSize": 40,
    "invert_y": true
  },
  "params": {                               // optional wrapper used by some UIs
    "style": {"alpha": 0.8},
    "groupBy": "wmo:1"                       // string gets normalised into a list
  }
}
```

**Responses**

1. Control header: `{"type":"plot_blob","requestId":"r5","contentType":"image/png","size":254321}`
2. Immediately one binary WS frame with PNG bytes.

### 4.5 Export CSV

**Request**

```json
{
  "type":"view.export",
  "requestId":"r6",
  "datasetKey":"ds1",
  "format":"csv",
  "columns":["time","latitude","longitude","pres","doxy"],
  "filename":"ds1_export.csv",
  "orderBy":[{"col":"time","dir":"asc"}],
  "filter":{"dsl":"doxy > 0"}
}
```

**Server stream**

1. `{"type":"file_start","requestId":"r6","contentType":"text/csv","filename":"ds1_export.csv"}`
2. N binary frames (CSV chunks, 256 KiB each). Optional `{"type":"file_chunk","size":...}` control frames precede each chunk.
3. `{"type":"file_end","requestId":"r6","sha256":"...","size":1234567}`

### 4.6 Errors

```json
{
  "type":"error",
  "requestId":"rX",
  "code":"FILTER_INVALID",
  "message":"Unknown identifier: prs",
  "hint":"Did you mean pres?",
  "details":{"position":8}
}
```

Canonical codes: `DATASET_OPEN_FAIL`, `UNSUPPORTED_ENGINE`, `DATASET_NOT_FOUND`,
`FILTER_INVALID`, `FILTER_UNSUPPORTED_OP`, `COLUMN_UNKNOWN`,
`PREVIEW_TOO_LARGE`, `ROW_LIMIT_EXCEEDED`, `BAD_BINS`, `BAD_GROUPBY`,
`PLOT_FAIL`, `EXPORT_FAIL`, `PLUGIN_NOT_AVAILABLE`, `PLUGIN_UNAVAILABLE`,
`INTERNAL_ERROR`.

---

## 5. CLI ⇄ Plugin NDJSON Protocol

Transport: one JSON line per message on stdout/stdin. Binary payload follows the
control header immediately (no separators).

**Common envelope**

* Every message has `op` and `msgId`.
* CLI reuses the WS `requestId` as `msgId`.

**Operations**

| `op`            | Direction | Payload (highlights)                                                                               |
| --------------- | --------- | --------------------------------------------------------------------------------------------------- |
| `open_dataset`  | CLI → plugin | `{path,datasetKey,enginePreference?,caseInsensitive?}`                                             |
| `open_dataset.ok` | plugin → CLI | `{datasetKey,summary}`                                                                            |
| `list_vars`     | CLI → plugin | `{datasetKey}`                                                                                     |
| `list_vars.ok`  | plugin → CLI | `{datasetKey,coords,vars}`                                                                         |
| `preview`       | CLI → plugin | `{datasetKey,subsetKey?,columns?,filter?,bbox?,start?,end?,cursor?,orderBy?,limit?,trimDimensions?}` |
| `preview.ok`    | plugin → CLI | `{datasetKey,columns,rows,limitHit,nextCursor,subsetKey?}`                                         |
| `plot`          | CLI → plugin | `{datasetKey,kind,x,y,z?,style?,params?,groupBy?,agg?,bins?,limit?,orderBy?,filter?,caseInsensitive?}` |
| `plot_blob`     | plugin → CLI | `{contentType,size}` followed by PNG bytes                                                         |
| `export`        | CLI → plugin | `{datasetKey,format="csv",columns?,filename?,orderBy?,filter?,caseInsensitive?}`                   |
| `file_start`    | plugin → CLI | `{contentType="text/csv",filename}`                                                                |
| `file_chunk`    | plugin → CLI | `{size}` followed by CSV bytes                                                                     |
| `file_end`      | plugin → CLI | `{sha256,size}`                                                                                    |
| `close_dataset` | CLI → plugin | `{datasetKey}`                                                                                     |
| `close_dataset.ok` | plugin → CLI | `{datasetKey}`                                                                                   |
| `error`         | plugin → CLI | `{code,message,hint?,details?}`                                                                    |
| `debug`         | plugin → CLI | Optional breadcrumbs forwarded to stdout when `ODBARGO_DEBUG=1`.                                   |

Binary data is written directly to stdout (stdio mode) or as WS binary frames
in WS mode. Chunk size for CSV export is 256 KiB.

---

## 6. Dataset Normalisation & ERDDAP Compatibility

* CLI flag `--case-insensitive-vars` defaults to `True` (override via
  `--no-case-insensitive-vars` or `ODBARGO_CASE_INSENSITIVE_VARS=0`).
* Viewer tracks a per-dataset case flag. When enabled:
  * All dims, coords, and data variables are lower-cased and conflicts are
    detected early.
  * Preview/list/plot/export responses are emitted in lower-case.
  * CLI lower-cases incoming command fields before forwarding them.
  * Filters and subsets keep a `column_map` so upper-case requests still work.
* ERDDAP tabledap downloads often lack CF coords. When case-insensitive mode is
  on and `longitude`/`latitude`/`time` exist as columns, the plugin promotes
  them to coords automatically. Files missing the trio are rejected with
  `DATASET_OPEN_FAIL`.

---

## 7. Filtering DSL & Helpers

* DSL grammar matches v0.2.1: identifiers, numeric/string literals, comparators
  (`=`, `!=`, `>`, `>=`, `<`, `<=`, `BETWEEN`). Logical operators: `AND`, `OR`,
  parentheses. No functions or regexes. Hard size cap: 2048 chars.
* JSON filters mirror the DSL AST (`{"and":[...]} | {"or":[...]}`) with leaves
  shaped as `{">":[field,value]}` or `{"between":[field,low,high]}`.
* Additional helpers merged into the filter spec:
  * `bbox`/`box`: `[lon0,lat0,lon1,lat1]` filters.
  * `start`/`end`: parsed via `pandas.to_datetime`.
  * `--json-filter` slash option allows front-ends to send exact JSON.
* Plugin merges subset filters + request filters + helpers into a single
  `FilterSpec` before storing subsets.

---

## 8. Preview & Subsets

* Defaults: `DEFAULT_PREVIEW_LIMIT=500`, `MAX_PREVIEW_ROWS=1000`,
  `MAX_PREVIEW_COLUMNS=16`.
* Pagination via `cursor` (string offset). `nextCursor` is returned when more
  rows exist.
* Columns:
  * When user does not specify columns, plugin chooses coords + vars up to the
    column cap (or subset-projected columns).
  * When user narrows columns, coords are preserved unless `trimDimensions=true`.
  * `--order` / `orderBy` ensures deterministic ordering (`mergesort`).
* Subsets:
  * `subsetKey` + filters register a derived dataset inside `DatasetStore`.
  * Stored columns include user-selected columns plus coords unless trimmed.
  * To avoid duplicates, subset keys cannot collide with dataset keys.

---

## 9. Plotting Kinds

### 9.1 Shared options

* `limit` caps the number of rows pulled before plotting.
* `style` supports `width`, `height`, `dpi`, `title`, `alpha`, `marker`,
  `line`, `linewidth`, `grid`, `legend`, `legend_loc` (`top`, `bottom`,
  `left`, `right`, `best`, standard Matplotlib positions), `legend_fontsize`,
  `legend_always`, `max_series`, `pointSize`, `scatterSize`, `cmap`.
* `groupBy` entries:
  * `COL` groups by column values.
  * `COL:<bin>` bins numeric columns (floored to multiples).
  * `TIME:<freq>` uses `pd.Grouper` resampling (e.g., `TIME:1D`, `TIME:1M`).

### 9.2 Timeseries

* Requires both `x` and `y`.
* Optional grouping:
  * Resamples on TIME (using pandas resample) and applies reducer (`mean` by
    default, `count` for frequency, etc.).
  * Additional discrete groups (e.g., `platform_number`, `wmo:5`) plot one series
    per group, limited by `style.max_series`.
* Automatic datetime axis formatting (daily/monthly/yearly tick logic).

### 9.3 Profile

* Plots `x` vs `y` (default `y="pres"`). `style.invert_y` defaults to `true`.
* Supports numeric binning (`groupBy=["lat:1"]`) and `bins.y` to collapse depths
  (e.g., bins of 10 dbar). Reducer applies per-depth per-group.
* Legend placement obeys `legend_loc` and `legend` toggles; spec describes
  multi-row external legends (top/bottom).

### 9.4 Map

* Requires `longitude`/`latitude`. Optional `z` determines colouring.
* With valid `bins.lon` and `bins.lat`, plugin performs a pivoted aggregation
  and renders a `pcolormesh`. Otherwise, falls back to scatter.
* `agg` chooses aggregator for gridded values (`mean`, `median`, `sum`,
  `min`, `max`, `count`).
* `style.pointSize` / `style.scatterSize` control scatter markers.

---

## 10. CSV Export

* Only `csv` is supported today.
* If a subset has restricted columns and the request omits `columns`, export
  inherits the subset column list.
* Export respects the same filters/orderings as preview/plot.
* Plugin streams CSV in 256 KiB chunks, prepended with a `file_chunk` control
  JSON so the CLI can mark the binary expectation. SHA-256 is computed over the
  final file and included in `file_end`.

---

## 11. Resource Limits, Errors, and Debugging

* Preview: 1k rows / 16 columns caps, limit may not exceed 10k even when the
  caller requests more.
* Filters longer than 2 KiB raise `FILTER_INVALID`.
* `ODBARGO_DEBUG=1` makes the plugin emit `{"op":"debug","msgId":"m1","message":"..."}`
  frames. CLI prints them when running interactively.
* `PluginClient` enforces sequential responses; if a plugin sends a response for
  an unknown `msgId`, the CLI ignores it and logs a breadcrumb.
* CLI exit codes map errors to friendly shell statuses:
  * `EXIT_DATASET` (3) for dataset problems.
  * `EXIT_FILTER` (4) for filter/column issues.
  * `EXIT_OUTPUT` (5) for plot/export failures.
  * `EXIT_PLUGIN` (10) for bridge issues.

---

## 12. Testing & Verification

* Unit targets: DSL parser/tokeniser, filter application (`bbox`, `start/end`),
  subset registration, group-by bin helpers.
* Integration sanity checks (manual until CI exists):
  1. `/view open <file>` → `view.dataset_opened`.
  2. `/view preview <key> --limit 5 --bbox ...` (verify pagination + subsets).
  3. `/view plot <key> timeseries --x time --y doxy --group-by TIME:1M` (PNG saved/opened).
  4. `/view export <key> csv --cols time,lat,lon,pres,doxy --out foo.csv`.
* After edits, run:
  * `uvicorn argo_app.app:app --reload --port 8090` (if API wrappers still rely on the spec).
  * `python odbargo-cli.py 5903377` (download), then `/view` commands.
* Document manual verification (curl to `/argo/api/test`, CLI download) in PRs
  until a pytest suite is added under `tests/`.

---

## Appendix A – Slash command cheatsheet

```
/view open <path> [as <key>] [--case-insensitive|--no-case-insensitive]
/view list_vars [<key>]
/view preview <key> [as <subset>] [--cols --filter --json-filter --bbox --start --end --order --cursor --limit --trim-dims]
/view plot <key> <timeseries|profile|map> [--x --y --z --group-by --agg --bins --size --dpi --title --cmap --point-size --legend --bbox --filter --start --end --order --out]
/view export <key> csv [--cols --filter --order --filename --out]
/view close [<key>]
/view help [command]
```

Flags like `--echo-json`, `--json-filter`, `--bbox/--box`, `--group-by`, `--agg`,
`--bins`, `--trim-dims`, `--legend-loc`, `--legend-fontsize`, `--point-size`,
and `--case-insensitive-vars` map 1:1 to the JSON payloads described above.

---

## Appendix B – Environment Reference

* `ODBARGO_CASE_INSENSITIVE_VARS`: `"1"`/`"0"` toggle for CLI default.
* `ODBARGO_VIEW_BINARY`: explicit path to the viewer executable.
* `ODBARGO_PLUGIN_TOKEN`: optional shared secret for plugin registration.
* `ODBARGO_CLI_WS`: when set, viewer connects back to CLI via WS instead of
  stdio (e.g., PyInstaller bundle launching `odbargo-view` next to the CLI).
* `ODBARGO_VIEW_STARTUP_TIMEOUT`: seconds to wait for WS registration before
  CLI falls back to stdio (default 8s).
* `ODBARGO_DEBUG`: enables debug breadcrumbs on both CLI and plugin.

---

This v0.2.8 spec supersedes both `specs/odbargo_view_spec_v_0_2_1.md` and
`specs/odbargo_view_spec_format_patch.md`. All future work (new plot kinds,
cancel/progress hooks, alternate export formats) should extend this file.
