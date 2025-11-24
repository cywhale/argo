# ODB Viz Spec v0.3.0

**Status:** Draft  
**Module name:** `odbViz` (generic viewer/plotter plugin)

`odbViz` is a general-purpose viewer/plotter module that can be mounted on:

- `odbargo_cli` (Argo / Seaglider / NetCDF/xarray data)
- `odbchat_cli` (MHW and other ODB Web API JSON data)
- Future CLIs using ODB Open APIs

It reuses and extends the `ODBArgo-View Spec v0.2.8`:

- Same transport and handshake (`plugin.hello_ok` + capabilities).
- Same basic dataset lifecycle (`open_dataset`, `preview`, `subset`, `export`).
- Adds **generic `open_records`** for API-derived/tabular data.
- Normalizes **plot kinds** across all data sources:

  - `"timeseries"`
  - `"climatology"` (monthly climatology)
  - `"map"`
  - `"profile"` (generic x–y section, not restricted to depth)


---

## 1. Transport & Bootstrap

### 1.1 Transport

Same as `ODBArgo-View v0.2.8`:

- **Stdio mode:** NDJSON on stdin/stdout.
- **WebSocket mode:** JSON messages (and optional binary frames for image/CSV).

Every control message:

- Has either `type` (for plugin-level messages) or `op` (for operations).
- Has `msgId` (string) that correlates request/response.

### 1.2 Stdio bootstrap

On startup in stdio mode, the plugin MUST emit:

```json
{
  "type": "plugin.hello_ok",
  "pluginProtocolVersion": "1.0",
  "capabilities": {
    "open_dataset": true,
    "open_records": true,
    "preview": true,
    "subset": true,
    "plot": true,
    "export": true
  }
}
````

Notes:

* `open_records` is new in v0.3.0.
  If the plugin does not support it, it MAY omit `open_records` from `capabilities`.
* All other capabilities retain backward-compatible semantics.

### 1.3 WebSocket registration

In WS mode, the plugin uses `plugin.register` / `plugin.register_ok` as in v0.2.8.

The `plugin.register_ok` MUST carry the same `capabilities` object as above.

After registration, all `op` messages described in this spec are delivered over WS.

---

## 2. Datasets, Records, and Dataset Keys

### 2.1 Dataset keys (`datasetKey`)

* A `datasetKey` is an **opaque string chosen by the caller**.

* Typical keys look like `"ds1"`, `"ds2"`, `"argo9wmo"`.

* `datasetKey` is what users see in the CLI:

  ```bash
  /view open tmp/argo_data9wmo.nc as ds1
  /view plot ds1 map --y TEMP ...
  ```

* `odbViz` treats `datasetKey` **purely as a handle**. It MUST:

  * Distinguish datasets by `datasetKey`.
  * Allow multiple datasetKeys to coexist and be referenced independently.

### 2.2 `open_dataset` (file-based datasets)

`open_dataset` keeps the original semantics from `ODBArgo-View v0.2.8`:

```jsonc
{
  "op": "open_dataset",
  "msgId": "m1",
  "path": "/path/to/file.nc",
  "datasetKey": "ds1",
  "enginePreferred": ["netcdf4", "h5netcdf"],
  "caseInsensitive": true
}
```

**Response:**

```jsonc
{
  "op": "open_dataset.ok",
  "msgId": "m1",
  "datasetKey": "ds1",
  "summary": {
    "dims": {
      "TIME": 1234,
      "DEPTH": 50,
      "LATITUDE": 721,
      "LONGITUDE": 1440
    },
    "vars": ["TEMP", "PSAL", "DOXY", "CHLA", "TIME", "LATITUDE", "LONGITUDE"],
    "source": "argo"   // OPTIONAL hint; may be "argo", "seaglider", "woa23", etc.
  }
}
```

Notes:

* `datasetKey` MUST be the same string provided in the request.
* `source` is optional; CLIs are encouraged to set a consistent `source` when
  calling `plot` (see §4.2).

### 2.3 `open_records` (API/tabular datasets)

`open_records` is the generic entry point for in-memory tabular data
(e.g. JSON from Web APIs converted to records or DataFrame rows).

**Request:**

```jsonc
{
  "op": "open_records",
  "msgId": "m2",
  "datasetKey": "ds_mhw_1",
  "source": "mhw",               // REQUIRED. e.g. "mhw", "argo", "seaglider", "woa23", ...
  "schema": {
    "timeField": "date",         // name of column used as time axis (if any)
    "lonField": "lon",           // name of longitude column (if any)
    "latField": "lat",           // name of latitude column (if any)
    "timeResolution": "auto"     // OPTIONAL: "auto" | "hourly" | "daily" | "monthly"
  },
  "records": [
    {"lon": 120.1, "lat": 23.5, "date": "2024-01-01", "sst_anomaly": 2.3, "mhw_category": 3},
    {"lon": 120.2, "lat": 23.6, "date": "2024-01-01", "sst_anomaly": 2.5, "mhw_category": 3}
    // ...
  ]
}
```

Notes:

* `source` MUST be a short identifier describing the origin of the data:

  * `"mhw"` for ODB Marine Heatwave API.
  * `"argo"` for Argo/Seaglider-like records.
  * `"woa23"`, `"currents"`, etc. for future sources.
* `schema` is a **light-weight hint**:

  * `timeField`, `lonField`, `latField` may be omitted if not applicable.
  * If `timeResolution` is omitted or `"auto"`, the plugin MAY infer the resolution
    from actual values in `timeField`.

**Response:**

```jsonc
{
  "op": "open_records.ok",
  "msgId": "m2",
  "datasetKey": "ds_mhw_1",
  "summary": {
    "nRows": 12345,
    "nColumns": 5,
    "timeField": "date",
    "lonField": "lon",
    "latField": "lat",
    "source": "mhw",
    "timeResolution": "monthly"  // plugin may refine from "auto"
  }
}
```

All subsequent operations (`preview`, `subset`, `plot`, `export`) can be applied
to a dataset created via `open_records` exactly as if it were opened with
`open_dataset`.

---

## 3. Preview, Filtering, and Derived Datasets

### 3.1 `preview` (sampling view)

`preview` returns a small table snapshot of a dataset:

```jsonc
{
  "op": "preview",
  "msgId": "p1",
  "datasetKey": "ds1",
  "maxRows": 1000,
  "columns": ["TIME", "PRES", "TEMP", "PSAL"],
  "filter": "PRES >= 0 AND PRES <= 400",
  "bbox": [-100, -30, -20, 30]  // [lon_min, lat_min, lon_max, lat_max]
}
```

**Response:**

```jsonc
{
  "op": "preview.ok",
  "msgId": "p1",
  "datasetKey": "ds1",
  "rows": [
    ["2002-01-01T00:00:00", 10.0, 23.5, 35.1],
    // ...
  ],
  "columns": ["TIME", "PRES", "TEMP", "PSAL"]
}
```

* `filter` is an optional SQL-like expression interpreted by the plugin or its
  data backend (semantics follow v0.2.8).
* `bbox` optionally restricts rows to a lon/lat bounding box.
* `maxRows` is a hard cap; the plugin MAY return fewer rows.

### 3.2 Derived datasets via `subset`

`odbViz` MUST support creating **derived datasets** from existing ones,
with a new `datasetKey`, so that CLI commands like:

```bash
/view preview ds1 as ds2 --filter "PRES > 0" --bbox -100,-30,-20,30
/view plot ds2 map --y TEMP ...
```

remain valid.

This is typically implemented via an explicit `subset` operation:

```jsonc
{
  "op": "subset",
  "msgId": "s1",
  "sourceDatasetKey": "ds1",
  "targetDatasetKey": "ds2",
  "filter": "PRES > 0",
  "bbox": [-100, -30, -20, 30]
}
```

**Response:**

```jsonc
{
  "op": "subset.ok",
  "msgId": "s1",
  "sourceDatasetKey": "ds1",
  "targetDatasetKey": "ds2",
  "summary": {
    "nRows": 3456,
    "nColumns": 4,
    "timeField": "TIME",
    "lonField": "LONGITUDE",
    "latField": "LATITUDE",
    "source": "argo"
  }
}
```

Semantics:

* `targetDatasetKey` MUST become a new handle, independent of `sourceDatasetKey`.
* Subsequent `preview` / `plot` / `export` on `ds2` operate on the filtered subset.
* CLI is free to expose this via `/view preview ds1 as ds2` or `/view subset ds1 as ds2`.

---

## 4. Plot API

`odbViz` exposes a single `plot` operation parameterized by:

* `kind` – type of plot.
* Plot axes / variable fields (`x`, `y`, and optionally `z`).
* Styling (`style`).
* Additional parameters (`params`) such as group-by and aggregation.

### 4.1 Plot kinds

Supported values:

* `"timeseries"` – line plots of value(s) vs time.
* `"climatology"` – monthly climatology plots (12 points per series).
* `"map"` – 2D lon/lat maps (with color field).
* `"profile"` – generic x–y section plots (vertical or horizontal).

### 4.2 Common plot request structure

```jsonc
{
  "op": "plot",
  "msgId": "pl1",
  "datasetKey": "ds1",
  "kind": "map",              // "timeseries" | "climatology" | "map" | "profile"
  "source": "mhw",            // OPTIONAL but recommended: "mhw", "argo", ...

  "x": null,                  // see per-kind semantics below
  "y": "sst_anomaly",         // "variable to plot" from CLI perspective
  "z": null,                  // used in map/profile as color field if needed

  "timeField": "date",        // OPTIONAL hint
  "groupBy": null,            // OPTIONAL group-by field name (e.g. "period_label")

  "style": {
    "cmap": null,             // colormap name; may be source-specific preset
    "vmin": null,
    "vmax": null,
    "title": null,
    "labelX": null,
    "labelY": null,
    "labelZ": null,
    "legend": true,
    "legendLoc": "best",
    "pointSize": 20,
    "grid": true
  },

  "params": {
    "bboxMode": "auto",       // "auto" | "split_antimeridian" | "wrap_0" | "wrap_180"
    "timeResolution": "auto", // "auto" | "hourly" | "daily" | "monthly"
    "agg": null               // optional aggregation description from CLI, if used
  },

  "export": {
    "format": "png",          // "none" | "png" | "svg" | "pdf"
    "filenameHint": "plot_1"
  }
}
```

Notes:

* `source` is a hint used for defaults:

  * When `source="mhw"` and `style.cmap` is null, plugin MAY choose an MHW-specific palette.
  * When `source="argo"`, plugin may choose oceanographic defaults, etc.
* `y` is always the **data variable to plot** from CLI’s perspective:

  * In `timeseries` / `climatology` / `profile`: `y` is plotted on vertical axis (standard).
  * In `map`: `y` is used as the **color field**; axes are lon/lat (see §4.4).

### 4.3 `"timeseries"` plots

Intended for:

* MHW area-mean time series.
* Argo / Seaglider station time series.
* Any value vs time.

Semantics:

* X-axis: `timeField` (if omitted, plugin infers from dataset).
* Y-axis: `y` variable, potentially with aggregation applied by CLI or plugin.
* `groupBy` (if set) creates **multiple series** on the same axes:

  * e.g. `groupBy="period_label"` to compare different multi-year periods.

Example:

```jsonc
{
  "op": "plot",
  "msgId": "ts1",
  "datasetKey": "ds_mhw_ts",
  "kind": "timeseries",
  "source": "mhw",
  "y": "sst_anomaly",
  "timeField": "date",
  "groupBy": "period_label",
  "style": {
    "title": "MHW SST anomaly – multiple periods",
    "labelY": "SST anomaly (°C)"
  }
}
```

The CLI is responsible for:

* Constructing `period_label` when `--periods` is used in `/mhw`.
* Ensuring that all desired periods are concatenated into a single dataset prior
  to plotting.

### 4.4 `"climatology"` (monthly climatology)

`kind="climatology"` is a **monthly climatology mode**:

* It replicates and generalizes the behavior of the legacy
  `plot_month_climatology()` function.
* Input is a time series with a `timeField` (typically `"date"`).

Key behaviors:

1. Plugin derives `month` from `timeField` (`1..12`).
2. For each group (see below) and each `month`, it computes the mean of `y`.
3. It plots one line per group with 12 points (Jan–Dec).

Grouping:

* If `groupBy` is NULL:

  * Single climatology curve for the entire dataset.
* If `groupBy` is a field (e.g. `"period_label"`):

  * One climatology curve per distinct value of that field.

This provides a direct replacement for the old behavior of:

* `/mhw` with `--periods "20100101-20191231,20200101-20250601"`:

  * CLI builds a `period_label` column.
  * `groupBy="period_label"` produces multiple climatology curves.

Example:

```jsonc
{
  "op": "plot",
  "msgId": "clim1",
  "datasetKey": "ds_mhw_clim",
  "kind": "climatology",
  "source": "mhw",
  "y": "sst_anomaly",
  "timeField": "date",
  "groupBy": "period_label",
  "style": {
    "title": "Monthly climatology of SST anomaly (by period)",
    "labelX": "Month",
    "labelY": "SST anomaly (°C)"
  }
}
```

The plugin is responsible for:

* Computing `month` from `timeField` (taking into account `timeResolution`).
* Grouping by `(groupBy, month)` and averaging `y`.
* Using a fixed 12-month x-axis with labels `"Jan".."Dec"` or equivalent.

All heavy **time aggregation** for climatology is performed **inside `odbViz`,
on the local machine where the viewer runs**, not on remote MCP servers.

### 4.5 `"map"` plots

`kind="map"` creates 2D maps:

* X-axis: longitude.
* Y-axis: latitude.
* Color: variable `y` from CLI, internally treated as `z`.

Semantics:

* Plugin uses `schema.lonField` / `schema.latField` (from `open_records`) or
  known coordinate names (`LONGITUDE`, `LATITUDE`) to identify axes.
* For compatibility with existing `/view` usage:

  * CLI passes `--y TEMP` for maps.
  * Inside the viewer, `y` is used as the **color field** (`z`), not the vertical axis.

Example:

```jsonc
{
  "op": "plot",
  "msgId": "map1",
  "datasetKey": "ds_mhw_map",
  "kind": "map",
  "source": "mhw",
  "y": "sst_anomaly",
  "timeField": "date",
  "style": {
    "cmap": "mhw_levels",
    "title": "MHW anomaly map – 2024-01"
  },
  "params": {
    "bboxMode": "auto"
  }
}
```

`bboxMode`:

* `"auto"` (default): plugin may auto-handle antimeridian (0°/180°) cases
  based on data extent.
* `"split_antimeridian"` etc.: optional hints from CLI if it has already
  preprocessed longitudes.

### 4.6 `"profile"` plots

`kind="profile"` draws generic 1D section plots:

* X-axis: `x` variable.
* Y-axis: `y` variable.
* It is **not restricted to vertical depth profiles**:

  * It can be a traditional `(PSAL vs PRES)` vertical profile.
  * Or a `TEMP vs LONGITUDE` section across a longitude range.
  * Or any x–y combination the CLI/user chooses.

Example (vertical profile):

```jsonc
{
  "op": "plot",
  "msgId": "prof1",
  "datasetKey": "ds2",
  "kind": "profile",
  "source": "argo",
  "x": "PSAL",
  "y": "PRES",
  "style": {
    "title": "Salinity profile",
    "labelX": "PSAL (psu)",
    "labelY": "PRES (dbar)"
  }
}
```

Example (longitude section):

```jsonc
{
  "op": "plot",
  "msgId": "prof2",
  "datasetKey": "ds_lon_section",
  "kind": "profile",
  "source": "mhw",
  "x": "LON",
  "y": "sst_anomaly",
  "style": {
    "title": "SST anomaly vs Longitude"
  }
}
```

If the dataset does not contain requested fields for `x`/`y`, the plugin MUST
return a `plot.error` message (see §5).

---

## 5. Errors, Limits, and Debug Messages

### 5.1 Error responses

On any failure, `odbViz` MUST reply with:

```jsonc
{
  "op": "<original_op>.error",
  "msgId": "pl1",
  "error": "PLOT_INVALID",
  "detail": "Missing 'y' field 'TEMP' in dataset ds1."
}
```

Common error codes (non-exhaustive):

* `DATASET_NOT_FOUND`
* `PLOT_INVALID`
* `PLOT_UNSUPPORTED_KIND`
* `FIELD_NOT_FOUND`
* `SUBSET_INVALID_FILTER`

### 5.2 Preview limits

As in `ODBArgo-View v0.2.8`, the plugin SHOULD:

* Impose conservative caps on `preview` rows (e.g. default 1k rows, hard cap 10k).
* Document any caps in implementation notes (not required in this spec).

### 5.3 Debug messages

When a debug environment variable is set (e.g. `ODB_VIZ_DEBUG=1`), the plugin
MAY emit debug frames:

```jsonc
{
  "op": "debug",
  "msgId": "pl1",
  "message": "Detected timeField=date, inferred timeResolution=monthly"
}
```

Debug messages are advisory and MUST NOT replace required `*.ok` / `*.error`
responses.

---

## 6. Backward Compatibility

* `odbViz` is a drop-in replacement for `odbargo_view` for existing `odbargo_cli`
  usages:

  * `open_dataset`, `preview`, `subset`, `plot` maintain their semantics.
  * CLI commands such as:

    ```bash
    /view open foo.nc as ds1
    /view preview ds1 as ds2 --filter ...
    /view plot ds2 profile --x PSAL --y PRES --agg mean ...
    /view plot ds1 map --y TEMP --filter "PRES >= 0 AND PRES <= 50" --agg mean
    ```

    remain valid.

* Additional capabilities (`open_records`, `climatology`, generic `source`) are
  strictly additive.

* For new consumers (e.g. `odbchat_cli`), `odbViz` provides a unified plot API
  for MHW, Argo, Seaglider, WOA23 and other ODB Open APIs.




