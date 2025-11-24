# ODB Viz v0.3.0 – Map Plot Patch

**File:** `odbViz_spec_v0_3_0_map_plot_patch.md`  
**Scope:** Map plotting behavior in `plugin.py` (kind = `"map"`)  
**Goal:** Make `odbViz` map plots visually and behaviorally consistent with the
legacy `/mhw plot` implementation in `map_plot.py::plot_map()`, and wire up the
missing spec parameters (`engine/backend`, `vmin/vmax`, bbox/antimeridian
handling, etc).

This document is a **patch spec**: it describes *what is wrong now* and *how to
refactor* `plugin.py` to reuse `map_plot.py` for gridded data (e.g. MHW API) maps, 
while keeping a scatter/grid fallback for generic point data (Argo/Seaglider).

---

## 1. Current Regression Summary

### 1.1 Visual differences

The new `plugin.py` map output (kind = `"map"`) is noticeably worse than the
legacy `/mhw plot map`:

- No coastlines or land shading.
- Default Matplotlib colorbar (position, size, style) instead of the tuned one
  in `map_plot.py`.
- Simple `ax.grid()` instead of the old smart geographic ticks and labels
  (`_smart_geo_ticks`, `_nice_ticks`, Cartopy/Basemap degree formatters).
- Font sizes, tick label density, and overall layout feel “flat” and crowded
  compared to the original.

Root cause: `plugin.py` currently draws maps using **plain Matplotlib**:

- `fig, ax = plt.subplots(...)`
- Either `ax.pcolormesh(...)` or `ax.scatter(...)`
- `fig.colorbar(...)` with default settings
- No Cartopy/Basemap backend usage.

Follow the codes and styles in the legacy `map_plot.plot_map()` functions, 
which encapsulate all the tuned behavior.

### 1.2 Missing / ignored parameters (spec drift)

Several parameters that were part of the older design are **not wired up** in
`plugin.py`’s map branch:

- **Engine / backend**:
  - Legacy `/mhw` had `--map-method {cartopy|basemap|plain}`.
  - `map_plot.plot_map(df, ..., method=...)` fully supports backend selection:
    - `"cartopy"`: PlateCarree projection, coastlines, land polygons.
    - `"basemap"`: Basemap instance with coastlines/fillcontinents.
    - `"plain"`: plain Matplotlib for environments without GIS libs.
    - If None is set, or any error occurs when using engine/backend, fallback to `plain`, refer to `map_plot._backend()`.

  - `plugin.py` currently:
    - Uses engine preference only for xarray IO (NetCDF engine selection).
    - Does **not** pass any `engine` / `method` into the map drawing logic.
- **`vmin` / `vmax`**:
  - `map_plot.plot_map()` already supports `vmin` / `vmax`, plus sensible
    defaults via `_default_cmap_norm(field)` (e.g. `sst_anomaly` → `[-3, 3]`).
  - `plugin.py` does *not* read style or message-level `vmin` / `vmax`.
- **Antimeridian / crossing-zero / bbox mode**:
  - `map_plot.py` has `bbox_mode` and helpers to deal with:
    - Crossing 0° vs crossing 180°.
    - Adjusting projection (`central_longitude`) and splitting grids across the
      180° seam.
  - `mhw_cli` still uses similar concepts (`_bbox_mode()` helpers).
  - `plugin.py` currently:
    - Does not accept `bboxMode` / `bbox_mode`.
    - Does no special handling for 0° / 180° crossing.
    - Just feeds lon/lat into pcolormesh/scatter.
- **Legend / colorbar**:
  - `map_plot.py`:
    - Uses a tuned colorbar layout (orientation, padding, tick fontsize, label).
  - `plugin.py`:
    - For map, always uses `fig.colorbar(...)` with default settings.
    - Legend handling (`legend_loc`) is implemented only for
      `"timeseries"` / `"profile"` and is not used for map (which is OK for
      continuous fields but not for discrete categorical levels).

In short:

> The spec mentions backend, vmin/vmax, bbox/antimeridian, and styles for
> colormaps, but `plugin.py`’s map implementation ignores most of them.

### 1.3 Data semantics hints (patch to original spec)

The viewer can optionally receive **hints** about the data associated with `y`.
These hints are never required, but when present they override auto–detection and
affect how maps are drawn (backend choice, colorbar vs legend, etc.).

#### 1.3.1 Fields

* `gridded` (boolean)
  Hint that the variable referenced by `y` is already on a **regular grid**
  (after any CLI-side binning), e.g. MHW 0.25° fields, WOA23 monthly maps.
  When `gridded: true`:

  * The plugin SHOULD prefer a gridded map backend (`pcolormesh` /
    `map_plot.plot_map`) instead of a pure scatter plot.
  * Antimeridian / 0°-crossing handling and nice map styling are enabled.

* `categorical` (boolean)
  Hint that the variable referenced by `y` is **categorical** (discrete
  classes), e.g. `mhw_category`, flags, QC codes.
  When `categorical: true`:

  * The plugin SHOULD use a discrete colormap and a legend (not a continuous
    colorbar).
  * `vmin` / `vmax` are normally ignored, and category → color mapping is used.

If both hints are omitted, the plugin MAY try to infer the data type, but the
hints take precedence when present.

#### 1.3.2 Slash commands (CLI)

Slash commands MAY expose these hints as flags that apply to the `y` variable:

```bash
/view plot ds1 map --y sst_anomaly   --gridded
/view plot ds1 map --y mhw_category  --categorical
```

The CLI then translates them into the `plot` message sent to the viewer.

#### 1.3.3 WebSocket / NDJSON `plot` messages

In the `plot` message, these hints are carried under `params`:

```jsonc
{
  "op": "plot",
  "datasetKey": "ds1",
  "kind": "map",
  "y": "sst_anomaly",
  "params": {
    "gridded": true,
    "categorical": false,
    "bboxMode": "auto",
    "timeResolution": "monthly"
  },
  "style": {
    "cmap": "coolwarm"
  }
}
```

For a categorical example:

```jsonc
{
  "op": "plot",
  "datasetKey": "ds1",
  "kind": "map",
  "y": "level",
  "params": {
    "gridded": true,
    "categorical": true
  }
}

---

## 2. Desired Target Behavior

### 2.1 Map style should match legacy `/mhw plot`

For map plot:
- **gridded data**, the map produced by `odbViz` **must visually match** the old `/mhw plot` at:
  - Same coastlines / land shading (Cartopy or Basemap).
  - Same tick density and degree-format ticks.
  - Same colorbar or discrete legend layout.

- `map_plot.py::plot_map()` already contains all the tuned backend logic, reuse it.
  - Including correct handling of :
    * Bbox crossing 0° vs 180°.
    * 0–360 vs -180–180 longitude conventions.

- For other not gridded datasets (irregular point data, Argo casts, etc.):
  - `plugin.py` may still use its current scatter / grid logic.

### 2.3 Spec patch for map-related fields

Within the `plot` message for `kind="map"` we standardize:

```jsonc
{
  "style": {
    "cmap": "coolwarm",                 // or null for defaults
    "vmin": -3.0,
    "vmax": 3.0,
    "engine": "basemap",                // or "cartopy" | "plain"
    "grid": true,                       // only for plain scatter maps
  },
  "params": {
    "bboxMode": "auto"                  // "auto" | "crossing-zero" | "antimeridian" | "none"
  },
  "source": "mhw"                       // strongly recommended
}
````

Rules:

* `style.vmin` / `style.vmax`:
  * If absent, follow what `map_plot._default_cmap_norm(field)` applies.
  * If present, they override defaults.

* `params.bboxMode`:
  * `"auto"`: let CLI or plugin choose; for MHW CLI likely sets it explicitly.
  * `"crossing-zero"`: treat bbox as crossing longitude 0°.
  * `"antimeridian"`: treat bbox as crossing 180° east/west.
  * `"none"`: do not apply special seam handling.

The spec does not dictate *how* CLI chooses these; only that `plugin.py` must
respect them and forward to `map_plot.plot_map()`.

---

## 3. Implementation Plan for `plugin.py` (Map Branch)

> **Note:** This section describes high-level structure and helper functions,
> not full code. Codex should follow the structure but is free to adjust naming
> as long as behavior matches.

### 3.1 High-level split (only suggestion for code)

In `_render_plot()`:

```python
elif kind == "map":
    if self._is_gridded_map(message, df):
        # MHW / WOA23-style monthly gridded data → use legacy backend
        png_bytes = self._render_gridded_map_with_legacy_backend(df, message, style, msg_id)
        return png_bytes
    else:
        # Argo / glider / generic points → keep current scatter/grid behavior
        png_bytes = self._render_point_map_with_scatter(df, message, style, msg_id)
        return png_bytes
```

Where:

* `_is_gridded_map(message, df)`: `--gridded` is specified

* `_render_point_map_with_scatter()`:

  * Encapsulates the **current** scatter / `pcolormesh` logic (bins/grid).


---

## 4. Legend / Colorbar Behavior

### 4.1 `legend_loc` in spec vs plugin

* The generic legend handling (`legend_loc` etc.) implemented via
  `_legend_params()` in `plugin.py` applies only to:

  * `"timeseries"`
  * `"profile"`

  and is **already guarded by**:

  ```python
  and kind in {"timeseries", "profile"}
  ```

  This is correct and should remain unchanged.

* For `kind="map"`, consider both continuous/discrecte fields:

  * Continuous fields:

    * Use colorbar only (no legend).
    * Refer to `map_plot.plot_map()`, defer colorbar layout to that function.
  * Discrete categorical field:

    * Consider the consistent legend behavior in plugin.py.

### 4.2 Colorbar style

By delegating to `map_plot.plot_map()`, we inherit:

* Orientation (horizontal/vertical) based on figure aspect.
* `extend='both'` or `extend='neither'` as appropriate.
* Tick label font sizes and padding.

For scatter maps (non-gridded, not using `plot_map()`), plugin should:

* At least set:

  ```python
  cbar.set_label(value_col or y_col)
  ```

* Optionally adjust tick size via `cbar.ax.tick_params(labelsize=...)` using a
  small default (e.g. 9–10) or `style.colorbar_fontsize`.

---

## 5. Checklist for Codex Implementation

1. **Refactor `_render_plot()`**:

   * Introduce `_is_gridded_map()`, `_render_gridded_map_with_legacy_backend()`,
     `_render_point_map_with_scatter()`.
   * Use these for `kind == "map"` branch.

2. **Wire spec parameters**:

   * `style.cmap`, `style.vmin`, `style.vmax` → map_plot and/or scatter branch.
   * `params.engine`, `params.bboxMode`, `params.gridded`, `params.categorical`

3. **For gridded maps, follow `map_plot.plot_map()` coding and styles**:

4. **Keep scatter for non-gridded data**:

   * Clean existing scatter/pcolormesh code into `_render_point_map_with_scatter`.
   * Make it respect `style.cmap`, `style.vmin/vmax`.

5. **Examples for existing CLI usage**:

   * For MHW:

     ```bash
     /mhw --bbox 135,-25,-60,25 \
          --fields sst_anomaly \
          --plot map \
          --plot-field sst_anomaly \
          --start 2007-12 \
          --map-method basemap \
          --cmap coolwarm \
          --vmin -3 --vmax 3 --gridded
     ```

   * For Argo-like usage:

     ```bash
     /view plot ds1 map --y TEMP \
         --agg mean \
         --cmap coolwarm \
         --engine plain
     ```

If all the above items are satisfied, `odbViz`’s map mode will regain the
“elegant” look of the old `/mhw plot` while remaining flexible for other data
sources.

