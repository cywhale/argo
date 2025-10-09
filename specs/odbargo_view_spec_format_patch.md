# ODBArgo-View Compatibility Patch – Case-Insensitive Variables (Draft)

> **Purpose** – Align the CLI/viewer with ERDDAP tabledap outputs where all
> columns are delivered in lower case and no longer expose CF-style dimension
> variables. We need the CLI and viewer to behave consistently for both legacy
> (dimension-aware, upper-case) and new (flat, lower-case) NetCDF downloads.

---

## Background – NetCDF formats we see today

### Legacy downloads (via argopy)

```
<xarray.Dataset>
Dimensions:  (N_POINTS: 10_691)
Coordinates:
    LATITUDE   (N_POINTS) float64
    LONGITUDE  (N_POINTS) float64
    TIME       (N_POINTS) datetime64[ns]
  * N_POINTS   (N_POINTS) int64
Data variables: CONFIG_MISSION_NUMBER, DOXY, TEMP, ... (mixed case)
```

* Dimension coordinates exist and are upper-case (`LATITUDE`, `LONGITUDE`, `TIME`).
* Data variables retain their canonical upper-case Argo names.
* Consumers can rely on CF-style dimensions and case-sensitive lookups.

### Direct ERDDAP tabledap downloads (current default)

```
<xarray.Dataset>
Dimensions:           (row: 60_638)
Dimensions w/o coords: row
Data variables: wmo_inst_type, time, longitude, latitude, ..., profile_temp_qc (all lower-case)
Attributes: cdm_profile_variables: cycle_number, latitude, longitude, time, ...
```

* No explicit dimension coordinates; everything arrives as table columns.
* All variable names are lower-case (`longitude`, `latitude`, `time`, `temp`, ...).
* Additional metadata (`cdm_*` attributes) describe which columns should be treated as coordinates.

The goal of this patch is to support both structures transparently, without
breaking existing front-end contracts or the `/view` websocket protocol.

---

## 1. CLI flag / environment switch

**Option name**: `--case-insensitive-vars` (default: `True`).

* Available on `odbargo-cli` (both REPL and `--plugin` modes).
* Also configurable via env variable `ODBARGO_CASE_INSENSITIVE_VARS=0|1`.
* The effective flag is stored on the CLI app and gets forwarded to the viewer
  (`view.open_dataset`, etc.).

### Behaviour when **True** (default)

* All incoming variable names from slash commands (`--x`, `--y`, `--cols`,
  filters, orderBy, groupBy, bins, etc.) are normalised to lower-case before
  they hit the plugin.
* The plugin exposes dataset columns/coordinates in lower-case regardless of
  how they appear in the NetCDF file, so downstream operations always see
  lower-case names.
* Preview responses also return lower-case column names to keep the UI/REPL
  consistent.

### Behaviour when **False**

* Current behaviour is preserved. Variable names and dataset columns retain
  their original case (legacy compatibility mode).
* Any lookups remain case-sensitive. Users typing upper-case variable names
  against lower-case datasets will still receive `COLUMN_UNKNOWN` errors.

---

## 2. Dataset normalisation inside the viewer plugin

When `case_insensitive_vars=True` is in effect:

1. The plugin normalises the columns of the incoming `xarray.Dataset` to
   lower-case (both data variables and coordinates). The lower-case mapping is
   used for the remainder of the request lifecycle.
2. A reverse lookup map is maintained should we want to surface an original
   name (e.g. in error messages).
3. Any columns added by preview/plot/export pipelines follow the lower-case
   convention as well.

### Dimension fallback for flattened NetCDF files

* If the dataset lacks dimension coordinates and the expected columns
  (`longitude`, `latitude`, `time`) exist, the plugin fabricates synthetic
  coordinates backed by those columns so downstream code can keep using the
  familiar dimension names.
* If neither dimension coordinates nor the expected columns are present, the
  dataset is considered invalid for our workflow: the plugin returns a warning
  (`PLOT_FAIL`/`DATASET_OPEN_FAIL`) and skips further processing.
* Legacy, dimension-aware files continue to use the original coordinates
  without modification.

---

## 3. API / contract updates

* `specs/odbargo_view_spec_v_0_2_1.md` will note the flag, indicating that
  all variable references become case-insensitive by default.
* Slash command help (both English and Traditional Chinese) should mention the
  new flag.
* Error messages when a column is missing should include the normalised name
  and, if available, hint at the original case (e.g. `Column 'psla' not found;
  did you mean 'PSLA'?`).

Compatibility note: the websocket contract (`view.*` message schema) remains
unchanged. Normalisation happens internally so existing front-end clients do
not need to adjust payloads.

---

## 4. Outstanding questions / follow-up

* Should we expose the case-insensitive behaviour as part of the viewer’s
  `capabilities` handshake so older CLIs can detect compatibility? (Optional.)
* For persisted subsets: normalise stored column lists as well so subsequent
  `/view preview ds2` calls continue to work after reopening.
* Validate filters/order-by definitions after normalisation to surface early
  errors instead of waiting for runtime errors.

---

## Implementation status

*Pending* – This document only captures the desired behaviour; code changes
will follow once the spec is agreed upon.
