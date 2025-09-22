# Install and Use odbargo-view

## Introduction

`odbargo-view` is an **optional viewer module** for `odbargo-cli` that lets you **open, preview, filter, plot, and export** Argo NetCDF data directly from the CLI using `/view ...` commands.

* **Why optional?** We keep the main CLI tiny. The viewer carries heavy scientific deps (xarray/h5py/matplotlib), so it’s installed only when you need it.
* **No viewer executable shipped.** Install from the source archive; the CLI will auto‑start it in *module mode*.

---

## Install

We ship a source archive that includes the viewer package, as mentioned in [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md):

```bash
pip install odbargo-0.x.y.tar.gz
```

Now, when you run `odbargo-cli` and type a `/view ...` command, the viewer starts automatically.

> If you built your own `odbargo-view` executable, you can just run it when `odbargo-cli` is running, or placed it next to `odbargo-cli`, the CLI will prefer that binary and automatically find it.

---

## Quick Start

```bash
odbargo-cli
argo> /view open /path/to/argo_data.nc as ds1
argo> /view list_vars ds1
argo> /view preview ds1 as ds2 --cols TIME,PRES,TEMP --order TIME:desc --limit 1000
argo> /view plot ds1 timeseries --x TIME --y TEMP --out temp.png
```

* `open` registers a dataset alias (e.g., `ds1`).
* `list_vars` prints coordinates and data variables.
* `preview` shows a bounded table (respects `--limit`, `--order`, filters). 
* `plot` streams back a PNG; add `--out` to save, or omit to open with your system viewer.

---

## Commands

All viewer features are available through slash commands in `odbargo-cli`.

### Open

```bash
/view open <path> [as <datasetKey>]
```

* Supports NetCDF via `h5netcdf` (preferred) or `netcdf4` (fallback).

### List variables

```bash
/view list_vars [<datasetKey>]
```

### Preview

```bash
/view preview <datasetKey> \
  [--cols TIME,PRES,TEMP] \
  [--filter "PRES BETWEEN 25 AND 100 AND TEMP > 0"] \
  [--order TIME:asc] [--cursor c1] [--limit 1000] \
  [--start 2010-01-01] [--end 2012-12-31] [--bbox lon1,lat1,lon2,lat2] \
  [--trim-dims]
```

* Shows a bounded table (the plugin enforces upper caps).
* **Filters** can combine: value conditions, time window (`--start/--end`), spatial box (`--bbox`).
* `--trim-dims` removes dimension coords from the output (default: keep TIME/LATITUDE/LONGITUDE).

### Plot

```bash
/view plot <datasetKey> <timeseries|profile|map> \
  --x <XCOL> --y <YCOL> \
  [--group-by COL[:BIN],COL2[:BIN]] [--agg mean|median|count] \
  [--filter …] [--order …] [--limit N] \
  [--cmap viridis] [--size 900x500] [--dpi 120] [--title "..."] \
  [--out plot.png]
```

* **timeseries**: usually `--x TIME --y <var>`; supports grouping/aggregation.
* **profile**: usually `--x <var> --y PRES` (Y is inverted by default).
* **map**: `--x LONGITUDE --y LATITUDE` (optional color by `--y <var>` or use `--cmap`).
* `--group-by` accepts `COL` or `COL:BIN` (e.g., `PRES:10.0`, `TIME:1D`). Use with `--agg` to plot grouped series with legends.

### Export

```bash
/view export <datasetKey> csv \
  [--cols TIME,LATITUDE,LONGITUDE,PRES,TEMP,DOXY] \
  [--filter …] [--order …] [--limit N] \
  --out data.csv
```

* Streams CSV in chunks; the CLI aggregates and writes to `--out`.

---

## Filtering (quick guide)

* DSL supports: `= != > >= < <=`, `BETWEEN a AND b`, `AND`, `OR`, parentheses.
* Time literals use ISO‑8601 strings, e.g., `TIME >= "2019-01-01"`.
* You can also pass a JSON filter with `--json-filter '{"and":[...]}'` (advanced).

Examples:

```bash
--filter "PRES BETWEEN 25 AND 100 AND DOXY > 0"
--start 2010-01-01 --end 2011-01-01 --bbox 120,20,130,26
```

---

## Tips

* Omit `--out` on `/view plot …` to let the CLI open the PNG via your system viewer; add `--out` to save instead.
* `--limit` works for **preview/export/plot** to keep responses small when piping results to a frontend.
* You can furthur save preview result as a dataframe by `preview ds1 as ds2` (ds1 or ds2 are examples for dataset-alias). If you need to overwrite a dataset alias, you should /view close dataset-alias first.

---

## Build (advanced users)

* Clone the repo and use the PyInstaller spec if you want a local viewer executable:

  * `pyinstaller/odbargo-view.spec` (Agg‑only; curated `mpl-data`).
* We do not publish the viewer binary by default due to size; the sdist install is the recommended path.

---

## Troubleshooting

* **Viewer didn’t start?** Ensure you installed the sdist and try again; or set `ODBARGO_DEBUG=1` to see handshake logs.
* **PNG didn’t open?** Add `--out plot.png` and open it manually; the CLI always writes the raw bytes.
* **Slow on first use?** Heavy libs are imported the first time; later commands are faster. On first plot the imports may be slow; that’s normal. Adjust `ODBARGO_VIEW_STARTUP_TIMEOUT` if needed.

---

*For downloader usage and platform integration (Ocean APIverse), see the main* [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md).
