# Install and Use odbargo-view

## Introduction

`odbargo-view` is an **optional viewer module** for `odbargo-cli` that lets you **open, preview, filter, plot, and export** Argo NetCDF data directly from the CLI using `/view ...` commands. The viewer implementation now lives in `odbViz` but keeps the same install/command name for compatibility.

* **Why optional?** We keep the main CLI tiny. The viewer carries heavy scientific deps (xarray/h5py/matplotlib), so it’s installed only when you need it.
* **No viewer executable shipped.** Install from the source archive; the CLI will auto‑start it in *module mode*.

---

## Install

We ship a source archive that includes the viewer package, as mentioned in [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md):

```bash
pip install "odbargo-0.x.y.tar.gz[view]"
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

### Open Argo NetCDF data file

```bash
/view open <path> [as <datasetKey>]
```

* For example: /view open path/argo_data.nc as ds1, then in subsequent commands, use ds1 as dataset alias. 

### List variables

```bash
/view list_vars [<datasetKey>]
```

### Dataset preview

```bash
/view preview <datasetKey> [as <new datasetKey>] \
  [--cols TIME,PRES,TEMP] \
  [--filter "PRES BETWEEN 25 AND 100 AND TEMP > 0"] \
  [--order TIME:asc] [--cursor c1] [--limit 1000] \
  [--start 2010-01-01] [--end 2012-12-31] [--bbox lon1,lat1,lon2,lat2] \
  [--trim-dims]
```

* Shows a bounded table, and this subset preview can be set as another new alias.
* **Filters** can combine: value conditions, time window (`--start/--end`), spatial box (`--bbox`).
* `--trim-dims` removes dimension coords from the output (default: keep TIME/LATITUDE/LONGITUDE).

---

### Filtering (quick guide)

* DSL supports: `= != > >= < <=`, `BETWEEN a AND b`, `AND`, `OR`, parentheses.
* Time literals use ISO‑8601 strings, e.g., `TIME >= "2019-01-01"`.
* You can also pass a JSON filter with `--json-filter '{"and":[...]}'` (advanced).

Examples:

```bash
--filter "PRES BETWEEN 25 AND 100 AND DOXY > 0"
--start 2010-01-01 --end 2011-01-01 --bbox 120,20,130,26
```

---

### Plot

```bash
/view plot <datasetKey> <timeseries|profile|map> \
  --x <XCOL> --y <YCOL> \
  [--group-by COL[:BIN],COL2[:BIN]] [--agg mean|median|count] \
  [--bins lon=0.5,lat=0.5|y=10] \
  [--filter …] [--order …] [--limit N] \
  [--cmap plasma] [--legend] [--legend-loc bottom] [--legend-fontsize small] \
  [--point-size 24] [--size 900x500] [--dpi 120] [--title "..."] \
  [--out plot.png]
```

* **timeseries**: typically `--x TIME --y <var>`; grouping with `--group-by` plus `--agg` plots one line per bucket.
* **profile**: `--x <var> --y PRES` with depth inverted. Add `--bins y=<ΔP>` (e.g. `y=10`) to combine near-depth samples into one point per bin per group; omit to keep exact-depth profiles. Sort the data by `--order` is usually required for profile plot.
* **map**: `--x LONGITUDE --y LATITUDE` (optionally color by `--z <var>` or `--cmap`). Use `--bins lon=…,lat=…` for gridded heatmaps; otherwise you get a scatter with `--point-size` control.
* `--legend`, `--legend-loc`, `--legend-fontsize` help position multi-series legends outside the plot window; bottom/top placements reserve space automatically.

Example (profile with depth bins and discrete groups):

```bash
/view plot ds1 profile --x TEMP --y PRES --group-by LATITUDE:2 --agg mean --bins lon=2,lat=2,y=2 --order PRES:desc --legend-loc bottom --cmap tab20

/view plot ds1 map --y TEMP --filter "PRES >= 0 AND PRES <= 50" --bbox -100,-30,-20,30 --agg mean --point-size 15

/view plot ds1 timeseries --x TEMP --y PSAL --group-by PRES:25 --agg mean --legend-loc "upper right" --cmap plasma --filter "PRES >= 0 AND PRES <= 250" --start 2002-01-01 --end 2017-01-01
```

### Export

```bash
/view export <datasetKey> csv \
  [--cols TIME,LATITUDE,LONGITUDE,PRES,TEMP,DOXY] \
  [--filter …] [--order …] [--limit N] \
  --out data.csv
```

---

## Tips

* Omit `--out` on `/view plot …` to let the CLI open the PNG via your system viewer; add `--out` to save instead.
* `--limit` works for **preview/export/plot** to keep responses small when piping results to a frontend.
* You can furthur save preview result as a dataframe by `preview ds1 as ds2` (ds1 or ds2 are examples for dataset-alias). If you need to overwrite a dataset alias, you should /view close dataset-alias first.

---

## Build (advanced users)

* Clone the repo and use the PyInstaller spec if you want a local viewer executable:
  * `pyinstaller cli.spec` and `pyinstaller view.spec`.
  * We do not publish the viewer binary by default due to size; the sdist install is the recommended path.
* If you need to build the whole package: `python -m build`

---

## Troubleshooting

* **Viewer didn’t start?** Ensure you installed the sdist and try again; or set `ODBARGO_DEBUG=1` to see handshake logs.
* **PNG didn’t open?** Add `--out plot.png` and open it manually; the CLI always writes the raw bytes.
* **Slow on first use?** Heavy libs are imported the first time; later commands are faster. On first plot the imports may be slow; that’s normal. Adjust `ODBARGO_VIEW_STARTUP_TIMEOUT` if needed.
* **Variable name is case-insensitive**: as mentioned in [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md)

---

*For downloader usage and platform integration (Ocean APIverse), see the main* [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md).
