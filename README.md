# Install and Use odbargo-cli

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15655311.svg)](https://doi.org/10.5281/zenodo.15655311)

## Introduction  
`odbargo-cli` is a lightweight command-line tool designed to download Biogeochemical (BGC) Argo float data (NetCDF) from the [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) server.
It is developed by [ODB](https://www.odb.ntu.edu.tw/) as a Argo plugin in ODB [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/), and also supports standalone interactive use.  Visit it for more [details](https://api.odb.ntu.edu.tw/hub/?help=Argo) about the integration with Ocean APIverse . 

We also provide a **new optional viewer module** (`odbargo-view`) that lets you preview, filter, plot, and export data directly from the CLI using `/view ...` commands. The viewer is **not** shipped as an executable to keep downloads small; install it from the source archive when needed (see **Add the viewer** below). Full usage is documented in [README_odbargo-view](https://github.com/cywhale/argo/blob/main/README_odbargo-view.md).

---

## Contents

- `odbargo-cli` — the downloader + single-port WS bridge.
- `odbargo-view` — the viewer plugin (now backed by `odbViz`; deps: xarray/matplotlib/h5netcdf/h5py; optional pandas).

The CLI can run standalone; the viewer is started on demand when you use `/view …`.

---

## Usage

### 1. Download

We publish **only** the small CLI binaries:

- **Use prebuilt binaries** (if available for your platform):  
  - Windows: [`odbargo-cli.exe`](https://github.com/cywhale/argo/blob/main/dist/win_cli/odbargo-cli.exe)
  - Linux: [`odbargo-cli`](https://github.com/cywhale/argo/blob/main/dist/linux_cli/odbargo-cli)
  - macOS: [`odbargo-cli`](https://github.com/cywhale/argo/blob/main/dist/mac_cli/odbargo-cli)

- **Add the viewer** (from source archive):
  - Find a full source archive: [odbargo-0.x.y.tar.gz](https://github.com/cywhale/argo/tree/main/dist)

```bash
pip install "odbargo-0.x.y.tar.gz[view]"
```

### 2. Run the CLI

You can run it as a background websocket server + interactive prompt:
*--port is optional, default port is 8765*

```bash
odbargo-cli

# if you git clone the whole repo, you can run it from project root by:
# python cli_entry.py
# python view_entry.py # for viewer
```

Or, if using a compiled binary:

```bash
./odbargo-cli.exe    # Windows. If Window Defender block it, click 【More info】>【Run anyway】
./odbargo-cli        # Linux/macOS Need `sudo chmod u+x ./odbargo-cli` to make it executable in Linux.
```

You will see:

```
[Argo-CLI] WebSocket listening on ws://localhost:8765
🟡 CLI interactive mode. Type '/view ...' commands or comma-separated WMO list. Type 'exit' to quit.
argo>
```

---

### 3. Options

* **Interactive mode**:
  Type WMO float numbers like:

  ```
  argo> 5903377, 5903594
  ```

  The tool will fetch and save NetCDF files into a temporary directory.
  For example, download ok and tell you: Saved to: /tmp/argo_xxx/argo_5903377_5903594.nc

  Viewer help (if you **Add the viewer** and installed it successfully):
  ```
  argo> /view help
  argo> /view open /tmp/argo_xxx/argo_5903377_5903594.nc as ds1
  argo> /view list_vars ds1
  argo> /view preview ds1 as ds2 --cols TIME,PRES,TEMP --filter "PRES >= 25 AND PRES <= 100" --order TIME:asc --limit 500
  ```
  It will open the dataset and alias as `ds1`. You can furthur list the variables and preview the data.
  See more: [README_odbargo-view](https://github.com/cywhale/argo/blob/main/README_odbargo-view.md).
  
#### Notes on Slow Networks

* Large WMO groups cause a huge NetCDF to download which may timeout in poor network conditions.
* The CLI retries **3 times with 10-second intervals** if a download fails.
* It's recommended to fetch ≤ 3 WMOs at a time on slow connections.

#### Case-insensitive variables

`odbargo-cli` treats variable names case-insensitively by default (flag
`--case-insensitive-vars`, or `ODBARGO_CASE_INSENSITIVE_VARS=0|1`). This keeps
commands compatible with both legacy upper-case NetCDF files and the newer
lower-case ERDDAP exports. Set the flag to `False` only if you need strict,
case-sensitive matching, use `--no-case-insensitive-vars` or `ODBARGO_CASE_INSENSITIVE_VARS=0`.

#### Frontend-connected mode
  If the [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/settings) > Options > Plugin > Enable OdbArgo detects this tool running locally, it will communicate via `ws://localhost:8765` to trigger downloads directly.

  Use ODB [Argofloats WMS layer](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats) directly on Ocean APIverse.
