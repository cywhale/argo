# Install and Use odbargo-cli

## Introduction  
`odbargo-cli` is a lightweight command-line tool designed to download Biogeochemical (BGC) Argo float data (NetCDF) from the [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) server.
It is developed by [ODB](https://www.odb.ntu.edu.tw/) as a Argo plugin in ODB [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/), and also supports standalone interactive use.  Visit it for more [details](https://api.odb.ntu.edu.tw/hub/?help=Argo) about the integration with Ocean APIverse . 

---

## Usage

### 1. Download

You can either:

- **Download the source**: [`odbargo-cli.py`](https://raw.githubusercontent.com/cywhale/argo/refs/heads/main/odbargo-cli.py)
- **Use prebuilt binaries** (if available for your platform):  
  - Windows: [`odbargo-cli.exe`](https://github.com/cywhale/argo/blob/main/dist/win_cli/odbargo-cli.exe)
  - Linux: [`odbargo-cli`](https://github.com/cywhale/argo/blob/main/dist/linux_cli/odbargo-cli)

### 2. Run the CLI

You can run it as a background websocket server + interactive prompt:
*--port is optional, default port is 8765*

```bash
python odbargo-cli.py --port 8765
```

Or, if using a compiled binary:

```bash
./odbargo-cli.exe --port 8765    # Windows
./odbargo-cli --port 8765        # Linux/macOS
```

You will see:

```
[Argo-CLI] WebSocket listening on ws://localhost:8765
🟡 CLI interactive mode. Type WMO list (comma separated) or 'exit':
>>>
```

---

### 3. Options

* **Interactive mode**:
  Type WMO float numbers like:

  ```
  >>> 5903377, 5903594
  ```

  The tool will fetch and save NetCDF files into a temporary directory.

#### Notes on Slow Networks

* Large WMO groups cause a huge NetCDF to download which may timeout in poor network conditions.
* The CLI retries **3 times with 10-second intervals** if a download fails.
* It's recommended to fetch ≤ 3 WMOs at a time on slow connections.

#### Frontend-connected mode
  If the [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/settings) > Options > Plugin > Enable OdbArgo detects this tool running locally, it will communicate via `ws://localhost:8765` to trigger downloads directly.

  Use ODB [Argofloats WMS layer](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats) directly on Ocean APIverse.

