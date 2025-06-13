# Install and Use odbargo-cli

* **Deprecated**, it works well but we drop the `argopy` dependency in new version of `odbargo-cli`

## Introduction  
`odbargo-cli` is a lightweight command-line tool designed to download Biogeochemical (BGC) Argo float data from the [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) server using [argopy](https://argopy.readthedocs.io/en/latest/).
It is developed by [ODB](https://www.odb.ntu.edu.tw/) as a backend companion for the ODB Argo Mapper plugin in [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/), and also supports standalone interactive use.

---

## Usage (Executable)

### 1. Download Executable  
Download the standalone executable `odbargo-cli.exe` from:
https://github.com/cywhale/argo/releases

> ⚠️ If Windows shows a warning from Defender SmartScreen, click **"More Info" → "Run anyway"** to proceed.
> This app is safe and verified internally.

---

### 2. Run CLI Tool  
Double-click `odbargo-cli.exe`, or run from terminal:
```cmd
odbargo-cli.exe
````

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

* **Frontend-connected mode**:
  If the [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/settings) > Options > Plugin > Enable OdbArgo detects this tool running locally, it will communicate via `ws://localhost:8765` to trigger downloads directly.

  Use ODB [Argofloats WMS layer](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats) directly on Ocean APIverse.

---

## Advanced: Python Source Version

If you prefer the source version (e.g., on Linux/macOS):

### 1. Install Python 3.10+

[https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Set up and run manually:

```bash
pip install argopy websockets
python argo_cli.py --port 8765
```

> This version also supports the interactive CLI and frontend plugin modes.

---

## Support

For Argofloats WMS layer integration, visit [Ocean APIverse with Argo Mapper](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats) or contact me.

```

---

