# 安裝與使用 odbargo-cli

## 簡介  
`odbargo_app` 是一款輕量級命令列工具，從 [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) 伺服器下載生地化 (BGC) Argo 剖面浮標資料 (NetCDF)。此應用由 [ODB](https://www.odb.ntu.edu.tw/) 開發，並做為在 [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/) 平台上與 ODB Argo WMS 圖層整合之擴充模組，也可獨立互動使用。更多整合使用細節請參考[這裡](https://api.odb.ntu.edu.tw/hub/?help=Argo)。

---  

## 使用方式

### 1. 下載程式

您可以選擇：

- **下載原始檔案**：[`odbargo-cli.py`](https://raw.githubusercontent.com/cywhale/argo/refs/heads/main/odbargo-cli.py)
- **使用預編譯版本**（若提供對應平台）：
  - Windows：[`odbargo-cli.exe`](https://github.com/cywhale/argo/blob/main/dist/win_cli/odbargo-cli.exe)
  - Linux：[`odbargo-cli`](https://github.com/cywhale/argo/blob/main/dist/linux_cli/odbargo-cli)

### 2. 執行 CLI

可啟動為背景 WebSocket 伺服器 + 互動模式：  
*（`--port` 可省略，預設為 8765）*

```bash
python odbargo-cli.py --port 8765
````

若使用編譯好的版本：

```bash
./odbargo-cli.exe --port 8765    # Windows
./odbargo-cli --port 8765        # Linux/macOS
```

啟動後畫面會顯示：

```
[Argo-CLI] WebSocket listening on ws://localhost:8765
🟡 CLI interactive mode. Type WMO list (comma separated) or 'exit':
>>>
```

---

### 3. 操作說明

* **互動模式**：
  輸入 WMO 漂流浮標編號（用逗號分隔）：

  ```
  >>> 5903377, 5903594
  ```

  系統會下載並儲存 NetCDF 資料於暫存資料夾中。

#### 網路緩慢時的建議

* 大量 WMO 編號會產生巨大的 NetCDF 檔，易因網速慢而下載失敗。
* CLI 工具會自動重試最多 3 次，每次間隔 10 秒。
* 建議一次查詢不超過 3 個 WMO 編號以提高成功率。

#### 前端連線模式

若您啟用 [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/settings) > 選項 > 擴充模組 > 啟用 OdbArgo，此工具會與前端透過 `ws://localhost:8765` 溝通並自動觸發下載任務。

也可直接連結 ODB APIverse 提供的 [Argofloats WMS 圖層](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats)

