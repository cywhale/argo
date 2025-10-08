# 安裝與使用 odbargo-cli

## 簡介
`odbargo-cli` 是一款輕量級命令列工具，可從 [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) 伺服器下載生地化 (BGC) Argo 浮標 NetCDF 資料。由 [ODB](https://www.odb.ntu.edu.tw/) 開發，可在 [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/) 平台整合使用，也能獨立運作。整合流程請參考[說明頁](https://api.odb.ntu.edu.tw/hub/?help=Argo)。

我們另外提供 **資料檢視模組** `odbargo-view`，可直接在 CLI 透過 `/view ...` 指令預覽、篩選、繪圖與匯出資料。為了保持主程式精簡，檢視模組需另行安裝（詳見「安裝 viewer」段落）。完整操作可參考 [README_odbargo-view](https://github.com/cywhale/argo/blob/main/README_odbargo-view_tw.md)。

---

## 內容

- `odbargo-cli`：下載器與單一埠 WebSocket 橋接器。
- `odbargo-view`：擴充模組（包含其他相依套件：如xarray等）。

CLI 可以獨立使用；需要視覺化時再啟動 viewer。

---

## 使用方式

### 1. 下載程式

- **預編譯 CLI**（依平台提供）：
  - Windows：[`odbargo-cli.exe`](https://github.com/cywhale/argo/blob/main/dist/win_cli/odbargo-cli.exe)
  - Linux：[`odbargo-cli`](https://github.com/cywhale/argo/blob/main/dist/linux_cli/odbargo-cli)
- **安裝 全部（含  viewer 原始套件）**：自 [dist](https://github.com/cywhale/argo/tree/main/dist) 下載 `odbargo-0.x.y.tar.gz`

```bash
pip install odbargo-0.2.6.tar.gz
```

### 2. 執行 CLI

CLI 可同時啟動背景 WebSocket 伺服器與互動模式（`--port` 預設 8765，可省略）：

```bash
odbargo-cli

# 若從原始碼執行：
# python cli_entry.py
# python view_entry.py  # 啟動 viewer（必要時）
```

若使用預編譯版本：

```bash
./odbargo-cli.exe    # Windows；若遭 Windows Defender 阻擋，選【更多資訊】→【仍要執行】
./odbargo-cli        # Linux/macOS；Linux 需先執行 sudo chmod u+x ./odbargo-cli
```

畫面會顯示：

```
[Argo-CLI] WebSocket listening on ws://localhost:8765
🟡 CLI interactive mode. Type '/view ...' commands or comma-separated WMO list. Type 'exit' to quit.
argo>
```

---

### 3. 操作說明

* **互動模式**：

  ```
  argo> 5903377, 5903594
  ```

  以逗號分隔輸入 WMO 編號，即可下載並儲存 NetCDF 於暫存資料夾，完成後會顯示：`Saved to: /tmp/argo_xxx/argo_5903377_5903594.nc`。
  
  若已安裝 viewer，可使用：

  ```
  argo> /view help
  argo> /view open /tmp/argo_xxx/argo_5903377_5903594.nc as ds1
  argo> /view list_vars ds1
  argo> /view preview ds1 as ds2 --cols TIME,PRES,TEMP --filter "PRES >= 25 AND PRES <= 100" --order TIME:asc --limit 500
  ```
  
  即可開啟資料集 (`ds1`)、檢視變數與預覽子集，詳細操作請參考 [README_odbargo-view](https://github.com/cywhale/argo/blob/main/README_odbargo-view.md)。

#### 網路緩慢時的建議

* 大量 WMO 編號會產生巨大的 NetCDF 檔，易因網速慢而下載失敗。
* CLI 工具會自動重試最多 3 次，每次間隔 10 秒。
* 建議一次查詢不超過 3 個 WMO 編號以提高成功率。

#### 前端連線模式

若您啟用 [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/settings) > 選項 > 擴充模組 > 啟用 OdbArgo，此工具會與前端透過 `ws://localhost:8765` 溝通並自動觸發下載任務。

也可直接連結 ODB APIverse 提供的 [Argofloats WMS 圖層](https://api.odb.ntu.edu.tw/hub/earth/settings?ogcurl=https://ecodata.odb.ntu.edu.tw/geoserver/odbargo/wms&service=WMS&layer=argofloats)
