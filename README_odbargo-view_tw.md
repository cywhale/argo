# 安裝與使用 odbargo-view

## 導覽

`odbargo-view` 是 `odbargo-cli` 資料檢視擴充模組，透過 `/view ...` 指令即可在命令列開啟、預覽、篩選、繪圖並匯出 Argo NetCDF 資料。

* **為何選配(optional)？** 為了讓主要 CLI （下載Argo資料檔主功能）保持精簡。而資料檢視模組帶有 xarray / h5py / matplotlib 等較大的套件，需要時再安裝即可。
* **未提供獨立執行檔。** 安裝原始套件後，CLI 會自動以 *module mode* 啟動檢視服務。

---

## 安裝

可從主專案的原始壓縮檔安裝（詳見 [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README_tw.md)）：

```bash
pip install odbargo-0.x.y.tar.gz
```

完成後執行 `odbargo-cli`，輸入 `/view ...` 指令，檢視模組便會自動啟動。

> 若您自行編譯 `odbargo-view` 執行檔，將其放在 `odbargo-cli` 同層目錄，CLI 也能自動偵測並優先使用。

---

## 快速上手

```bash
odbargo-cli
argo> /view open /path/to/argo_data.nc as ds1
argo> /view list_vars ds1
argo> /view preview ds1 as ds2 --cols TIME,PRES,TEMP --order TIME:desc --limit 1000
argo> /view plot ds1 timeseries --x TIME --y TEMP --out temp.png
```

* `open` 指定資料檔並建立別名（如 `ds1`）。
* `list_vars` 列出座標及變數。
* `preview` 顯示表格，可搭配 `--limit`、`--order` 與篩選條件。
* `plot` 回傳 PNG；加上 `--out` 另存檔案，省略則會嘗試開啟系統預設檢視器。

---

## 指令速覽

以下操作皆在 `odbargo-cli` 中輸入。

### 開啟資料

```bash
/view open <path> [as <datasetKey>]
```

例如 `/view open argo_data.nc as ds1`，後續便可用 `ds1` 作為資料集別名。

### 列出變數

```bash
/view list_vars [<datasetKey>]
```

### 預覽資料

```bash
/view preview <datasetKey> [as <new datasetKey>] \
  [--cols TIME,PRES,TEMP] \
  [--filter "PRES BETWEEN 25 AND 100 AND TEMP > 0"] \
  [--order TIME:asc] [--cursor c1] [--limit 1000] \
  [--start 2010-01-01] [--end 2012-12-31] [--bbox lon1,lat1,lon2,lat2] \
  [--trim-dims]
```

* 可另存為新別名（例如 `preview ds1 as ds2`）。
* `--filter`、`--start/--end`、`--bbox` 可混合使用。
* `--trim-dims` 會移除預設附帶的維度軸向欄位（TIME / LATITUDE / LONGITUDE）。

---

### 篩選語法提醒

* DSL 支援 `= != > >= < <=`、`BETWEEN ... AND ...`、`AND`、`OR` 及括號。
* 時間請使用 ISO-8601 字串，例如 `TIME >= "2019-01-01"`。
* 亦可用 `--json-filter '{"and":[...]}'` 傳 JSON 條件（進階使用）。

範例：

```bash
--filter "PRES BETWEEN 25 AND 100 AND DOXY > 0"
--start 2010-01-01 --end 2011-01-01 --bbox 120,20,130,26
```

---

### 繪圖

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

* **timeseries**：常見設定為 `--x TIME --y <var>`。搭配 `--group-by` + `--agg` 可一次繪多條線。
* **profile**：`--x <var> --y PRES`，預設反轉深度。加入 `--bins y=<ΔP>`（例如 `y=10`）可將相近深度合併成單一點；若未指定則僅合併完全相同的 PRES。通常會搭配 `--order PRES:desc/asc` 排序資料。
* **map**：`--x LONGITUDE --y LATITUDE`，可再用 `--z` 或 `--cmap` 設定色階。`--bins lon=...,lat=...` 會以網格化作圖；未設定時則是散佈圖（可用 `--point-size` 控制點大小）。
* `--legend`、`--legend-loc`、`--legend-fontsize` 可調整圖例位置；放在上下方時會自動預留空間。

範例：

```bash
/view plot ds1 profile --x TEMP --y PRES --group-by LATITUDE:2 --agg mean --bins lon=2,lat=2,y=2 --order PRES:desc --legend-loc bottom --cmap tab20

/view plot ds1 map --y TEMP --filter "PRES >= 0 AND PRES <= 50" --bbox -100,-30,-20,30 --agg mean --point-size 15

/view plot ds1 timeseries --x TIME --y PSAL --group-by PRES:25 --agg mean --legend-loc "upper right" --cmap plasma --filter "PRES >= 0 AND PRES <= 250" --start 2002-01-01 --end 2017-01-01
```

### 匯出 CSV

```bash
/view export <datasetKey> csv \
  [--cols TIME,LATITUDE,LONGITUDE,PRES,TEMP,DOXY] \
  [--filter …] [--order …] [--limit N] \
  --out data.csv
```

---

## 使用小撇步

* `/view plot …` 預設嘗試開啟圖檔；加上 `--out` 可直接存檔。
* `--limit` 對 `preview / plot / export` 都有效，適合串接前端時控制回傳量。
* 若想保留子集資料，可用 `preview ds1 as ds2` 另存別名；若要覆寫相同別名，記得先 `/view close` 釋放舊資料。

---

## 進階：自行打包

* 想建立本機執行檔可使用 PyInstaller 規格：
  * `pyinstaller cli.spec`、`pyinstaller view.spec`
* 若需要完整發行檔：
  * `python -m build`

---

## 疑難排解

* **Viewer 沒啟動？** 確認已安裝 sdist，可設 `ODBARGO_DEBUG=1` 觀察啟動過程交握訊息。
* **PNG 沒開啟？** 用 `--out plot.png` 儲存後自行開啟；CLI 仍會寫出原始位元組。
* **首次啟動較慢？** 初次載入套件會花時間，之後會快很多。若需要調整等待時間，可修改 `ODBARGO_VIEW_STARTUP_TIMEOUT`。
* **變數大小寫問題** 詳見 [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README_tw.md)，預設不分大小寫。

---

*CLI 與 APIverse platform 整合操作，請參考主專案* [README: odbargo-cli](https://github.com/cywhale/argo/blob/main/README.md)。
