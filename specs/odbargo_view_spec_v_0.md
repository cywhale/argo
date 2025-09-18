# ODBArgo-View Spec v0.1 (Spec‑Driven Development)

**Purpose:** Define the developer‑facing contract for a *single‑port* WebSocket workflow where the core `odbargo-cli.py` acts as the **only** WS server and brokers visualization, filtering, and export requests to a **subprocess plugin** named `odbargo-view` (this repo). This spec is self‑contained: it documents the plugin interface and the broker protocol the *main CLI* must implement. Platform apps (e.g., APIverse/Preact) can integrate later by following the same WS frames defined here, but that’s out of scope for this file.

---

## 0. Scope & Goals

**In scope**

- View/Query plugin as an **independent Python package**: `odbargo_view` (subprocess run‑mode), minimal public CLI entrypoint `python -m odbargo_view`.
- Communication between **Main CLI** (`odbargo-cli.py`) and **Plugin** (this package) via **STDIN/STDOUT NDJSON + binary**.
- Communication between **Main CLI** and **Front‑end** (WS client) via **single WebSocket**: JSON control frames + **binary image/file frames**.
- Core features: open dataset, list variables/coords, preview tabular rows (bounded), filtering, plotting (time‑series/profile/map), CSV export.
- Dependencies: `xarray`, `h5netcdf` + `h5py`, `matplotlib`. `pandas` is **optional** (extra: `view[pandas]`).

**Out of scope**

- Platform UI flows, ReactFlow nodes, storage of flow JSON (handled by APIverse project).
- FastAPI/HTTP servers (we keep single WS port; no extra port).

**Design principles**

- **Single port** end‑to‑end. No local HTTP.
- **Binary‑first** for images/files on WS to avoid base64 overhead.
- **Spec first**: messages are versioned and JSON‑schema’d; small DSL parser for filters (no `eval`).
- **Back‑pressure aware**: chunked file streaming and bounded preview.

---

## 1. High‑Level Architecture

```
+------------------+                  +-------------------------+
|  Front-end (WS)  |  JSON + Binary  |  Main CLI: odbargo-cli  |
| (APIverse, etc.) | <--------------> |  - Single WS server     |
+------------------+                  |  - Job mgr / router     |
                                       |  - Subprocess bridge    |
                                       +-----------+-------------+
                                                   |
                                         NDJSON + Binary (stdio)
                                                   |
                                       +-----------v-------------+
                                       |   Plugin: odbargo-view  |
                                       |   - xarray/h5netcdf     |
                                       |   - matplotlib          |
                                       |   - (opt) pandas        |
                                       +-------------------------+
```

- Main CLI stays small; the heavy deps live in the plugin, spawned **on demand**.
- The **same** WS is used for: downloads (existing), view/plot/export (new).

---

## 2. Versioning & Handshake

### 2.1 Protocol versions

- `wsProtocolVersion`: `1.0`
- `pluginProtocolVersion`: `1.0`

### 2.2 WS handshake (Front‑end ⇄ Main CLI)

**Request**

```json
{
  "type": "hello",
  "from": "apiverse",
  "wanted": ["download", "view"],
  "wsProtocolVersion": "1.0"
}
```

**Response**

```json
{
  "type": "hello_ok",
  "capabilities": {
    "download": true,
    "view": true,
    "binaryFrames": true
  },
  "wsProtocolVersion": "1.0"
}
```

### 2.3 CLI⇄Plugin bootstrap (internal)

Main CLI spawns plugin on first `view.*` command. Plugin must respond with:

```json
{"type":"plugin.hello_ok","pluginProtocolVersion":"1.0","capabilities":{"open_dataset":true,"list_vars":true,"preview":true,"plot":true,"export":true}}
```

---

## 3. Message Types (WS: Front‑end ⇄ Main CLI)

> All messages carry `requestId` (string). Responses echo the same `requestId`.

### 3.1 Dataset lifecycle

- **Open dataset**

```json
{"type":"view.open_dataset","requestId":"r1","path":"/tmp/argo_data.nc","enginePreference":["h5netcdf","netcdf4"],"datasetKey":"ds1"}
```

**Response**

```json
{
  "type":"view.dataset_opened",
  "requestId":"r1",
  "datasetKey":"ds1",
  "summary":{
    "dims":{"N_POINTS":304919},
    "coords":[{"name":"LATITUDE","dtype":"float64","size":304919,"attrs":{"units":"degree_north"}}, {"name":"LONGITUDE","dtype":"float64","size":304919}],
    "vars":[{"name":"PRES","dtype":"float32","shape":[304919],"attrs":{"units":"dbar","long_name":"Pressure"}}, {"name":"DOXY","dtype":"float32","shape":[304919],"attrs":{"units":"umol/kg"}}]
  }
}
```

- **List variables/coords**

```json
{"type":"view.list_vars","requestId":"r2","datasetKey":"ds1"}
```

**Response**

```json
{"type":"view.vars","requestId":"r2","datasetKey":"ds1","coords":[...],"vars":[...]}
```

- **Close dataset**

```json
{"type":"view.close_dataset","requestId":"r3","datasetKey":"ds1"}
```

**Response**

```json
{"type":"view.dataset_closed","requestId":"r3","datasetKey":"ds1"}
```

### 3.2 Preview (bounded table)

```json
{
  "type":"view.preview",
  "requestId":"r4",
  "datasetKey":"ds1",
  "columns":["TIME","PRES","DOXY"],
  "filter": {"dsl":"PRES >= 25 AND PRES <= 100 AND DOXY > 0"},
  "limit": 2000,
  "cursor": null,
  "orderBy": [{"col":"TIME","dir":"asc"}]
}
```

**Response**

```json
{
  "type":"view.preview_result",
  "requestId":"r4",
  "datasetKey":"ds1",
  "columns":["TIME","PRES","DOXY"],
  "rows":[["2019-01-01T00:00:00Z", 26.1, 210.0], ...],
  "limitHit": true,
  "nextCursor": "c1"
}
```

### 3.3 Plot (image returned via *binary WS frame*)

**Control header (JSON):**

```json
{
  "type":"view.plot",
  "requestId":"r5",
  "datasetKey":"ds1",
  "kind":"timeseries",  
  "x":"TIME",
  "y":"DOXY",
  "filter":{"dsl":"PRES BETWEEN 25 AND 100"},
  "style":{"width":900,"height":500,"dpi":120,"title":"DOXY @ 25–100 dbar"}
}
```

**Server sends:**

1. **Plot header**

```json
{"type":"plot_blob","requestId":"r5","contentType":"image/png","size": 254321}
```

2. **Immediately one *****binary***** WS frame** containing PNG bytes.

> For chunking large files, server MAY instead send:
>
> - `{"type":"plot_blob_start",...}` → multiple `binary` frames → `{"type":"plot_blob_end",...}`.

### 3.4 Export CSV (chunked or single binary)

**Request**

```json
{"type":"view.export","requestId":"r6","datasetKey":"ds1","format":"csv","columns":["TIME","LATITUDE","LONGITUDE","PRES","DOXY"],"filter":{"json":{"and":[{">=":["PRES",25]},{"<=":["PRES",100]},{">":["DOXY",0]}]}}}
```

**Server sends (chunked example)**

```json
{"type":"file_start","requestId":"r6","contentType":"text/csv","filename":"argo_ds1_export.csv"}
```

→ N × **binary** frames (CSV bytes)

```json
{"type":"file_end","requestId":"r6","sha256":"...","size":1234567}
```

### 3.5 Errors (uniform)

```json
{
  "type":"error",
  "requestId":"rX",
  "code":"FILTER_INVALID",
  "message":"Unknown identifier: PRS",
  "hint":"Did you mean PRES?",
  "details": {"position": 8}
}
```

**Canonical error codes**

- `DATASET_OPEN_FAIL`, `UNSUPPORTED_ENGINE`, `DATASET_NOT_FOUND`
- `FILTER_INVALID`, `FILTER_UNSUPPORTED_OP`, `COLUMN_UNKNOWN`
- `PREVIEW_TOO_LARGE`, `ROW_LIMIT_EXCEEDED`
- `PLOT_FAIL`, `EXPORT_FAIL`
- `PLUGIN_NOT_AVAILABLE`, `INTERNAL_ERROR`

---

## 4. CLI ⇄ Plugin (STDIN/STDOUT) Protocol

**Transport:** NDJSON (one JSON per line) for control; binary payload follows specific headers.

**Common envelope**

- Every control message has `msgId` (string) and `op` (string). Responses echo `msgId`.

### 4.1 Operations

- `op: "open_dataset"`

```json
{"op":"open_dataset","msgId":"m1","datasetKey":"ds1","path":"/tmp/argo_data.nc","enginePreference":["h5netcdf","netcdf4"]}
```

**Response**

```json
{"op":"open_dataset.ok","msgId":"m1","datasetKey":"ds1","summary":{...}}
```

- `op: "list_vars"`

```json
{"op":"list_vars","msgId":"m2","datasetKey":"ds1"}
```

**Response**

```json
{"op":"list_vars.ok","msgId":"m2","datasetKey":"ds1","coords":[...],"vars":[...]}
```

- `op: "preview"`

```json
{"op":"preview","msgId":"m3","datasetKey":"ds1","columns":["TIME","PRES","DOXY"],"filter":{"dsl":"PRES BETWEEN 25 AND 100"},"limit":2000,"cursor":null,"orderBy":[{"col":"TIME","dir":"asc"}]}
```

**Response**

```json
{"op":"preview.ok","msgId":"m3","datasetKey":"ds1","columns":[...],"rows":[...],"limitHit":true,"nextCursor":"c1"}
```

- `op: "plot"` (binary follows)

```json
{"op":"plot","msgId":"m4","datasetKey":"ds1","kind":"profile","x":"DOXY","y":"PRES","filter":{...},"style":{"width":800,"height":600,"dpi":120,"invert_y":true}}
```

**Plugin writes:**

1. `{"op":"plot_blob","msgId":"m4","contentType":"image/png","size":123456}`\n
2. Immediately **raw PNG bytes** to stdout.

- `op: "export"` (binary may be large)

```json
{"op":"export","msgId":"m5","datasetKey":"ds1","format":"csv","columns":[...],"filter":{...}}
```

**Plugin writes (chunked)**

- `{"op":"file_start","msgId":"m5","contentType":"text/csv","filename":"argo_ds1_export.csv"}`\n
- N × binary chunks (plugin may choose a fixed chunk size, e.g., 256 KiB)
- `{"op":"file_end","msgId":"m5","sha256":"...","size":1234567}`\n **Error**

```json
{"op":"error","msgId":"mX","code":"PLOT_FAIL","message":"Matplotlib crashed: ..."}
```

---

## 5. Filtering DSL (minimal, safe)

### 5.1 Grammar (subset)

- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*` (must match known coord/var names)
- **Literals**: integers, floats, ISO datetime strings (quoted), booleans
- **Comparators**: `=`, `!=`, `>`, `>=`, `<`, `<=`, `BETWEEN a AND b`
- **Logical**: `AND`, `OR`, parentheses `(...)`
- **Geo helpers** (optional v0.2): `WITHIN lon1,lat1,lon2,lat2`

### 5.2 Examples

- `PRES BETWEEN 25 AND 100 AND DOXY > 0`
- `TIME >= "2019-01-01" AND TIME < "2020-01-01"`

### 5.3 Semantics

- DSL → AST → NumPy boolean mask over *common index* (`N_POINTS` by default).
- **No function calls, no attribute access, no eval.**

### 5.4 JSON equivalent (for machines)

```json
{"and":[{"between":["PRES",25,100]},{">":["DOXY",0]}]}
```

---

## 6. Plotting Kinds

### 6.1 `timeseries`

- Inputs: `x: TIME`, `y: <var>`, optional `groupBy` (e.g., by WMO or CYCLE)
- Style: `width,height,dpi,title,marker,line,alpha`

### 6.2 `profile`

- Inputs: `x: <var>`, `y: PRES` (default invert Y)
- Style: `invert_y (default true)`, `grid`, `median/mean overlay`

### 6.3 `map`

- Inputs: `x: LONGITUDE`, `y: LATITUDE`, `z: <var>` (optional), `agg` (`mean|median|count`), `bins` (e.g., `{lon:0.5, lat:0.5}`)
- Rendering: **no Cartopy** initially; simple scatter/hexbin. (Cartopy may be optional extra in future.)

---

## 7. Resource & Safety Limits

- `maxPreviewRows` (default 5000)
- `maxColumns` in preview (default 16)
- `maxImagePixels` (e.g., 4k×4k)
- Export enforced by **chunked streaming** only.
- Timeouts per op; cancellation via `view.cancel` (future v0.2).

---

## 8. Packaging & Installation

- Package: `odbargo_view`
- Minimum deps: `xarray>=2024.6.0`, `h5netcdf`, `h5py`, `matplotlib`
- Optionals: `pandas` via extra `view[pandas]`
- Entry point: `python -m odbargo_view` (reads NDJSON on stdin).

---

## 9. Main CLI Integration (odbargo-cli.py)

### 9.1 Responsibilities

- Maintain single WS server and job manager.
- On first `view.*` WS command: spawn plugin (`subprocess.Popen([sys.executable, '-m', 'odbargo_view'])`).
- Bridge logic:
  - WS JSON → write NDJSON to plugin stdin (append `\n`).
  - When plugin writes JSON control lines → convert to WS JSON (map fields, add `requestId`).
  - When plugin starts binary stream → immediately relay to WS binary frames.
- Echo results to terminal for human visibility when the command originated from CLI.

### 9.2 Mapping table (WS ⇄ Plugin)

| WS `type`           | Plugin `op`    | Notes                                   |
| ------------------- | -------------- | --------------------------------------- |
| `view.open_dataset` | `open_dataset` | path, enginePreference, datasetKey      |
| `view.list_vars`    | `list_vars`    | datasetKey                              |
| `view.preview`      | `preview`      | columns, filter, limit, cursor, orderBy |
| `view.plot`         | `plot`         | kind, x, y, style, filter               |
| `view.export`       | `export`       | format=csv, columns, filter             |

---

## 10. JSON Schemas (abridged)

> JSON Schema drafts omitted for brevity; implement with `jsonschema`. Key fields:

- `FilterDSL`: string ≤ 2 KiB.
- `FilterJSON`: structured as `{and:[...]} | {or:[...]}` nodes; leaves are comparator tuples.
- `PreviewRequest`: `{datasetKey, columns?, filter?, limit<=10000, cursor?, orderBy?}`
- `PlotRequest`: `{datasetKey, kind in [timeseries,profile,map], x,y?,style?,filter?}`
- `ExportRequest`: `{datasetKey, format='csv', columns?, filter?}`

---

## 11. Example End‑to‑End Flows

### Flow A: Timeseries preview + plot

1. WS: `view.open_dataset(ds1, /tmp/argo_data.nc)` → `view.dataset_opened`.
2. WS: `view.preview(ds1, cols=[TIME,PRES,DOXY], filter=PRES between 25..100)` → rows.
3. WS: `view.plot(kind=timeseries, x=TIME, y=DOXY, filter=...)` → `plot_blob` header + **binary PNG**.

### Flow B: Export CSV (bounded)

1. WS: `view.export(format=csv, cols=[TIME,LAT,LON,PRES,DOXY], filter=...)` → `file_start` + **binary chunks** + `file_end`.

---

## 12. Testing Checklist

- Unit: DSL parser, mask application vs. xarray `.where`.
- Integration: open→preview→plot→export happy paths.
- Big file: ensure chunking and memory cap; verify no event‑loop blocking (use `asyncio.to_thread`).
- Cross‑engine: open with `h5netcdf`, fallback `netcdf4`.
- Optional pandas: with/without extras produce identical preview semantics.

---

## 13. Security Notes

- WS origin allow‑list enforced by Main CLI.
- No `eval` anywhere. DSL parser is closed‑world over known identifiers.
- Path opening limited to known temp dirs unless explicitly allowed.

---

## 14. Roadmap (v0.2+)

- Cancellation API, progress events for long exports.
- Map with hexbin/2D histogram; optional Cartopy extra.
- Server‑side throttling and per‑client quotas.

---

# Appendix A — Minimal Plugin Skeleton (pseudocode)

```python
# python -m odbargo_view
import sys, json, struct
from typing import IO

class Bridge:
    def __init__(self, out: IO[bytes]):
        self.out = out
    def send(self, obj):
        self.out.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.out.flush()
    def send_blob(self, header_obj, blob_bytes: bytes):
        self.send(header_obj)
        self.out.write(blob_bytes)
        self.out.flush()

# read lines from stdin, dispatch to handlers (open_dataset, list_vars, preview, plot, export)
```

# Appendix B — Main CLI Bridge Notes

- Set `ws.binaryType = 'blob'` on front‑end.
- For plot/export: forward plugin `plot_blob/file_start` as WS JSON, then forward raw bytes as WS binary frames.

---

# Appendix C — Dependency Matrix

- Required: `xarray`, `h5netcdf`, `h5py`, `matplotlib`
- Optional: `pandas` (extra)

---

# CODEx Implementation Prompt (for this repo)

**System goal** Implement `` per spec above, as a subprocess plugin that communicates over **STDIN/STDOUT** (NDJSON + binary). Assume the **Main CLI is **`` — parse it first to understand its WS server, job handling, and message routing. Do *not* change its port behavior; extend it only to add the *bridge* from WS to this plugin.

**Key tasks**

1. Read this spec end‑to‑end. Extract all message types and fields.
2. Open the repository, **start with **``: identify where WS messages are received and jobs are tracked; add a bridge that spawns the plugin on first `view.*` message and proxies messages bidirectionally.
3. Create a new package `odbargo_view` with:
   - `__init__.py`, `__main__.py` (entrypoint `python -m odbargo_view`).
   - `bridge.py` (stdio helpers), `dsl.py` (safe parser → AST → NumPy mask), `dataset.py` (open/list/close), `preview.py`, `plot.py`, `export.py`.
   - Dependencies: `xarray`, `h5netcdf`, `h5py`, `matplotlib`; extras `[pandas]` optional.
4. Implement operations: `open_dataset`, `list_vars`, `preview`, `plot` (timeseries, profile, map scatter), `export` (CSV chunked). Follow the **binary framing** rules (send `plot_blob`/`file_start` then raw bytes).
5. Enforce limits (Section 7) and errors (Section 3.5). Add `plugin.hello_ok` at startup.
6. Write unit tests for DSL and basic flows. Provide a smoke script to pipe NDJSON in and receive a PNG to stdout (redirect to file for validation).

**Acceptance**

- Running `python -m odbargo_view` starts listening on stdin, prints `plugin.hello_ok` as first line.
- From Main CLI, sending `view.*` over WS results in valid responses and PNG/CSV binary frames to the WS client.
- No additional ports opened. Images render in APIverse ImageViewNode when bound to the WS binary frame.

**Notes**

- Be conservative on memory usage. Use `asyncio.to_thread` on Main CLI side for any blocking IO/CPU when bridging.
- Prefer `h5netcdf` engine first; fallback to `netcdf4` if import/IO fails.

---

**End of Spec**



---

## 15. CLI Mode (Standalone) — Slash Commands

> 目標：`odbargo-cli.py` 在**無平台/前端**情況下，也能完成開檔、探索、預覽、篩選、繪圖與匯出。CLI 指令採 **slash 語法**，與 WS/Plugin 介面**一對一**對照，利於平台將來做命令鏡像。

### 15.1 指令風格與入口

- 兩種啟動模式：
  1. **單發命令**（非互動）：`odbargo-cli "/view open /tmp/argo.nc as ds1"`
  2. **互動 REPL**：`odbargo-cli` → 出現 `argo>` 提示符，輸入 `/view ...`
- 每條命令產生一個 **requestId**，與 WS/Plugin 交握相同；回覆以人類可讀 +（可選）JSON 列印。

### 15.2 指令總覽（與 WS 對照）

| CLI Slash 命令                               | WS `type`            | Plugin `op`    |
| ------------------------------------------ | -------------------- | -------------- |
| `/view open <path> [as <datasetKey>]`      | `view.open_dataset`  | `open_dataset` |
| `/view list_vars [<datasetKey>]`           | `view.list_vars`     | `list_vars`    |
| `/view preview <datasetKey> [opts...]`     | `view.preview`       | `preview`      |
| `/view plot <datasetKey> <kind> [opts...]` | `view.plot`          | `plot`         |
| `/view export <datasetKey> csv [opts...]`  | `view.export`        | `export`       |
| `/view close <datasetKey>`                 | `view.close_dataset` | —              |

> `kind` ∈ `timeseries|profile|map`

### 15.3 通用選項（opts）

- `--cols TIME,PRES,DOXY` → `columns`
- `--filter "PRES BETWEEN 25 AND 100 AND DOXY>0"` → `filter.dsl`
- `--json-filter '{"and":[...]}'` → `filter.json`
- `--limit 2000`、`--order TIME:asc`、`--cursor c1`
- `--size 900x500`（寬x高像素）/ `--dpi 120` / `--title "..."`
- `--invert-y`（profile 預設 true，可用 `--no-invert-y` 關）
- `--bins lon=0.5,lat=0.5`（map 聚合）
- `--out <file>`（plot/export 將二進位直接存檔；若不指定，CLI 以人類訊息回覆並提示以 WS/平台收取二進位）
- `--echo-json`（額外輸出對等的 WS 請求 JSON，便於除錯/教學）

### 15.4 範例

- 開檔並命名：
  - `argo> /view open /tmp/argo_data.nc as ds1`
- 變數清單：
  - `argo> /view list_vars ds1`
- 預覽（限制 1200 列、排序）：
  - `argo> /view preview ds1 --cols TIME,PRES,DOXY --filter "PRES BETWEEN 25 AND 100 AND DOXY>0" --limit 1200 --order TIME:asc`
- 畫時序圖並存檔：
  - `argo> /view plot ds1 timeseries --x TIME --y DOXY --filter "PRES BETWEEN 25 AND 100" --size 900x500 --dpi 120 --title "DOXY @ 25–100 dbar" --out doxy_ts.png`
- 畫剖面圖（倒 Y）：
  - `argo> /view plot ds1 profile --x DOXY --y PRES --invert-y --out profile.png`
- 匯出 CSV（分塊寫檔，由 CLI 負責聚合）：
  - `argo> /view export ds1 csv --cols TIME,LATITUDE,LONGITUDE,PRES,DOXY --filter "DOXY>0" --out ds1_doxy.csv`
- 關閉資料集：
  - `argo> /view close ds1`

### 15.5 輸出慣例

- **人類可讀**：表格頭 + 最多前後各數十列（自動截斷，顯示 `...`）。
- **JSON（選配）**：加 `--echo-json` 時，最後附上 `preview_result`/`dataset_opened` 等對等 JSON。
- **影像/CSV**：若 `--out`，直接寫檔；否則顯示「可透過 WS 取回二進位」的提示。

### 15.6 退出碼（Exit codes）

- `0` 成功
- `2` 使用錯誤（參數/語法）
- `3` 檔案/資料集錯誤
- `4` 篩選語法錯誤
- `5` 圖像/匯出失敗
- `10` 插件不可用/崩潰

### 15.7 REPL 補充

- 支援上下鍵歷史、`/help`、`/quit`。
- `;` 可串多命令：`/view open ...; /view list_vars`（逐一執行）。

---

## 16. CLI↔WS↔Plugin 對照（擴充）

| CLI Slash                                      | WS Request (要點)                                    | Plugin op            | 備註                            |
| ---------------------------------------------- | -------------------------------------------------- | -------------------- | ----------------------------- |
| `/view open <path> [as <ds>]`                  | `type:view.open_dataset, path, datasetKey`         | `open_dataset`       | 引擎優先 `h5netcdf` → 退 `netcdf4` |
| `/view list_vars [<ds>]`                       | `type:view.list_vars, datasetKey`                  | `list_vars`          | 若略 ds 則使用最近一次 `open` 的 ds     |
| `/view preview <ds> [--cols ... --filter ...]` | `type:view.preview, columns, filter, limit, order` | `preview`            | 超量截斷、回 `limitHit/nextCursor`  |
| `/view plot <ds> <kind> [--x --y ... --out f]` | `type:view.plot, kind, x, y, style, filter`        | `plot` + `plot_blob` | 若 `--out`：CLI 直接寫 PNG         |
| `/view export <ds> csv [--cols ... --out f]`   | `type:view.export, format=csv, columns, filter`    | `export` + `file_*`  | 若 `--out`：CLI 聚合分塊寫 CSV       |
| `/view close <ds>`                             | `type:view.close_dataset, datasetKey`              | —                    | 釋放記憶體                         |

---

## 17. CODEx Prompt Addendum — 實作 CLI Slash 層

**請在啟動時解析 **``，新增：

1. **Argparse 單發模式**：若啟動參數包含一個以 `/` 開頭的字串，直接走 Slash Parser，對應為同名 WS 流程或直連 Plugin（本機模式）。
2. **互動 REPL**：無參數則開啟 REPL（`argo>`），內建 Slash Parser（詞法：空白分隔、引號保留；語義：子命令與 `--opts` 轉成結構化請求）。
3. **與 WS/Plugin 橋接對齊**：Slash 指令 → 產生與 §3 對等的請求 JSON；若 `--out`，攔截 `plot_blob/file_*` 二進位並直接寫檔；否則把控制訊息與簡短結果列印到 stdout。
4. **錯誤與退出碼**：依 §15.6 對應，並在 REPL 顯示人類可讀的錯誤與 `hint`。
5. **測試**：為各 Slash 指令寫 smoke 測試（含 `--echo-json` 快照比對）。

---

**本章追加的 Slash 指令層不改變任何既有 WS/Plugin 規格，只是薄薄的一層 UX，確保 **``** 在離線/單機也能跑完整流程，且與平台的訊息語意一對一對映。**

