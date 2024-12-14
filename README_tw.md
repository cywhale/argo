# 安裝與使用 odbargo_app

## 簡介  
`odbargo_app` 是生地化 (BGC) Argo 剖面浮標資料下載的一個API本地伺服器，採用了 [argopy](https://argopy.readthedocs.io/en/latest/) 功能從 [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) 伺服器下載 NetCDF。此應用由 [ODB](https://www.odb.ntu.edu.tw/) 開發，並作為在 [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/) 平台上與 ODB Argo WMS 圖層整合之 Argo Mapper 擴充模組。更多細節請參考[這裡](https://api.odb.ntu.edu.tw/hub/?help=Argo)。
請依照以下步驟安裝 Python、建立虛擬環境，以及設置應用程式。

---  

## 安裝步驟  

### 1. 安裝 Python  
若系統尚未安裝 Python：  
- 請從官方網站下載 Python（建議版本：3.12，最低版本：3.10）：  
  https://www.python.org/downloads/  
- 安裝時請確認勾選以下選項：  
  - Add Python to PATH（將 Python 路徑加入環境變數 PATH）  
  - Install for all users（為所有使用者安裝）  

### 2. 建立虛擬環境  
為隔離應用程式相依性（非必要但建議使用）：  

1. 開啟命令提示字元，執行以下指令：  
   ```cmd
   python -m venv odbargo_env
   ```  

2. 啟動虛擬環境：  
   - 在 Windows 上：  
     ```cmd
     odbargo_env\Scripts\activate
     ```  

3. 更新 `pip` 至最新版本：  
   ```cmd
   python -m pip install --upgrade pip
   ```  

### 3. 安裝 `odbargo_app`  
1. 從以下網址下載 `odbargo_app.tar.gz`：  
   https://github.com/cywhale/argo/blob/main/dist/odbargo_app.tar.gz  

2. 在啟動的虛擬環境中執行以下指令：  
   ```cmd
   pip install path\to\odbargo_app.tar.gz
   ```  

3. 安裝成功後啟動應用：  
   ```cmd
   odbargo_app
   ```  
