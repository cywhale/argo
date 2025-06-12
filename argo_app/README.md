# Install and Use odbargo_app
* Deprecated, it's only used before odbargo_app v0.0.6, but no longer maintained *

## Introduction  
`odbargo_app` is a local API server for downloading Biogeochemical (BGC) Argo float data (NetCDF) from the [ERDDAP](https://erddap.ifremer.fr/erddap/index.html) server using [argopy](https://argopy.readthedocs.io/en/latest/) functions. This application is developed by [ODB](https://www.odb.ntu.edu.tw/) and serves as an Argo Mapper plugin to integrate with the ODB Argo WMS layer on the web platform [Ocean APIverse](https://api.odb.ntu.edu.tw/hub/). Visit it for more [details](https://api.odb.ntu.edu.tw/hub/?help=Argo). Follow the steps below to install Python, create a virtual environment, and set up the application.  

---  

## Installation Steps  

### 1. Install Python  
If Python is not installed on your system:  
- Download Python (*recommended version: 3.12, minimum version: 3.10*) from the official Python website:  
  https://www.python.org/downloads/  
- During installation, ensure the following options are checked:  
  - Add Python to PATH  
  - Install for all users  

### 2. Create a Virtual Environment  
To isolate the app's dependencies (*Optional but recommended*):  

1. Open a command prompt and run:  
   ```cmd
   python -m venv odbargo_env
   ```  

2. Activate the virtual environment:  
   - On Windows:  
     ```cmd
     odbargo_env\Scripts\activate
     ```  

3. Update `pip` to the latest version:  
   ```cmd
   python -m pip install --upgrade pip
   ```  

### 3. Install `odbargo_app`  
1. Download the `odbargo_app.tar.gz` file from:  
   https://github.com/cywhale/argo/blob/main/dist/odbargo_app.tar.gz  

2. Run the following command in the activated virtual environment:  
   ```cmd
   pip install path\to\odbargo_app.tar.gz
   ```  

3. Start the application after installation (*in virtual environment if it exists*):  
   ```cmd
   odbargo_app
   ```  
