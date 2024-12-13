#### ver 0.0.1 create argo_app.py
#### ver 0.0.2 endpoint: /api/floats/download, try websocket notify (not yet)
#### ver 0.0.3 move to argo_app and setup.py for package/Docker (work for net=host)/PyInstaller(not yet)
    -- create window installer by Inno Setup
	-- Make window installer can use Conda
	-- Installer/uninstall works for Conda/Python

#### ver 0.0.4 More customized functions/n1: adjustable running port
    -- modify Inno Setup to check_port feed %PORT% into odbargo_app
    -- add odbargo_app-0.0.4.installer.zip (.exe) in dist/ for downloading test

#### ver 0.0.5 Handle weekly update logic to POSTGIS db/n1
    -- convert numpy types to python natvie types when writing into db
    -- change build workflow, using pyproject.toml/drop some dummy columns in argofloats db
    -- update window installer (Inno Setup), reduce dummy file size greatly(not yet)/n1
    -- pyproject.toml build workflow trials (not yet)/n2
    -- drop pyproject.toml, use setup.py (python -m build) to fix no entry: No module named argo_app/n3
    -- need uvicorn[standard] to include Websocket after v0.30.6
    -- cancel argo_app app.py zipfile (equal size with .nc file)
    -- Filter duplicated WMS/add websocket onclose and error handling
    -- add TIMEOUT for powershell download python installer
    -- Change to BITS download python and fallback to web client method/n1, correctly detect result by enabledelayedexpansion/n2
    -- package upgrade but argopy1.0.0 still have bug (#412, wait official release)/n1
    -- Fix window installer for case without python installed
    -- Fixed argopy==0.1.17 (wait argopy1.0.0 bug fixation official release)/n2 
    -- try to use fixed requirements.txt in odbargo_app istallation (wait argopy1.0.0 bugfix releas)/n3
