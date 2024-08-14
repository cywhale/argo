#### ver 0.0.1 create argo_app.py
#### ver 0.0.2 endpoint: /api/floats/download, try websocket notify (not yet)
#### ver 0.0.3 move to argo_app and setup.py for package/Docker (work for net=host)/PyInstaller(not yet)
    -- create window installer by Inno Setup
	-- Make window installer can use Conda
	-- Installer/uninstall works for Conda/Python

#### ver 0.0.4 More customized functions/n1: adjustable running port
    -- modify Inno Setup to check_port feed %PORT% into odbargo_app
    -- add odbargo_app-0.0.4.installer.zip (.exe) in dist/ for downloading test

#### ver 0.0.5 Handle update logic to POSTGIS db/n1
