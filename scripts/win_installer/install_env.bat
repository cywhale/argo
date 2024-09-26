@echo off
setlocal

:: Define global desired Python version and app version
set desired_python_version=3.11.3
set env_name=odbargo_env
set app_version=0.0.5

:: Create a log file
set log_file=%~dp0\..\log\install_env.log
if not exist %~dp0\..\log mkdir %~dp0\..\log
echo Setting up environment at %date% %time% > %log_file%
goto MAIN

:: Function to install odbargo_app
:install_app
echo Installing odbargo_app... >> %log_file% 2>&1
pip install %~dp0\odbargo_app-%app_version%.tar.gz >> %log_file% 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error installing odbargo_app >> %log_file%
    exit /b %ERRORLEVEL%
)
goto eof

:MAIN
:: Check if Conda is available
if exist "%CONDA_EXE%" (
    echo Conda detected. >> %log_file%
    goto Conda_block
)

goto Python_block

:Conda_block
conda env list | findstr /i /c:"%env_name%" > nul
if %ERRORLEVEL% neq 0 (
    echo Creating new Conda environment... >> %log_file%
	goto Conda_venv
)

echo Conda environment %env_name% already exists. Activating it... >> %log_file%
goto Conda_activate

:Conda_venv
call conda create -n %env_name% python=%desired_python_version% openssl pyopenssl certifi -y >> %log_file% 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error creating Conda environment >> %log_file%
    exit /b 1
)
goto Conda_activate

:Conda_activate
call conda activate %env_name% >> %log_file% 2>&1
set python_path=%CONDA_PREFIX%\python.exe
echo Using Conda Python path: %python_path% >> %log_file%
echo environment: Conda > %~dp0\..\config_app.yaml
goto install_app

:Python_block
:: Check if any Python is already installed
echo Checking if Python is already installed... >> %log_file%
python --version >> %log_file% 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found. Proceeding with installation... >> %log_file%
    :: Old method: use PowerShell
	:: powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/%desired_python_version%/python-%desired_python_version%-amd64.exe -OutFile python_installer.exe -TimeoutSec 300" >> %log_file% 2>&1
	
	:: using BITS (Background Intelligent Transfer Service)
    powershell -ExecutionPolicy Bypass -File "python_downloader_bits.ps1" -PythonVersion "%desired_python_version%" >> %log_file% 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Failed to download Python installer. Check %log_file% for details. >> %log_file%
        exit /b 1
    )
	
    echo Running Python installer... >> %log_file%
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 >> %log_file% 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Error running Python installer >> %log_file%
        exit /b 1
    )
    set python_path="C:\Program Files\Python%desired_python_version:~0,4%\python.exe"
    echo Python installer finished. Checking installation at %python_path% >> %log_file%
) else (
    :: Find the actual path of the installed Python using PowerShell
    for /f "usebackq delims=" %%i in (`powershell -Command "Get-Command python | Select-Object -ExpandProperty Source"`) do set python_path=%%i
    echo Python already installed at %python_path% >> %log_file%
)

:: Check the Python version using the detected path
"%python_path%" --version >> %log_file% 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python installation failed >> %log_file%
    exit /b 1
)
echo Python installed successfully >> %log_file%

:: Check if the virtual environment already exists
if exist "%~dp0\..\%env_name%" (
    echo Virtual environment %env_name% already exists. Activating it... >> %log_file%
    call %~dp0\..\%env_name%\Scripts\activate >> %log_file% 2>&1
) else (
    :: Create a virtual environment using the detected Python
    echo Creating virtual environment... >> %log_file% 2>&1
    "%python_path%" -m venv %~dp0\..\%env_name% >> %log_file% 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Error creating virtual environment >> %log_file%
        exit /b %ERRORLEVEL%
    )
    echo Activating virtual environment... >> %log_file%
    call %~dp0\..\%env_name%\Scripts\activate >> %log_file% 2>&1
)
set python_path=%~dp0\..\%env_name%\Scripts\python.exe
echo Using venv Python path: %python_path% >> %log_file%
echo environment: Regular > %~dp0\..\config_app.yaml
goto install_app

:eof
echo env_name: %env_name% >> %~dp0\..\config_app.yaml
echo Environment setup completed at %date% %time% >> %log_file%
call %~dp0\check_port.bat >> %log_file% 2>&1
endlocal
