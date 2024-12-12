@echo off
setlocal

:: Define global desired Python version and app version
set desired_python_version=3.12.5
set env_name=odbargo_env
set app_version=0.0.5

:: Create a log file
set log_file=%~dp0..\log\install_env.log
if not exist %~dp0..\log mkdir %~dp0..\log
echo Setting up environment at %date% %time% > %log_file%
goto MAIN

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
echo environment: Conda > %~dp0..\config_app.yaml
goto install_app

:Python_block
:: Check if any Python is already installed
:: Enable delayed expansion
setlocal enabledelayedexpansion

:: Try to detect Python version
for /f "delims=" %%i in ('python --version 2^>nul') do set "py_version=%%i"

:: Persist py_version after endlocal
endlocal & set "py_version=%py_version%"

:: Log and check if py_version is defined
echo "Now test py_version is: [%py_version%]" >> %log_file%

:: Check if Python is not installed or version is empty
if not defined py_version (
    echo Python not found. Proceeding with installation... >> %log_file%
    goto install_default_python
)

echo "Debugging: Python detected: and find python_path now..." >> %log_file%
:: Find the actual path of the installed Python using PowerShell
for /f "usebackq delims=" %%i in (`powershell -Command "Get-Command python | Select-Object -ExpandProperty Source"`) do set python_path=%%i
echo Python already installed at %python_path% >> %log_file%
goto install_default_venv

:install_default_python
    echo Try to installed python version %desired_python_version%... >> %log_file%
    :: Old method: use PowerShell
	:: powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/%desired_python_version%/python-%desired_python_version%-amd64.exe -OutFile python_installer.exe -TimeoutSec 300" >> %log_file% 2>&1
	
	:: using BITS (Background Intelligent Transfer Service)
    powershell -ExecutionPolicy Bypass -File "python_downloader_bits.ps1" -PythonVersion "%desired_python_version%" -LogFilePath "download_python.log"

    :: Check if the PowerShell script succeeded by looking at the status file
    :: Enable delayed expansion
    setlocal enabledelayedexpansion
    if exist status.txt (
        echo Reading status.txt file >> %log_file%
        
        :: Try using findstr to get the result value
        for /f "tokens=2 delims=:" %%a in ('findstr "result" status.txt') do (
            ::echo "Debug: Found line: %%a" >> %log_file%
            set "result=%%a"
        )
        
        :: Log the result with delayed expansion
        echo "Debugging: Result is: !result!" >> %log_file%
        
        if "!result!"=="success" (
            echo Python installer downloaded successfully. >> %log_file%
        ) else (
            echo Failed to download Python installer. Check download_python.log for details. >> %log_file%
            exit /b 1
        )
    ) else (
        echo Status file not found. Download might have failed. Check %log_file% for details. >> %log_file%
        exit /b 1
    )
    endlocal
    move status.txt %~dp0..\log\download_python_status.txt 2>nul
    move download_python.log %~dp0..\log\ 2>nul

    echo Running Python installer... >> %log_file%
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 /log "python_install_log.txt" >> %log_file% 2>&1
    set installer_exit_code=%ERRORLEVEL%
    ::echo "Debugging: Installer exit code: %installer_exit_code%" >> %log_file%

    if %installer_exit_code% neq 0 (
        echo Error running Python installer with exit code %installer_exit_code% >> %log_file%
        exit /b 1
    )
	
    setlocal enabledelayedexpansion
    set "verx=Python%desired_python_version:~0,4%"
    set "vers=!verx:.=!"
    endlocal & set "vers=%vers%"
	
    set python_path="C:\Program Files\%vers%\python.exe"
    if exist 4 (
        echo Python installation verified successfully at %python_path% >> %log_file%
    ) else (
        echo Python installation failed: python.exe not found at %python_path% >> %log_file%
        exit /b 1
    )
    move python_install_log.txt %~dp0..\log\ 2>nul
    echo Python installer finished. Checking installation at %python_path% >> %log_file%	
:: goto install_default_venv

:: Check the Python version using the detected path
:: "%python_path%" --version >> %log_file% 2>&1
:: if %ERRORLEVEL% neq 0 (
::    echo Error: Python installation failed >> %log_file%
::    exit /b 1
:: )
:: echo Python installed successfully >> %log_file%

:install_default_venv
:: Check if the virtual environment already exists
if exist "%~dp0..\%env_name%" (
    echo Virtual environment %env_name% already exists. Activating it... >> %log_file%
    call %~dp0..\%env_name%\Scripts\activate >> %log_file% 2>&1
) else (
    :: Create a virtual environment using the detected Python
    echo Creating virtual environment by %python_path%... >> %log_file% 2>&1
    "%python_path%" -m venv "%~dp0..\%env_name%" >> %log_file% 2>&1
    if %ERRORLEVEL% neq 0 (
        echo Error creating virtual environment >> %log_file%
        exit /b %ERRORLEVEL%
    )
    echo Activating virtual environment... >> %log_file%
    call %~dp0..\%env_name%\Scripts\activate >> %log_file% 2>&1
)
set python_path=%~dp0..\%env_name%\Scripts\python.exe
echo Using venv Python path: %python_path% >> %log_file%
echo environment: Regular > %~dp0..\config_app.yaml
goto install_app

:: Function to install %~dp0\odbargo_app-%app_version%.tar.gz
:install_app
echo Installing odbargo_app... >> %log_file% 2>&1
"%python_path%" -m pip install %~dp0\odbargo_app.tar.gz >> %log_file% 2>&1

if %ERRORLEVEL% neq 0 (
    echo Error installing odbargo_app >> %log_file%
    exit /b %ERRORLEVEL%
)
goto eof

:eof
echo env_name: %env_name% >> %~dp0..\config_app.yaml
echo Environment setup completed at %date% %time% >> %log_file%
call %~dp0\check_port.bat >> %log_file% 2>&1
endlocal
