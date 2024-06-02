@echo off
setlocal

:: Create a log file
set log_file=%~dp0\..\log\install_app.log
if not exist %~dp0\..\log mkdir %~dp0\..\log
echo Starting installation at %date% %time% > %log_file%

:: Create a virtual environment
echo Creating virtual environment... >> %log_file% 2>&1
"C:\Program Files\Python311\python.exe" -m venv %~dp0\..\venv >> %log_file% 2>&1
if ERRORLEVEL 1 (
    echo Error creating virtual environment >> %log_file%
    exit /b %ERRORLEVEL%
)

:: Activate the virtual environment
echo Activating virtual environment... >> %log_file% 2>&1
call %~dp0\..\venv\Scripts\activate >> %log_file% 2>&1
if ERRORLEVEL 1 (
    echo Error activating virtual environment >> %log_file%
    exit /b %ERRORLEVEL%
)

:: Install the package
echo Installing odbargo_app... >> %log_file% 2>&1
pip install %~dp0\odbargo_app-0.0.3.tar.gz >> %log_file% 2>&1
if ERRORLEVEL 1 (
    echo Error installing odbargo_app >> %log_file%
    exit /b %ERRORLEVEL%
)

echo Installation completed at %date% %time% >> %log_file%
endlocal
