@echo off
setlocal

:: Create a log file
set log_file=%~dp0\log\start_odbargo.log
if not exist %~dp0\log mkdir %~dp0\log
echo Starting odbargo_app at %date% %time% > %log_file%

call %~dp0\venv\Scripts\activate >> %log_file% 2>&1
if ERRORLEVEL 1 (
    echo Error activating virtual environment >> %log_file%
    exit /b 1
)

echo Running odbargo_app... >> %log_file%
%~dp0\venv\Scripts\odbargo_app
if ERRORLEVEL 1 (
    echo Error running odbargo_app >> %log_file%
    exit /b 1
)

echo odbargo_app started at %date% %time% >> %log_file%
endlocal
