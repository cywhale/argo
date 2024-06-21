@echo off
setlocal

:: Create a log file
set log_file=%~dp0\log\start_odbargo.log
if not exist %~dp0\log mkdir %~dp0\log
echo Starting odbargo_app at %date% %time% > %log_file%

:: Read config_app.yaml
for /f "tokens=2 delims=:" %%a in ('findstr "environment" %~dp0\config_app.yaml') do set "env_type=%%a"
for /f "tokens=2 delims=:" %%a in ('findstr "env_name" %~dp0\config_app.yaml') do set "env_name=%%a"
for /f "tokens=2 delims=:" %%a in ('findstr "port" %~dp0\config_app.yaml') do set "PORT=%%a"

:: Trim spaces
set "env_type=%env_type: =%"
set "PORT=%PORT: =%"
set "env_name=%env_name: =%"

:: Debugging output
echo env_type=%env_type% >> %log_file%
echo PORT=%PORT% >> %log_file%
echo env_name=%env_name% >> %log_file%

if "%env_type%"=="Conda" (
    echo In Conda >> %log_file%
    cmd /k "conda activate %env_name% && odbargo_app %PORT%"
    if %ERRORLEVEL% neq 0 (
        echo Error activating Conda environment and run odbargo_app >> %log_file%
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo In normal Python >> %log_file%
    cmd /k "%~dp0\%env_name%\Scripts\activate && %~dp0\%env_name%\Scripts\odbargo_app %PORT%"
    if %ERRORLEVEL% neq 0 (
        echo Error activating virtual environment and run odbargo_app >> %log_file%
        exit /b %ERRORLEVEL%
    )
)

echo odbargo_app started at %date% %time% >> %log_file%
endlocal