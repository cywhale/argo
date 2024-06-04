@echo off
setlocal

:: Set environment name
set env_name=odbargo_env

:: Create a log file
set log_file=%~dp0\log\start_odbargo.log
if not exist %~dp0\log mkdir %~dp0\log
echo Starting odbargo_app at %date% %time% > %log_file%

:: Check if Conda environment is available
if exist "%CONDA_EXE%" (
    echo Conda detected at %CONDA_EXE%. Activating Conda environment... >> %log_file%
    cmd /k "conda activate %env_name% && odbargo_app"
    if %ERRORLEVEL% neq 0 (
        echo Error running odbargo_app >> %log_file%
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo Activating virtual environment... >> %log_file%
    cmd /k "%~dp0%env_name%\Scripts\activate && %~dp0%env_name%\Scripts\odbargo_app"
    if %ERRORLEVEL% neq 0 (
        echo Error running odbargo_app >> %log_file%
        pause
        exit /b %ERRORLEVEL%
    )
)

echo odbargo_app started at %date% %time% >> %log_file%
pause
endlocal
