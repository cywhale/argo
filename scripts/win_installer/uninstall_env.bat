@echo off
setlocal

:: Set environment name
set env_name=odbargo_env

:: Create a log file in a temporary location
set log_file=%temp%\uninstall_env.log
echo Uninstalling odbargo_app at %date% %time% > %log_file%

:: Read the parameter to determine whether to remove the virtual environment
set remove_env=%1

if "%remove_env%" == "true" (
    echo User chose to remove the virtual environment %remove_env% >> %log_file%
    if exist "%CONDA_EXE%" (
        echo Uninstalling odbargo_app package by conda >> %log_file%
        call conda activate %env_name% >> %log_file% 2>&1
        pip uninstall odbargo_app -y >> %log_file% 2>&1
        if %ERRORLEVEL% neq 0 (
            echo Error uninstalling odbargo_app by conda %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )        
        echo Deactivating and removing Conda environment... >> %log_file%
        call conda deactivate >> %log_file% 2>&1
        call conda env remove -n %env_name% >> %log_file% 2>&1
        if %ERRORLEVEL% neq 0 (
            echo Error removing Conda environment %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )
    ) else (
        echo Uninstalling odbargo_app package... >> %log_file%
        call "%~dp0%env_name%\Scripts\activate" >> %log_file% 2>&1
        pip uninstall odbargo_app -y >> %log_file% 2>&1
        if %ERRORLEVEL% neq 0 (
            echo Error uninstalling odbargo_app %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )
        echo Deactivating and removing virtual environment... >> %log_file%
        call "%~dp0%env_name%\Scripts\deactivate" >> %log_file% 2>&1
        rmdir /s /q %~dp0\%env_name% >> %log_file% 2>&1        
        if %ERRORLEVEL% neq 0 (
            echo Error removing virtual environment %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )
    )
) else (
    echo User chose not to remove the virtual environment %remove_env% >> %log_file%
    if exist "%CONDA_EXE%" (
        echo Uninstalling odbargo_app package by conda... >> %log_file%
        call conda activate %env_name% >> %log_file% 2>&1
        pip uninstall odbargo_app -y >> %log_file% 2>&1
        if %ERRORLEVEL% neq 0 (
            echo Error uninstalling odbargo_app by conda %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )
        echo Deactivating Conda environment... >> %log_file%
        call conda deactivate >> %log_file% 2>&1
    ) else (
        echo Uninstalling odbargo_app package... >> %log_file%
        call "%~dp0%env_name%\Scripts\activate" >> %log_file% 2>&1
        pip uninstall odbargo_app -y >> %log_file% 2>&1
        if %ERRORLEVEL% neq 0 (
            echo Error uninstalling odbargo_app %ERRORLEVEL% >> %log_file%
            pause
            exit /b %ERRORLEVEL%
        )
        echo Deactivating virtual environment... >> %log_file%
        call "%~dp0%env_name%\Scripts\deactivate" >> %log_file% 2>&1
    )
)

echo Uninstallation completed at %date% %time% >> %log_file%
pause
endlocal
