@echo off
setlocal

:: Create a log file
set log_file=%~dp0\..\log\install_python.log
if not exist %~dp0\..\log mkdir %~dp0\..\log
echo Checking for Python installation at %date% %time% > %log_file%

:: Check if Python is installed
echo Checking if Python is already installed... >> %log_file%
python --version >> %log_file% 2>&1
if ERRORLEVEL 1 (
    echo Python not found. Proceeding with installation... >> %log_file%
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.3/python-3.11.3-amd64.exe -OutFile python_installer.exe" >> %log_file% 2>&1
    if ERRORLEVEL 1 (
        echo Error downloading Python installer >> %log_file%
        exit /b 1
    )
    echo Running Python installer... >> %log_file%
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 >> %log_file% 2>&1
    if ERRORLEVEL 1 (
        echo Error running Python installer >> %log_file%
        exit /b 1
    )
    echo Python installer finished. Checking installation... >> %log_file%
    :: Temporarily add Python to PATH
    :: set "PATH=%ProgramFiles%\Python311\Scripts\;%ProgramFiles%\Python311\;%PATH%"
    
    "C:\Program Files\Python311\python.exe" --version >> %log_file% 2>&1
    if ERRORLEVEL 1 (
        echo Error: Python installation failed >> %log_file%
        exit /b 1
    )
    echo Python installed successfully >> %log_file%
) else (
    echo Python is already installed. >> %log_file%
)

endlocal
