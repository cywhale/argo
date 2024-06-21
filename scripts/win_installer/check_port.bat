@echo off
setlocal

:: Create a log file
set log_file=%~dp0\..\log\check_port.log
if not exist %~dp0\..\log mkdir %~dp0\..\log
echo Check port at %date% %time% > %log_file%
goto MAIN

:check_port
:: Check if the default port is available or find the next available one
set /a PORT=8090
:retry_port
echo Checking if port %PORT% is available...
netstat -aon | findstr /R /C:":%PORT% " > nul
if %ERRORLEVEL% neq 0 (
    echo Port %PORT% is available.
    goto end_check
) else (
    echo Port %PORT% is in use.
    goto prompt_port
)

:prompt_port
set /a try_count=0
:input_port
set /p "PORT=Enter a different port number between 1024 and 65535: "
echo User set %PORT% at try: %try_count%
:: Check if the input is an integer
echo %PORT%| findstr /R "^[0-9][0-9]*$" > nul
if %ERRORLEVEL% neq 0 (
    echo Error: Input is not a valid number. Please try again.
    goto input_retry
)

if %PORT% lss 1024 (
    echo Error: Input port number too small (1024-65535). Please try again.
    goto input_retry
)

if %PORT% gtr 65535 (
    echo Error: Input port number too large (1024-65535). Please try again.
    goto input_retry
)

:: Validate the user's port choice
netstat -aon | findstr /R /C:":%PORT% " > nul
if %ERRORLEVEL% neq 0 (
    goto end_check
) else (
    echo Error: Port %PORT% is in use. Please try again.
    goto input_retry
)  

:input_retry
set /a try_count+=1
if %try_count% lss 3 (
    goto input_port
)
echo Error: Tried too many times. Using default port and can be altered in config_app.yaml.
set PORT=8090
goto end_check

:end_check
echo Final port configuration written to config_app.yaml. >> %log_file%
echo Port %PORT% will be used for the application. >> %log_file%
echo port: %PORT% >> %~dp0\..\config_app.yaml
goto eof

:MAIN
:: Setup the environment and install the application
goto check_port

:eof	
echo End checking port at %date% %time% >> %log_file%
endlocal