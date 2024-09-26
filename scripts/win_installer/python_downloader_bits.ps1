param(
    [Parameter(Mandatory=$true)]
    [string]$PythonVersion,

    [Parameter(Mandatory=$true)]
    [string]$LogFilePath
)

$LogFile = $LogFilePath
$ErrorActionPreference = "Stop"

# Write output to both console and log file
function Log {
    param([string]$message)
    $message | Tee-Object -FilePath $LogFile -Append
}

try {
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $output = "python_installer.exe"
    $statusFile = "status.txt"

    # Check if BITS service is running
    $bitsService = Get-Service -Name BITS
    if ($bitsService.Status -ne 'Running') {
        Log "BITS service is not running. Attempting to start..."
        Start-Service -Name BITS
        Start-Sleep -Seconds 5
    }

    $job = Start-BitsTransfer -Source $url -Destination $output -Asynchronous

    do {
        Start-Sleep -Seconds 5
        $job = Get-BitsTransfer -JobId $job.JobId
        Log "Download status: $($job.JobState)"
    } while ($job.JobState -eq 'Transferring' -or $job.JobState -eq 'Connecting')

    switch ($job.JobState) {
        'Transferred' {
            Complete-BitsTransfer -BitsJob $job
            Log "Download completed successfully."
            Set-Content -Path $statusFile -Value "result:success"
        }
        'Error' {
            Log "Error downloading file. Error details: $($job.ErrorDescription)"
            Remove-BitsTransfer -BitsJob $job
            Set-Content -Path $statusFile -Value "result:error"
            exit 1
        }
    }
} catch {
    Log "An error occurred: $_"
    Set-Content -Path $statusFile -Value "result:error"
    exit 1
}

# Fallback to WebClient if BITS fails
if (!(Test-Path $output)) {
    try {
        Log "BITS download failed. Attempting to download using System.Net.WebClient..."
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($url, $output)
        Log "Download completed successfully using WebClient."
        Set-Content -Path $statusFile -Value "result:success"
    } catch {
        Log "WebClient download failed. An error occurred: $_"
        Set-Content -Path $statusFile -Value "result:error"
        exit 1
    }
}
