param(
    [Parameter(Mandatory=$true)]
    [string]$PythonVersion
)

$ErrorActionPreference = "Stop"

$url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$output = "python_installer.exe"

try {
    # Check if BITS service exists
    $bitsService = Get-Service -Name BITS -ErrorAction SilentlyContinue

    if ($null -eq $bitsService) {
        Write-Host "BITS service is not installed or not available. Falling back to WebClient download."
        throw "BITS service unavailable"
    } elseif ($bitsService.Status -ne 'Running') {
        Write-Host "BITS service is not running. Attempting to start..."
        Start-Service -Name BITS
        Start-Sleep -Seconds 5
    }

    # Start downloading using BITS
    $job = Start-BitsTransfer -Source $url -Destination $output -Asynchronous

    do {
        Start-Sleep -Seconds 5
        $job = Get-BitsTransfer -JobId $job.JobId
        Write-Host "Download status: $($job.JobState)"
    } while ($job.JobState -eq 'Transferring' -or $job.JobState -eq 'Connecting')

    switch ($job.JobState) {
        'Transferred' {
            Complete-BitsTransfer -BitsJob $job
            Write-Host "Download completed successfully."
        }
        'Error' {
            Write-Host "Error downloading file. Error details: $($job.ErrorDescription)"
            Remove-BitsTransfer -BitsJob $job
            exit 1
        }
    }
} catch {
    Write-Host "BITS download failed or BITS service not available. Error: $_"
    Write-Host "Falling back to WebClient download..."

    try {
        # Fallback to WebClient if BITS fails or is unavailable
        Write-Host "Attempting to download using System.Net.WebClient..."
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($url, $output)
        Write-Host "Download completed successfully using WebClient."
    } catch {
        Write-Host "WebClient download failed. An error occurred: $_"
        Write-Host "Stack Trace: $($_.ScriptStackTrace)"
        exit 1
    }
}
