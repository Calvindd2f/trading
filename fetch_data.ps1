# Define global variables and constants
$cryptoId = "maga-hat"
$vsCurrency = "eur"
$days = 2
$tempJsonFile = "data/historical_data.json"

# Function to send API requests with retry and pagination
function Send-ApiRequest {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory=$true)]
        [string]$Url,

        [Parameter(Mandatory=$false)]
        [int]$retryCount = 5,

        [Parameter(Mandatory=$false)]
        [int]$waitTime = 30,

        [Parameter(Mandatory=$false)]
        [int[]]$retryCodes = @(503, 504, 520, 521, 522, 524)
    )

    $success = $false
    $attempt = 0
    $response = $null

    while ($attempt -lt $retryCount -and -not $success) {
        try {
            $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 10
            $success = $true
        } catch {
            Write-Host "ERROR: $($_.Exception.Message)"
            $errorCode = $_.Exception.Response.StatusCode.Value__
            if ($errorCode -in $retryCodes) {
                $attempt++
                Write-Host "Waiting $waitTime seconds before retrying..."
                Start-Sleep -Seconds $waitTime
                $waitTime *= 2  # Exponential backoff
            } else {
                throw $_
            }
        }
    }

    if ($success -eq $false) {
        throw "Failed to get a successful response after $retryCount attempts."
    }

    return $response
}

# Function to process the API response and save raw JSON to a file
function Save-RawJson {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory=$true)]
        [object]$apiResponse,

        [Parameter(Mandatory=$true)]
        [string]$outputFile
    )

    $content = $apiResponse | ConvertTo-Json

    try {
        $content | Out-File -FilePath $outputFile -Encoding utf8
        Write-Host "Raw data saved to $outputFile"
    } catch {
        throw "Failed to save JSON content to file: $_"
    }
}

function Execute() {
    # Main script execution
    try {
        $url = "https://api.coingecko.com/api/v3/coins/$cryptoId/market_chart?vs_currency=$vsCurrency&days=$days"
        $apiResponse = Send-ApiRequest -Url $url
        Save-RawJson -ApiResponse $apiResponse -OutputFile $tempJsonFile
    } catch {
        Write-Host "Failed to execute script: $($_.Exception.Message)"
    }
}

Execute
