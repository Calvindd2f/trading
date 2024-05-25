# Define global variables and constants
[string]$CryptoId = "maga-hat"
[string]$VsCurrency = "eur"
[int]$Days = 2
$TempJsonFile = "data/historical_data.json"

# Function to send API requests with retry and pagination
function Send-ApiRequest {
    param (
        [string]$Url,
        [int]$RetryCount = 5,
        [int]$WaitTime = 30,
        [array]$RetryCodes = @(503, 504, 520, 521, 522, 524)
    )

    $success = $false
    $attempt = 0
    $response = $null

    while ($attempt -lt $RetryCount -and -not $success) {
        try {
            $request = [System.Net.HttpWebRequest]::Create($Url)
            $request.Method = "GET"
            $request.ContentType = "application/json"
            $request.Timeout = 10000

            $response = $request.GetResponse()
            $success = $true
        } catch {
            Write-Host "ERROR: $($_.Exception.Message)"
            $ErrorCode = $_.Exception.InnerException.Response.StatusCode
            if ($ErrorCode -in $RetryCodes) {
                $attempt++
                Write-Host "Waiting $WaitTime seconds before retrying..."
                Start-Sleep -Seconds $WaitTime
                $WaitTime *= 2  # Exponential backoff
            } else {
                throw $_
            }
        }
    }
    if ($success -eq $false) {
        throw "Failed to get a successful response after $RetryCount attempts."
    }
    return $response
}

# Function to process the API response and save raw JSON to a file
function Save-RawJson {
    param (
        [System.Net.HttpWebResponse]$ApiResponse,
        [string]$OutputFile
    )

    $reader = [System.IO.StreamReader]::new($ApiResponse.GetResponseStream())
    $content = $reader.ReadToEnd()
    $reader.Close()

    try {
        $jsonContent = $content | ConvertFrom-Json
        if ($null -eq $jsonContent) {
            throw "Received empty or invalid JSON content."
        }
    } catch {
        throw "Failed to parse JSON content: $_"
    }

    # Save the raw JSON to a file
    $content | Out-File -FilePath $OutputFile -Encoding utf8
    Write-Host "Raw data saved to $OutputFile"
}

Function Execute()
{
    # Main script execution
    try 
    {
        $url = "https://api.coingecko.com/api/v3/coins/$CryptoId/market_chart?vs_currency=$VsCurrency&days=$Days"#&interval=hourly"
        $apiResponse = Send-ApiRequest -Url $url
        Save-RawJson -ApiResponse $apiResponse -OutputFile $TempJsonFile
    } 
    catch 
    {
        Write-Host "Failed to execute script: $($_.Exception.Message)"
    }
}

Execute