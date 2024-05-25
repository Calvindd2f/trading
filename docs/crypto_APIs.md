# Popular Crypto Data APIs

1. CoinGecko API
2. CoinMarketCap API
3. CryptoCompare API
4. Binance API

CoinGecko API is used in this sample.

https://api.coingecko.com/api/v3/coins/maga-hat/market_chart?vs_currency=eur&days=0.004167

## Script to Fetch and Save Historical Data:

`fetch_data.py`

**Run the Data Fetching Script:**

`python fetch_data.py`

This will save the fetched data to `data/historical_data.csv`.

--------------------

**Fetch and Save Historical Data:**

```bash
python fetch_data.py
```

**Generate the Model:**

```bash
python src/retraining/training.py
```

**Run the Flask App:**

```bash
python src/app.py
```

----------------------------------


**request with params**

[Sample call]('https://api.coingecko.com/api/v3/coins/maga-hat/market_chart?vs_currency=eur&days=2')

**response**

```json
{"prices":[[1716473051076,6.869869027086961e-05],[1716476553869,7.437018230769331e-05],[1716480290213,7.669183276485923e-05],[1716483765662,6.757604053669255e-05],[1716487515278,5.514399085485353e-05],[1716491124006,5.5947438953494326e-05],[1716494679952,5.734390570683495e-05],[1716498189649,4.1657473632554656e-05],[1716501807780,4.8145914374609355e-05],[1716505377713,5.6874268060607936e-05],[1716508984853,5.1634881644526815e-05],[1716512578506,5.1603883786639196e-05],[1716516214073,5.5145004722449985e-05],[1716519779881,6.433726951253531e-05],[1716523376753,6.37417518070178e-05],[1716526979122,6.752960905933076e-05],[1716530601828,6.096587955742428e-05],[1716534224369,5.474407365704235e-05],[1716537781164,5.3449446191428514e-05],[1716541346324,5.338554441455465e-05],[1716544849871,5.3081342459348364e-05],[1716548673204,6.241255138286887e-05],[1716552121408,6.025720060532938e-05],[1716555781587,6.202058318357517e-05],[1716559373979,8.501725695770363e-05],[1716563356256,9.271574697828745e-05],[1716566885293,8.396246121622139e-05],[1716570019866,9.195769049533387e-05],[1716573842275,9.607955055020627e-05],[1716577565386,0.00010320501213332599],[1716581220023,0.0001049690274046829],[1716584831984,8.90886253495731e-05],[1716588263941,9.436631666522922e-05],[1716592077602,8.506443084369698e-05],[1716595284026,8.777568255610852e-05],[1716599006951,8.792129947105094e-05],[1716602888300,9.764004318867397e-05],[1716606327326,0.00012226492207684486],[1716609963351,0.0001295079350282893],[1716613602082,0.00012769610866420923],[1716617238544,0.00013275964236140382],[1716620461032,0.00012090729744933047],[1716624451759,0.00012706974506043175],[1716627717665,0.00012891719048846104],[1716631344617,0.00012884082874330542],[1716635177121,0.0001240573964880492],[1716638801300,0.00012433985346762936],[1716642120180,0.00012214562137345967],[1716645580000,0.00013556790057101786]],"market_caps":[[1716473051076,0.0],[1716476553869,0.0],[1716480290213,0.0],[1716483765662,0.0],[1716487515278,0.0],[1716491124006,0.0],[1716494679952,0.0],[1716498189649,0.0],[1716501807780,0.0],[1716505377713,0.0],[1716508984853,0.0],[1716512578506,0.0],[1716516214073,0.0],[1716519779881,0.0],[1716523376753,0.0],[1716526979122,0.0],[1716530601828,0.0],[1716534224369,0.0],[1716537781164,0.0],[1716541346324,21350001.636579756],[1716544849871,22093717.316998836],[1716548673204,25053064.40708239],[1716552121408,24806229.230435953],[1716555781587,25036127.078795094],[1716559373979,31752381.727100406],[1716563356256,38987876.306966916],[1716566885293,34184256.146631755],[1716570019866,37419004.44112064],[1716573842275,39416060.53632439],[1716577565386,42490208.855480924],[1716581220023,42491899.774551846],[1716584831984,36337388.98369431],[1716588263941,36861076.227344066],[1716592077602,34675852.39919452],[1716595284026,35745962.05425763],[1716599006951,35525203.646762885],[1716602888300,39524034.05036559],[1716606327326,49238372.87629112],[1716609963351,49817141.603640035],[1716613602082,51588251.852743216],[1716617238544,54035187.59578466],[1716620461032,49334273.318254866],[1716624451759,50413847.8322584],[1716627717665,52954771.08473483],[1716631344617,52583537.208207995],[1716635177121,49729554.24455735],[1716638801300,50088076.56197766],[1716642120180,50056300.946593314],[1716645580000,53861938.7018816]],"total_volumes":[[1716473051076,23659012.668240894],[1716476553869,23756890.22061064],[1716480290213,23812397.323846936],[1716483765662,23491375.87383248],[1716487515278,28218467.458962753],[1716491124006,27937429.83648728],[1716494679952,27279715.37068319],[1716498189649,28922811.454076834],[1716501807780,29942880.181218628],[1716505377713,31548702.704371117],[1716508984853,28516893.76714719],[1716512578506,27771127.3667276],[1716516214073,26703369.257558774],[1716519779881,25623972.394044172],[1716523376753,27315899.456169434],[1716526979122,26371773.523569617],[1716530601828,25130364.032757375],[1716534224369,24973039.34878798],[1716537781164,24368355.48726723],[1716541346324,23317810.764813665],[1716544849871,16132059.770376567],[1716548673204,24058376.209654097],[1716552121408,24866309.6608291],[1716555781587,23539311.057409912],[1716559373979,25595509.73823137],[1716563356256,29518128.639433194],[1716566885293,30366751.8232586],[1716570019866,28199241.602434937],[1716573842275,28720992.43407948],[1716577565386,8166094.380935632],[1716581220023,29214664.04257035],[1716584831984,26761502.452321738],[1716588263941,25107339.73486529],[1716592077602,25775818.130555768],[1716595284026,24657712.071596347],[1716599006951,25007078.353201073],[1716602888300,25240840.683596656],[1716606327326,35964659.9015317],[1716609963351,35957083.02846393],[1716613602082,39532847.22000076],[1716617238544,41576873.06119633],[1716620461032,41109518.620104834],[1716624451759,41454215.72483055],[1716627717665,43178278.92443006],[1716631344617,44412741.8945111],[1716635177121,44463708.1661022],[1716638801300,45709888.99891201],[1716642120180,46028606.77091352],[1716645580000,45440947.71948549]]}
```

--------------------------

# Regression

#### Modify the PowerShell Script:
+ Ensure the PowerShell script saves the raw JSON data to a temporary file instead of processing it into CSV directly.
+ This will allow Python to read and process this data.

```powershell
# Define global variables and constants
$CryptoId = "bitcoin"
$VsCurrency = "usd"
$Days = "30"
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

    # Save the raw JSON to a file
    $content | Out-File -FilePath $OutputFile -Encoding utf8
    Write-Host "Raw data saved to $OutputFile"
}

Function Execute()
{
    # Main script execution
    try 
    {
        $url = "https://api.coingecko.com/api/v3/coins/$CryptoId/market_chart?vs_currency=$VsCurrency&days=$Days&interval=hourly"
        $apiResponse = Send-ApiRequest -Url $url
        Save-RawJson -ApiResponse $apiResponse -OutputFile $TempJsonFile
    } 
    catch 
    {
        Write-Host "Failed to execute script: $($_.Exception.Message)"
    }
}

Execute
```

#### Python Script to Call PowerShell and Process Data:
+ Use subprocess to call the PowerShell script from Python.
+ Read the JSON data saved by the PowerShell script.
+ Use NumPy and pandas to process the data, then save it to a CSV file.

```python
import subprocess
import json
import pandas as pd
import numpy as np

# Call the PowerShell script
ps_script_path = "path_to_your_script.ps1"
subprocess.call(["powershell.exe", "-File", ps_script_path])

# Read the JSON data saved by PowerShell
with open("data/historical_data.json", "r") as file:
    data = json.load(file)

# Check if the required keys are in the response
if 'prices' not in data or 'total_volumes' not in data:
    raise KeyError("Expected keys not found in the API response")

prices = data['prices']
volumes = data['total_volumes']

# Create a DataFrame using pandas
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['volume'] = [v[1] for v in volumes]
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Process the DataFrame using NumPy (example processing)
df['price'] = np.log(df['price'])  # Example processing: taking log of price

# Save the processed DataFrame to a CSV file
output_csv_file = "data/processed_historical_data.csv"
df.to_csv(output_csv_file, index=False)
print(f"Data saved to {output_csv_file}")
```

## Summary:

+ **PowerShell Script:**
  + Fetches data from the API.
  + Saves the raw JSON data to a file.

+ **Python Script:**
  + Calls the PowerShell script using subprocess.
  + Reads the JSON data from the file.
  + Processes the data using pandas and NumPy.
  + Saves the processed data to a CSV file.
