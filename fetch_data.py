import asyncio
import subprocess
import json
import pandas as pd
import numpy as np
import os

# Function to call PowerShell script asynchronously
async def call_powershell_script(ps_script_path):
    process = await asyncio.create_subprocess_exec(
        'powershell.exe', '-File', ps_script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise Exception(f"PowerShell script failed with error: {stderr.decode()}")
    else:
        print(stdout.decode())

# Function to remove BOM from the file content
def remove_bom(file_content):
    bom = b'\xef\xbb\xbf'
    if file_content.startswith(bom):
        return file_content[len(bom):]
    return file_content

async def main():
    # Path to the PowerShell script
    ps_script_path = "fetch_data.ps1"

    # Ensure the output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call the PowerShell script asynchronously
    await call_powershell_script(ps_script_path)

    # Path to the JSON file saved by PowerShell
    json_file_path = "data/historical_data.json"

    # Check if the JSON file exists and is not empty
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, "rb") as file:
            file_content = file.read()
            file_content = remove_bom(file_content)
            file_content = file_content.decode('utf-8')

            if not file_content.strip():
                raise ValueError("JSON file is empty.")
            try:
                data = json.loads(file_content)
            except json.JSONDecodeError as e:
                print(f"Content of the JSON file:\n{file_content}")
                raise ValueError(f"Error decoding JSON: {e}") from e

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
        output_csv_file = "data/historical_data.csv"
        df.to_csv(output_csv_file, index=False)
        print(f"Data saved to {output_csv_file}")
    else:
        raise FileNotFoundError(f"The JSON file {json_file_path} does not exist or is empty.")

# Run the main function
asyncio.run(main())
