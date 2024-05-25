#!/usr/bin/env bash

    """
    Verify function checks if the script should execute or not.

    This function does not take any parameters.

    Returns:
        int: Returns 0 if the script should execute, otherwise it does not return anything.
    """
function Verify()
{
    # This only returns true for now but this script executes based off this function returning true
    return 0
}

    """
    Unfuck function downloads and installs PyPy based on the system architecture and Python version.

    This function does not take any parameters.

    Returns:
        None
    """
function Unfuck()
{
    local retryCount=5
    local success=false
    local count=0
    local pypyVersion
    local url
    local pythonVersion

    pypyVersion=$(uname -m | sed 's/x86_64/64/g')
    if [[ $pypyVersion == "x86_64" ]]; then
        pypyVersion=64
    else
        pypyVersion=32
    fi

    url=""
    pythonVersion=$(python3 --version | sed 's/.*\([0-9]\.[0-9]\.[0-9]*\).*/\1/g')
    if [[ $pythonVersion -ge "3.11" ]]; then
        url="https://downloads.python.org/pypy/pypy3.10-v7.3.16-win$($pypyVersion).zip"
    else
        url="https://downloads.python.org/pypy/pypy3.10-v7.3.16-win$($pypyVersion).zip"
    fi

    while [[ $count -lt $retry && $success == false ]]; do
        if [[ $count -gt 0 ]]; then
            echo "TIMEOUT: Will retry in 10 seconds."
            sleep 10
        fi
        ((count++))

        echo "Downloading PyPy $url"
        wget -c --user-agent="Mozilla/5.0 (Windows NT; Windows NT 10.0; en-AU) WindowsPowerShell/5.1.19041.1682" -O "C:\temp\pypy.zip" "$url"
        if [[ $? -ne 0 ]]; then
            if [[ $url == "" ]]; then
                echo "No url found"
                exit 1
            fi
            echo "Failed to download PyPy"
            exit 1
        fi
        success=true
    done
    echo "Unpackaging PyPy"
    unzip -o -d "C:\temp\pypy" "C:\temp\pypy.zip"
    echo "Adding PyPy to PATH"
    export PATH="$PATH;C:\temp\pypy"
}

if Verify; then
    Unfuck
fi
