#!/usr/bin/env bash

# Verify function checks if the script should execute or not.
function Verify() {
    # Check if the script is being run with sufficient permissions
    if [[ $(id -u) -ne 0 ]]; then
        echo "ERROR: This script must be run as root."
        return 1
    fi

    # Check if the required tools are installed
    if ! command -v wget &> /dev/null || ! command -v unzip &> /dev/null; then
        echo "ERROR: wget and unzip are required to run this script."
        return 1
    fi

    return 0
}

# Unfuck function downloads and installs PyPy based on the system architecture and Python version.
function Unfuck() {
    local -r MAX_RETRY=5
    local -r WAIT_TIME=10
    local -r PYPY_ARCH=$(uname -m | sed 's/x86_64/64/g')
    local -r PYPY_VERSION=""
    local -r URL=""
    local -r PYTHON_VERSION=$(python3 --version 2> /dev/null | sed 's/.*\([0-9]\.[0-9]\.[0-9]*\).*/\1/g')
    local -r COUNT=0

    if [[ $PYTHON_VERSION == "" ]]; then
        echo "WARNING: Unable to determine Python version."
    elif [[ ${PYTHON_VERSION%.*} -ge 311 ]]; then
        PYPY_VERSION="3.10-v7.3.16"
    else
        PYPY_VERSION="3.9-v7.3.9"
    fi

    if [[ $PYPY_ARCH == "x86_64" ]]; then
        URL="https://downloads.python.org/pypy/pypy${PYPY_VERSION}-win64.zip"
    else
        URL="https://downloads.python.org/pypy/pypy${PYPY_VERSION}-win32.zip"
    fi

    while [[ $COUNT -lt $MAX_RETRY && "$URL" != "" ]]; do
        if [[ $COUNT -gt 0 ]]; then
            echo "TIMEOUT: Will retry in $WAIT_TIME seconds."
            sleep $WAIT_TIME
        fi

        echo "Downloading PyPy $URL"
        if ! wget -c --user-agent="Mozilla/5.0 (Windows NT; Windows NT 10.0; en-AU) WindowsPowerShell/5.1.19041.1682" -O "C:\temp\pypy.zip" "$URL" &> /dev/null; then
            if [[ $URL == "" ]]; then
                echo "No url found"
                exit 1
            fi
            echo "Failed to download PyPy"
            exit 1
        fi

        if unzip -t "C:\temp\pypy.zip" &> /dev/null; then
            echo "Unpackaging PyPy"
            unzip -o -d "C:\temp\pypy" "C:\temp\pypy.zip"
            echo "Adding PyPy to PATH"
            export PATH="$PATH;C:\temp\pypy"
            break
        else
            echo "ERROR: The downloaded zip file is corrupted. Retrying..."
            URL=""
        fi

        COUNT=$((COUNT + 1))
    done

    if [[ "$URL" == "" ]]; then
        echo "ERROR: Failed to download and install PyPy."
        exit 1
    fi
}

if Verify; then
    Unfuck
fi
