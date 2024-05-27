# Download and build TA-Lib source code
@echo on

# Set the version of TA-Lib and Visual Studio platform
set TALIB_C_VER=0.4.0
set TALIB_PY_VER=0.4.29
set VS_PLATFORM=x64

# Download TA-Lib source code
curl -L -o talib.zip "https://sourceforge.net/projects/ta-lib/files/ta-lib/%TALIB_C_VER%/ta-lib-%TALIB_C_VER%-msvc.zip"
if errorlevel 1 (
  echo Error downloading TA-Lib source code
  exit /B 1
)

7z x talib.zip -y
if errorlevel 1 (
  echo Error extracting TA-Lib source code
  exit /B 1
)

del talib\ta_lib.c

# Download TA-Lib-Python source code
curl -L -o talib-python.zip "https://github.com/TA-Lib/ta-lib-python/archive/refs/tags/TA_Lib-%TALIB_PY_VER%.zip"
if errorlevel 1 (
  echo Error downloading TA-Lib-Python source code
  exit /B 1
)

7z x talib-python.zip -y --strip-components=1
if errorlevel 1 (
  echo Error extracting TA-Lib-Python source code
  exit /B 1
)

# Apply patch
git --version >nul 2>&1 || (
  echo Git is not installed. Skipping patch application.
)

if exist talib-TA_Lib-%TALIB_PY_VER%\ta_lib\c\mfiles\ta_func.h (
  git -C talib-TA_Lib-%TALIB_PY_VER% apply --verbose --binary talib.diff
  if errorlevel 1 (
    echo Error applying patch
    exit /B 1
  )
)

# Build TA-Lib
msbuild talib-TA_Lib-%TALIB_PY_VER%\ta-lib\c\ide\vs2022\lib_proj\ta_lib.sln /m /t:Clean;Rebuild /p:Configuration=cdr /p:Platform=%VS_PLATFORM%
if errorlevel 1 (
  echo Error building TA-Lib
  exit /B 1
)

copy /Y talib-TA_Lib-%TALIB_PY_VER%\ta-lib\c\include\*.h talib-TA_Lib-%TALIB_PY_VER%\

# Clean up
del talib.zip
del talib-python.zip
del talib-TA_Lib-%TALIB_PY_VER%\ta-lib\c\mfiles\*.obj
del talib-TA_Lib-%TALIB_PY_VER%\ta-lib\c\mfiles\*.exp
del talib-TA_Lib-%TALIB_PY_VER%\ta-lib\c\mfiles\*.lib
