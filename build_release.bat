@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo   Vivestream Revived - Building Whisper Native Binary
echo ======================================================
echo.

where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Cargo / Rust not found in PATH!
    pause
    exit /b 1
)

echo [*] Compiling optimized release binary...
cargo build --release -j 2
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [*] Build successful! Binary located at:
echo target\release\vivestream-whisper.exe
echo.
dir target\release\vivestream-whisper.exe
echo.
echo ======================================================
echo   Done! Ready for deployment in Vivestream Revived.
echo ======================================================
echo.
pause
