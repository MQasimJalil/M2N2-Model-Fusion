@echo off
setlocal

:: This script runs the full evaluation suite.
:: 1. Standard Clean Evaluation (via corrupt.py which handles clean + corrupted)
:: 2. Corrupted Evaluation (generating data if needed)

echo Starting Full Evaluation Suite...
echo.

set PYTHONPATH=.
python scripts/corrupt.py --batch_size 128

if ERRORLEVEL 1 goto :error

echo.
echo Evaluation Suite Complete.
goto :eof

:error
echo An error occurred during evaluation.
exit /b 1