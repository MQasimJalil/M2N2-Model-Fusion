@echo off
setlocal

set SEEDS=42 43 44
set BASELINE_CONFIG=configs/baseline.yaml
set FUSED_CONFIG=configs/fused.yaml

echo Starting Baseline Model Training...
for %%s in (%SEEDS%) do (
    echo.
    echo Training Baseline Model with seed %%s
    python -m src.train --config %BASELINE_CONFIG% --seed %%s
    if ERRORLEVEL 1 goto :error
)

echo.
echo Starting Hybrid Model Training...
for %%s in (%SEEDS%) do (
    echo.
    echo Training Hybrid Model with seed %%s
    python -m src.train --config %FUSED_CONFIG% --seed %%s
    if ERRORLEVEL 1 goto :error
)

echo.
echo All training complete!
goto :eof

:error
echo.
echo An error occurred during training.
exit /b 1
