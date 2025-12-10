@echo off
setlocal

set SEEDS_BASELINE=42 43 44
set SEEDS_HYBRID=42 43 44
set BASELINE_CONFIG=configs/baseline.yaml
set FUSED_CONFIG=configs/fused.yaml

echo Starting Baseline Model Training (Resuming from Seed 44)...
for %%s in (%SEEDS_BASELINE%) do (
    echo.
    echo Training Baseline Model with seed %%s
    python -m src.train --config %BASELINE_CONFIG% --seed %%s
    if ERRORLEVEL 1 goto :error
)

echo.
echo Starting Hybrid Model Training (All Seeds)...
for %%s in (%SEEDS_HYBRID%) do (
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