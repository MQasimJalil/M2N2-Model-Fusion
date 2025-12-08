@echo off
setlocal

:: Example usage:
:: call scripts\run_eval.bat --model_name EfficientNetBaseline --checkpoint checkpoints\baseline\EfficientNetBaseline_best_seed_42.pth --batch_size 64

python -m src.eval %*
if ERRORLEVEL 1 goto :error
echo Evaluation complete.
goto :eof

:error
echo An error occurred during evaluation.
exit /b 1

