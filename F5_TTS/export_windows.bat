@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0" || exit /b 1
set "SCRIPT_DIR=%CD%"
set "EXPORT_PY_FILE=%SCRIPT_DIR%\Export_F5_TTS.py"
set "OPTIMIZE_PY_FILE=%SCRIPT_DIR%\Optimize_ONNX_DML.py"
set "INFER_PY_FILE=%SCRIPT_DIR%\Inference_F5_TTS_ONNX.py"
set "EXPORT_DIR=%SCRIPT_DIR%\F5_ONNX"
set "EXPORT_OP_DIR=%SCRIPT_DIR%\F5_Optimized"

set "F5_TTS_MODEL_DIR=%USERPROFILE%\Downloads\F5TTS_v1_Base"
set "F5_TTS_CHECKPOINT=%F5_TTS_MODEL_DIR%\model_1250000.safetensors"
set "F5_TTS_VOCAB=%F5_TTS_MODEL_DIR%\vocab.txt"
set "VOCOS_DIR=%DOWNLOAD_DIR%\vocos-mel-24khz"
set "VOCOS_CONFIG=%VOCOS_DIR%\config.yaml"
set "VOCOS_WEIGHTS=%VOCOS_DIR%\pytorch_model.bin"
set "OPTIMIZE_MODE=no"
if /i "%~1"=="--optimize" set "OPTIMIZE_MODE=yes"

echo Working directory: %SCRIPT_DIR%
echo.

echo Exporting F5-TTS ONNX models...
python "%EXPORT_PY_FILE%"
echo Exported models: %EXPORT_DIR%
echo.

if /i "%OPTIMIZE_MODE%"=="yes" (
    echo Optimizing models for DirectML...
    python "%OPTIMIZE_PY_FILE%"
    echo Optimized models: %EXPORT_OP_DIR%
)

echo.
echo Completed successfully.
echo To run inference:
echo   python "%INFER_PY_FILE%" --onnx-folder "%EXPORT_DIR%" --vocab-path "%F5_TTS_VOCAB%"
popd
endlocal
exit /b 0
