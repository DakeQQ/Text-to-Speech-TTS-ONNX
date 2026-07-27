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

set "DOWNLOAD_DIR=%USERPROFILE%\Downloads"
set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_v1_Base"
set "F5_TTS_CHECKPOINT=%F5_TTS_MODEL_DIR%\model_1250000.safetensors"
set "F5_TTS_VOCAB=%F5_TTS_MODEL_DIR%\vocab.txt"
set "VOCOS_DIR=%DOWNLOAD_DIR%\vocos-mel-24khz"
set "VOCOS_CONFIG=%VOCOS_DIR%\config.yaml"
set "VOCOS_WEIGHTS=%VOCOS_DIR%\pytorch_model.bin"
set "REQUIRED_IMPORTS=import f5_tts, huggingface_hub, omegaconf, onnx, onnxruntime, onnxslim, pypinyin, rjieba, safetensors, torch, torchaudio, vocos, x_transformers, yaml"

set "OPTIMIZE_MODE=prompt"
if /i "%~1"=="--optimize" set "OPTIMIZE_MODE=yes"
if /i "%~1"=="--no-optimize" set "OPTIMIZE_MODE=no"
if /i "%~1"=="--help" goto :Usage
if /i "%~1"=="/?" goto :Usage
if not "%~1"=="" if /i not "%~1"=="--optimize" if /i not "%~1"=="--no-optimize" goto :InvalidArgument
if not "%~2"=="" goto :InvalidArgument

echo Working directory: %SCRIPT_DIR%
echo.

where python >nul 2>&1 || (
    echo [ERROR] Python was not found on PATH.
    goto :Failed
)
python -c "import sys; assert sys.version_info >= (3, 10), sys.version" >nul 2>&1 || (
    echo [ERROR] Python 3.10 or newer is required.
    goto :Failed
)
python -m pip --version >nul 2>&1 || (
    echo [ERROR] pip is unavailable for the selected Python interpreter.
    goto :Failed
)

echo [1/4] Checking Python dependencies...
python -c "%REQUIRED_IMPORTS%" >nul 2>&1
if errorlevel 1 (
    echo Installing missing export dependencies...
    python -m pip install f5-tts huggingface-hub omegaconf onnx onnxruntime onnxslim onnxconverter-common pypinyin PyYAML rjieba safetensors x-transformers || goto :Failed
    python -c "%REQUIRED_IMPORTS%" >nul 2>&1 || (
        echo [ERROR] One or more Python dependencies still cannot be imported.
        goto :Failed
    )
)

if not exist "%DOWNLOAD_DIR%" (
    mkdir "%DOWNLOAD_DIR%" || goto :Failed
)

echo [2/4] Checking model files...
set "DOWNLOAD_F5="
if not exist "%F5_TTS_CHECKPOINT%" set "DOWNLOAD_F5=1"
if not exist "%F5_TTS_VOCAB%" set "DOWNLOAD_F5=1"
if defined DOWNLOAD_F5 (
    echo Downloading F5-TTS v1 model...
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', local_dir=os.environ['DOWNLOAD_DIR'], allow_patterns=['F5TTS_v1_Base/*'])" || goto :Failed
)
set "DOWNLOAD_VOCOS="
if not exist "%VOCOS_CONFIG%" set "DOWNLOAD_VOCOS=1"
if not exist "%VOCOS_WEIGHTS%" set "DOWNLOAD_VOCOS=1"
if defined DOWNLOAD_VOCOS (
    echo Downloading Vocos model...
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='charactr/vocos-mel-24khz', local_dir=os.environ['VOCOS_DIR'])" || goto :Failed
)
if not exist "%F5_TTS_CHECKPOINT%" (
    echo [ERROR] F5-TTS checkpoint was not found at "%F5_TTS_CHECKPOINT%".
    goto :Failed
)
if not exist "%F5_TTS_VOCAB%" (
    echo [ERROR] F5-TTS vocabulary was not found at "%F5_TTS_VOCAB%".
    goto :Failed
)
if not exist "%VOCOS_CONFIG%" (
    echo [ERROR] Vocos download did not create "%VOCOS_CONFIG%".
    goto :Failed
)
if not exist "%VOCOS_WEIGHTS%" (
    echo [ERROR] Vocos download did not create "%VOCOS_WEIGHTS%".
    goto :Failed
)

echo [3/4] Exporting F5-TTS ONNX models...
python "%EXPORT_PY_FILE%" || goto :Failed
for %%F in (F5_Preprocess.onnx F5_Transformer.onnx F5_Decode.onnx F5_Metadata.onnx) do (
    if not exist "%EXPORT_DIR%\%%F" (
        echo [ERROR] Export completed without creating "%EXPORT_DIR%\%%F".
        goto :Failed
    )
)
echo Exported models: %EXPORT_DIR%
echo.

if /i "%OPTIMIZE_MODE%"=="prompt" (
    choice /c YN /n /m "Optimize the models for DirectML? [Y/N]: "
    if errorlevel 3 goto :Failed
    if errorlevel 2 (
        set "OPTIMIZE_MODE=no"
    ) else if errorlevel 1 (
        set "OPTIMIZE_MODE=yes"
    ) else (
        goto :Failed
    )
)

if /i "%OPTIMIZE_MODE%"=="yes" (
    echo [4/4] Optimizing models for DirectML...
    python "%OPTIMIZE_PY_FILE%" || goto :Failed
    for %%F in (F5_Preprocess.onnx F5_Transformer.onnx F5_Decode.onnx F5_Metadata.onnx) do (
        if not exist "%EXPORT_OP_DIR%\%%F" (
            echo [ERROR] Optimization completed without creating "%EXPORT_OP_DIR%\%%F".
            goto :Failed
        )
    )
    echo Optimized models: %EXPORT_OP_DIR%
) else (
    echo [4/4] DirectML optimization skipped.
)

echo.
echo Completed successfully.
echo To run inference:
echo   python "%INFER_PY_FILE%" --onnx-folder "%EXPORT_DIR%" --vocab-path "%F5_TTS_VOCAB%"
goto :Success

:InvalidArgument
echo [ERROR] Invalid command line.
echo Usage: %~nx0 [--optimize ^| --no-optimize]
goto :Failed

:Usage
echo Usage: %~nx0 [--optimize ^| --no-optimize]
echo.
echo   --optimize       Export and optimize for DirectML without prompting.
echo   --no-optimize    Export only without prompting.
goto :Success

:Failed
echo.
echo Process failed.
popd
endlocal
exit /b 1

:Success
popd
endlocal
exit /b 0
