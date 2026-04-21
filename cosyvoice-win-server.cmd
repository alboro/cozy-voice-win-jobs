@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "PROJECT_ROOT=%~dp0"
set "ENV_NAME=cosyvoice-win-jobs"
if not "%COSYVOICE_ENV_NAME%"=="" set "ENV_NAME=%COSYVOICE_ENV_NAME%"

set "PYTHONPATH=%PROJECT_ROOT%src;%PROJECT_ROOT%vendor\CosyVoice;%PROJECT_ROOT%vendor\CosyVoice\third_party\Matcha-TTS"
set "TEMP=%PROJECT_ROOT%.tmp"
set "TMP=%PROJECT_ROOT%.tmp"
set "XDG_DATA_HOME=%PROJECT_ROOT%.data"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%XDG_DATA_HOME%" mkdir "%XDG_DATA_HOME%"

conda run -n "%ENV_NAME%" python -m cosyvoice_win.server %*
exit /b %ERRORLEVEL%
