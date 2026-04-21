@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0bootstrap_windows.ps1" %*
exit /b %ERRORLEVEL%
