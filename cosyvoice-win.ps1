$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = if ($env:COSYVOICE_ENV_NAME) { $env:COSYVOICE_ENV_NAME } else { "cosyvoice-win-jobs" }

$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\vendor\CosyVoice;$ProjectRoot\vendor\CosyVoice\third_party\Matcha-TTS"
$env:TEMP = "$ProjectRoot\.tmp"
$env:TMP = "$ProjectRoot\.tmp"
$env:XDG_DATA_HOME = "$ProjectRoot\.data"

New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
New-Item -ItemType Directory -Force -Path $env:XDG_DATA_HOME | Out-Null

conda run -n $EnvName python -m cosyvoice_win.cli @args
