$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python scripts/start_sematrace.py @args
