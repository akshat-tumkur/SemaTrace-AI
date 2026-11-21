"""Stop only the product process tree recorded by this workspace's launcher."""

import json, os, signal, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / ".local/runtime/server.json"
    if not path.exists():
        print("No managed SemaTrace server is recorded.")
        return
    state = json.loads(path.read_text())
    pid = int(state["pid"])
    command = str(ROOT / "backend/product_server.py")
    if state.get("command") != command:
        raise SystemExit(
            "Server identity does not match this workspace. Nothing was stopped."
        )
    if os.name == "nt":
        script = """$ErrorActionPreference='Stop'
$taskPid=[int]$env:SEMATRACE_STOP_PID
$needle=$env:SEMATRACE_STOP_COMMAND
$processes=@(Get-CimInstance Win32_Process)
$rootProcess=$processes | Where-Object ProcessId -eq $taskPid
if (!$rootProcess) { Write-Output 'SemaTrace is already stopped.'; exit 0 }
if (!$rootProcess.CommandLine.Contains($needle)) { throw 'Recorded process identity changed. Nothing was stopped.' }
$owned=@($taskPid)
do {
    $added=@($processes | Where-Object { $owned -contains $_.ParentProcessId -and $owned -notcontains $_.ProcessId -and $_.CommandLine -and $_.CommandLine.Contains($needle) } | Select-Object -ExpandProperty ProcessId)
    $owned+= $added
} while ($added.Count)
[array]::Reverse($owned)
foreach ($taskProcess in $owned) { Stop-Process -Id $taskProcess -ErrorAction SilentlyContinue }
Write-Output 'SemaTrace stopped.'
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            env=dict(
                os.environ, SEMATRACE_STOP_PID=str(pid), SEMATRACE_STOP_COMMAND=command
            ),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="], text=True, capture_output=True
        )
        if result.returncode:
            print("SemaTrace is already stopped.")
        elif command not in result.stdout:
            raise SystemExit("Recorded process identity changed. Nothing was stopped.")
        else:
            os.kill(pid, signal.SIGTERM)
            print("SemaTrace stopped.")
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
