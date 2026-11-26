# Copied unchanged from gap_study/power_watch.ps1 (agent B, 2026-11-26) so gap_study_v3 does not depend on untracked files.
# Power-gated launcher for gap_study runs (Ansel laptop). Detached: survives the Claude session.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','gap_study\power_watch.ps1','-Jobs','gap_study\jobs-phase2.json'
# Behaviour:
#   - waits until the laptop is on AC power, then starts every job of the jobs file (each with --resume)
#   - on battery for more than $GraceSeconds while jobs run: stops them (each model's results are written as soon as
#     it finishes, so --resume loses at most the model in progress) and waits for AC again
#   - keeps Windows awake (SetThreadExecutionState) while jobs run
#   - a job that exits with code 0 is done; a non-zero exit is retried up to $MaxRetries times (logged)
#   - when every job is done, runs the jobs file's "final" command once, writes <jobs>.done and exits
param(
    [Parameter(Mandatory = $true)][string]$Jobs,
    [int]$PollSeconds = 30,
    [int]$GraceSeconds = 180,
    [int]$MaxRetries = 2,
    [switch]$AssumeAC,         # test only: treat the laptop as plugged in
    [string]$WaitFor = ""       # optional: do not start any job before this file exists (e.g. the previous phase's .done)
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
Add-Type -AssemblyName System.Windows.Forms
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$cfg = Get-Content $Jobs -Raw | ConvertFrom-Json
$py = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONPATH = "bit-reduction-test"
$env:PYTHONUNBUFFERED = "1"
$logDir = Join-Path $Root $cfg.log_dir
New-Item -ItemType Directory -Force $logDir | Out-Null
$watchLog = Join-Path $logDir ("power-watch-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
function Write-Log([string]$m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "  " + $m; Add-Content -Path $watchLog -Value $line }
function Test-OnAC { if ($AssumeAC) { return $true }; return ([System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus -eq "Online") }

$state = @{}
foreach ($j in $cfg.jobs) { $state[$j.name] = @{ proc = $null; done = $false; retries = 0 } }
Write-Log "watcher started (pid $PID), jobs: $(($cfg.jobs | ForEach-Object { $_.name }) -join ', ')"
$offSince = $null
if ($WaitFor) { Write-Log "waiting for $WaitFor"; while (-not (Test-Path $WaitFor)) { Start-Sleep -Seconds $PollSeconds }; Write-Log "$WaitFor exists" }
while ($true) {
    $ac = Test-OnAC
    $running = @($cfg.jobs | Where-Object { $state[$_.name].proc -ne $null -and -not $state[$_.name].proc.HasExited })
    # collect exits
    foreach ($j in $cfg.jobs) {
        $s = $state[$j.name]
        if ($s.proc -ne $null -and $s.proc.HasExited) {
            $code = $s.proc.ExitCode
            if ($s.stopped) { Write-Log "$($j.name): stopped (battery)"; }
            elseif ($code -eq 0) { $s.done = $true; Write-Log "$($j.name): finished (exit 0)" }
            else { $s.retries++; Write-Log "$($j.name): exit code $code (retry $($s.retries) of $MaxRetries)"; if ($s.retries -gt $MaxRetries) { $s.done = $true; Write-Log "$($j.name): giving up" } }
            $s.proc = $null; $s.stopped = $false
        }
    }
    if (@($cfg.jobs | Where-Object { -not $state[$_.name].done }).Count -eq 0) { break }
    if ($ac) {
        $offSince = $null
        [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        foreach ($j in $cfg.jobs) {
            $s = $state[$j.name]
            if (-not $s.done -and $s.proc -eq $null) {
                $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
                $out = Join-Path $logDir "$($j.name)-$stamp.log"
                $err = Join-Path $logDir "$($j.name)-$stamp.err.log"
                $s.proc = Start-Process -FilePath $py -ArgumentList $j.args -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
                $null = $s.proc.Handle   # cache the handle so ExitCode is available after exit
                $s.stopped = $false
                Write-Log "$($j.name): started pid $($s.proc.Id) on AC: $($j.args -join ' ')"
            }
        }
    }
    else {
        if ($running.Count -gt 0) {
            if ($offSince -eq $null) { $offSince = Get-Date; Write-Log "on battery with $($running.Count) job(s) running; grace $GraceSeconds s" }
            elseif (((Get-Date) - $offSince).TotalSeconds -ge $GraceSeconds) {
                foreach ($j in $running) { $s = $state[$j.name]; $s.stopped = $true; Stop-Process -Id $s.proc.Id -Force; Write-Log "$($j.name): stopped pid $($s.proc.Id) (battery > $GraceSeconds s)" }
                [Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
                $offSince = $null
            }
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
if ($cfg.final) {
    $fo = Join-Path $logDir ("final-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
    $p = Start-Process -FilePath $py -ArgumentList $cfg.final -WorkingDirectory $Root -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $fo -RedirectStandardError ($fo + ".err")
    Write-Log "final command exit $($p.ExitCode): $($cfg.final -join ' ')"
}
Set-Content -Path ($Jobs + ".done") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Write-Log "all jobs done; watcher exits"
