# Keeps Windows awake while any gap_study_v4 python worker runs (the watcher died at ~13:3x on 2026-11-26; workers kept running).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
while (@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'gap_study_v4' }).Count -gt 0) {
    [Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
