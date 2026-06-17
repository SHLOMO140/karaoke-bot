@echo off
chcp 65001 >nul

echo Stopping Hebrew Karaoke Telegram Bot...
taskkill /FI "WINDOWTITLE eq Hebrew Karaoke Telegram Bot" /F >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$project = (Resolve-Path '%~dp0').Path; " ^
  "$project = $project.TrimEnd('\'); " ^
  "$projectPattern = [regex]::Escape($project); " ^
  "$starterCmds = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'start_bot\.bat' -and $_.CommandLine -match $projectPattern }; " ^
  "if ($starterCmds) { $starterCmds | ForEach-Object { Start-Process -FilePath taskkill.exe -ArgumentList @('/PID', $_.ProcessId, '/T', '/F') -WindowStyle Hidden -Wait } } " ^
  "else { " ^
  "  $botProcs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'bot\.py' }; " ^
  "  if ($botProcs) { $botProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } " ^
  "}"

if %ERRORLEVEL% == 0 (
    echo Done - bot stopped.
) else (
    echo Bot was not running.
)

timeout /t 2 /nobreak >nul
