@echo off
rem Daily workspace reset - invoked manually or by the "KaraokeBot Daily Reset"
rem scheduled task. Deletes stale jobs and transient junk; never touches
rem model caches, the venv, secrets or source code.
cd /d "%~dp0"
call runtime_env.bat
chcp 65001 >nul
if not exist logs mkdir logs
"%VENV_DIR%\Scripts\python.exe" tools\reset_workspace.py %* >> logs\cleanup_task.log 2>&1
