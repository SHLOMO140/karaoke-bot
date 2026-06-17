@echo off
rem -- cd BEFORE chcp to avoid Hebrew-path encoding issues --
cd /d "%~dp0"
call "runtime_env.bat"
chcp 65001 >nul
title Hebrew Karaoke Telegram Bot

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [!] הסביבה המקומית לא קיימת עדיין.
    echo     הרץ קודם install.bat
    echo.
    pause
    exit /b 1
)

:restart_loop
echo.
echo [%date% %time%] Starting bot...
.venv\Scripts\python.exe bot.py
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% == 0 (
    echo.
    echo Bot stopped normally. Press any key to exit.
    pause
    exit /b 0
)

echo.
echo [%date% %time%] Bot exited with code %EXIT_CODE%. Restarting in 5 seconds...
echo Press Ctrl+C to cancel restart.
timeout /t 5 /nobreak >nul
goto restart_loop
