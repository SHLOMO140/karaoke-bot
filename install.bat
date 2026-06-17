@echo off
title Install Hebrew Karaoke Bot
chcp 65001 >nul

cd /d "%~dp0"
call "%~dp0runtime_env.bat"

set "BASE_PYTHON=C:\Users\shlom\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%BASE_PYTHON%" (
    echo לא נמצא Python בסיסי בנתיב:
    echo %BASE_PYTHON%
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating local virtual environment at:
    echo %VENV_DIR%
    "%BASE_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo יצירת הסביבה המקומית נכשלה.
        pause
        exit /b 1
    )
)

echo Installing Python packages for the karaoke bot into:
echo %VENV_DIR%
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ההתקנה נכשלה. החבילות נשארו בתיקייה המקומית של הפרויקט ולא על C.
    pause
    exit /b 1
)

echo.
echo ההתקנה הושלמה.
echo כל החבילות וה-cache יישמרו תחת:
echo %PROJECT_DIR%\.venv
echo %PROJECT_DIR%\.cache
echo.
echo Set TELEGRAM_BOT_TOKEN before running the bot.
pause
