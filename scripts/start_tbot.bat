@echo off
title T-BOT V2
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_tbot.ps1" %*
if errorlevel 1 (
    echo.
    echo Bot exited with error. See logs\bot_*.log
    pause
)
