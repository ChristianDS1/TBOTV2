@echo off
title T-BOT V2 (72h loop)
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_tbot_loop.ps1" %*
echo.
echo Loop stopped.
pause
