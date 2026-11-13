@echo off
title Hudasoft Attendance Dashboard
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Dashboard starting. Open http://localhost:5000 in the browser. Leave this window open (minimise it).
waitress-serve --host=0.0.0.0 --port=5000 web_dashboard:app
pause
