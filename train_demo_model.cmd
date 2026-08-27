@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\train_demo_model.ps1" %*

