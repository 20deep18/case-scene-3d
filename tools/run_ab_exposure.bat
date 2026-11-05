@echo off
REM A/B: per-frame affine exposure adaptation (--exposure-adapt) on/off.
REM ASCII + CRLF only; must call msvc-env.bat so TORCH_EXTENSIONS_DIR points into the workspace.
setlocal
chcp 65001 >nul
set "DSH_ROOT=%~dp0.."
cd /d "%DSH_ROOT%"
call "%DSH_ROOT%\tools\msvc-env.bat" >nul 2>&1
set "PY=%DSH_ROOT%\pyruntime312\python.exe"
set "W=%DSH_ROOT%\output\ab-prune"
echo [1/2] A: exposure-adapt OFF
"%PY%" -u pipeline\train_gsplat.py --data "%W%\colmap" --images "%W%\images" --out "%W%\exp-off" --iterations 7000 --downscale 2 --sh-degree 1 --eval-views 4 --save-every 0 --prune-scale 0.1
echo [2/2] B: exposure-adapt ON
"%PY%" -u pipeline\train_gsplat.py --data "%W%\colmap" --images "%W%\images" --out "%W%\exp-on" --iterations 7000 --downscale 2 --sh-degree 1 --eval-views 4 --save-every 0 --prune-scale 0.1 --exposure-adapt
echo DONE
exit /b 0
