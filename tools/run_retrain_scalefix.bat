@echo off
REM Retrain the small-video model with the new feature-scale cap.
setlocal
chcp 65001 >nul
set "DSH_ROOT=%~dp0.."
cd /d "%DSH_ROOT%"
call "%DSH_ROOT%\tools\msvc-env.bat" >nul 2>&1
set "PY=%DSH_ROOT%\pyruntime312\python.exe"
set "J=%DSH_ROOT%\data\projects\case-20261003\recon\recon-muvf7pbo-k9nz1"
"%PY%" -u pipeline\train_gsplat.py --data "%J%\colmap" --images "%J%\images" --out "%J%\train-fix" --iterations 30000 --downscale 2 --sh-degree 3 --lr-scale-factor 0.4 --prune-scale 0.1 --eval-views 4 --save-every 0
echo TRAINDONE
exit /b 0
