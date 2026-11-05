@echo off
REM Densification A/B: default vs aggressive, 6000 steps each.
setlocal
chcp 65001 >nul
set "DSH_ROOT=%~dp0.."
cd /d "%DSH_ROOT%"
call "%DSH_ROOT%\tools\msvc-env.bat" >nul 2>&1
set "PY=%DSH_ROOT%\pyruntime312\python.exe"
set "J=%DSH_ROOT%\data\projects\case-20261003\recon\recon-muwcu0a4-889pb"
echo === A: default (grad 2e-4, until 15000) ===
"%PY%" -u pipeline\train_gsplat.py --data "%J%\colmap" --images "%J%\images" --out "%J%\dens-default" --iterations 6000 --downscale 2 --sh-degree 3 --lr-scale-factor 0.4 --prune-scale 0.1 --eval-views 0 --save-every 0
echo === B: aggressive (grad 2e-5, until 6000, percent 0.001) ===
"%PY%" -u pipeline\train_gsplat.py --data "%J%\colmap" --images "%J%\images" --out "%J%\dens-aggressive" --iterations 6000 --downscale 2 --sh-degree 3 --lr-scale-factor 0.4 --prune-scale 0.1 --eval-views 0 --save-every 0 --densify-grad-threshold 2e-5 --densify-until 6000 --densify-percent 0.001
echo ABDONE
exit /b 0
