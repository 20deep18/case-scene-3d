@echo off
REM ============================================================================
REM  A/B experiment: prune-scale 0.05 (web default) vs 0.10 (trainer default)
REM  Usage: tools\run_ab_prune.bat [12mb|24mb|...]   (clip tag, default 12mb)
REM
REM  NOTE: keep this file ASCII-only AND CRLF. A UTF-8 BOM or LF-only endings
REM  makes cmd.exe misparse it (observed: "t" eaten -> "rainer is not recognized").
REM ============================================================================
chcp 65001 >nul
set "DSH_ROOT=%~dp0.."
cd /d "%DSH_ROOT%"
call "%DSH_ROOT%\tools\msvc-env.bat" >nul 2>&1
set "PY=%DSH_ROOT%\pyruntime312\python.exe"
set "TAG=%~1"
if "%TAG%"=="" set "TAG=12mb"
set "WORK=%DSH_ROOT%\output\ab-prune-%TAG%"
set "CLIP=%DSH_ROOT%\output\threshold\clips\clip-%TAG%.mp4"

echo [1/4] extract frames
if not exist "%WORK%\images\frame_0001.png" (
  "%PY%" -u pipeline\extract_frames.py --video "%CLIP%" --out "%WORK%\images" --fps 2 --long-side 1600 --blur-threshold 60
)
echo [2/4] colmap (shared by both arms)
if not exist "%WORK%\colmap\sparse\0\images.bin" (
  "%PY%" -u pipeline\colmap_recon.py --images "%WORK%\images" --workspace "%WORK%\colmap" --matcher exhaustive --camera-model OPENCV --use-gpu 1
)
echo [3/4] train A: prune-scale 0.05
"%PY%" -u pipeline\train_gsplat.py --data "%WORK%\colmap" --images "%WORK%\images" --out "%WORK%\a-prune005" --iterations 7000 --downscale 2 --sh-degree 1 --eval-views 4 --save-every 0 --prune-scale 0.05
echo [4/4] train B: prune-scale 0.10
"%PY%" -u pipeline\train_gsplat.py --data "%WORK%\colmap" --images "%WORK%\images" --out "%WORK%\b-prune010" --iterations 7000 --downscale 2 --sh-degree 1 --eval-views 4 --save-every 0 --prune-scale 0.10
echo DONE
exit /b 0
