@echo off
REM ============================================================================
REM  A/B: does background separation (mask) actually help?
REM
REM  Same material, same params, only --mask-dir differs.
REM  Why A/B and not a single run: masking removes sparse points BY DESIGN,
REM  so "fewer points" proves nothing. Only a controlled pair answers
REM  "did background separation make the model cleaner".
REM
REM  Read afterwards with: python tools/analyze_model_fog.py <out>\point_cloud.ply
REM  (gaussian count, scale P50/P95, >0.5m ratio, fog ratio, robust span)
REM
REM  Usage: cmd /c "tools\run_mask_ab.bat"
REM  NOTE: keep this file ASCII + CRLF (handoff lesson).
REM ============================================================================
call "%~dp0msvc-env.bat" || exit /b 1

set "ROOT=%~dp0.."
set "PY=%ROOT%\pyruntime312\python.exe"
set "IMAGES=%ROOT%\data\projects\case-20261003\recon\recon-muwcu0a4-889pb\images"
set "MASKED=%ROOT%\output\mask-ab\colmap-mask2"
set "PLAIN=%ROOT%\output\mask-ab\colmap-nomask"
set "MASKS=%ROOT%\output\mask-ab\masks"

echo === A: with mask ===
"%PY%" "%ROOT%\pipeline\train_gsplat.py" --data "%MASKED%" --images "%IMAGES%" --out "%ROOT%\output\mask-ab\train-masked" --mask-dir "%MASKS%" --iterations 7000 --downscale 2 --sh-degree 3 --prune-scale 0.1 --eval-views 4 --save-every 0
echo masked-train-exit=%ERRORLEVEL%

echo === B: without mask (control) ===
"%PY%" "%ROOT%\pipeline\train_gsplat.py" --data "%PLAIN%" --images "%IMAGES%" --out "%ROOT%\output\mask-ab\train-plain" --iterations 7000 --downscale 2 --sh-degree 3 --prune-scale 0.1 --eval-views 4 --save-every 0
echo plain-train-exit=%ERRORLEVEL%

echo === done ===
