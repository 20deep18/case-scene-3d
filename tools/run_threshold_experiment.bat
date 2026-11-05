@echo off
REM 建模最小素材体积实测：先进入 CUDA/MSVC 编译环境（gsplat 训练需要），再跑驱动脚本。
chcp 65001 >nul
set "DSH_ROOT=%~dp0.."
cd /d "%DSH_ROOT%"
call "%DSH_ROOT%\tools\msvc-env.bat" >nul 2>&1
"%DSH_ROOT%\pyruntime312\python.exe" -u "%DSH_ROOT%\tools\threshold_experiment.py" %*
exit /b %errorlevel%
