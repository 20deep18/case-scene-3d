@echo off
REM ============================================================================
REM  Enter a CUDA-capable build environment.
REM
REM  Measured on this machine (important):
REM    CUDA 12.8 host_config.h accepts host compilers up to VS 2022 (_MSC_VER < 1950).
REM    VS 2026 ships MSVC 14.51 (_MSC_VER 1951):
REM        - without -allow-unsupported-compiler: hard #error
REM        - with that flag                    : cudafe++ crashes 0xC0000005
REM    The v143 toolset (MSVC 14.44) that VS 2026 also installs compiles sm_120 fine.
REM  So we pin the 14.44 toolset here.
REM
REM  Usage:  call tools\msvc-env.bat
REM ============================================================================

set "VSDIR=C:\Program Files\Microsoft Visual Studio\18\Community"
set "CUDADIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "TOOLSET=14.44.35207"

if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" exit /b 1

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%TOOLSET% >nul 2>&1
if errorlevel 1 exit /b 1

REM ninja and cmake shipped with VS (PyTorch JIT needs ninja)
set "NINJADIR=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "CMAKEDIR=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "PATH=%NINJADIR%;%CMAKEDIR%;%PATH%"

set "CUDA_PATH=%CUDADIR%"
set "PATH=%CUDADIR%\bin;%PATH%"

REM compile only this machine's compute capability
if "%TORCH_CUDA_ARCH_LIST%"=="" set "TORCH_CUDA_ARCH_LIST=12.0"
REM gsplat defaults to MAX_JOBS=10; 8 cores here, too high risks OOM kills
if "%MAX_JOBS%"=="" set "MAX_JOBS=6"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM Build cache and scratch space must live inside the workspace:
REM   - %LOCALAPPDATA%\torch_extensions is not writable here (WinError 5)
REM   - the sandbox gives a fresh, restricted TMP per invocation, and nvcc
REM     fails to create its temp files there ("Could not open output file")
set "TORCH_EXTENSIONS_DIR=%~dp0..\.buildtmp\torch_ext"
set "TMP=%~dp0..\.buildtmp"
set "TEMP=%~dp0..\.buildtmp"
if not exist "%TORCH_EXTENSIONS_DIR%" mkdir "%TORCH_EXTENSIONS_DIR%" >nul 2>&1
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1

REM matplotlib needs a writable cache dir (%LOCALAPPDATA% is denied here)
set "MPLCONFIGDIR=%~dp0..\.buildtmp\mpl"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>&1

exit /b 0
