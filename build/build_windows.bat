@echo off
REM =============================================================================
REM Universal DOE Platform - Windows 빌드 스크립트
REM =============================================================================
REM 이 스크립트는 Windows용 실행파일과 인스톨러를 생성합니다.
REM 요구사항: Python 3.8+, PyInstaller, Inno Setup 6
REM =============================================================================

setlocal enabledelayedexpansion

REM 색상 코드 설정
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set MAGENTA=[95m
set CYAN=[96m
set WHITE=[97m
set RESET=[0m

REM 빌드 정보
set APP_NAME=UniversalDOE
set APP_VERSION=2.0.0
set BUILD_DATE=%DATE% %TIME%

echo %CYAN%=============================================================================%RESET%
echo %BLUE%  Universal DOE Platform - Windows Build Script v1.0%RESET%
echo %CYAN%=============================================================================%RESET%
echo.
echo %WHITE%앱 이름:%RESET% %APP_NAME%
echo %WHITE%버전:%RESET% %APP_VERSION%
echo %WHITE%빌드 시작:%RESET% %BUILD_DATE%
echo.

REM -----------------------------------------------------------------------------
REM 1. 환경 확인
REM -----------------------------------------------------------------------------
echo %YELLOW%[1/8] 환경 확인 중...%RESET%

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Python이 설치되어 있지 않습니다!%RESET%
    echo    https://www.python.org/downloads/ 에서 Python 3.8+ 설치
    goto :error
)
echo ✅ Python 확인 완료

REM PyInstaller 확인
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️  PyInstaller가 없습니다. 설치 중...%RESET%
    pip install pyinstaller
    if errorlevel 1 (
        echo %RED%❌ PyInstaller 설치 실패!%RESET%
        goto :error
    )
)
echo ✅ PyInstaller 확인 완료

REM Inno Setup 확인
set INNO_PATH=
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
    set INNO_PATH=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
) else if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" (
    set INNO_PATH=%ProgramFiles%\Inno Setup 6\ISCC.exe
) else (
    echo %YELLOW%⚠️  Inno Setup이 설치되어 있지 않습니다.%RESET%
    echo    인스톨러 생성을 건너뜁니다.
    echo    https://jrsoftware.org/isdl.php 에서 다운로드 가능
)

REM -----------------------------------------------------------------------------
REM 2. 가상환경 활성화 (있는 경우)
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[2/8] 가상환경 확인 중...%RESET%

if exist "venv\Scripts\activate.bat" (
    echo 가상환경 활성화 중...
    call venv\Scripts\activate.bat
    echo ✅ 가상환경 활성화 완료
) else if exist ".venv\Scripts\activate.bat" (
    echo 가상환경 활성화 중...
    call .venv\Scripts\activate.bat
    echo ✅ 가상환경 활성화 완료
) else (
    echo ℹ️  가상환경 없음 (시스템 Python 사용)
)

REM -----------------------------------------------------------------------------
REM 3. 의존성 설치
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[3/8] 의존성 확인 및 설치 중...%RESET%

REM requirements.txt 확인
if not exist "requirements.txt" (
    echo %RED%❌ requirements.txt 파일을 찾을 수 없습니다!%RESET%
    goto :error
)

REM 빌드 전용 requirements 확인 및 설치
if exist "requirements_build.txt" (
    echo 빌드 의존성 설치 중...
    pip install -r requirements_build.txt >nul 2>&1
    if errorlevel 1 (
        echo %YELLOW%⚠️  일부 빌드 의존성 설치 실패 (계속 진행)%RESET%
    ) else (
        echo ✅ 빌드 의존성 설치 완료
    )
)

REM -----------------------------------------------------------------------------
REM 4. 빌드 디렉토리 정리
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[4/8] 빌드 디렉토리 정리 중...%RESET%

if exist "build\temp" (
    echo 임시 파일 삭제 중...
    rmdir /s /q "build\temp" 2>nul
)

if exist "dist\%APP_NAME%" (
    echo 이전 빌드 삭제 중...
    rmdir /s /q "dist\%APP_NAME%" 2>nul
)

echo ✅ 정리 완료

REM -----------------------------------------------------------------------------
REM 5. 애셋 생성
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[5/8] 애셋 파일 생성 중...%RESET%

cd build\assets
if exist "create_assets.py" (
    python create_assets.py
    if errorlevel 1 (
        echo %YELLOW%⚠️  애셋 생성 중 경고 발생 (계속 진행)%RESET%
    ) else (
        echo ✅ 애셋 생성 완료
    )
) else (
    echo %YELLOW%⚠️  create_assets.py 없음 (기본 애셋 사용)%RESET%
)
cd ..\..

REM 필수 파일 확인
if not exist "launcher.py" (
    echo %RED%❌ launcher.py 파일을 찾을 수 없습니다!%RESET%
    goto :error
)

REM -----------------------------------------------------------------------------
REM 6. PyInstaller 빌드
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[6/8] PyInstaller로 실행파일 생성 중...%RESET%
echo 이 작업은 몇 분 정도 걸릴 수 있습니다...

REM 빌드 설정이 있는 경우
if exist "build\build_config.py" (
    echo 커스텀 빌드 설정 사용...
    python build\build_config.py --platform Windows
    if errorlevel 1 (
        echo %RED%❌ 빌드 실패!%RESET%
        goto :error
    )
) else (
    REM 기본 PyInstaller 명령
    echo 기본 빌드 설정 사용...
    python -m PyInstaller ^
        --name "%APP_NAME%" ^
        --windowed ^
        --onedir ^
        --clean ^
        --noconfirm ^
        --icon="build\assets\icon.ico" ^
        --add-data="pages;pages" ^
        --add-data="modules;modules" ^
        --add-data="utils;utils" ^
        --add-data="config;config" ^
        --add-data=".streamlit;.streamlit" ^
        --add-data="assets;assets" ^
        --hidden-import="streamlit" ^
        --hidden-import="plotly" ^
        --hidden-import="pandas" ^
        --collect-all="streamlit" ^
        launcher.py
        
    if errorlevel 1 (
        echo %RED%❌ 빌드 실패!%RESET%
        goto :error
    )
)

echo ✅ 실행파일 생성 완료

REM -----------------------------------------------------------------------------
REM 7. 인스톨러 생성
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[7/8] 인스톨러 생성 중...%RESET%

if defined INNO_PATH (
    if exist "build\installer_config.iss" (
        echo Inno Setup 실행 중...
        "%INNO_PATH%" /Q "build\installer_config.iss"
        if errorlevel 1 (
            echo %YELLOW%⚠️  인스톨러 생성 중 경고 발생%RESET%
        ) else (
            echo ✅ 인스톨러 생성 완료
        )
    ) else (
        echo %YELLOW%⚠️  installer_config.iss 파일 없음%RESET%
    )
) else (
    echo ⏭️  인스톨러 생성 건너뜀 (Inno Setup 없음)
)

REM -----------------------------------------------------------------------------
REM 8. 빌드 검증 및 정리
REM -----------------------------------------------------------------------------
echo.
echo %YELLOW%[8/8] 빌드 검증 중...%RESET%

REM 실행파일 확인
if exist "dist\%APP_NAME%\%APP_NAME%.exe" (
    echo ✅ 실행파일: dist\%APP_NAME%\%APP_NAME%.exe
    
    REM 파일 크기 표시
    for %%F in ("dist\%APP_NAME%\%APP_NAME%.exe") do (
        set /a SIZE=%%~zF/1048576
        echo    크기: !SIZE! MB
    )
) else (
    echo %RED%❌ 실행파일 생성 실패!%RESET%
    goto :error
)

REM 인스톨러 확인
if exist "dist\%APP_NAME%_Setup_v%APP_VERSION%.exe" (
    echo ✅ 인스톨러: dist\%APP_NAME%_Setup_v%APP_VERSION%.exe
    
    REM 파일 크기 표시
    for %%F in ("dist\%APP_NAME%_Setup_v%APP_VERSION%.exe") do (
        set /a SIZE=%%~zF/1048576
        echo    크기: !SIZE! MB
    )
)

REM -----------------------------------------------------------------------------
REM 완료
REM -----------------------------------------------------------------------------
echo.
echo %GREEN%=============================================================================%RESET%
echo %GREEN%✅ 빌드 완료!%RESET%
echo %GREEN%=============================================================================%RESET%
echo.
echo %WHITE%생성된 파일:%RESET%
echo   - 실행파일: dist\%APP_NAME%\%APP_NAME%.exe
if exist "dist\%APP_NAME%_Setup_v%APP_VERSION%.exe" (
    echo   - 인스톨러: dist\%APP_NAME%_Setup_v%APP_VERSION%.exe
)
echo.
echo %CYAN%실행 방법:%RESET%
echo   1. 직접 실행: dist\%APP_NAME%\%APP_NAME%.exe
echo   2. 인스톨러 실행: dist\%APP_NAME%_Setup_v%APP_VERSION%.exe
echo.
echo %WHITE%빌드 완료 시간:%RESET% %DATE% %TIME%
echo.

REM 실행 여부 확인
choice /C YN /M "지금 프로그램을 실행하시겠습니까?"
if errorlevel 2 goto :end
if errorlevel 1 (
    echo.
    echo %CYAN%프로그램 실행 중...%RESET%
    start "" "dist\%APP_NAME%\%APP_NAME%.exe"
)

goto :end

REM -----------------------------------------------------------------------------
REM 에러 처리
REM -----------------------------------------------------------------------------
:error
echo.
echo %RED%=============================================================================%RESET%
echo %RED%❌ 빌드 실패!%RESET%
echo %RED%=============================================================================%RESET%
echo.
echo 위의 오류 메시지를 확인하고 문제를 해결한 후 다시 시도하세요.
echo.
echo %YELLOW%도움말:%RESET%
echo   - Python 3.8 이상이 필요합니다
echo   - 가상환경 사용을 권장합니다
echo   - requirements.txt의 모든 패키지가 설치되어야 합니다
echo   - 관리자 권한이 필요할 수 있습니다
echo.
pause
exit /b 1

:end
echo.
pause
endlocal
