#!/bin/bash
# =============================================================================
# Universal DOE Platform - macOS 빌드 스크립트
# =============================================================================
# 이 스크립트는 macOS용 앱 번들과 DMG 설치 이미지를 생성합니다.
# 요구사항: Python 3.8+, PyInstaller, create-dmg (선택)
# =============================================================================

# 색상 코드 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
RESET='\033[0m'

# 빌드 정보
APP_NAME="UniversalDOE"
APP_VERSION="2.0.0"
BUILD_DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo -e "${CYAN}=============================================================================${RESET}"
echo -e "${BLUE}  Universal DOE Platform - macOS Build Script v1.0${RESET}"
echo -e "${CYAN}=============================================================================${RESET}"
echo
echo -e "${WHITE}앱 이름:${RESET} $APP_NAME"
echo -e "${WHITE}버전:${RESET} $APP_VERSION"
echo -e "${WHITE}빌드 시작:${RESET} $BUILD_DATE"
echo

# -----------------------------------------------------------------------------
# 1. 환경 확인
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/8] 환경 확인 중...${RESET}"

# macOS 버전 확인
OS_VERSION=$(sw_vers -productVersion)
echo -e "macOS 버전: $OS_VERSION"

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3가 설치되어 있지 않습니다!${RESET}"
    echo "   brew install python3 명령으로 설치하세요"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "✅ Python $PYTHON_VERSION 확인 완료"

# PyInstaller 확인
if ! python3 -m pip show pyinstaller &> /dev/null; then
    echo -e "${YELLOW}⚠️  PyInstaller가 없습니다. 설치 중...${RESET}"
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ PyInstaller 설치 실패!${RESET}"
        exit 1
    fi
fi
echo -e "✅ PyInstaller 확인 완료"

# create-dmg 확인 (선택사항)
if command -v create-dmg &> /dev/null; then
    echo -e "✅ create-dmg 확인 완료"
    CREATE_DMG_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  create-dmg가 설치되어 있지 않습니다.${RESET}"
    echo "   DMG 생성을 건너뜁니다."
    echo "   설치: brew install create-dmg"
    CREATE_DMG_AVAILABLE=false
fi

# Xcode Command Line Tools 확인
if ! xcode-select -p &> /dev/null; then
    echo -e "${YELLOW}⚠️  Xcode Command Line Tools가 필요합니다.${RESET}"
    echo "   설치 중..."
    xcode-select --install
    echo "   설치 완료 후 다시 실행하세요."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. 가상환경 활성화 (있는 경우)
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[2/8] 가상환경 확인 중...${RESET}"

if [ -f "venv/bin/activate" ]; then
    echo "가상환경 활성화 중..."
    source venv/bin/activate
    echo -e "✅ 가상환경 활성화 완료"
elif [ -f ".venv/bin/activate" ]; then
    echo "가상환경 활성화 중..."
    source .venv/bin/activate
    echo -e "✅ 가상환경 활성화 완료"
else
    echo -e "ℹ️  가상환경 없음 (시스템 Python 사용)"
fi

# -----------------------------------------------------------------------------
# 3. 의존성 설치
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[3/8] 의존성 확인 및 설치 중...${RESET}"

# requirements.txt 확인
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt 파일을 찾을 수 없습니다!${RESET}"
    exit 1
fi

# 빌드 전용 requirements 확인 및 설치
if [ -f "requirements_build.txt" ]; then
    echo "빌드 의존성 설치 중..."
    pip3 install -r requirements_build.txt > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "✅ 빌드 의존성 설치 완료"
    else
        echo -e "${YELLOW}⚠️  일부 빌드 의존성 설치 실패 (계속 진행)${RESET}"
    fi
fi

# -----------------------------------------------------------------------------
# 4. 빌드 디렉토리 정리
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[4/8] 빌드 디렉토리 정리 중...${RESET}"

if [ -d "build/temp" ]; then
    echo "임시 파일 삭제 중..."
    rm -rf "build/temp"
fi

if [ -d "dist/$APP_NAME.app" ]; then
    echo "이전 빌드 삭제 중..."
    rm -rf "dist/$APP_NAME.app"
fi

if [ -f "dist/$APP_NAME.dmg" ]; then
    rm -f "dist/$APP_NAME.dmg"
fi

echo -e "✅ 정리 완료"

# -----------------------------------------------------------------------------
# 5. 애셋 생성
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[5/8] 애셋 파일 생성 중...${RESET}"

cd build/assets
if [ -f "create_assets.py" ]; then
    python3 create_assets.py
    if [ $? -eq 0 ]; then
        echo -e "✅ 애셋 생성 완료"
    else
        echo -e "${YELLOW}⚠️  애셋 생성 중 경고 발생 (계속 진행)${RESET}"
    fi
else
    echo -e "${YELLOW}⚠️  create_assets.py 없음 (기본 애셋 사용)${RESET}"
fi
cd ../..

# 필수 파일 확인
if [ ! -f "launcher.py" ]; then
    echo -e "${RED}❌ launcher.py 파일을 찾을 수 없습니다!${RESET}"
    exit 1
fi

# -----------------------------------------------------------------------------
# 6. PyInstaller 빌드
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[6/8] PyInstaller로 앱 번들 생성 중...${RESET}"
echo "이 작업은 몇 분 정도 걸릴 수 있습니다..."

# 빌드 설정이 있는 경우
if [ -f "build/build_config.py" ]; then
    echo "커스텀 빌드 설정 사용..."
    python3 build/build_config.py --platform Darwin
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 빌드 실패!${RESET}"
        exit 1
    fi
else
    # 기본 PyInstaller 명령
    echo "기본 빌드 설정 사용..."
    python3 -m PyInstaller \
        --name "$APP_NAME" \
        --windowed \
        --onedir \
        --clean \
        --noconfirm \
        --icon="build/assets/icon.icns" \
        --osx-bundle-identifier="com.doeteam.universaldoe" \
        --add-data="pages:pages" \
        --add-data="modules:modules" \
        --add-data="utils:utils" \
        --add-data="config:config" \
        --add-data=".streamlit:.streamlit" \
        --add-data="assets:assets" \
        --hidden-import="streamlit" \
        --hidden-import="plotly" \
        --hidden-import="pandas" \
        --collect-all="streamlit" \
        launcher.py
        
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 빌드 실패!${RESET}"
        exit 1
    fi
fi

echo -e "✅ 앱 번들 생성 완료"

# -----------------------------------------------------------------------------
# 7. 코드 서명 및 공증 (선택사항)
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[7/8] 코드 서명 확인 중...${RESET}"

# 개발자 ID 확인
DEVELOPER_ID=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | awk -F'"' '{print $2}')

if [ -n "$DEVELOPER_ID" ]; then
    echo -e "개발자 ID 발견: $DEVELOPER_ID"
    echo "코드 서명 중..."
    
    # 앱 번들 서명
    codesign --deep --force --verify --verbose \
        --sign "$DEVELOPER_ID" \
        --options runtime \
        --entitlements build/entitlements.plist \
        "dist/$APP_NAME.app"
        
    if [ $? -eq 0 ]; then
        echo -e "✅ 코드 서명 완료"
        
        # 서명 확인
        codesign --verify --verbose "dist/$APP_NAME.app"
        
        # 공증 (선택사항)
        echo -e "${YELLOW}앱 공증은 Apple Developer Program 가입이 필요합니다.${RESET}"
    else
        echo -e "${YELLOW}⚠️  코드 서명 실패 (계속 진행)${RESET}"
    fi
else
    echo -e "${YELLOW}⚠️  개발자 ID가 없습니다. 코드 서명을 건너뜁니다.${RESET}"
    echo "   코드 서명 없이 배포 시 사용자가 보안 경고를 받을 수 있습니다."
fi

# -----------------------------------------------------------------------------
# 8. DMG 생성
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}[8/8] DMG 설치 이미지 생성 중...${RESET}"

if [ "$CREATE_DMG_AVAILABLE" = true ]; then
    echo "create-dmg 실행 중..."
    
    # DMG 생성
    create-dmg \
        --volname "$APP_NAME" \
        --volicon "build/assets/icon.icns" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "$APP_NAME.app" 175 120 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 425 120 \
        --background "build/assets/dmg-background.png" \
        --format UDZO \
        "dist/${APP_NAME}_v${APP_VERSION}.dmg" \
        "dist/$APP_NAME.app"
        
    if [ $? -eq 0 ]; then
        echo -e "✅ DMG 생성 완료"
        
        # DMG 서명 (개발자 ID가 있는 경우)
        if [ -n "$DEVELOPER_ID" ]; then
            echo "DMG 서명 중..."
            codesign --sign "$DEVELOPER_ID" "dist/${APP_NAME}_v${APP_VERSION}.dmg"
        fi
    else
        echo -e "${YELLOW}⚠️  DMG 생성 실패${RESET}"
    fi
else
    echo -e "⏭️  DMG 생성 건너뜀 (create-dmg 없음)"
    
    # 대체 방법: 간단한 DMG 생성
    echo "간단한 DMG 생성 중..."
    
    # 임시 디렉토리 생성
    TEMP_DIR=$(mktemp -d)
    cp -R "dist/$APP_NAME.app" "$TEMP_DIR/"
    
    # Applications 심볼릭 링크 생성
    ln -s /Applications "$TEMP_DIR/Applications"
    
    # DMG 생성
    hdiutil create -volname "$APP_NAME" \
        -srcfolder "$TEMP_DIR" \
        -ov -format UDZO \
        "dist/${APP_NAME}_v${APP_VERSION}.dmg"
        
    # 임시 디렉토리 삭제
    rm -rf "$TEMP_DIR"
    
    echo -e "✅ 간단한 DMG 생성 완료"
fi

# -----------------------------------------------------------------------------
# 빌드 검증
# -----------------------------------------------------------------------------
echo
echo -e "${YELLOW}빌드 검증 중...${RESET}"

# 앱 번들 확인
if [ -d "dist/$APP_NAME.app" ]; then
    echo -e "✅ 앱 번들: dist/$APP_NAME.app"
    
    # 앱 크기 표시
    APP_SIZE=$(du -sh "dist/$APP_NAME.app" | cut -f1)
    echo "   크기: $APP_SIZE"
else
    echo -e "${RED}❌ 앱 번들 생성 실패!${RESET}"
    exit 1
fi

# DMG 확인
if [ -f "dist/${APP_NAME}_v${APP_VERSION}.dmg" ]; then
    echo -e "✅ DMG 파일: dist/${APP_NAME}_v${APP_VERSION}.dmg"
    
    # DMG 크기 표시
    DMG_SIZE=$(du -sh "dist/${APP_NAME}_v${APP_VERSION}.dmg" | cut -f1)
    echo "   크기: $DMG_SIZE"
fi

# -----------------------------------------------------------------------------
# 완료
# -----------------------------------------------------------------------------
echo
echo -e "${GREEN}=============================================================================${RESET}"
echo -e "${GREEN}✅ 빌드 완료!${RESET}"
echo -e "${GREEN}=============================================================================${RESET}"
echo
echo -e "${WHITE}생성된 파일:${RESET}"
echo "  - 앱 번들: dist/$APP_NAME.app"
if [ -f "dist/${APP_NAME}_v${APP_VERSION}.dmg" ]; then
    echo "  - DMG 파일: dist/${APP_NAME}_v${APP_VERSION}.dmg"
fi
echo
echo -e "${CYAN}실행 방법:${RESET}"
echo "  1. Finder에서 dist/$APP_NAME.app 더블클릭"
echo "  2. 터미널: open dist/$APP_NAME.app"
if [ -f "dist/${APP_NAME}_v${APP_VERSION}.dmg" ]; then
    echo "  3. DMG 파일 마운트 후 Applications로 드래그"
fi
echo
echo -e "${WHITE}빌드 완료 시간:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"
echo

# 실행 여부 확인
read -p "지금 앱을 실행하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo -e "${CYAN}앱 실행 중...${RESET}"
    open "dist/$APP_NAME.app"
fi

echo
echo "빌드 스크립트를 종료합니다."
