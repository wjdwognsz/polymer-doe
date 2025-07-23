"""
🎨 Universal DOE Platform - 리소스 생성 스크립트
================================================================================
데스크톱 애플리케이션에 필요한 모든 이미지 리소스를 자동 생성
아이콘, 스플래시 스크린, 설치 마법사 이미지 등
================================================================================
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import struct

# 색상 팔레트
COLORS = {
    'primary': '#1E88E5',        # 메인 블루
    'primary_dark': '#1565C0',   # 진한 블루
    'primary_light': '#64B5F6',  # 연한 블루
    'secondary': '#43A047',      # 서브 그린
    'accent': '#E53935',         # 강조 레드
    'white': '#FFFFFF',
    'gray': '#F5F5F5',
    'dark_gray': '#424242',
    'text': '#212121',
    'text_light': '#757575'
}

# 리소스 생성 경로
ASSETS_DIR = Path(__file__).parent
ASSETS_DIR.mkdir(exist_ok=True)

class AssetGenerator:
    """애플리케이션 리소스 생성기"""
    
    def __init__(self):
        self.assets_dir = ASSETS_DIR
        
    def generate_all(self):
        """모든 리소스 생성"""
        print("🎨 Universal DOE Platform 리소스 생성 시작...")
        
        # 1. 아이콘 생성
        self.create_app_icon()
        
        # 2. 스플래시 스크린 생성
        self.create_splash_screen()
        
        # 3. 설치 마법사 이미지 생성
        self.create_wizard_images()
        
        # 4. 텍스트 파일 생성
        self.create_text_files()
        
        print("✅ 모든 리소스 생성 완료!")
        
    def create_app_icon(self):
        """앱 아이콘 생성 (모든 플랫폼)"""
        print("🖼️ 아이콘 생성 중...")
        
        # 마스터 아이콘 생성 (1024x1024)
        master_size = 1024
        icon = self._create_icon_design(master_size)
        
        # Windows ICO (다중 크기)
        self._save_ico(icon, self.assets_dir / 'icon.ico')
        
        # macOS ICNS
        self._save_icns(icon, self.assets_dir / 'icon.icns')
        
        # Linux PNG
        icon_256 = icon.resize((256, 256), Image.Resampling.LANCZOS)
        icon_256.save(self.assets_dir / 'icon.png', 'PNG')
        
        # 추가 PNG 크기들
        for size in [16, 32, 48, 64, 128, 512]:
            sized_icon = icon.resize((size, size), Image.Resampling.LANCZOS)
            sized_icon.save(self.assets_dir / f'icon_{size}.png', 'PNG')
            
    def _create_icon_design(self, size):
        """아이콘 디자인 생성"""
        # 투명 배경
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # 배경 원
        margin = size // 10
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=COLORS['primary']
        )
        
        # 그라데이션 효과를 위한 오버레이
        overlay = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 상단 하이라이트
        overlay_draw.ellipse(
            [margin, margin, size - margin, size // 2],
            fill=(255, 255, 255, 40)
        )
        
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        
        # 플라스크 아이콘 그리기
        flask_width = size // 3
        flask_height = size // 2
        flask_x = (size - flask_width) // 2
        flask_y = (size - flask_height) // 2 + size // 20
        
        # 플라스크 본체
        points = [
            (flask_x + flask_width // 3, flask_y),  # 상단 왼쪽
            (flask_x + 2 * flask_width // 3, flask_y),  # 상단 오른쪽
            (flask_x + 2 * flask_width // 3, flask_y + flask_height // 4),  # 목 오른쪽
            (flask_x + flask_width, flask_y + flask_height),  # 하단 오른쪽
            (flask_x, flask_y + flask_height),  # 하단 왼쪽
            (flask_x + flask_width // 3, flask_y + flask_height // 4),  # 목 왼쪽
        ]
        draw.polygon(points, fill=COLORS['white'], outline=None)
        
        # 액체 (파란색)
        liquid_points = [
            (flask_x + flask_width // 5, flask_y + flask_height * 2 // 3),
            (flask_x + 4 * flask_width // 5, flask_y + flask_height * 2 // 3),
            (flask_x + flask_width - 10, flask_y + flask_height - 10),
            (flask_x + 10, flask_y + flask_height - 10),
        ]
        draw.polygon(liquid_points, fill=COLORS['primary_light'])
        
        # 버블 효과
        bubble_sizes = [size // 40, size // 50, size // 60]
        bubble_positions = [
            (flask_x + flask_width // 2 - 10, flask_y + flask_height * 3 // 4),
            (flask_x + flask_width // 2 + 15, flask_y + flask_height * 4 // 5),
            (flask_x + flask_width // 3, flask_y + flask_height * 5 // 6),
        ]
        
        for pos, bubble_size in zip(bubble_positions, bubble_sizes):
            draw.ellipse(
                [pos[0] - bubble_size, pos[1] - bubble_size,
                 pos[0] + bubble_size, pos[1] + bubble_size],
                fill=COLORS['white']
            )
        
        # DOE 텍스트
        try:
            # 시스템 폰트 사용 시도
            font_size = size // 8
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # 기본 폰트 사용
            font = ImageFont.load_default()
            
        text = "DOE"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (size - text_width) // 2
        text_y = flask_y + flask_height + margin // 2
        
        # 텍스트 그림자
        shadow_offset = 2
        draw.text(
            (text_x + shadow_offset, text_y + shadow_offset),
            text, fill=(0, 0, 0, 100), font=font
        )
        # 메인 텍스트
        draw.text((text_x, text_y), text, fill=COLORS['white'], font=font)
        
        return img
        
    def _save_ico(self, img, path):
        """Windows ICO 파일 저장"""
        # ICO에 포함할 크기들
        sizes = [(16, 16), (32, 32), (48, 48), (256, 256)]
        
        # PIL의 save 메서드 사용
        img.save(path, format='ICO', sizes=sizes)
        
    def _save_icns(self, img, path):
        """macOS ICNS 파일 저장"""
        # ICNS에 포함할 크기들
        sizes = [16, 32, 48, 128, 256, 512, 1024]
        
        # 임시 PNG 파일들 생성
        temp_files = []
        for size in sizes:
            temp_path = self.assets_dir / f'temp_icon_{size}.png'
            sized_img = img.resize((size, size), Image.Resampling.LANCZOS)
            sized_img.save(temp_path, 'PNG')
            temp_files.append(temp_path)
        
        # iconutil 명령어로 ICNS 생성 (macOS에서만 작동)
        # Windows/Linux에서는 PNG를 ICNS로 변환하는 도구 필요
        try:
            import subprocess
            iconset_path = self.assets_dir / 'icon.iconset'
            iconset_path.mkdir(exist_ok=True)
            
            # iconset 디렉토리에 파일 복사
            for size in sizes:
                if size <= 512:
                    shutil.copy(
                        self.assets_dir / f'temp_icon_{size}.png',
                        iconset_path / f'icon_{size}x{size}.png'
                    )
                    # Retina 디스플레이용
                    if size <= 256:
                        shutil.copy(
                            self.assets_dir / f'temp_icon_{size*2}.png',
                            iconset_path / f'icon_{size}x{size}@2x.png'
                        )
            
            # ICNS 생성
            subprocess.run(['iconutil', '-c', 'icns', str(iconset_path)])
            
            # 임시 파일 정리
            shutil.rmtree(iconset_path)
            
        except:
            # iconutil이 없는 경우 PNG로 대체
            img.save(path, 'PNG')
        
        # 임시 파일 정리
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                
    def create_splash_screen(self):
        """스플래시 스크린 생성"""
        print("🎨 스플래시 스크린 생성 중...")
        
        width, height = 600, 400
        img = Image.new('RGBA', (width, height), COLORS['white'])
        draw = ImageDraw.Draw(img)
        
        # 배경 그라데이션
        for y in range(height):
            # 상단은 흰색, 하단은 연한 회색
            gray_value = 255 - int((y / height) * 10)
            draw.line([(0, y), (width, y)], fill=(gray_value, gray_value, gray_value))
        
        # 중앙 로고
        logo_size = 120
        logo = self._create_icon_design(logo_size * 2)
        logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
        logo_x = (width - logo_size) // 2
        logo_y = height // 3 - logo_size // 2
        img.paste(logo, (logo_x, logo_y), logo)
        
        # 제목
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            subtitle_font = ImageFont.truetype("arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # 메인 타이틀
        title = "Universal DOE Platform"
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = bbox[2] - bbox[0]
        title_x = (width - title_width) // 2
        title_y = logo_y + logo_size + 30
        draw.text((title_x, title_y), title, fill=COLORS['text'], font=title_font)
        
        # 서브타이틀
        subtitle = "AI 기반 실험 설계 플랫폼"
        bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = bbox[2] - bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        subtitle_y = title_y + 45
        draw.text((subtitle_x, subtitle_y), subtitle, fill=COLORS['text_light'], font=subtitle_font)
        
        # 로딩 바 영역
        bar_width = 300
        bar_height = 6
        bar_x = (width - bar_width) // 2
        bar_y = height - 80
        
        # 로딩 바 배경
        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
            fill=COLORS['gray'],
            outline=COLORS['primary_light']
        )
        
        # 버전 정보
        version_text = "Version 2.0.0"
        bbox = draw.textbbox((0, 0), version_text, font=subtitle_font)
        version_width = bbox[2] - bbox[0]
        version_x = (width - version_width) // 2
        version_y = height - 30
        draw.text((version_x, version_y), version_text, fill=COLORS['text_light'], font=subtitle_font)
        
        # 저장
        img.save(self.assets_dir / 'splash.png', 'PNG')
        
    def create_wizard_images(self):
        """설치 마법사 이미지 생성"""
        print("🖼️ 설치 마법사 이미지 생성 중...")
        
        # 큰 이미지 (164x314)
        self._create_wizard_large()
        
        # 작은 이미지 (55x58)
        self._create_wizard_small()
        
    def _create_wizard_large(self):
        """설치 마법사 큰 이미지"""
        width, height = 164, 314
        img = Image.new('RGB', (width, height), COLORS['primary'])
        draw = ImageDraw.Draw(img)
        
        # 패턴 배경
        for i in range(0, width, 20):
            for j in range(0, height, 20):
                if (i + j) % 40 == 0:
                    draw.ellipse(
                        [i, j, i + 15, j + 15],
                        fill=COLORS['primary_light']
                    )
        
        # 중앙 아이콘
        icon_size = 80
        icon = self._create_icon_design(icon_size * 2)
        icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        icon_x = (width - icon_size) // 2
        icon_y = height // 3 - icon_size // 2
        
        # 아이콘 배경 (흰색 원)
        draw.ellipse(
            [icon_x - 10, icon_y - 10, icon_x + icon_size + 10, icon_y + icon_size + 10],
            fill=COLORS['white']
        )
        
        # 아이콘 붙이기
        img.paste(icon, (icon_x, icon_y), icon)
        
        # 텍스트
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 11)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # 타이틀
        lines = ["Universal", "DOE", "Platform"]
        y_offset = icon_y + icon_size + 30
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            draw.text((x, y_offset), line, fill=COLORS['white'], font=font)
            y_offset += 20
        
        # 저장 (BMP 형식)
        img.save(self.assets_dir / 'wizard-image.bmp', 'BMP')
        
    def _create_wizard_small(self):
        """설치 마법사 작은 이미지"""
        width, height = 55, 58
        img = Image.new('RGB', (width, height), COLORS['white'])
        draw = ImageDraw.Draw(img)
        
        # 아이콘
        icon_size = 48
        icon = self._create_icon_design(icon_size * 4)
        icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        icon_x = (width - icon_size) // 2
        icon_y = (height - icon_size) // 2
        img.paste(icon, (icon_x, icon_y), icon)
        
        # 저장 (BMP 형식)
        img.save(self.assets_dir / 'wizard-small-image.bmp', 'BMP')
        
    def create_text_files(self):
        """텍스트 파일 생성"""
        print("📄 텍스트 파일 생성 중...")
        
        # .env.example
        self._create_env_example()
        
        # README.md
        self._create_readme()
        
    def _create_env_example(self):
        """.env.example 파일 생성"""
        env_content = """# ============================================================================
# 🔧 Universal DOE Platform - 환경 설정 템플릿
# ============================================================================
# 이 파일을 .env로 복사하여 사용하세요: cp .env.example .env
# 주의: .env 파일은 절대 Git에 커밋하지 마세요!
# ============================================================================

# -----------------------------------------------------------------------------
# 🤖 AI API 키 설정
# -----------------------------------------------------------------------------
# Google Gemini API (필수) - https://makersuite.google.com/app/apikey
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# xAI Grok API (선택) - https://x.ai/api
XAI_GROK_API_KEY=

# Groq API (선택) - https://console.groq.com
GROQ_API_KEY=

# DeepSeek API (선택) - https://platform.deepseek.com
DEEPSEEK_API_KEY=

# SambaNova API (선택) - https://cloud.sambanova.ai
SAMBANOVA_API_KEY=

# HuggingFace Token (선택) - https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=

# -----------------------------------------------------------------------------
# 📊 Google Sheets 설정
# -----------------------------------------------------------------------------
# Google Sheets URL (필수)
GOOGLE_SHEETS_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit

# Google 서비스 계정 JSON (선택 - 고급 사용자용)
# GOOGLE_SERVICE_ACCOUNT_JSON=path/to/service-account.json

# -----------------------------------------------------------------------------
# 🔐 보안 설정
# -----------------------------------------------------------------------------
# 세션 암호화 키 (자동 생성 권장)
SESSION_SECRET_KEY=your-secret-key-here-min-32-chars-recommended

# JWT 비밀 키
JWT_SECRET_KEY=your-jwt-secret-key-here

# -----------------------------------------------------------------------------
# 📁 로컬 설정 (데스크톱 앱)
# -----------------------------------------------------------------------------
# 데이터 저장 경로 (기본값: 앱 디렉토리)
# DATA_DIR=./data

# 로그 레벨: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# 오프라인 모드 (기본값: false)
OFFLINE_MODE=false

# -----------------------------------------------------------------------------
# 🌐 네트워크 설정
# -----------------------------------------------------------------------------
# Streamlit 포트 (기본값: 8501)
STREAMLIT_PORT=8501

# 프록시 설정 (필요한 경우)
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=https://proxy.example.com:8080

# -----------------------------------------------------------------------------
# 🚀 개발 설정
# -----------------------------------------------------------------------------
# 환경: development, staging, production
APP_ENV=development

# 디버그 모드
DEBUG_MODE=false

# Mock 데이터 사용
USE_MOCK_DATA=false
"""
        
        with open(self.assets_dir / '.env.example', 'w', encoding='utf-8') as f:
            f.write(env_content)
            
    def _create_readme(self):
        """README.md 파일 생성"""
        readme_content = """# 📁 Universal DOE Platform - Assets Directory

이 디렉토리는 Universal DOE Platform의 모든 시각적 리소스를 포함합니다.

## 📋 파일 목록

### 🎨 아이콘 파일
- **icon.ico** - Windows 실행파일 아이콘 (16x16, 32x32, 48x48, 256x256)
- **icon.icns** - macOS 애플리케이션 아이콘 (16x16 ~ 1024x1024)
- **icon.png** - Linux 및 범용 아이콘 (256x256)
- **icon_[size].png** - 다양한 크기의 PNG 아이콘

### 🖼️ 스플래시 스크린
- **splash.png** - 애플리케이션 시작 화면 (600x400)

### 📦 설치 프로그램 이미지
- **wizard-image.bmp** - Inno Setup 큰 이미지 (164x314)
- **wizard-small-image.bmp** - Inno Setup 작은 이미지 (55x58)

### 📄 설정 파일
- **.env.example** - 환경 설정 템플릿

## 🎨 디자인 가이드라인

### 색상 팔레트
- **Primary Blue**: #1E88E5
- **Dark Blue**: #1565C0
- **Light Blue**: #64B5F6
- **Secondary Green**: #43A047
- **Accent Red**: #E53935
- **White**: #FFFFFF
- **Gray**: #F5F5F5

### 아이콘 디자인 원칙
1. **심플함** - 작은 크기에서도 인식 가능
2. **일관성** - 모든 플랫폼에서 동일한 느낌
3. **의미** - 실험/과학을 상징하는 플라스크 모티프
4. **명확성** - 높은 대비와 선명한 윤곽선

## 🛠️ 리소스 재생성

리소스를 다시 생성하려면:

```bash
cd build/assets
python create_assets.py

