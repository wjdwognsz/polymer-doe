"""
📦 Universal DOE Platform - PyInstaller 빌드 설정
================================================================================
데스크톱 애플리케이션 빌드 및 패키징 자동화 스크립트
크로스 플랫폼 지원, 리소스 번들링, 코드 서명, 인스톨러 생성
================================================================================
"""

import os
import sys
import shutil
import platform
import subprocess
import json
import hashlib
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# PyInstaller
import PyInstaller.__main__

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / 'build'
DIST_DIR = PROJECT_ROOT / 'dist'
WORK_DIR = BUILD_DIR / 'temp'
ASSETS_DIR = BUILD_DIR / 'assets'

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================================================
# 🔧 빌드 설정
# ===========================================================================

# 앱 정보 (app_config.py에서 가져옴)
try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from config.app_config import APP_INFO
except ImportError:
    APP_INFO = {
        'name': 'UniversalDOE',
        'version': '2.0.0',
        'description': 'Universal Design of Experiments Platform',
        'author': 'DOE Team',
        'website': 'https://universaldoe.com',
        'email': 'support@universaldoe.com'
    }

# 빌드 설정
BUILD_CONFIG = {
    'app_name': APP_INFO['name'].replace(' ', ''),
    'app_version': APP_INFO['version'],
    'app_description': APP_INFO['description'],
    'app_author': APP_INFO['author'],
    'app_website': APP_INFO['website'],
    'app_email': APP_INFO['email'],
    'app_id': 'com.doeteam.universaldoe',  # 역방향 도메인
    'main_script': 'launcher.py',
    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
}

# 플랫폼별 설정
PLATFORM_CONFIG = {
    'Windows': {
        'icon': ASSETS_DIR / 'icon.ico',
        'file_extension': '.exe',
        'installer': 'inno',
        'architecture': platform.machine()
    },
    'Darwin': {  # macOS
        'icon': ASSETS_DIR / 'icon.icns',
        'file_extension': '.app',
        'installer': 'dmg',
        'bundle_identifier': BUILD_CONFIG['app_id']
    },
    'Linux': {
        'icon': ASSETS_DIR / 'icon.png',
        'file_extension': '',
        'installer': 'appimage',
        'desktop_file': True
    }
}

# PyInstaller 기본 옵션
PYINSTALLER_BASE_OPTIONS = [
    '--name', BUILD_CONFIG['app_name'],
    '--clean',
    '--noconfirm',
    '--log-level', 'INFO',
    '--distpath', str(DIST_DIR),
    '--workpath', str(WORK_DIR),
    '--specpath', str(BUILD_DIR),
]

# 데이터 파일 및 디렉토리
DATA_FILES = [
    ('pages', 'pages'),
    ('modules', 'modules'),
    ('utils', 'utils'),
    ('config', 'config'),
    ('.streamlit', '.streamlit'),
    ('assets', 'assets'),
]

# 숨겨진 임포트 (PyInstaller가 자동 감지 못하는 모듈)
HIDDEN_IMPORTS = [
    # Streamlit 관련
    'streamlit',
    'streamlit.components.v1',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.uploaded_file_manager',
    
    # AI/ML 라이브러리
    'google.generativeai',
    'openai',
    'transformers',
    'torch',
    'huggingface_hub',
    
    # 데이터 과학
    'pandas',
    'numpy',
    'scipy',
    'sklearn',
    'pyDOE3',
    'statsmodels',
    
    # 시각화
    'plotly',
    'matplotlib',
    'seaborn',
    'altair',
    
    # Google 통합
    'google.auth',
    'google.oauth2',
    'gspread',
    'gspread_pandas',
    
    # 유틸리티
    'dotenv',
    'requests',
    'aiohttp',
    'cryptography',
    'bcrypt',
    'jwt',
    'pytz',
    
    # 플랫폼별
    'tkinter',
    'webview',  # pywebview
]

# 제외할 모듈 (크기 최적화)
EXCLUDED_MODULES = [
    'test',
    'tests',
    'pytest',
    'sphinx',
    'notebook',
    'IPython',
    'ipykernel',
    'jupyter',
    'pylint',
    'black',
    'flake8',
    'mypy',
]

# ===========================================================================
# 🔨 빌드 함수
# ===========================================================================

class DOEAppBuilder:
    """Universal DOE Platform 빌드 관리자"""
    
    def __init__(self, platform_name: Optional[str] = None):
        self.platform = platform_name or platform.system()
        self.config = PLATFORM_CONFIG.get(self.platform, PLATFORM_CONFIG['Linux'])
        self.build_time = datetime.now()
        
    def clean_build_dirs(self):
        """빌드 디렉토리 정리"""
        logger.info("빌드 디렉토리 정리 중...")
        
        for dir_path in [WORK_DIR, DIST_DIR]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def prepare_assets(self):
        """애셋 파일 준비"""
        logger.info("애셋 파일 준비 중...")
        
        # 아이콘 확인
        icon_path = self.config['icon']
        if not icon_path.exists():
            logger.warning(f"아이콘 파일 없음: {icon_path}")
            # 기본 아이콘 생성 또는 다운로드
            self._create_default_icon(icon_path)
            
        # 스플래시 스크린
        splash_path = ASSETS_DIR / 'splash.png'
        if not splash_path.exists():
            logger.warning("스플래시 스크린 없음")
            
    def build_spec_file(self) -> Path:
        """PyInstaller spec 파일 생성"""
        logger.info("Spec 파일 생성 중...")
        
        spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-
# Universal DOE Platform PyInstaller Spec
# Generated: {self.build_time.isoformat()}

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Streamlit 데이터 수집
streamlit_datas = []
streamlit_binaries = []
streamlit_hiddenimports = []

# Streamlit 전체 수집
datas, binaries, hiddenimports = collect_all('streamlit')
streamlit_datas += datas
streamlit_binaries += binaries
streamlit_hiddenimports += hiddenimports

# Plotly 데이터 수집
datas, binaries, hiddenimports = collect_all('plotly')
streamlit_datas += datas

# Altair 데이터 수집
streamlit_datas += collect_data_files('altair')

# 추가 데이터 파일
added_files = {DATA_FILES}

a = Analysis(
    ['{PROJECT_ROOT / BUILD_CONFIG["main_script"]}'],
    pathex=['{PROJECT_ROOT}'],
    binaries=streamlit_binaries,
    datas=streamlit_datas + added_files,
    hiddenimports={HIDDEN_IMPORTS} + streamlit_hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={EXCLUDED_MODULES},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 바이너리 제거 (크기 최적화)
a.binaries = [x for x in a.binaries if not x[0].startswith('scipy.test')]
a.binaries = [x for x in a.binaries if not x[0].startswith('numpy.test')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{BUILD_CONFIG["app_name"]}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 앱
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {self._get_platform_exe_options()}
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{BUILD_CONFIG["app_name"]}',
)

{self._get_platform_bundle_options()}
"""
        
        spec_path = BUILD_DIR / f"{BUILD_CONFIG['app_name']}.spec"
        with open(spec_path, 'w', encoding='utf-8') as f:
            f.write(spec_content)
            
        return spec_path
        
    def _get_platform_exe_options(self) -> str:
        """플랫폼별 EXE 옵션"""
        options = []
        
        if self.platform == 'Windows':
            options.extend([
                f"icon='{self.config['icon']}'",
                f"version='{BUILD_DIR / 'version.txt'}'",
                "uac_admin=False",
                "uac_uiaccess=False",
            ])
        elif self.platform == 'Darwin':
            options.extend([
                f"icon='{self.config['icon']}'",
                "bundle_identifier='com.doeteam.universaldoe'",
            ])
            
        return ',\n    '.join(options)
        
    def _get_platform_bundle_options(self) -> str:
        """플랫폼별 번들 옵션"""
        if self.platform == 'Darwin':
            return f"""
app = BUNDLE(
    coll,
    name='{BUILD_CONFIG["app_name"]}.app',
    icon='{self.config['icon']}',
    bundle_identifier='{self.config['bundle_identifier']}',
    info_plist={{
        'CFBundleName': '{BUILD_CONFIG["app_name"]}',
        'CFBundleDisplayName': '{APP_INFO["name"]}',
        'CFBundleVersion': '{BUILD_CONFIG["app_version"]}',
        'CFBundleShortVersionString': '{BUILD_CONFIG["app_version"]}',
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.12.0',
    }},
)
"""
        return ""
        
    def build_executable(self):
        """실행파일 빌드"""
        logger.info(f"{self.platform} 실행파일 빌드 시작...")
        
        # PyInstaller 옵션
        options = PYINSTALLER_BASE_OPTIONS.copy()
        
        # 플랫폼별 옵션
        if self.platform == 'Windows':
            options.extend(['--windowed', '--onedir'])
            if self.config['icon'].exists():
                options.extend(['--icon', str(self.config['icon'])])
                
            # 스플래시 스크린
            splash = ASSETS_DIR / 'splash.png'
            if splash.exists():
                options.extend(['--splash', str(splash)])
                
        elif self.platform == 'Darwin':
            options.extend(['--windowed', '--onedir'])
            
        elif self.platform == 'Linux':
            options.extend(['--onedir'])
            
        # 데이터 파일 추가
        for src, dst in DATA_FILES:
            src_path = PROJECT_ROOT / src
            if src_path.exists():
                options.extend(['--add-data', f'{src_path}{os.pathsep}{dst}'])
                
        # 숨겨진 임포트 추가
        for module in HIDDEN_IMPORTS:
            options.extend(['--hidden-import', module])
            
        # 제외 모듈
        for module in EXCLUDED_MODULES:
            options.extend(['--exclude-module', module])
            
        # 메인 스크립트
        options.append(str(PROJECT_ROOT / BUILD_CONFIG['main_script']))
        
        # PyInstaller 실행
        try:
            PyInstaller.__main__.run(options)
            logger.info("빌드 성공!")
        except Exception as e:
            logger.error(f"빌드 실패: {e}")
            raise
            
    def create_version_file(self):
        """Windows 버전 정보 파일 생성"""
        if self.platform != 'Windows':
            return
            
        logger.info("Windows 버전 파일 생성 중...")
        
        version_info = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({','.join(BUILD_CONFIG['app_version'].split('.'))}),
    prodvers=({','.join(BUILD_CONFIG['app_version'].split('.'))}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'{BUILD_CONFIG["app_author"]}'),
        StringStruct(u'FileDescription', u'{BUILD_CONFIG["app_description"]}'),
        StringStruct(u'FileVersion', u'{BUILD_CONFIG["app_version"]}'),
        StringStruct(u'InternalName', u'{BUILD_CONFIG["app_name"]}'),
        StringStruct(u'LegalCopyright', u'Copyright (c) {datetime.now().year} {BUILD_CONFIG["app_author"]}'),
        StringStruct(u'OriginalFilename', u'{BUILD_CONFIG["app_name"]}.exe'),
        StringStruct(u'ProductName', u'{APP_INFO["name"]}'),
        StringStruct(u'ProductVersion', u'{BUILD_CONFIG["app_version"]}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
        
        version_file = BUILD_DIR / 'version.txt'
        with open(version_file, 'w', encoding='utf-8') as f:
            f.write(version_info)
            
    def code_sign(self):
        """코드 서명"""
        logger.info("코드 서명 중...")
        
        if self.platform == 'Windows':
            self._sign_windows()
        elif self.platform == 'Darwin':
            self._sign_macos()
            
    def _sign_windows(self):
        """Windows 코드 서명"""
        # signtool.exe 사용
        exe_path = DIST_DIR / BUILD_CONFIG['app_name'] / f"{BUILD_CONFIG['app_name']}.exe"
        
        if not exe_path.exists():
            logger.warning("실행파일을 찾을 수 없어 서명을 건너뜁니다")
            return
            
        # 인증서가 있는 경우만 서명
        cert_path = os.environ.get('WINDOWS_CERT_PATH')
        cert_password = os.environ.get('WINDOWS_CERT_PASSWORD')
        
        if cert_path and cert_password:
            cmd = [
                'signtool', 'sign',
                '/f', cert_path,
                '/p', cert_password,
                '/t', 'http://timestamp.digicert.com',
                '/fd', 'SHA256',
                str(exe_path)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info("Windows 코드 서명 완료")
            except subprocess.CalledProcessError as e:
                logger.error(f"코드 서명 실패: {e}")
        else:
            logger.info("인증서가 없어 코드 서명을 건너뜁니다")
            
    def _sign_macos(self):
        """macOS 코드 서명"""
        app_path = DIST_DIR / f"{BUILD_CONFIG['app_name']}.app"
        
        if not app_path.exists():
            logger.warning("앱 번들을 찾을 수 없어 서명을 건너뜁니다")
            return
            
        # 개발자 ID가 있는 경우만 서명
        developer_id = os.environ.get('MACOS_DEVELOPER_ID')
        
        if developer_id:
            cmd = [
                'codesign',
                '--deep',
                '--force',
                '--verify',
                '--verbose',
                '--sign', developer_id,
                str(app_path)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info("macOS 코드 서명 완료")
                
                # 공증 (notarization) - 선택적
                self._notarize_macos(app_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"코드 서명 실패: {e}")
        else:
            logger.info("개발자 ID가 없어 코드 서명을 건너뜁니다")
            
    def create_installer(self):
        """인스톨러 생성"""
        logger.info(f"{self.platform} 인스톨러 생성 중...")
        
        if self.platform == 'Windows':
            self._create_windows_installer()
        elif self.platform == 'Darwin':
            self._create_macos_installer()
        elif self.platform == 'Linux':
            self._create_linux_installer()
            
    def _create_windows_installer(self):
        """Windows 인스톨러 생성 (Inno Setup)"""
        inno_script = BUILD_DIR / 'installer_config.iss'
        
        if not inno_script.exists():
            logger.warning("Inno Setup 스크립트가 없습니다")
            return
            
        # Inno Setup 컴파일러 경로
        iscc_path = "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe"
        
        if not Path(iscc_path).exists():
            # 다른 경로 시도
            iscc_path = shutil.which('iscc')
            
        if iscc_path:
            cmd = [iscc_path, str(inno_script)]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info("Windows 인스톨러 생성 완료")
            except subprocess.CalledProcessError as e:
                logger.error(f"인스톨러 생성 실패: {e}")
        else:
            logger.warning("Inno Setup이 설치되어 있지 않습니다")
            
    def _create_macos_installer(self):
        """macOS DMG 생성"""
        app_name = BUILD_CONFIG['app_name']
        dmg_name = f"{app_name}-{BUILD_CONFIG['app_version']}.dmg"
        
        # create-dmg 사용 (brew install create-dmg)
        cmd = [
            'create-dmg',
            '--volname', app_name,
            '--window-pos', '200', '120',
            '--window-size', '600', '400',
            '--icon-size', '100',
            '--icon', f"{app_name}.app", '175', '120',
            '--hide-extension', f"{app_name}.app",
            '--app-drop-link', '425', '120',
            str(DIST_DIR / dmg_name),
            str(DIST_DIR)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("macOS DMG 생성 완료")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("create-dmg가 설치되어 있지 않습니다. 기본 방법 사용...")
            self._create_simple_dmg()
            
    def _create_linux_installer(self):
        """Linux AppImage 생성"""
        # AppImage 도구 사용
        logger.info("Linux AppImage 생성은 별도 도구가 필요합니다")
        
        # .desktop 파일 생성
        desktop_content = f"""[Desktop Entry]
Name={APP_INFO['name']}
Comment={APP_INFO['description']}
Exec={BUILD_CONFIG['app_name']}
Icon={BUILD_CONFIG['app_name']}
Terminal=false
Type=Application
Categories=Education;Science;
"""
        
        desktop_file = DIST_DIR / f"{BUILD_CONFIG['app_name']}.desktop"
        with open(desktop_file, 'w') as f:
            f.write(desktop_content)
            
        logger.info(".desktop 파일 생성 완료")
        
    def create_archive(self):
        """배포용 압축 파일 생성"""
        logger.info("배포 아카이브 생성 중...")
        
        # 압축할 디렉토리
        if self.platform == 'Windows':
            source_dir = DIST_DIR / BUILD_CONFIG['app_name']
            archive_name = f"{BUILD_CONFIG['app_name']}-{BUILD_CONFIG['app_version']}-win-{platform.machine()}.zip"
        elif self.platform == 'Darwin':
            source_dir = DIST_DIR
            archive_name = f"{BUILD_CONFIG['app_name']}-{BUILD_CONFIG['app_version']}-macos.zip"
        else:
            source_dir = DIST_DIR / BUILD_CONFIG['app_name']
            archive_name = f"{BUILD_CONFIG['app_name']}-{BUILD_CONFIG['app_version']}-linux-{platform.machine()}.tar.gz"
            
        # 압축
        if archive_name.endswith('.zip'):
            with zipfile.ZipFile(DIST_DIR / archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(source_dir.parent)
                        zf.write(file_path, arcname)
        else:
            # tar.gz for Linux
            import tarfile
            with tarfile.open(DIST_DIR / archive_name, 'w:gz') as tf:
                tf.add(source_dir, arcname=BUILD_CONFIG['app_name'])
                
        logger.info(f"아카이브 생성 완료: {archive_name}")
        
        # 체크섬 생성
        self._create_checksum(DIST_DIR / archive_name)
        
    def _create_checksum(self, file_path: Path):
        """체크섬 파일 생성"""
        checksums = {}
        
        # SHA256
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        checksums['sha256'] = sha256_hash.hexdigest()
        
        # MD5 (호환성)
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        checksums['md5'] = md5_hash.hexdigest()
        
        # 체크섬 파일 작성
        checksum_file = file_path.with_suffix('.checksums')
        with open(checksum_file, 'w') as f:
            f.write(f"SHA256: {checksums['sha256']}\n")
            f.write(f"MD5: {checksums['md5']}\n")
            
        logger.info("체크섬 파일 생성 완료")
        
    def _create_default_icon(self, icon_path: Path):
        """기본 아이콘 생성"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 아이콘 크기
            size = (256, 256) if self.platform == 'Windows' else (512, 512)
            
            # 이미지 생성
            img = Image.new('RGBA', size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 배경
            draw.ellipse([10, 10, size[0]-10, size[1]-10], 
                        fill=(30, 136, 229, 255))  # 파란색
            
            # 텍스트
            text = "DOE"
            try:
                # 폰트 로드 시도
                font = ImageFont.truetype("arial.ttf", size[0]//4)
            except:
                font = ImageFont.load_default()
                
            # 텍스트 위치 계산
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((size[0] - text_width) // 2, 
                       (size[1] - text_height) // 2)
            
            # 텍스트 그리기
            draw.text(position, text, fill=(255, 255, 255, 255), font=font)
            
            # 저장
            icon_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.platform == 'Windows':
                img.save(icon_path, 'ICO', sizes=[(256, 256)])
            elif self.platform == 'Darwin':
                img.save(icon_path, 'ICNS')
            else:
                img.save(icon_path, 'PNG')
                
            logger.info(f"기본 아이콘 생성 완료: {icon_path}")
            
        except ImportError:
            logger.error("PIL/Pillow가 설치되어 있지 않아 아이콘을 생성할 수 없습니다")
            
    def build_all(self):
        """전체 빌드 프로세스 실행"""
        logger.info(f"{'='*60}")
        logger.info(f"Universal DOE Platform 빌드 시작")
        logger.info(f"플랫폼: {self.platform}")
        logger.info(f"버전: {BUILD_CONFIG['app_version']}")
        logger.info(f"Python: {BUILD_CONFIG['python_version']}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 빌드 디렉토리 정리
            self.clean_build_dirs()
            
            # 2. 애셋 준비
            self.prepare_assets()
            
            # 3. 버전 파일 생성 (Windows)
            if self.platform == 'Windows':
                self.create_version_file()
                
            # 4. 실행파일 빌드
            self.build_executable()
            
            # 5. 코드 서명
            self.code_sign()
            
            # 6. 인스톨러 생성
            self.create_installer()
            
            # 7. 배포 아카이브 생성
            self.create_archive()
            
            # 8. 빌드 보고서
            self._generate_build_report()
            
            logger.info(f"{'='*60}")
            logger.info(f"빌드 완료! 출력 디렉토리: {DIST_DIR}")
            logger.info(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"빌드 실패: {e}")
            raise
            
    def _generate_build_report(self):
        """빌드 보고서 생성"""
        report = {
            'build_info': {
                'app_name': BUILD_CONFIG['app_name'],
                'app_version': BUILD_CONFIG['app_version'],
                'build_time': self.build_time.isoformat(),
                'platform': self.platform,
                'python_version': BUILD_CONFIG['python_version'],
                'architecture': platform.machine()
            },
            'outputs': []
        }
        
        # 출력 파일 목록
        for file in DIST_DIR.iterdir():
            if file.is_file():
                stat = file.stat()
                report['outputs'].append({
                    'name': file.name,
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2)
                })
                
        # 보고서 저장
        report_file = DIST_DIR / 'build_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"빌드 보고서 생성: {report_file}")

# ===========================================================================
# 🚀 빌드 헬퍼 함수
# ===========================================================================

def optimize_build():
    """빌드 최적화 설정"""
    optimizations = {
        'upx': {
            'enabled': True,
            'level': 9,  # 최대 압축
            'excludes': ['vcruntime', 'ucrtbase']  # 압축 제외
        },
        'strip': {
            'enabled': True,
            'symbols': True
        },
        'tree_shaking': {
            'enabled': True,
            'aggressive': False
        }
    }
    return optimizations

def get_runtime_hooks():
    """런타임 훅 스크립트"""
    hooks = []
    
    # Streamlit 런타임 훅
    streamlit_hook = """
import os
import sys

# Streamlit 설정
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
"""
    
    hook_file = BUILD_DIR / 'runtime_streamlit.py'
    with open(hook_file, 'w') as f:
        f.write(streamlit_hook)
    hooks.append(str(hook_file))
    
    return hooks

def validate_environment():
    """빌드 환경 검증"""
    checks = {
        'python_version': sys.version_info >= (3, 8),
        'pyinstaller': True,  # requirements_build.txt로 설치됨
        'disk_space': shutil.disk_usage('.').free > 5 * 1024**3,  # 5GB
    }
    
    # 플랫폼별 도구 확인
    if platform.system() == 'Windows':
        checks['inno_setup'] = shutil.which('iscc') is not None
    elif platform.system() == 'Darwin':
        checks['create_dmg'] = shutil.which('create-dmg') is not None
    elif platform.system() == 'Linux':
        checks['appimage'] = shutil.which('appimagetool') is not None
        
    return all(checks.values()), checks

# ===========================================================================
# 🎯 메인 함수
# ===========================================================================

def main():
    """메인 빌드 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal DOE Platform 빌드 스크립트'
    )
    parser.add_argument(
        '--platform',
        choices=['Windows', 'Darwin', 'Linux'],
        default=platform.system(),
        help='대상 플랫폼'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='빌드 디렉토리 정리만 수행'
    )
    parser.add_argument(
        '--no-installer',
        action='store_true',
        help='인스톨러 생성 건너뛰기'
    )
    parser.add_argument(
        '--no-sign',
        action='store_true',
        help='코드 서명 건너뛰기'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 빌드'
    )
    
    args = parser.parse_args()
    
    # 환경 검증
    valid, checks = validate_environment()
    if not valid:
        logger.error("빌드 환경 검증 실패:")
        for check, result in checks.items():
            if not result:
                logger.error(f"  - {check}: ❌")
        sys.exit(1)
        
    # 빌더 생성
    builder = DOEAppBuilder(args.platform)
    
    # 정리만 수행
    if args.clean:
        builder.clean_build_dirs()
        logger.info("빌드 디렉토리 정리 완료")
        return
        
    # 전체 빌드
    try:
        builder.build_all()
    except Exception as e:
        logger.error(f"빌드 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
