"""
🚀 Universal DOE Platform - 데스크톱 애플리케이션 실행기
================================================================================
Streamlit 기반 데스크톱 앱을 실행하는 메인 런처
크로스플랫폼 지원, 자동 포트 관리, 프로세스 모니터링, 시스템 트레이
================================================================================
"""

import sys
import os
import subprocess
import threading
import time
import webbrowser
import socket
import signal
import atexit
import json
import logging
import argparse
import psutil
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from contextlib import closing
import platform

# 프로젝트 루트 경로 설정
if getattr(sys, 'frozen', False):
    # PyInstaller로 패키징된 경우
    BASE_DIR = Path(sys._MEIPASS)
    DATA_DIR = Path(sys.executable).parent / 'data'
else:
    # 개발 환경
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'

# 경로를 sys.path에 추가
sys.path.insert(0, str(BASE_DIR))

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

# 앱 정보
APP_NAME = "Universal DOE Platform"
APP_VERSION = "2.0.0"
APP_ID = "com.universaldoe.platform"
APP_ICON = str(BASE_DIR / 'assets' / 'icon.ico') if (BASE_DIR / 'assets' / 'icon.ico').exists() else None

# 서버 설정
DEFAULT_PORT = 8501
PORT_RANGE = (8501, 8510)
STARTUP_TIMEOUT = 30  # 초
CHECK_INTERVAL = 0.5  # 초
HEALTH_CHECK_INTERVAL = 10  # 초

# 로그 설정
LOG_DIR = DATA_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f'launcher_{datetime.now().strftime("%Y%m%d")}.log'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_LOG_BACKUPS = 5

# PID 파일
PID_FILE = DATA_DIR / 'app.pid'

# 설정 파일
CONFIG_FILE = DATA_DIR / 'launcher_config.json'

# ===========================================================================
# 🔍 로깅 설정
# ===========================================================================

def setup_logging(debug: bool = False):
    """로깅 시스템 설정"""
    from logging.handlers import RotatingFileHandler
    
    log_level = logging.DEBUG if debug else logging.INFO
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러 (순환)
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=MAX_LOG_SIZE,
        backupCount=MAX_LOG_BACKUPS,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('launcher')

# ===========================================================================
# 🔧 유틸리티 함수
# ===========================================================================

def find_free_port(start_port: int = DEFAULT_PORT, 
                  end_port: int = PORT_RANGE[1]) -> Optional[int]:
    """사용 가능한 포트 찾기"""
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    return None

def check_port_in_use(port: int) -> bool:
    """포트 사용 여부 확인"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(('localhost', port)) == 0

def get_streamlit_processes() -> List[psutil.Process]:
    """실행 중인 Streamlit 프로세스 찾기"""
    streamlit_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('streamlit' in arg for arg in cmdline):
                streamlit_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return streamlit_processes

def kill_existing_processes():
    """기존 Streamlit 프로세스 종료"""
    processes = get_streamlit_processes()
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

def save_pid(pid: int):
    """PID 파일 저장"""
    with open(PID_FILE, 'w') as f:
        f.write(str(pid))

def load_pid() -> Optional[int]:
    """저장된 PID 로드"""
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                return int(f.read().strip())
        except Exception:
            pass
    return None

def remove_pid():
    """PID 파일 제거"""
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass

# ===========================================================================
# 💾 설정 관리
# ===========================================================================

class LauncherConfig:
    """런처 설정 관리"""
    
    DEFAULT_CONFIG = {
        'theme': 'light',
        'auto_open_browser': True,
        'use_system_tray': True,
        'minimize_to_tray': True,
        'start_minimized': False,
        'check_updates': True,
        'webview_mode': False,
        'kiosk_mode': False,
        'debug_mode': False,
        'custom_port': None,
        'window_size': [1280, 800],
        'window_position': None
    }
    
    @classmethod
    def load(cls) -> Dict[str, Any]:
        """설정 로드"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return {**cls.DEFAULT_CONFIG, **config}
            except Exception:
                pass
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save(cls, config: Dict[str, Any]):
        """설정 저장"""
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"설정 저장 실패: {e}")

# ===========================================================================
# 🔍 시스템 체크
# ===========================================================================

class SystemChecker:
    """시스템 요구사항 체크"""
    
    @staticmethod
    def check_requirements() -> Tuple[bool, Dict[str, Any]]:
        """시스템 요구사항 확인"""
        results = {
            'passed': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Python 버전 확인
        min_version = (3, 8)
        python_version = sys.version_info
        
        results['checks']['python_version'] = {
            'current': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'required': f"{min_version[0]}.{min_version[1]}+",
            'passed': python_version >= min_version
        }
        
        if not results['checks']['python_version']['passed']:
            results['errors'].append(f"Python {min_version[0]}.{min_version[1]} 이상이 필요합니다.")
            results['passed'] = False
        
        # 운영체제 확인
        os_name = platform.system()
        results['checks']['os'] = {
            'name': os_name,
            'version': platform.version(),
            'architecture': platform.machine(),
            'passed': os_name in ['Windows', 'Darwin', 'Linux']
        }
        
        # 메모리 확인
        try:
            memory = psutil.virtual_memory()
            min_memory_gb = 4  # 권장 4GB
            
            results['checks']['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'required_gb': min_memory_gb,
                'passed': memory.total >= min_memory_gb * (1024**3)
            }
            
            if not results['checks']['memory']['passed']:
                results['warnings'].append(f"권장 메모리: {min_memory_gb}GB 이상")
        except Exception:
            results['warnings'].append("메모리 확인 불가")
        
        # 디스크 공간 확인
        try:
            disk_usage = psutil.disk_usage(str(DATA_DIR.parent))
            min_disk_mb = 2000  # 2GB
            
            results['checks']['disk_space'] = {
                'free_mb': round(disk_usage.free / (1024**2), 2),
                'required_mb': min_disk_mb,
                'passed': disk_usage.free >= min_disk_mb * (1024**2)
            }
            
            if not results['checks']['disk_space']['passed']:
                results['warnings'].append(f"여유 디스크 공간 부족 (최소 {min_disk_mb}MB 필요)")
        except Exception:
            results['warnings'].append("디스크 공간 확인 불가")
        
        # 필수 모듈 확인
        required_modules = ['streamlit', 'pandas', 'numpy', 'plotly']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        results['checks']['modules'] = {
            'required': required_modules,
            'missing': missing_modules,
            'passed': len(missing_modules) == 0
        }
        
        if missing_modules:
            results['errors'].append(f"필수 모듈 누락: {', '.join(missing_modules)}")
            results['passed'] = False
        
        return results['passed'], results

# ===========================================================================
# 🖥️ 시스템 트레이
# ===========================================================================

class SystemTray:
    """시스템 트레이 관리"""
    
    def __init__(self, launcher):
        self.launcher = launcher
        self.icon = None
        self._running = False
        
    def create_tray_icon(self):
        """트레이 아이콘 생성"""
        try:
            import pystray
            from PIL import Image
            
            # 아이콘 이미지 로드
            if APP_ICON and Path(APP_ICON).exists():
                image = Image.open(APP_ICON)
            else:
                # 기본 아이콘 생성
                image = Image.new('RGB', (64, 64), color='blue')
            
            # 메뉴 생성
            menu = pystray.Menu(
                pystray.MenuItem("열기", self.on_open),
                pystray.MenuItem("대시보드", self.on_dashboard),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("로그 보기", self.on_view_logs),
                pystray.MenuItem("설정", self.on_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("재시작", self.on_restart),
                pystray.MenuItem("종료", self.on_quit)
            )
            
            # 아이콘 생성
            self.icon = pystray.Icon(
                APP_NAME,
                image,
                APP_NAME,
                menu
            )
            
            return True
            
        except ImportError:
            logging.warning("pystray 모듈이 없어 시스템 트레이를 사용할 수 없습니다.")
            return False
        except Exception as e:
            logging.error(f"트레이 아이콘 생성 실패: {e}")
            return False
    
    def run(self):
        """트레이 아이콘 실행"""
        if self.create_tray_icon():
            self._running = True
            self.icon.run()
    
    def stop(self):
        """트레이 아이콘 중지"""
        self._running = False
        if self.icon:
            self.icon.stop()
    
    def on_open(self, icon, item):
        """앱 열기"""
        self.launcher.open_browser()
    
    def on_dashboard(self, icon, item):
        """대시보드 열기"""
        webbrowser.open(f"{self.launcher.app_url}/1_📊_Dashboard")
    
    def on_view_logs(self, icon, item):
        """로그 보기"""
        if platform.system() == 'Windows':
            os.startfile(LOG_FILE)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', LOG_FILE])
        else:
            subprocess.run(['xdg-open', LOG_FILE])
    
    def on_settings(self, icon, item):
        """설정 열기"""
        webbrowser.open(f"{self.launcher.app_url}/settings")
    
    def on_restart(self, icon, item):
        """앱 재시작"""
        self.launcher.restart()
    
    def on_quit(self, icon, item):
        """앱 종료"""
        self.launcher.shutdown()

# ===========================================================================
# 🌐 WebView 관리
# ===========================================================================

class WebViewManager:
    """WebView 관리자"""
    
    def __init__(self, url: str, config: Dict[str, Any]):
        self.url = url
        self.config = config
        self.window = None
        
    def create_window(self):
        """WebView 창 생성"""
        try:
            import webview
            
            # 창 설정
            window_config = {
                'title': APP_NAME,
                'width': self.config.get('window_size', [1280, 800])[0],
                'height': self.config.get('window_size', [1280, 800])[1],
                'resizable': True,
                'fullscreen': self.config.get('kiosk_mode', False),
                'min_size': (800, 600)
            }
            
            # 위치 설정
            if self.config.get('window_position'):
                window_config['x'] = self.config['window_position'][0]
                window_config['y'] = self.config['window_position'][1]
            
            # 창 생성
            self.window = webview.create_window(
                **window_config,
                url=self.url
            )
            
            # 이벤트 핸들러
            self.window.events.closed += self.on_closed
            
            return True
            
        except ImportError:
            logging.warning("pywebview 모듈이 없어 WebView를 사용할 수 없습니다.")
            return False
        except Exception as e:
            logging.error(f"WebView 창 생성 실패: {e}")
            return False
    
    def start(self):
        """WebView 시작"""
        try:
            import webview
            
            if self.create_window():
                webview.start()
        except Exception as e:
            logging.error(f"WebView 시작 실패: {e}")
    
    def on_closed(self):
        """창 닫힘 이벤트"""
        logging.info("WebView 창이 닫혔습니다.")

# ===========================================================================
# 🚀 메인 런처 클래스
# ===========================================================================

class DOELauncher:
    """Universal DOE Platform 실행기"""
    
    def __init__(self, debug: bool = False, port: Optional[int] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.debug = debug
        self.config = config or LauncherConfig.load()
        self.logger = setup_logging(debug or self.config.get('debug_mode', False))
        
        # 서버 설정
        self.port = port or self.config.get('custom_port') or DEFAULT_PORT
        self.app_url = f"http://localhost:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[datetime] = None
        
        # 시스템 트레이
        self.tray: Optional[SystemTray] = None
        self.tray_thread: Optional[threading.Thread] = None
        
        # WebView
        self.webview: Optional[WebViewManager] = None
        
        # 상태
        self._running = False
        self._shutting_down = False
        self._restart_requested = False
        
        # 신호 핸들러 등록
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 종료 시 정리
        atexit.register(self.cleanup)
    
    def signal_handler(self, signum, frame):
        """신호 핸들러"""
        self.logger.info(f"신호 수신: {signum}")
        self.shutdown()
    
    def create_data_directories(self):
        """데이터 디렉토리 생성"""
        directories = [
            DATA_DIR,
            DATA_DIR / 'cache',
            DATA_DIR / 'logs',
            DATA_DIR / 'exports',
            DATA_DIR / 'backups',
            DATA_DIR / 'modules',
            DATA_DIR / 'templates'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"디렉토리 생성: {directory}")
    
    def initialize_database(self):
        """데이터베이스 초기화"""
        try:
            from utils.database_manager import DatabaseManager
            
            db_path = DATA_DIR / 'app.db'
            db_manager = DatabaseManager(str(db_path))
            
            # 테이블 생성
            if not db_path.exists():
                self.logger.info("데이터베이스 초기화 중...")
                db_manager.create_tables()
                self.logger.info("데이터베이스 초기화 완료")
            else:
                self.logger.info("기존 데이터베이스 발견")
                
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def find_available_port(self) -> int:
        """사용 가능한 포트 찾기"""
        # 설정된 포트 확인
        if self.port and not check_port_in_use(self.port):
            return self.port
        
        # 포트 범위에서 찾기
        port = find_free_port()
        if port:
            self.logger.info(f"사용 가능한 포트 발견: {port}")
            return port
        
        raise RuntimeError("사용 가능한 포트를 찾을 수 없습니다.")
    
    def start_streamlit(self):
        """Streamlit 서버 시작"""
        try:
            # 포트 확인
            self.port = self.find_available_port()
            self.app_url = f"http://localhost:{self.port}"
            
            # Streamlit 명령어 구성
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                str(BASE_DIR / 'polymer_platform.py'),
                '--server.port', str(self.port),
                '--server.address', 'localhost',
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ]
            
            if self.debug:
                cmd.extend(['--logger.level', 'debug'])
            
            # 환경 변수 설정
            env = os.environ.copy()
            env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
            
            # 프로세스 시작
            self.logger.info(f"Streamlit 서버 시작: {' '.join(cmd)}")
            
            if platform.system() == 'Windows':
                # Windows: 콘솔 창 숨기기
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                self.process = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    universal_newlines=True,
                    bufsize=1
                )
            else:
                # Unix-like
                self.process = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
            
            # PID 저장
            save_pid(self.process.pid)
            
            # 로그 스레드 시작
            log_thread = threading.Thread(
                target=self.read_process_output,
                daemon=True
            )
            log_thread.start()
            
        except Exception as e:
            self.logger.error(f"Streamlit 서버 시작 실패: {e}")
            raise
    
    def read_process_output(self):
        """프로세스 출력 읽기"""
        if not self.process:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    if self.debug:
                        print(f"[Streamlit] {line}")
                    self.logger.debug(f"Streamlit: {line}")
        except Exception as e:
            self.logger.error(f"출력 읽기 오류: {e}")
    
    def wait_for_server(self) -> bool:
        """서버 시작 대기"""
        start_time = time.time()
        
        while time.time() - start_time < STARTUP_TIMEOUT:
            if check_port_in_use(self.port):
                # 서버 응답 확인
                try:
                    import requests
                    response = requests.get(self.app_url, timeout=1)
                    if response.status_code == 200:
                        self.logger.info("서버가 성공적으로 시작되었습니다.")
                        return True
                except Exception:
                    pass
            
            # 프로세스 상태 확인
            if self.process and self.process.poll() is not None:
                self.logger.error(f"프로세스가 예기치 않게 종료됨: {self.process.returncode}")
                return False
            
            time.sleep(CHECK_INTERVAL)
        
        return False
    
    def open_browser(self):
        """브라우저 열기"""
        if not self.config.get('auto_open_browser', True):
            return
        
        if self.config.get('webview_mode', False):
            # WebView 모드
            self.webview = WebViewManager(self.app_url, self.config)
            webview_thread = threading.Thread(
                target=self.webview.start,
                daemon=True
            )
            webview_thread.start()
        else:
            # 시스템 브라우저
            time.sleep(1)  # 서버 안정화 대기
            webbrowser.open(self.app_url)
    
    def monitor_process(self):
        """프로세스 모니터링"""
        self._running = True
        last_health_check = time.time()
        
        while self._running and not self._shutting_down:
            try:
                # 프로세스 확인
                if self.process and self.process.poll() is not None:
                    if not self._shutting_down:
                        self.logger.error("프로세스가 예기치 않게 종료되었습니다.")
                        if self.config.get('auto_restart', True):
                            self.logger.info("자동 재시작 시도...")
                            self.restart()
                        else:
                            break
                
                # 헬스 체크
                current_time = time.time()
                if current_time - last_health_check >= HEALTH_CHECK_INTERVAL:
                    if not check_port_in_use(self.port):
                        self.logger.warning("서버가 응답하지 않습니다.")
                    last_health_check = current_time
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                time.sleep(5)
    
    def start_system_tray(self):
        """시스템 트레이 시작"""
        if not self.config.get('use_system_tray', True):
            return
        
        try:
            self.tray = SystemTray(self)
            self.tray_thread = threading.Thread(
                target=self.tray.run,
                daemon=True
            )
            self.tray_thread.start()
            self.logger.info("시스템 트레이 시작됨")
        except Exception as e:
            self.logger.warning(f"시스템 트레이 시작 실패: {e}")
    
    def cleanup(self):
        """정리 작업"""
        if self._shutting_down:
            return
        
        self._shutting_down = True
        self._running = False
        
        # 트레이 아이콘 제거
        if self.tray:
            self.tray.stop()
        
        # 프로세스 종료
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
        
        # PID 파일 제거
        remove_pid()
        
        self.logger.info("정리 작업 완료")
    
    def shutdown(self):
        """앱 종료"""
        self.logger.info("앱 종료 중...")
        self._shutting_down = True
        self._running = False
        self.cleanup()
        sys.exit(0)
    
    def restart(self):
        """앱 재시작"""
        self.logger.info("앱 재시작 중...")
        self._restart_requested = True
        
        # 현재 프로세스 종료
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                if self.process.poll() is None:
                    self.process.kill()
        
        # 새 프로세스 시작
        time.sleep(2)
        self.start_streamlit()
        
        if self.wait_for_server():
            self.logger.info("재시작 완료")
            self._restart_requested = False
        else:
            self.logger.error("재시작 실패")
    
    def run(self) -> int:
        """런처 실행"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 {APP_NAME} v{APP_VERSION}")
            print(f"{'='*60}\n")
            
            # 단일 인스턴스 확인
            old_pid = load_pid()
            if old_pid:
                try:
                    old_process = psutil.Process(old_pid)
                    if 'streamlit' in ' '.join(old_process.cmdline()).lower():
                        print(f"\n⚠️  이미 실행 중인 인스턴스가 있습니다. (PID: {old_pid})")
                        print("기존 앱을 종료하거나 브라우저에서 http://localhost:8501 을 열어보세요.\n")
                        return 1
                except psutil.NoSuchProcess:
                    remove_pid()
            
            # 시스템 요구사항 확인
            print(f"\n🔍 시스템 요구사항 확인 중...")
            passed, results = SystemChecker.check_requirements()
            
            if not passed:
                print("\n❌ 시스템 요구사항을 충족하지 못했습니다:")
                for error in results['errors']:
                    print(f"  - {error}")
                return 1
            
            if results['warnings']:
                print("\n⚠️  경고:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
            
            print("✅ 시스템 요구사항 확인 완료\n")
            
            # 데이터 디렉토리 생성
            print("📁 데이터 디렉토리 초기화 중...")
            self.create_data_directories()
            
            # 데이터베이스 초기화
            print("🗄️  데이터베이스 확인 중...")
            self.initialize_database()
            
            # Streamlit 서버 시작
            print(f"\n🚀 {APP_NAME} 시작 중...")
            self.start_streamlit()
            
            # 서버 대기
            if not self.wait_for_server():
                print("\n❌ 서버 시작 실패")
                return 1
            
            # 시스템 트레이 시작
            if self.config.get('use_system_tray', True):
                self.start_system_tray()
            
            # 브라우저 열기
            if not self.config.get('start_minimized', False):
                print("\n🌐 브라우저 열기...")
                self.open_browser()
            
            # 실행 정보 출력
            print(f"\n{'='*60}")
            print(f"✨ {APP_NAME} v{APP_VERSION}이(가) 실행 중입니다!")
            print(f"{'='*60}")
            print(f"🔗 주소: {self.app_url}")
            print(f"📝 로그: {LOG_FILE}")
            
            if self.tray:
                print(f"🔔 시스템 트레이에서 제어 가능")
            
            print(f"\n종료하려면 Ctrl+C를 누르세요...")
            print(f"{'='*60}\n")
            
            self.start_time = datetime.now()
            
            # 프로세스 모니터링
            self.monitor_process()
            
        except KeyboardInterrupt:
            print("\n\n👋 사용자에 의해 종료됩니다...")
        except Exception as e:
            self.logger.error(f"실행 중 오류 발생: {e}", exc_info=True)
            print(f"\n❌ 오류 발생: {e}")
            return 1
        finally:
            # 정리 작업
            self.cleanup()
            
            # 실행 시간 출력
            if self.start_time:
                runtime = datetime.now() - self.start_time
                print(f"\n⏱️  총 실행 시간: {runtime}")
            
            print(f"\n👍 {APP_NAME}이(가) 종료되었습니다.\n")
            
        return 0

# ===========================================================================
# 🎯 메인 진입점
# ===========================================================================

def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} - 데스크톱 애플리케이션 실행기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python launcher.py                    # 기본 실행
  python launcher.py --debug           # 디버그 모드
  python launcher.py --port 8502       # 특정 포트 지정
  python launcher.py --no-browser      # 브라우저 자동 열기 비활성화
  python launcher.py --webview         # WebView 모드로 실행
  python launcher.py --tray            # 시스템 트레이 최소화 시작
        """
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='서버 포트 (기본값: 8501)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='브라우저 자동 열기 비활성화'
    )
    
    parser.add_argument(
        '--webview', '-w',
        action='store_true',
        help='WebView 모드로 실행'
    )
    
    parser.add_argument(
        '--no-tray',
        action='store_true',
        help='시스템 트레이 비활성화'
    )
    
    parser.add_argument(
        '--tray', '-t',
        action='store_true',
        help='시스템 트레이로 최소화 시작'
    )
    
    parser.add_argument(
        '--kiosk', '-k',
        action='store_true',
        help='키오스크 모드 (전체화면)'
    )
    
    parser.add_argument(
        '--reset-config',
        action='store_true',
        help='설정 초기화'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'{APP_NAME} v{APP_VERSION}'
    )
    
    args = parser.parse_args()
    
    # 설정 로드 및 적용
    if args.reset_config:
        CONFIG_FILE.unlink(missing_ok=True)
        print("설정이 초기화되었습니다.")
    
    config = LauncherConfig.load()
    
    # 명령줄 옵션 적용
    if args.no_browser:
        config['auto_open_browser'] = False
    if args.webview:
        config['webview_mode'] = True
    if args.no_tray:
        config['use_system_tray'] = False
    if args.tray:
        config['start_minimized'] = True
    if args.kiosk:
        config['kiosk_mode'] = True
        config['webview_mode'] = True
    
    # 설정 저장
    LauncherConfig.save(config)
    
    # 런처 실행
    try:
        launcher = DOELauncher(
            debug=args.debug, 
            port=args.port,
            config=config
        )
        
        return launcher.run()
        
    except Exception as e:
        print(f"\n❌ 치명적인 오류: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
