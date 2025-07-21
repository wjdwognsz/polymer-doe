"""
🚀 Universal DOE Platform - 데스크톱 애플리케이션 실행기
================================================================================
Streamlit 기반 데스크톱 앱을 실행하는 메인 런처
크로스 플랫폼 지원, 자동 포트 관리, 프로세스 모니터링
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
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from contextlib import closing

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
APP_ICON = str(BASE_DIR / 'assets' / 'icon.ico') if (BASE_DIR / 'assets' / 'icon.ico').exists() else None

# 서버 설정
DEFAULT_PORT = 8501
PORT_RANGE = (8501, 8510)
STARTUP_TIMEOUT = 30  # 초
CHECK_INTERVAL = 0.5  # 초

# 로그 설정
LOG_DIR = DATA_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f'launcher_{datetime.now().strftime("%Y%m%d")}.log'

# PID 파일
PID_FILE = DATA_DIR / 'app.pid'

# ===========================================================================
# 🔍 로깅 설정
# ===========================================================================

def setup_logging(debug: bool = False):
    """로깅 시스템 설정"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
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
    
    return logging.getLogger(__name__)

# ===========================================================================
# 🖥️ 시스템 유틸리티
# ===========================================================================

class SystemChecker:
    """시스템 요구사항 검사"""
    
    @staticmethod
    def check_requirements() -> Tuple[bool, Dict[str, Any]]:
        """시스템 요구사항 확인"""
        import platform
        
        results = {
            'passed': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Python 버전 확인
        python_version = sys.version_info
        min_version = (3, 8)
        
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
            import psutil
            memory = psutil.virtual_memory()
            min_memory_gb = 2
            
            results['checks']['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'required_gb': min_memory_gb,
                'passed': memory.total >= min_memory_gb * (1024**3)
            }
            
            if not results['checks']['memory']['passed']:
                results['warnings'].append(f"권장 메모리: {min_memory_gb}GB 이상")
        except ImportError:
            results['warnings'].append("메모리 확인 불가 (psutil 미설치)")
        
        # 디스크 공간 확인
        try:
            import shutil
            disk_usage = shutil.disk_usage(DATA_DIR.parent)
            min_disk_mb = 500
            
            results['checks']['disk_space'] = {
                'free_mb': round(disk_usage.free / (1024**2), 2),
                'required_mb': min_disk_mb,
                'passed': disk_usage.free >= min_disk_mb * (1024**2)
            }
            
            if not results['checks']['disk_space']['passed']:
                results['warnings'].append(f"여유 디스크 공간 부족 (최소 {min_disk_mb}MB 필요)")
        except Exception:
            results['warnings'].append("디스크 공간 확인 불가")
        
        return results['passed'], results

# ===========================================================================
# 🚀 메인 런처 클래스
# ===========================================================================

class DOELauncher:
    """Universal DOE Platform 실행기"""
    
    def __init__(self, debug: bool = False, port: Optional[int] = None):
        self.debug = debug
        self.logger = setup_logging(debug)
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.port = port or DEFAULT_PORT
        self.app_url = None
        self.is_running = False
        self.start_time = None
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"{APP_NAME} v{APP_VERSION} 실행기 시작")
        self.logger.info(f"{'='*60}")
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신 - 종료 중...")
        self.cleanup()
        sys.exit(0)
    
    def check_single_instance(self) -> bool:
        """단일 인스턴스 확인"""
        if PID_FILE.exists():
            try:
                with open(PID_FILE, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # 프로세스 존재 확인
                try:
                    import psutil
                    if psutil.pid_exists(old_pid):
                        process = psutil.Process(old_pid)
                        if 'python' in process.name().lower():
                            self.logger.warning(f"이미 실행 중입니다 (PID: {old_pid})")
                            return False
                except ImportError:
                    # psutil이 없으면 기본 방법 사용
                    try:
                        os.kill(old_pid, 0)
                        self.logger.warning(f"이미 실행 중일 수 있습니다 (PID: {old_pid})")
                        return False
                    except OSError:
                        pass
                
            except (ValueError, IOError):
                pass
            
            # 오래된 PID 파일 제거
            PID_FILE.unlink()
        
        # 새 PID 파일 생성
        try:
            with open(PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"PID 파일 생성: {os.getpid()}")
            return True
        except Exception as e:
            self.logger.error(f"PID 파일 생성 실패: {e}")
            return True  # 실패해도 계속 진행
    
    def find_free_port(self) -> int:
        """사용 가능한 포트 찾기"""
        # 지정된 포트 먼저 확인
        if self.port and self._is_port_free(self.port):
            return self.port
        
        # 포트 범위에서 검색
        for port in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
            if self._is_port_free(port):
                self.logger.info(f"사용 가능한 포트 발견: {port}")
                return port
        
        raise RuntimeError(f"포트 {PORT_RANGE[0]}-{PORT_RANGE[1]} 범위에서 사용 가능한 포트를 찾을 수 없습니다")
    
    def _is_port_free(self, port: int) -> bool:
        """포트 사용 가능 여부 확인"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return True
            except socket.error:
                return False
    
    def create_data_directories(self):
        """필요한 데이터 디렉토리 생성"""
        directories = [
            DATA_DIR / 'db',
            DATA_DIR / 'cache', 
            DATA_DIR / 'logs',
            DATA_DIR / 'temp',
            DATA_DIR / 'backups',
            BASE_DIR / 'modules' / 'user_modules'
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"디렉토리 생성/확인: {directory}")
            except Exception as e:
                self.logger.error(f"디렉토리 생성 실패 {directory}: {e}")
    
    def initialize_database(self):
        """데이터베이스 초기화"""
        try:
            # 여기서는 최소한의 DB 체크만 수행
            # 실제 초기화는 앱 시작 시 수행
            db_path = DATA_DIR / 'db' / 'app.db'
            if not db_path.exists():
                self.logger.info("데이터베이스 파일이 없습니다. 앱 시작 시 생성됩니다.")
            else:
                self.logger.info(f"기존 데이터베이스 발견: {db_path}")
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 오류: {e}")
    
    def start_streamlit(self):
        """Streamlit 서버 시작"""
        self.port = self.find_free_port()
        self.app_url = f"http://localhost:{self.port}"
        
        # Streamlit 명령어 구성
        cmd = [
            sys.executable,
            "-m", "streamlit", "run",
            str(BASE_DIR / "polymer_platform.py"),
            "--server.port", str(self.port),
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",  # 파일 감시 비활성화
            "--logger.level", "error" if not self.debug else "info"
        ]
        
        # 테마 설정 (있으면)
        if (BASE_DIR / ".streamlit" / "config.toml").exists():
            cmd.extend(["--config", str(BASE_DIR / ".streamlit" / "config.toml")])
        
        self.logger.info(f"Streamlit 서버 시작 중... (포트: {self.port})")
        self.logger.debug(f"명령어: {' '.join(cmd)}")
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        env['PYTHONUNBUFFERED'] = '1'  # 실시간 출력
        
        try:
            # 프로세스 시작
            if sys.platform == "win32":
                # Windows: 새 콘솔 창 숨기기
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                self.streamlit_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Unix 계열
                self.streamlit_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            self.logger.info(f"Streamlit 프로세스 시작됨 (PID: {self.streamlit_process.pid})")
            
            # 출력 모니터링 스레드 시작
            if self.debug:
                threading.Thread(target=self._monitor_output, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Streamlit 시작 실패: {e}")
            raise
    
    def _monitor_output(self):
        """Streamlit 출력 모니터링 (디버그용)"""
        try:
            for line in iter(self.streamlit_process.stdout.readline, b''):
                if line:
                    self.logger.debug(f"[Streamlit] {line.decode().strip()}")
        except Exception as e:
            self.logger.error(f"출력 모니터링 오류: {e}")
    
    def wait_for_server(self) -> bool:
        """서버 시작 대기"""
        self.logger.info("서버가 준비될 때까지 대기 중...")
        start_time = time.time()
        
        while time.time() - start_time < STARTUP_TIMEOUT:
            # 프로세스 상태 확인
            if self.streamlit_process and self.streamlit_process.poll() is not None:
                self.logger.error(f"Streamlit 프로세스가 예기치 않게 종료됨 (코드: {self.streamlit_process.returncode})")
                return False
            
            # 서버 응답 확인
            try:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', self.port))
                    if result == 0:
                        self.logger.info("✅ 서버가 준비되었습니다!")
                        self.is_running = True
                        return True
            except Exception:
                pass
            
            # 진행 표시
            elapsed = int(time.time() - start_time)
            print(f"\r⏳ 서버 시작 중... ({elapsed}/{STARTUP_TIMEOUT}초)", end='', flush=True)
            time.sleep(CHECK_INTERVAL)
        
        print()  # 줄바꿈
        self.logger.error("서버 시작 시간 초과")
        return False
    
    def open_browser(self):
        """브라우저 열기"""
        if not self.app_url:
            return
        
        self.logger.info(f"브라우저 열기: {self.app_url}")
        
        try:
            # 플랫폼별 최적화
            if sys.platform == "win32":
                # Windows
                os.startfile(self.app_url)
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["open", self.app_url])
            else:
                # Linux/Unix
                subprocess.run(["xdg-open", self.app_url])
        except Exception:
            # 실패 시 기본 방법 사용
            try:
                webbrowser.open(self.app_url)
            except Exception as e:
                self.logger.error(f"브라우저 열기 실패: {e}")
                self.logger.info(f"수동으로 브라우저를 열고 다음 주소로 접속하세요: {self.app_url}")
    
    def monitor_process(self):
        """프로세스 모니터링"""
        self.logger.info("프로세스 모니터링 시작...")
        
        try:
            while self.is_running:
                # 프로세스 상태 확인
                if self.streamlit_process and self.streamlit_process.poll() is not None:
                    self.logger.warning(f"Streamlit 프로세스가 종료됨 (코드: {self.streamlit_process.returncode})")
                    self.is_running = False
                    break
                
                # CPU/메모리 사용량 모니터링 (옵션)
                if self.debug:
                    try:
                        import psutil
                        process = psutil.Process(self.streamlit_process.pid)
                        cpu_percent = process.cpu_percent(interval=1)
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        self.logger.debug(f"리소스 사용: CPU {cpu_percent:.1f}%, 메모리 {memory_mb:.1f}MB")
                    except:
                        pass
                
                time.sleep(5)  # 5초마다 확인
                
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"모니터링 오류: {e}")
    
    def cleanup(self):
        """정리 작업"""
        self.logger.info("정리 작업 시작...")
        
        # Streamlit 프로세스 종료
        if self.streamlit_process:
            try:
                self.logger.info("Streamlit 프로세스 종료 중...")
                self.streamlit_process.terminate()
                
                # 종료 대기 (최대 5초)
                try:
                    self.streamlit_process.wait(timeout=5)
                    self.logger.info("Streamlit 프로세스가 정상 종료되었습니다")
                except subprocess.TimeoutExpired:
                    self.logger.warning("강제 종료 중...")
                    self.streamlit_process.kill()
                    self.streamlit_process.wait()
                    
            except Exception as e:
                self.logger.error(f"프로세스 종료 오류: {e}")
        
        # PID 파일 제거
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
                self.logger.info("PID 파일 제거됨")
        except Exception as e:
            self.logger.error(f"PID 파일 제거 실패: {e}")
        
        # 임시 파일 정리
        try:
            temp_dir = DATA_DIR / 'temp'
            if temp_dir.exists():
                import shutil
                for item in temp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                self.logger.info("임시 파일 정리 완료")
        except Exception as e:
            self.logger.error(f"임시 파일 정리 오류: {e}")
        
        self.logger.info("정리 작업 완료")
    
    def run(self):
        """메인 실행 함수"""
        try:
            # 단일 인스턴스 확인
            if not self.check_single_instance():
                print(f"\n❌ {APP_NAME}이(가) 이미 실행 중입니다.")
                print("기존 앱을 종료하거나 브라우저에서 http://localhost:8501 을 열어보세요.\n")
                return 1
            
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
            
            # 브라우저 열기
            print("\n🌐 브라우저 열기...")
            self.open_browser()
            
            # 실행 정보 출력
            print(f"\n{'='*60}")
            print(f"✨ {APP_NAME} v{APP_VERSION}이(가) 실행 중입니다!")
            print(f"{'='*60}")
            print(f"🔗 주소: {self.app_url}")
            print(f"📝 로그: {LOG_FILE}")
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
        '--version', '-v',
        action='version',
        version=f'{APP_NAME} v{APP_VERSION}'
    )
    
    args = parser.parse_args()
    
    # 런처 실행
    try:
        launcher = DOELauncher(debug=args.debug, port=args.port)
        
        # no-browser 옵션 처리
        if args.no_browser:
            launcher.open_browser = lambda: None
            
        return launcher.run()
        
    except Exception as e:
        print(f"\n❌ 치명적인 오류: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
