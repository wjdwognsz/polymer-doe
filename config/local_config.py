"""
config/local_config.py
======================
Universal DOE Platform - 로컬 환경 설정
OS별 경로 관리, 시스템 리소스 모니터링, 프로세스 관리
"""

import os
import sys
import platform
import socket
import json
import logging
import psutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import shutil


# ===========================================================================
# 🏗️ 시스템 요구사항
# ===========================================================================

SYSTEM_REQUIREMENTS = {
    'minimum': {
        'python_version': '3.8',
        'ram_gb': 4,
        'disk_gb': 2,
        'cpu_cores': 2
    },
    'recommended': {
        'python_version': '3.10',
        'ram_gb': 8,
        'disk_gb': 5,
        'cpu_cores': 4
    },
    'supported_os': ['Windows', 'Darwin', 'Linux'],
    'supported_architectures': ['x86_64', 'AMD64', 'arm64', 'aarch64']
}


# ===========================================================================
# 🗂️ OS별 경로 정의
# ===========================================================================

class OSType(Enum):
    """운영체제 타입"""
    WINDOWS = "Windows"
    MACOS = "Darwin"
    LINUX = "Linux"
    UNKNOWN = "Unknown"


@dataclass
class DirectoryPaths:
    """애플리케이션 디렉토리 경로"""
    app_data_dir: Path
    config_dir: Path
    cache_dir: Path
    log_dir: Path
    temp_dir: Path
    documents_dir: Path
    
    def to_dict(self) -> Dict[str, str]:
        """딕셔너리로 변환 (JSON 저장용)"""
        return {k: str(v) for k, v in asdict(self).items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'DirectoryPaths':
        """딕셔너리에서 생성"""
        return cls(**{k: Path(v) for k, v in data.items()})


# ===========================================================================
# 📁 OS별 경로 관리자
# ===========================================================================

class PathManager:
    """OS별 경로 관리"""
    
    @staticmethod
    def get_os_type() -> OSType:
        """현재 OS 타입 반환"""
        system = platform.system()
        try:
            return OSType(system)
        except ValueError:
            return OSType.UNKNOWN
    
    @staticmethod
    def get_app_name() -> str:
        """애플리케이션 이름"""
        return "UniversalDOE"
    
    @staticmethod
    def get_app_directories() -> DirectoryPaths:
        """OS별 애플리케이션 디렉토리 경로 반환"""
        os_type = PathManager.get_os_type()
        app_name = PathManager.get_app_name()
        
        if os_type == OSType.WINDOWS:
            return PathManager._get_windows_paths(app_name)
        elif os_type == OSType.MACOS:
            return PathManager._get_macos_paths(app_name)
        elif os_type == OSType.LINUX:
            return PathManager._get_linux_paths(app_name)
        else:
            return PathManager._get_fallback_paths(app_name)
    
    @staticmethod
    def _get_windows_paths(app_name: str) -> DirectoryPaths:
        """Windows 경로"""
        # 환경 변수 사용
        app_data = Path(os.environ.get('LOCALAPPDATA', '')) / app_name
        if not app_data.parent.exists():
            app_data = Path.home() / 'AppData' / 'Local' / app_name
        
        documents = Path.home() / 'Documents' / app_name
        
        return DirectoryPaths(
            app_data_dir=app_data,
            config_dir=app_data / 'Config',
            cache_dir=app_data / 'Cache',
            log_dir=app_data / 'Logs',
            temp_dir=Path(tempfile.gettempdir()) / app_name,
            documents_dir=documents
        )
    
    @staticmethod
    def _get_macos_paths(app_name: str) -> DirectoryPaths:
        """macOS 경로"""
        app_support = Path.home() / 'Library' / 'Application Support' / app_name
        caches = Path.home() / 'Library' / 'Caches' / app_name
        documents = Path.home() / 'Documents' / app_name
        
        return DirectoryPaths(
            app_data_dir=app_support,
            config_dir=app_support / 'Config',
            cache_dir=caches,
            log_dir=Path.home() / 'Library' / 'Logs' / app_name,
            temp_dir=Path(tempfile.gettempdir()) / app_name,
            documents_dir=documents
        )
    
    @staticmethod
    def _get_linux_paths(app_name: str) -> DirectoryPaths:
        """Linux 경로 (XDG Base Directory 규격)"""
        # XDG 환경 변수 확인
        xdg_data = Path(os.environ.get('XDG_DATA_HOME', 
                                      Path.home() / '.local' / 'share'))
        xdg_config = Path(os.environ.get('XDG_CONFIG_HOME', 
                                        Path.home() / '.config'))
        xdg_cache = Path(os.environ.get('XDG_CACHE_HOME', 
                                       Path.home() / '.cache'))
        
        app_name_lower = app_name.lower()
        documents = Path.home() / 'Documents' / app_name
        
        return DirectoryPaths(
            app_data_dir=xdg_data / app_name_lower,
            config_dir=xdg_config / app_name_lower,
            cache_dir=xdg_cache / app_name_lower,
            log_dir=xdg_data / app_name_lower / 'logs',
            temp_dir=Path(tempfile.gettempdir()) / app_name_lower,
            documents_dir=documents
        )
    
    @staticmethod
    def _get_fallback_paths(app_name: str) -> DirectoryPaths:
        """폴백 경로 (알 수 없는 OS)"""
        base_dir = Path.home() / f'.{app_name.lower()}'
        
        return DirectoryPaths(
            app_data_dir=base_dir,
            config_dir=base_dir / 'config',
            cache_dir=base_dir / 'cache',
            log_dir=base_dir / 'logs',
            temp_dir=Path(tempfile.gettempdir()) / app_name.lower(),
            documents_dir=Path.home() / 'Documents' / app_name
        )
    
    @staticmethod
    def ensure_directories(paths: DirectoryPaths) -> bool:
        """모든 디렉토리 생성"""
        try:
            for path in asdict(paths).values():
                Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logging.error(f"디렉토리 생성 실패: {e}")
            return False


# ===========================================================================
# 💻 시스템 정보 수집
# ===========================================================================

class SystemInfo:
    """시스템 정보 수집 및 분석"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """전체 시스템 정보 수집"""
        return {
            'os': SystemInfo.get_os_info(),
            'hardware': SystemInfo.get_hardware_info(),
            'python': SystemInfo.get_python_info(),
            'network': SystemInfo.get_network_info(),
            'process': SystemInfo.get_process_info()
        }
    
    @staticmethod
    def get_os_info() -> Dict[str, str]:
        """OS 정보"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'platform': platform.platform()
        }
    
    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """하드웨어 정보"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'cores_physical': psutil.cpu_count(logical=False),
                'cores_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'usage_percent': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        }
    
    @staticmethod
    def get_python_info() -> Dict[str, str]:
        """Python 정보"""
        return {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': sys.executable
        }
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """네트워크 정보"""
        hostname = socket.gethostname()
        
        # 로컬 IP 주소 찾기
        local_ip = "127.0.0.1"
        try:
            # 외부 연결을 통해 로컬 IP 확인
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
        except:
            pass
        
        return {
            'hostname': hostname,
            'local_ip': local_ip,
            'interfaces': SystemInfo._get_network_interfaces(),
            'is_online': SystemInfo.check_internet_connection()
        }
    
    @staticmethod
    def _get_network_interfaces() -> List[Dict[str, str]]:
        """네트워크 인터페이스 목록"""
        interfaces = []
        for name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    interfaces.append({
                        'name': name,
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
        return interfaces
    
    @staticmethod
    def check_internet_connection() -> bool:
        """인터넷 연결 확인"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """현재 프로세스 정보"""
        process = psutil.Process()
        
        return {
            'pid': os.getpid(),
            'ppid': os.getppid(),
            'name': process.name(),
            'exe': process.exe(),
            'cwd': process.cwd(),
            'memory': process.memory_info().rss,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'num_threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
        }


# ===========================================================================
# ⚙️ 로컬 설정 클래스
# ===========================================================================

@dataclass
class LocalConfig:
    """로컬 환경 설정"""
    # 기본 정보
    app_name: str = "Universal DOE Platform"
    app_id: str = "com.universaldoe.platform"
    version: str = "2.0.0"
    
    # 디렉토리 경로
    paths: DirectoryPaths = None
    
    # 시스템 설정
    port_range: Tuple[int, int] = (8501, 8510)
    default_port: int = 8501
    localhost_only: bool = True
    single_instance: bool = True
    
    # 리소스 제한
    max_memory_percent: float = 50.0  # 최대 메모리 사용률
    max_cpu_percent: float = 80.0     # 최대 CPU 사용률
    cache_max_size_mb: int = 2048     # 최대 캐시 크기 (MB)
    log_max_size_mb: int = 100        # 로그 파일 최대 크기
    log_max_files: int = 10           # 로그 파일 최대 개수
    temp_max_age_hours: int = 24      # 임시 파일 최대 보관 시간
    
    # 백업 설정
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_max_count: int = 5
    
    # 성능 설정
    enable_gpu: bool = False
    parallel_workers: int = 4
    chunk_size_mb: int = 10
    
    # 네트워크 설정
    timeout_seconds: int = 30
    retry_count: int = 3
    proxy_auto_detect: bool = True
    
    # 기타 설정
    debug_mode: bool = False
    offline_mode_default: bool = True
    check_for_updates: bool = True
    telemetry_enabled: bool = False
    locale: str = "ko_KR"
    
    # 자동 시작 설정
    autostart_enabled: bool = False
    autostart_minimized: bool = True
    
    # 정리 설정
    auto_cleanup: bool = True
    cleanup_on_start: bool = True
    cleanup_on_exit: bool = True
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.paths is None:
            self.paths = PathManager.get_app_directories()
        
        # 프로퍼티 추가
        self._setup_properties()
    
    def _setup_properties(self):
        """편의 프로퍼티 설정"""
        # 디렉토리 경로 직접 접근
        self.app_data_dir = self.paths.app_data_dir
        self.config_dir = self.paths.config_dir
        self.cache_dir = self.paths.cache_dir
        self.log_dir = self.paths.log_dir
        self.temp_dir = self.paths.temp_dir
        self.documents_dir = self.paths.documents_dir
        
        # 특수 파일 경로
        self.db_path = self.app_data_dir / "db" / "universal_doe.db"
        self.config_file = self.config_dir / "local_config.json"
        self.pid_file = self.temp_dir / "universal_doe.pid"
        self.log_file = self.log_dir / "universal_doe.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['paths'] = self.paths.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LocalConfig':
        """딕셔너리에서 생성"""
        paths_data = data.pop('paths', None)
        if paths_data:
            paths = DirectoryPaths.from_dict(paths_data)
        else:
            paths = None
        
        return cls(paths=paths, **data)
    
    def save_to_file(self, filepath: Optional[Path] = None) -> bool:
        """설정을 파일로 저장"""
        if filepath is None:
            filepath = self.config_file
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logging.error(f"설정 저장 실패: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: Optional[Path] = None) -> 'LocalConfig':
        """파일에서 설정 로드"""
        if filepath is None:
            # 기본 경로에서 찾기
            paths = PathManager.get_app_directories()
            filepath = paths.config_dir / "local_config.json"
        
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls.from_dict(data)
        except Exception as e:
            logging.error(f"설정 로드 실패: {e}")
        
        # 기본값 반환
        return cls()
    
    def validate_system_requirements(self) -> Tuple[bool, List[str]]:
        """시스템 요구사항 검증"""
        errors = []
        
        # Python 버전 체크
        current_version = tuple(map(int, platform.python_version().split('.')))
        required_version = tuple(map(int, SYSTEM_REQUIREMENTS['minimum']['python_version'].split('.')))
        
        if current_version < required_version:
            errors.append(f"Python {SYSTEM_REQUIREMENTS['minimum']['python_version']} 이상 필요 (현재: {platform.python_version()})")
        
        # 메모리 체크
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < SYSTEM_REQUIREMENTS['minimum']['ram_gb']:
            errors.append(f"메모리 {SYSTEM_REQUIREMENTS['minimum']['ram_gb']}GB 이상 필요 (현재: {memory_gb:.1f}GB)")
        
        # 디스크 공간 체크
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_gb < SYSTEM_REQUIREMENTS['minimum']['disk_gb']:
            errors.append(f"디스크 공간 {SYSTEM_REQUIREMENTS['minimum']['disk_gb']}GB 이상 필요 (현재: {disk_gb:.1f}GB)")
        
        # OS 체크
        if platform.system() not in SYSTEM_REQUIREMENTS['supported_os']:
            errors.append(f"지원되지 않는 OS: {platform.system()}")
        
        return len(errors) == 0, errors
    
    def get_available_port(self) -> Optional[int]:
        """사용 가능한 포트 찾기"""
        for port in range(self.port_range[0], self.port_range[1] + 1):
            if not ProcessManager.is_port_in_use(port):
                return port
        return None
    
    def check_single_instance(self) -> bool:
        """단일 인스턴스 확인"""
        if not self.single_instance:
            return True
        
        return ProcessManager.create_pid_file(self.pid_file)
    
    def cleanup_on_exit(self):
        """종료 시 정리 작업"""
        try:
            # PID 파일 제거
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            # 임시 파일 정리
            if self.cleanup_on_exit and self.auto_cleanup:
                DirectoryManager.cleanup_temp_files(self.temp_dir, 0)
            
            # 오래된 로그 정리
            DirectoryManager.rotate_logs(self.log_dir, self.log_max_files, self.log_max_size_mb)
            
        except Exception as e:
            logging.error(f"종료 정리 작업 실패: {e}")


# ===========================================================================
# 🚀 프로세스 관리
# ===========================================================================

class ProcessManager:
    """프로세스 및 포트 관리"""
    
    @staticmethod
    def create_pid_file(pid_file: Path) -> bool:
        """PID 파일 생성 (중복 실행 방지)"""
        try:
            # 디렉토리 생성
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 기존 PID 파일 확인
            if pid_file.exists():
                with open(pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # 프로세스 존재 확인
                if psutil.pid_exists(old_pid):
                    try:
                        process = psutil.Process(old_pid)
                        # Python 프로세스인지 확인
                        if 'python' in process.name().lower():
                            return False  # 이미 실행 중
                    except:
                        pass
            
            # 새 PID 기록
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            return True
            
        except Exception as e:
            logging.error(f"PID 파일 생성 실패: {e}")
            return True  # 에러 시에도 실행 허용
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """포트 사용 여부 확인"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    @staticmethod
    def kill_process_on_port(port: int) -> bool:
        """특정 포트를 사용하는 프로세스 종료"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    process.wait(timeout=5)
                    return True
        except:
            pass
        return False
    
    @staticmethod
    def get_process_by_name(name: str) -> List[psutil.Process]:
        """이름으로 프로세스 찾기"""
        processes = []
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if name.lower() in proc.info['name'].lower():
                    processes.append(proc)
            except:
                pass
        return processes


# ===========================================================================
# 📁 디렉토리 관리
# ===========================================================================

class DirectoryManager:
    """디렉토리 관리 및 정리"""
    
    @staticmethod
    def get_directory_size(path: Path) -> int:
        """디렉토리 크기 계산 (bytes)"""
        total_size = 0
        
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except:
            pass
        
        return total_size
    
    @staticmethod
    def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24):
        """오래된 임시 파일 정리"""
        if not temp_dir.exists():
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # 파일 삭제
        for item in temp_dir.rglob('*'):
            if item.is_file():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        item.unlink()
                except:
                    pass
        
        # 빈 디렉토리 삭제
        for item in reversed(list(temp_dir.rglob('*'))):
            if item.is_dir() and not any(item.iterdir()):
                try:
                    item.rmdir()
                except:
                    pass
    
    @staticmethod
    def cleanup_cache(cache_dir: Path, max_size_mb: int):
        """캐시 크기 제한"""
        if not cache_dir.exists():
            return
        
        # 현재 크기 확인
        current_size = DirectoryManager.get_directory_size(cache_dir)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if current_size <= max_size_bytes:
            return
        
        # 파일 목록 (수정 시간 순)
        files = []
        for item in cache_dir.rglob('*'):
            if item.is_file():
                try:
                    files.append((item, item.stat().st_mtime, item.stat().st_size))
                except:
                    pass
        
        # 오래된 파일부터 삭제
        files.sort(key=lambda x: x[1])
        
        for filepath, _, size in files:
            try:
                filepath.unlink()
                current_size -= size
                
                if current_size <= max_size_bytes * 0.8:  # 80%까지 정리
                    break
            except:
                pass
    
    @staticmethod
    def rotate_logs(log_dir: Path, max_files: int, max_size_mb: int):
        """로그 파일 순환"""
        if not log_dir.exists():
            return
        
        # 로그 파일 목록
        log_files = []
        for item in log_dir.glob('*.log*'):
            if item.is_file():
                try:
                    log_files.append((item, item.stat().st_mtime))
                except:
                    pass
        
        # 날짜순 정렬
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        # 개수 제한
        for filepath, _ in log_files[max_files:]:
            try:
                filepath.unlink()
            except:
                pass
        
        # 크기 확인 및 순환
        for filepath, _ in log_files[:max_files]:
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    # 백업 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = filepath.with_suffix(f'.{timestamp}.log')
                    filepath.rename(backup_path)
            except:
                pass
    
    @staticmethod
    def create_backup(source_dir: Path, backup_dir: Path, prefix: str = "backup") -> Optional[Path]:
        """디렉토리 백업 생성"""
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{prefix}_{timestamp}"
            backup_path = backup_dir / backup_name
            
            # 압축 백업
            shutil.make_archive(str(backup_path), 'zip', source_dir)
            
            return backup_path.with_suffix('.zip')
            
        except Exception as e:
            logging.error(f"백업 생성 실패: {e}")
            return None


# ===========================================================================
# 🖥️ OS별 특화 기능
# ===========================================================================

class WindowsConfig:
    """Windows 특화 설정"""
    
    @staticmethod
    def set_registry_run(app_name: str, exe_path: Path, enable: bool = True) -> bool:
        """Windows 레지스트리 자동 시작 등록"""
        try:
            import winreg
            
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, 
                               winreg.KEY_ALL_ACCESS) as key:
                if enable:
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, str(exe_path))
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                    except:
                        pass
            
            return True
            
        except Exception as e:
            logging.error(f"레지스트리 설정 실패: {e}")
            return False
    
    @staticmethod
    def create_shortcut(target_path: Path, shortcut_path: Path, 
                       description: str = "", icon_path: Optional[Path] = None) -> bool:
        """바로가기 생성"""
        try:
            import win32com.client
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            
            shortcut.Targetpath = str(target_path)
            shortcut.WorkingDirectory = str(target_path.parent)
            shortcut.Description = description
            
            if icon_path:
                shortcut.IconLocation = str(icon_path)
            
            shortcut.save()
            return True
            
        except:
            # win32com 없을 때 PowerShell 사용
            try:
                ps_script = f'''
                $WshShell = New-Object -comObject WScript.Shell
                $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
                $Shortcut.TargetPath = "{target_path}"
                $Shortcut.Save()
                '''
                
                subprocess.run(["powershell", "-Command", ps_script], 
                             capture_output=True, text=True)
                return True
            except:
                return False


class MacOSConfig:
    """macOS 특화 설정"""
    
    @staticmethod
    def create_launch_agent(app_id: str, exe_path: Path, app_name: str) -> bool:
        """LaunchAgent 생성 (자동 시작)"""
        try:
            launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
            launch_agents_dir.mkdir(parents=True, exist_ok=True)
            
            plist_path = launch_agents_dir / f"{app_id}.plist"
            
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{app_id}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>LaunchOnlyOnce</key>
    <true/>
</dict>
</plist>"""
            
            with open(plist_path, 'w') as f:
                f.write(plist_content)
            
            # 권한 설정
            os.chmod(plist_path, 0o644)
            
            # LaunchAgent 로드
            subprocess.run(["launchctl", "load", str(plist_path)], 
                         capture_output=True)
            
            return True
            
        except Exception as e:
            logging.error(f"LaunchAgent 생성 실패: {e}")
            return False
    
    @staticmethod
    def remove_launch_agent(app_id: str):
        """LaunchAgent 제거"""
        try:
            plist_path = Path.home() / "Library" / "LaunchAgents" / f"{app_id}.plist"
            
            if plist_path.exists():
                subprocess.run(["launchctl", "unload", str(plist_path)], 
                             capture_output=True)
                plist_path.unlink()
                
        except:
            pass


class LinuxConfig:
    """Linux 특화 설정"""
    
    @staticmethod
    def create_desktop_entry(app_id: str, exe_path: Path, app_name: str,
                           icon_path: Optional[Path] = None) -> bool:
        """Desktop Entry 생성 (자동 시작)"""
        try:
            autostart_dir = Path.home() / ".config" / "autostart"
            autostart_dir.mkdir(parents=True, exist_ok=True)
            
            desktop_path = autostart_dir / f"{app_id}.desktop"
            
            desktop_content = f"""[Desktop Entry]
Type=Application
Name={app_name}
Exec={exe_path}
Icon={icon_path or 'applications-science'}
Terminal=false
Categories=Science;Education;
StartupNotify=true
"""
            
            with open(desktop_path, 'w') as f:
                f.write(desktop_content)
            
            # 실행 권한
            os.chmod(desktop_path, 0o755)
            
            return True
            
        except Exception as e:
            logging.error(f"Desktop Entry 생성 실패: {e}")
            return False
    
    @staticmethod
    def update_desktop_database():
        """데스크톱 데이터베이스 업데이트"""
        try:
            apps_dir = Path.home() / ".local" / "share" / "applications"
            subprocess.run(["update-desktop-database", str(apps_dir)], 
                         capture_output=True)
        except:
            pass


# ===========================================================================
# 🌐 네트워크 관리
# ===========================================================================

class NetworkManager:
    """네트워크 연결 및 프록시 관리"""
    
    @staticmethod
    def get_proxy_settings() -> Dict[str, Optional[str]]:
        """시스템 프록시 설정 감지"""
        proxies = {
            'http': None,
            'https': None,
            'no_proxy': None
        }
        
        # 환경 변수 확인
        for key in ['http_proxy', 'https_proxy', 'no_proxy']:
            value = os.environ.get(key) or os.environ.get(key.upper())
            if value:
                proxies[key.replace('_proxy', '')] = value
        
        # Windows 레지스트리 확인
        if platform.system() == "Windows":
            try:
                import winreg
                
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                   r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
                    proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
                    
                    if proxy_enable:
                        proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                        if proxy_server:
                            proxies['http'] = f"http://{proxy_server}"
                            proxies['https'] = f"http://{proxy_server}"
            except:
                pass
        
        return proxies
    
    @staticmethod
    def test_connectivity(url: str = "https://www.google.com", 
                         timeout: int = 5) -> Tuple[bool, Optional[float]]:
        """연결 테스트 및 응답 시간 측정"""
        try:
            import urllib.request
            
            start_time = time.time()
            
            # 프록시 설정 적용
            proxies = NetworkManager.get_proxy_settings()
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            
            response = opener.open(url, timeout=timeout)
            response.close()
            
            elapsed_time = time.time() - start_time
            
            return True, elapsed_time
            
        except Exception:
            return False, None
    
    @staticmethod
    def get_public_ip() -> Optional[str]:
        """공인 IP 주소 확인"""
        try:
            import urllib.request
            
            response = urllib.request.urlopen('https://api.ipify.org', timeout=5)
            ip = response.read().decode('utf-8').strip()
            
            return ip
            
        except:
            return None


# ===========================================================================
# 🔧 싱글톤 인스턴스 관리
# ===========================================================================

_local_config_instance: Optional[LocalConfig] = None


def get_local_config() -> LocalConfig:
    """LocalConfig 싱글톤 인스턴스 반환"""
    global _local_config_instance
    
    if _local_config_instance is None:
        # 설정 파일에서 로드 시도
        _local_config_instance = LocalConfig.load_from_file()
        
        # 디렉토리 생성
        PathManager.ensure_directories(_local_config_instance.paths)
        
        # 설정 파일이 없으면 저장
        if not _local_config_instance.config_file.exists():
            _local_config_instance.save_to_file()
    
    return _local_config_instance


def reset_local_config():
    """LocalConfig 인스턴스 리셋"""
    global _local_config_instance
    _local_config_instance = None


# ===========================================================================
# 🚀 초기화 및 유틸리티
# ===========================================================================

def initialize_local_environment() -> bool:
    """로컬 환경 초기화"""
    try:
        config = get_local_config()
        
        # 1. 시스템 요구사항 검증
        is_valid, errors = config.validate_system_requirements()
        if not is_valid:
            logging.warning(f"시스템 요구사항 미충족: {errors}")
        
        # 2. 단일 인스턴스 확인
        if config.single_instance and not config.check_single_instance():
            logging.error("이미 실행 중인 인스턴스가 있습니다.")
            return False
        
        # 3. 디렉토리 생성
        PathManager.ensure_directories(config.paths)
        
        # 4. 임시 파일 정리
        if config.cleanup_on_start and config.auto_cleanup:
            DirectoryManager.cleanup_temp_files(config.temp_dir, config.temp_max_age_hours)
        
        # 5. 로그 순환
        DirectoryManager.rotate_logs(config.log_dir, config.log_max_files, config.log_max_size_mb)
        
        # 6. 캐시 크기 관리
        DirectoryManager.cleanup_cache(config.cache_dir, config.cache_max_size_mb)
        
        logging.info("로컬 환경 초기화 완료")
        return True
        
    except Exception as e:
        logging.error(f"로컬 환경 초기화 실패: {e}")
        return False


def get_system_summary() -> Dict[str, Any]:
    """시스템 정보 요약"""
    info = SystemInfo.get_system_info()
    config = get_local_config()
    
    return {
        'os': f"{info['os']['system']} {info['os']['release']}",
        'python': info['python']['version'],
        'cpu': f"{info['hardware']['cpu']['cores_logical']} cores",
        'memory': f"{info['hardware']['memory']['total'] / (1024**3):.1f} GB",
        'disk_free': f"{info['hardware']['disk']['free'] / (1024**3):.1f} GB",
        'network': "Online" if info['network']['is_online'] else "Offline",
        'app_version': config.version,
        'data_dir': str(config.app_data_dir)
    }


def setup_autostart(enable: bool = True) -> bool:
    """자동 시작 설정"""
    config = get_local_config()
    
    # 실행 파일 경로
    if getattr(sys, 'frozen', False):
        exe_path = Path(sys.executable)
    else:
        # 개발 모드에서는 launcher.py 경로
        exe_path = Path(sys.argv[0]).resolve()
    
    system = platform.system()
    
    try:
        if system == "Windows":
            return WindowsConfig.set_registry_run(config.app_name, exe_path, enable)
        elif system == "Darwin":
            if enable:
                return MacOSConfig.create_launch_agent(config.app_id, exe_path, config.app_name)
            else:
                MacOSConfig.remove_launch_agent(config.app_id)
                return True
        elif system == "Linux":
            if enable:
                return LinuxConfig.create_desktop_entry(config.app_id, exe_path, config.app_name)
            else:
                desktop_path = Path.home() / ".config" / "autostart" / f"{config.app_id}.desktop"
                if desktop_path.exists():
                    desktop_path.unlink()
                return True
    except Exception as e:
        logging.error(f"자동 시작 설정 실패: {e}")
    
    return False


def create_shortcuts() -> Dict[str, bool]:
    """바로가기 생성"""
    config = get_local_config()
    results = {}
    
    # 실행 파일 경로
    if getattr(sys, 'frozen', False):
        exe_path = Path(sys.executable)
    else:
        exe_path = Path(sys.argv[0]).resolve()
    
    # 바탕화면 바로가기
    desktop_path = Path.home() / "Desktop" / f"{config.app_name}.lnk"
    
    if platform.system() == "Windows":
        results['desktop'] = WindowsConfig.create_shortcut(
            exe_path, desktop_path, config.app_name
        )
    
    return results


# ===========================================================================
# 📊 디버그 및 진단
# ===========================================================================

def print_environment_info():
    """환경 정보 출력 (디버깅용)"""
    print("=" * 70)
    print("Universal DOE Platform - 환경 정보")
    print("=" * 70)
    
    summary = get_system_summary()
    for key, value in summary.items():
        print(f"{key.ljust(15)}: {value}")
    
    print("\n디렉토리 구조:")
    config = get_local_config()
    for name, path in asdict(config.paths).items():
        exists = "✓" if Path(path).exists() else "✗"
        size = DirectoryManager.get_directory_size(Path(path)) / (1024**2)
        print(f"  {exists} {name.ljust(20)}: {path} ({size:.1f} MB)")
    
    print("\n네트워크 상태:")
    proxies = NetworkManager.get_proxy_settings()
    for key, value in proxies.items():
        if value:
            print(f"  {key}: {value}")
    
    # 연결 테스트
    is_connected, response_time = NetworkManager.test_connectivity()
    if is_connected:
        print(f"  인터넷 연결: ✓ (응답시간: {response_time:.2f}초)")
    else:
        print("  인터넷 연결: ✗")
    
    print("=" * 70)


# ===========================================================================
# 🔄 정리 함수 (앱 종료 시 호출)
# ===========================================================================

def cleanup_on_exit():
    """앱 종료 시 정리 작업"""
    config = get_local_config()
    config.cleanup_on_exit()
    
    logging.info("정리 작업 완료")


# 모듈 테스트
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 환경 초기화
    if initialize_local_environment():
        print("✅ 로컬 환경 초기화 성공")
    else:
        print("❌ 로컬 환경 초기화 실패")
        sys.exit(1)
    
    # 환경 정보 출력
    print_environment_info()
    
    # 자동 시작 테스트
    print("\n자동 시작 설정 테스트:")
    if setup_autostart(True):
        print("✅ 자동 시작 등록 성공")
    else:
        print("❌ 자동 시작 등록 실패")
    
    # 종료 시 정리
    cleanup_on_exit()
