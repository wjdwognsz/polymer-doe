"""
🖥️ Universal DOE Platform - 로컬 실행 환경 설정
================================================================================
데스크톱 애플리케이션의 로컬 실행 환경을 관리하는 설정 모듈
OS별 경로 자동 설정, 시스템 리소스 관리, 프로세스 제어
================================================================================
"""

import os
import sys
import platform
import socket
import psutil
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import subprocess

# ============================================================================
# 🖥️ 시스템 정보
# ============================================================================

class SystemInfo:
    """시스템 정보 및 환경 감지"""
    
    @staticmethod
    def get_os_info() -> Dict[str, str]:
        """운영체제 정보"""
        return {
            'system': platform.system(),  # Windows, Darwin, Linux
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),  # x86_64, arm64 등
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation()
        }
    
    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """시스템 리소스 정보"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'count': cpu_count,
                'count_logical': psutil.cpu_count(logical=True),
                'percent': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        }
    
    @staticmethod
    def check_internet_connection() -> bool:
        """인터넷 연결 확인"""
        try:
            # Google DNS 서버에 연결 시도
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except (socket.error, socket.timeout):
            return False
    
    @staticmethod
    def get_available_port(start_port: int = 8501, max_attempts: int = 10) -> int:
        """사용 가능한 포트 찾기"""
        for i in range(max_attempts):
            port = start_port + i
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
                sock.close()
                return port
            except socket.error:
                continue
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


# ============================================================================
# 📂 경로 관리
# ============================================================================

class PathManager:
    """OS별 표준 경로 관리"""
    
    @staticmethod
    def get_app_data_dir() -> Path:
        """OS별 앱 데이터 디렉토리"""
        app_name = "UniversalDOE"
        system = platform.system()
        
        if system == "Windows":
            # Windows: %APPDATA%\UniversalDOE
            base = os.environ.get('APPDATA')
            if not base:
                base = Path.home() / "AppData" / "Roaming"
            return Path(base) / app_name
            
        elif system == "Darwin":  # macOS
            # macOS: ~/Library/Application Support/UniversalDOE
            return Path.home() / "Library" / "Application Support" / app_name
            
        else:  # Linux 및 기타
            # Linux: ~/.local/share/universaldoe
            base = os.environ.get('XDG_DATA_HOME')
            if not base:
                base = Path.home() / ".local" / "share"
            return Path(base) / app_name.lower()
    
    @staticmethod
    def get_config_dir() -> Path:
        """OS별 설정 디렉토리"""
        app_name = "UniversalDOE"
        system = platform.system()
        
        if system == "Windows":
            # Windows: %APPDATA%\UniversalDOE\config
            return PathManager.get_app_data_dir() / "config"
            
        elif system == "Darwin":  # macOS
            # macOS: ~/Library/Preferences/UniversalDOE
            return Path.home() / "Library" / "Preferences" / app_name
            
        else:  # Linux
            # Linux: ~/.config/universaldoe
            base = os.environ.get('XDG_CONFIG_HOME')
            if not base:
                base = Path.home() / ".config"
            return Path(base) / app_name.lower()
    
    @staticmethod
    def get_cache_dir() -> Path:
        """OS별 캐시 디렉토리"""
        app_name = "UniversalDOE"
        system = platform.system()
        
        if system == "Windows":
            # Windows: %LOCALAPPDATA%\UniversalDOE\Cache
            base = os.environ.get('LOCALAPPDATA')
            if not base:
                base = Path.home() / "AppData" / "Local"
            return Path(base) / app_name / "Cache"
            
        elif system == "Darwin":  # macOS
            # macOS: ~/Library/Caches/UniversalDOE
            return Path.home() / "Library" / "Caches" / app_name
            
        else:  # Linux
            # Linux: ~/.cache/universaldoe
            base = os.environ.get('XDG_CACHE_HOME')
            if not base:
                base = Path.home() / ".cache"
            return Path(base) / app_name.lower()
    
    @staticmethod
    def get_log_dir() -> Path:
        """OS별 로그 디렉토리"""
        app_name = "UniversalDOE"
        system = platform.system()
        
        if system == "Windows":
            # Windows: %LOCALAPPDATA%\UniversalDOE\Logs
            base = os.environ.get('LOCALAPPDATA')
            if not base:
                base = Path.home() / "AppData" / "Local"
            return Path(base) / app_name / "Logs"
            
        elif system == "Darwin":  # macOS
            # macOS: ~/Library/Logs/UniversalDOE
            return Path.home() / "Library" / "Logs" / app_name
            
        else:  # Linux
            # Linux: ~/.local/share/universaldoe/logs
            return PathManager.get_app_data_dir() / "logs"
    
    @staticmethod
    def ensure_directories() -> Dict[str, Path]:
        """필요한 모든 디렉토리 생성"""
        directories = {
            'app_data': PathManager.get_app_data_dir(),
            'config': PathManager.get_config_dir(),
            'cache': PathManager.get_cache_dir(),
            'logs': PathManager.get_log_dir(),
            'database': PathManager.get_app_data_dir() / "db",
            'temp': PathManager.get_app_data_dir() / "temp",
            'modules': PathManager.get_app_data_dir() / "modules",
            'exports': PathManager.get_app_data_dir() / "exports",
            'backups': PathManager.get_app_data_dir() / "backups"
        }
        
        # 디렉토리 생성
        for name, path in directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Windows에서 숨김 속성 설정
                if platform.system() == "Windows" and name in ['config', 'cache']:
                    import ctypes
                    FILE_ATTRIBUTE_HIDDEN = 0x02
                    ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                logging.error(f"Failed to create directory {path}: {e}")
                # 대체 경로 사용 (임시 디렉토리)
                directories[name] = Path(tempfile.gettempdir()) / "UniversalDOE" / name
                directories[name].mkdir(parents=True, exist_ok=True)
        
        return directories


# ============================================================================
# 🔧 로컬 설정
# ============================================================================

# 디렉토리 경로
PATHS = PathManager.ensure_directories()

LOCAL_CONFIG = {
    # 시스템 정보
    'system': SystemInfo.get_os_info(),
    
    # 경로 설정
    'paths': {
        'app_data': PATHS['app_data'],
        'config': PATHS['config'],
        'cache': PATHS['cache'],
        'logs': PATHS['logs'],
        'database': PATHS['database'],
        'temp': PATHS['temp'],
        'modules': PATHS['modules'],
        'exports': PATHS['exports'],
        'backups': PATHS['backups']
    },
    
    # 데이터베이스 설정
    'database': {
        'type': 'sqlite',
        'path': PATHS['database'] / 'app.db',
        'backup_path': PATHS['backups'],
        'backup_interval': 3600,  # 1시간마다 백업
        'max_backups': 5,
        'wal_mode': True,
        'connection_pool_size': 5
    },
    
    # 캐시 설정
    'cache': {
        'path': PATHS['cache'],
        'max_size_mb': 500,
        'ttl_days': 30,
        'cleanup_on_start': True,
        'compression': True
    },
    
    # 로그 설정
    'logging': {
        'path': PATHS['logs'],
        'level': 'INFO',
        'max_files': 10,
        'max_size_mb': 10,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'console_output': True
    },
    
    # 임시 파일 설정
    'temp': {
        'path': PATHS['temp'],
        'cleanup_on_start': True,
        'max_age_hours': 24
    },
    
    # Streamlit 서버 설정
    'server': {
        'host': 'localhost',
        'port': SystemInfo.get_available_port(8501),
        'base_url_path': '',
        'enable_cors': False,
        'enable_xsrf_protection': True,
        'max_upload_size': 200,  # MB
        'headless': True,
        'run_on_save': False
    },
    
    # UI 설정
    'ui': {
        'use_system_browser': False,  # 내장 WebView 사용
        'window_title': 'Universal DOE Platform',
        'window_size': (1280, 800),
        'window_position': 'center',
        'start_maximized': False,
        'enable_dev_tools': False,
        'show_toolbar': True,
        'show_status_bar': True
    },
    
    # 오프라인 모드
    'offline_mode': {
        'default': True,  # 기본적으로 오프라인 모드
        'check_interval': 30,  # 30초마다 연결 확인
        'auto_sync': True,  # 온라인 시 자동 동기화
        'cache_ai_responses': True,
        'fallback_to_cached': True
    },
    
    # 프로세스 관리
    'process': {
        'pid_file': PATHS['app_data'] / 'app.pid',
        'lock_file': PATHS['app_data'] / 'app.lock',
        'single_instance': True,  # 단일 인스턴스만 허용
        'auto_restart_on_crash': True,
        'startup_timeout': 30  # 초
    },
    
    # 업데이트 설정
    'updates': {
        'check_on_startup': True,
        'auto_download': False,
        'channel': 'stable',  # stable, beta, dev
        'server_url': 'https://api.universaldoe.com/updates'
    },
    
    # 보안 설정
    'security': {
        'enable_sandbox': True,
        'verify_signatures': True,
        'secure_temp_files': True,
        'clear_clipboard_on_exit': True
    }
}

# ============================================================================
# 🖥️ 시스템 요구사항
# ============================================================================

SYSTEM_REQUIREMENTS = {
    'minimum': {
        'ram_gb': 4,
        'disk_gb': 2,
        'cpu_cores': 2,
        'python_version': '3.8',
        'os_versions': {
            'Windows': '10',
            'Darwin': '10.14',  # macOS Mojave
            'Linux': 'Ubuntu 18.04'
        }
    },
    'recommended': {
        'ram_gb': 8,
        'disk_gb': 5,
        'cpu_cores': 4,
        'python_version': '3.10',
        'gpu': 'Optional for ML features'
    },
    'supported_architectures': ['x86_64', 'arm64', 'aarch64']
}

# ============================================================================
# 🔍 시스템 체크 함수
# ============================================================================

def check_system_requirements() -> Dict[str, Any]:
    """시스템 요구사항 확인"""
    resources = SystemInfo.get_system_resources()
    os_info = SystemInfo.get_os_info()
    
    checks = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'info': []
    }
    
    # RAM 체크
    ram_gb = resources['memory']['total'] / (1024**3)
    if ram_gb < SYSTEM_REQUIREMENTS['minimum']['ram_gb']:
        checks['errors'].append(f"메모리 부족: {ram_gb:.1f}GB (최소 {SYSTEM_REQUIREMENTS['minimum']['ram_gb']}GB 필요)")
        checks['passed'] = False
    elif ram_gb < SYSTEM_REQUIREMENTS['recommended']['ram_gb']:
        checks['warnings'].append(f"메모리: {ram_gb:.1f}GB (권장 {SYSTEM_REQUIREMENTS['recommended']['ram_gb']}GB)")
    
    # 디스크 공간 체크
    disk_gb = resources['disk']['free'] / (1024**3)
    if disk_gb < SYSTEM_REQUIREMENTS['minimum']['disk_gb']:
        checks['errors'].append(f"디스크 공간 부족: {disk_gb:.1f}GB (최소 {SYSTEM_REQUIREMENTS['minimum']['disk_gb']}GB 필요)")
        checks['passed'] = False
    
    # CPU 체크
    cpu_cores = resources['cpu']['count']
    if cpu_cores < SYSTEM_REQUIREMENTS['minimum']['cpu_cores']:
        checks['warnings'].append(f"CPU 코어: {cpu_cores}개 (권장 {SYSTEM_REQUIREMENTS['minimum']['cpu_cores']}개 이상)")
    
    # Python 버전 체크
    python_version = sys.version_info
    min_version = tuple(map(int, SYSTEM_REQUIREMENTS['minimum']['python_version'].split('.')))
    if python_version[:2] < min_version[:2]:
        checks['errors'].append(f"Python 버전: {python_version.major}.{python_version.minor} (최소 {SYSTEM_REQUIREMENTS['minimum']['python_version']} 필요)")
        checks['passed'] = False
    
    # 아키텍처 체크
    if os_info['machine'] not in SYSTEM_REQUIREMENTS['supported_architectures']:
        checks['warnings'].append(f"지원되지 않는 아키텍처: {os_info['machine']}")
    
    # 정보 추가
    checks['info'].extend([
        f"운영체제: {os_info['system']} {os_info['release']}",
        f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}",
        f"CPU: {cpu_cores}코어",
        f"메모리: {ram_gb:.1f}GB",
        f"여유 디스크: {disk_gb:.1f}GB"
    ])
    
    return checks

# ============================================================================
# 🚀 프로세스 관리
# ============================================================================

class ProcessManager:
    """애플리케이션 프로세스 관리"""
    
    @staticmethod
    def create_pid_file() -> bool:
        """PID 파일 생성"""
        pid_file = LOCAL_CONFIG['process']['pid_file']
        try:
            # 기존 PID 파일 확인
            if pid_file.exists():
                # 프로세스가 실행 중인지 확인
                with open(pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                if psutil.pid_exists(old_pid):
                    try:
                        p = psutil.Process(old_pid)
                        if 'python' in p.name().lower():
                            return False  # 이미 실행 중
                    except:
                        pass
            
            # 새 PID 파일 생성
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create PID file: {e}")
            return True  # 에러 시에도 실행 허용
    
    @staticmethod
    def remove_pid_file():
        """PID 파일 제거"""
        pid_file = LOCAL_CONFIG['process']['pid_file']
        try:
            if pid_file.exists():
                pid_file.unlink()
        except Exception as e:
            logging.error(f"Failed to remove PID file: {e}")
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """포트 사용 여부 확인"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    @staticmethod
    def kill_process_on_port(port: int):
        """특정 포트를 사용하는 프로세스 종료"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    pass

# ============================================================================
# 🔧 설정 헬퍼 함수
# ============================================================================

def get_local_config(key: str, default: Any = None) -> Any:
    """로컬 설정값 가져오기"""
    keys = key.split('.')
    value = LOCAL_CONFIG
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def update_local_config(key: str, value: Any) -> None:
    """로컬 설정 업데이트"""
    keys = key.split('.')
    config = LOCAL_CONFIG
    
    for k in keys[:-1]:
        if k not in config:
            config[k] = {}
        config = config[k]
    
    config[keys[-1]] = value

def save_local_config():
    """로컬 설정을 파일로 저장"""
    config_file = LOCAL_CONFIG['paths']['config'] / 'local_config.json'
    try:
        # Path 객체를 문자열로 변환
        config_to_save = json.loads(
            json.dumps(LOCAL_CONFIG, default=str)
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Failed to save local config: {e}")

def load_local_config():
    """파일에서 로컬 설정 로드"""
    config_file = LOCAL_CONFIG['paths']['config'] / 'local_config.json'
    try:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                
            # Path 문자열을 Path 객체로 변환
            for key in ['paths', 'database', 'cache', 'logging', 'temp', 'process']:
                if key in saved_config:
                    for subkey, value in saved_config[key].items():
                        if isinstance(value, str) and ('/' in value or '\\' in value):
                            saved_config[key][subkey] = Path(value)
            
            LOCAL_CONFIG.update(saved_config)
            
    except Exception as e:
        logging.error(f"Failed to load local config: {e}")

# ============================================================================
# 🌐 네트워크 상태 관리
# ============================================================================

class NetworkManager:
    """네트워크 상태 및 연결 관리"""
    
    @staticmethod
    def get_network_status() -> Dict[str, Any]:
        """네트워크 상태 정보"""
        return {
            'is_online': SystemInfo.check_internet_connection(),
            'timestamp': datetime.now().isoformat(),
            'interfaces': NetworkManager.get_network_interfaces(),
            'proxy': NetworkManager.get_proxy_settings()
        }
    
    @staticmethod
    def get_network_interfaces() -> List[Dict[str, Any]]:
        """네트워크 인터페이스 정보"""
        interfaces = []
        for name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:  # IPv4
                    interfaces.append({
                        'name': name,
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
        return interfaces
    
    @staticmethod
    def get_proxy_settings() -> Dict[str, Optional[str]]:
        """프록시 설정 확인"""
        return {
            'http': os.environ.get('HTTP_PROXY'),
            'https': os.environ.get('HTTPS_PROXY'),
            'no_proxy': os.environ.get('NO_PROXY')
        }

# ============================================================================
# 📤 Export
# ============================================================================

__all__ = [
    'LOCAL_CONFIG',
    'SYSTEM_REQUIREMENTS',
    'SystemInfo',
    'PathManager',
    'ProcessManager',
    'NetworkManager',
    'check_system_requirements',
    'get_local_config',
    'update_local_config',
    'save_local_config',
    'load_local_config'
]

# 초기 설정 로드
if __name__ != "__main__":
    load_local_config()
