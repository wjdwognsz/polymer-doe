# utils/database_manager.py

"""
Universal DOE Platform - SQLite 데이터베이스 관리자

오프라인 우선 설계로 모든 데이터를 안전하게 로컬에 저장합니다.
스레드 안전성, 자동 백업, 암호화를 지원합니다.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
import shutil
import hashlib
from contextlib import contextmanager
import time
from queue import Queue
import atexit
import os
from dataclasses import dataclass, asdict, field
from enum import Enum
from cryptography.fernet import Fernet
import base64
import bcrypt

logger = logging.getLogger(__name__)


# ==================== 열거형 정의 ====================

class UserRole(Enum):
    """사용자 역할"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    PREMIUM = "premium"


class ProjectStatus(Enum):
    """프로젝트 상태"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    SHARED = "shared"


class ExperimentStatus(Enum):
    """실험 상태"""
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncStatus(Enum):
    """동기화 상태"""
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


# ==================== 데이터 모델 ====================

@dataclass
class User:
    """사용자 데이터 모델"""
    id: Optional[int] = None
    email: str = ""
    password_hash: str = ""
    name: str = ""
    role: str = UserRole.USER.value
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Project:
    """프로젝트 데이터 모델"""
    id: Optional[int] = None
    user_id: int = 0
    name: str = ""
    description: str = ""
    field: str = ""
    module_id: str = ""
    status: str = ProjectStatus.ACTIVE.value
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Experiment:
    """실험 데이터 모델"""
    id: Optional[int] = None
    project_id: int = 0
    name: str = ""
    design_type: str = ""
    factors: Dict[str, Any] = field(default_factory=dict)
    responses: Dict[str, Any] = field(default_factory=dict)
    design_matrix: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = ExperimentStatus.PLANNING.value
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


# ==================== 데이터베이스 관리자 ====================

class DatabaseManager:
    """SQLite 데이터베이스 관리자"""
    
    def __init__(self, db_path: Path, config: Optional[Dict[str, Any]] = None):
        """
        데이터베이스 매니저 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            config: 추가 설정 옵션
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 설정
        self.config = config or {}
        self.wal_mode = self.config.get('wal_mode', True)
        self.connection_pool_size = self.config.get('connection_pool_size', 5)
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.backup_path = self.config.get('backup_path', self.db_path.parent / 'backups')
        self.backup_interval = self.config.get('backup_interval', 3600)  # 1시간
        self.max_backups = self.config.get('max_backups', 5)
        
        # 스레드 안전성을 위한 락
        self._lock = threading.RLock()
        self._connections = {}
        self._backup_thread = None
        self._backup_stop_event = threading.Event()
        
        # 캐시
        self._cache = {}
        self._cache_ttl = {}
        
        # 암호화 설정
        self._setup_encryption()
        
        # 초기화
        self._init_database()
        
        # 백업 스레드 시작
        if self.backup_enabled:
            self._start_backup_thread()
        
        # 종료 시 정리
        atexit.register(self.close)
    
    def _setup_encryption(self):
        """암호화 설정"""
        key_file = self.db_path.parent / ".encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # 키 파일 권한 설정 (Windows 제외)
            if os.name != 'nt':
                os.chmod(key_file, 0o600)
        
        self._cipher = Fernet(key)
        logger.info("Encryption initialized")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with self._get_connection() as conn:
            # WAL 모드 설정 (성능 향상)
            if self.wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # 외래 키 제약 활성화
            conn.execute("PRAGMA foreign_keys = ON")
            
            # 테이블 생성
            self._create_tables(conn)
            
            # 인덱스 생성
            self._create_indexes(conn)
            
            # 초기 데이터 삽입
            self._insert_initial_data(conn)
            
            conn.commit()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """테이블 생성"""
        
        # 사용자 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                settings TEXT DEFAULT '{}',
                is_active INTEGER DEFAULT 1,
                UNIQUE(email)
            )
        ''')
        
        # 프로젝트 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                field TEXT,
                module_id TEXT,
                status TEXT DEFAULT 'active',
                data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 실험 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                design_type TEXT,
                factors TEXT DEFAULT '{}',
                responses TEXT DEFAULT '{}',
                design_matrix TEXT DEFAULT '[]',
                results TEXT DEFAULT '{}',
                status TEXT DEFAULT 'planning',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
        ''')
        
        # API 키 테이블 (암호화 저장)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                service TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, service)
            )
        ''')
        
        # 동기화 로그 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                sync_status TEXT DEFAULT 'pending',
                sync_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced_at TIMESTAMP,
                error_message TEXT
            )
        ''')
        
        # 캐시 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 활동 로그 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id INTEGER,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        ''')
        
        # 모듈 사용 기록 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS module_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id INTEGER,
                module_id TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL
            )
        ''')
        
        # 프로젝트 공유 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS project_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                shared_with_email TEXT NOT NULL,
                permission TEXT DEFAULT 'view',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
                UNIQUE(project_id, shared_with_email)
            )
        ''')
        
        # 알림 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT,
                data TEXT DEFAULT '{}',
                is_read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                read_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 스키마 버전 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 초기 버전 설정
        conn.execute('''
            INSERT OR IGNORE INTO schema_version (version, description)
            VALUES (1, 'Initial schema')
        ''')

        # 고분자 용매 시스템 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS polymer_solvent_systems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                polymer_name TEXT NOT NULL,
                polymer_mw TEXT,
                solvent_system TEXT NOT NULL,
                solvent_ratio TEXT,
                temperature REAL,
                phase_behavior TEXT,
                hansen_distance REAL,
                dissolution_time INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
        ''')
    
        # 고분자 가공 조건 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS polymer_processing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                process_type TEXT NOT NULL,  -- electrospinning, extrusion, etc
                temperature REAL,
                pressure REAL,
                flow_rate REAL,
                voltage REAL,  -- for electrospinning
                distance REAL,  -- for electrospinning
                parameters TEXT,  -- JSON for additional params
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
        ''')
    
        # 추출된 프로토콜 테이블 (기존 protocols 테이블 확장)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS extracted_protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,  -- pdf, text, html, etc
                source_url TEXT,
                doi TEXT,
                title TEXT,
                authors TEXT,
                materials JSON NOT NULL,
                equipment JSON,
                conditions JSON,
                procedure JSON,
                safety JSON,
                confidence_score REAL,
                extraction_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """인덱스 생성 (성능 최적화)"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
            "CREATE INDEX IF NOT EXISTS idx_sync_log_status ON sync_log(sync_status)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_log(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_log(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, is_read)",
            "CREATE INDEX IF NOT EXISTS idx_module_usage_user ON module_usage(user_id, module_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)"
        ]
        
        for index in indexes:
            conn.execute(index)
        
        # 트리거 생성 (자동 updated_at)
        tables_with_updated_at = ['users', 'projects', 'experiments', 'api_keys']
        for table in tables_with_updated_at:
            conn.execute(f'''
                CREATE TRIGGER IF NOT EXISTS update_{table}_timestamp
                AFTER UPDATE ON {table}
                BEGIN
                    UPDATE {table} SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END
            ''')
    
    def _insert_initial_data(self, conn: sqlite3.Connection):
        """초기 데이터 삽입"""
        # 게스트 사용자 확인/생성
        guest_exists = conn.execute(
            "SELECT COUNT(*) FROM users WHERE email = ?",
            ("guest@universaldoe.local",)
        ).fetchone()[0]
        
        if not guest_exists:
            conn.execute('''
                INSERT INTO users (email, password_hash, name, role)
                VALUES (?, ?, ?, ?)
            ''', ("guest@universaldoe.local", "no_password", "게스트 사용자", UserRole.GUEST.value))
    
    @contextmanager
    def _get_connection(self, readonly: bool = False) -> sqlite3.Connection:
        """데이터베이스 연결 컨텍스트 매니저"""
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._connections:
                conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0
                )
                conn.row_factory = sqlite3.Row
                if readonly:
                    conn.execute("PRAGMA query_only = ON")
                self._connections[thread_id] = conn
            
            conn = self._connections[thread_id]
        
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if not readonly:
                conn.commit()
    
    # ==================== 백업 관리 ====================
    
    def _start_backup_thread(self):
        """백업 스레드 시작"""
        def backup_worker():
            while not self._backup_stop_event.is_set():
                try:
                    self.backup_database()
                    self._cleanup_old_backups()
                except Exception as e:
                    logger.error(f"Backup error: {e}")
                
                # 백업 간격만큼 대기
                self._backup_stop_event.wait(self.backup_interval)
        
        self._backup_thread = threading.Thread(target=backup_worker, daemon=True)
        self._backup_thread.start()
        logger.info("Backup thread started")
    
    def backup_database(self) -> Path:
        """데이터베이스 백업"""
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_path / f"backup_{timestamp}.db"
        
        with self._lock:
            with self._get_connection() as conn:
                # SQLite 백업 API 사용
                backup = sqlite3.connect(str(backup_file))
                conn.backup(backup)
                backup.close()
        
        logger.info(f"Database backed up to {backup_file}")
        return backup_file
    
    def restore_database(self, backup_file: Path) -> bool:
        """데이터베이스 복원"""
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            # 현재 DB 백업
            temp_backup = self.db_path.with_suffix('.db.temp')
            shutil.copy2(self.db_path, temp_backup)
            
            # 복원
            shutil.copy2(backup_file, self.db_path)
            
            # 복원된 DB 테스트
            with self._get_connection() as conn:
                conn.execute("SELECT COUNT(*) FROM users")
            
            # 성공 시 임시 백업 삭제
            temp_backup.unlink()
            logger.info(f"Database restored from {backup_file}")
            return True
            
        except Exception as e:
            # 실패 시 원래 DB로 복구
            if temp_backup.exists():
                shutil.copy2(temp_backup, self.db_path)
                temp_backup.unlink()
            logger.error(f"Restore failed: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """오래된 백업 삭제"""
        if not self.backup_path.exists():
            return
        
        backups = sorted(
            self.backup_path.glob("backup_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # 지정된 개수만 유지
        for old_backup in backups[self.max_backups:]:
            old_backup.unlink()
            logger.info(f"Deleted old backup: {old_backup}")
    
    # ==================== 사용자 관리 ====================
    
    def create_user(self, user: User) -> int:
        """사용자 생성"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO users (email, password_hash, name, role, settings)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    user.email,
                    user.password_hash,
                    user.name,
                    user.role,
                    json.dumps(user.settings)
                ))
                
                user_id = cursor.lastrowid
                
                # 활동 로그
                self._log_activity(conn, user_id, "user_created", "users", user_id)
                
                # 동기화 로그
                self._add_sync_log(conn, "users", user_id, "create")
                
                return user_id
    
    def get_user(self, user_id: int = None, email: str = None) -> Optional[User]:
        """사용자 조회"""
        with self._get_connection(readonly=True) as conn:
            if user_id:
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ).fetchone()
            elif email:
                row = conn.execute(
                    "SELECT * FROM users WHERE email = ?", (email,)
                ).fetchone()
            else:
                return None
            
            if row:
                return self._row_to_user(row)
            return None
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """사용자 정보 업데이트"""
        with self._lock:
            with self._get_connection() as conn:
                allowed_fields = ['name', 'role', 'settings', 'is_active']
                filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
                
                if 'settings' in filtered_updates:
                    filtered_updates['settings'] = json.dumps(filtered_updates['settings'])
                
                set_clause = ', '.join([f"{k} = ?" for k in filtered_updates.keys()])
                values = list(filtered_updates.values()) + [user_id]
                
                conn.execute(
                    f"UPDATE users SET {set_clause} WHERE id = ?",
                    values
                )
                
                # 활동 로그
                self._log_activity(conn, user_id, "user_updated", "users", user_id, 
                                 {"fields": list(filtered_updates.keys())})
                
                # 동기화 로그
                self._add_sync_log(conn, "users", user_id, "update")
                
                return True
    
    def update_last_login(self, user_id: int):
        """마지막 로그인 시간 업데이트"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
    
    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Row를 User 객체로 변환"""
        data = dict(row)
        data['settings'] = json.loads(data.get('settings', '{}'))
        
        # datetime 변환
        for field in ['created_at', 'updated_at', 'last_login']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return User(**data)
    
    # ==================== 프로젝트 관리 ====================
    
    def create_project(self, project: Project) -> int:
        """프로젝트 생성"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO projects (user_id, name, description, field, module_id, status, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project.user_id,
                    project.name,
                    project.description,
                    project.field,
                    project.module_id,
                    project.status,
                    json.dumps(project.data)
                ))
                
                project_id = cursor.lastrowid
                
                # 활동 로그
                self._log_activity(conn, project.user_id, "project_created", 
                                 "projects", project_id, {"name": project.name})
                
                # 동기화 로그
                self._add_sync_log(conn, "projects", project_id, "create")
                
                return project_id
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """프로젝트 조회"""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            
            if row:
                return self._row_to_project(row)
            return None
    
    def list_user_projects(self, user_id: int, 
                          status: Optional[str] = None,
                          include_shared: bool = True) -> List[Project]:
        """사용자의 프로젝트 목록"""
        with self._get_connection(readonly=True) as conn:
            # 소유한 프로젝트
            query = "SELECT * FROM projects WHERE user_id = ?"
            params = [user_id]
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            owned_projects = conn.execute(query, params).fetchall()
            projects = [self._row_to_project(row) for row in owned_projects]
            
            # 공유받은 프로젝트
            if include_shared:
                user = self.get_user(user_id)
                if user:
                    shared_query = '''
                        SELECT p.* FROM projects p
                        JOIN project_shares ps ON p.id = ps.project_id
                        WHERE ps.shared_with_email = ?
                    '''
                    shared_rows = conn.execute(shared_query, (user.email,)).fetchall()
                    projects.extend([self._row_to_project(row) for row in shared_rows])
            
            # 최근 업데이트 순 정렬
            projects.sort(key=lambda p: p.updated_at, reverse=True)
            return projects
    
    def update_project(self, project_id: int, updates: Dict[str, Any]) -> bool:
        """프로젝트 업데이트"""
        with self._lock:
            with self._get_connection() as conn:
                allowed_fields = ['name', 'description', 'field', 'module_id', 'status', 'data']
                filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
                
                if 'data' in filtered_updates:
                    filtered_updates['data'] = json.dumps(filtered_updates['data'])
                
                set_clause = ', '.join([f"{k} = ?" for k in filtered_updates.keys()])
                values = list(filtered_updates.values()) + [project_id]
                
                conn.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
                
                # 프로젝트 소유자 찾기
                project = self.get_project(project_id)
                if project:
                    self._log_activity(conn, project.user_id, "project_updated", 
                                     "projects", project_id, {"fields": list(filtered_updates.keys())})
                
                # 동기화 로그
                self._add_sync_log(conn, "projects", project_id, "update")
                
                return True
    
    def share_project(self, project_id: int, shared_with_email: str, 
                     permission: str = 'view') -> bool:
        """프로젝트 공유"""
        with self._lock:
            with self._get_connection() as conn:
                try:
                    conn.execute('''
                        INSERT INTO project_shares (project_id, shared_with_email, permission)
                        VALUES (?, ?, ?)
                    ''', (project_id, shared_with_email, permission))
                    
                    # 공유받은 사용자에게 알림
                    shared_user = self.get_user(email=shared_with_email)
                    if shared_user:
                        project = self.get_project(project_id)
                        if project:
                            self._create_notification(
                                conn, shared_user.id, 'project_shared',
                                "프로젝트 공유됨",
                                f"'{project.name}' 프로젝트가 공유되었습니다.",
                                {"project_id": project_id, "permission": permission}
                            )
                    
                    return True
                except sqlite3.IntegrityError:
                    # 이미 공유된 경우
                    conn.execute('''
                        UPDATE project_shares SET permission = ?
                        WHERE project_id = ? AND shared_with_email = ?
                    ''', (permission, project_id, shared_with_email))
                    return True
    
    def _row_to_project(self, row: sqlite3.Row) -> Project:
        """Row를 Project 객체로 변환"""
        data = dict(row)
        data['data'] = json.loads(data.get('data', '{}'))
        
        # datetime 변환
        for field in ['created_at', 'updated_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return Project(**data)
    
    # ==================== 실험 관리 ====================
    
    def create_experiment(self, experiment: Experiment) -> int:
        """실험 생성"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO experiments 
                    (project_id, name, design_type, factors, responses, design_matrix, results, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experiment.project_id,
                    experiment.name,
                    experiment.design_type,
                    json.dumps(experiment.factors),
                    json.dumps(experiment.responses),
                    json.dumps(experiment.design_matrix),
                    json.dumps(experiment.results),
                    experiment.status
                ))
                
                experiment_id = cursor.lastrowid
                
                # 프로젝트 소유자 찾기
                project = self.get_project(experiment.project_id)
                if project:
                    self._log_activity(conn, project.user_id, "experiment_created", 
                                     "experiments", experiment_id, 
                                     {"name": experiment.name, "project_id": experiment.project_id})
                
                # 동기화 로그
                self._add_sync_log(conn, "experiments", experiment_id, "create")
                
                return experiment_id
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """실험 조회"""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            
            if row:
                return self._row_to_experiment(row)
            return None
    
    def list_project_experiments(self, project_id: int, 
                               status: Optional[str] = None) -> List[Experiment]:
        """프로젝트의 실험 목록"""
        with self._get_connection(readonly=True) as conn:
            query = "SELECT * FROM experiments WHERE project_id = ?"
            params = [project_id]
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_experiment(row) for row in rows]
    
    def update_experiment(self, experiment_id: int, updates: Dict[str, Any]) -> bool:
        """실험 업데이트"""
        with self._lock:
            with self._get_connection() as conn:
                allowed_fields = ['name', 'design_type', 'factors', 'responses', 
                                'design_matrix', 'results', 'status']
                filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
                
                # JSON 필드 변환
                json_fields = ['factors', 'responses', 'design_matrix', 'results']
                for field in json_fields:
                    if field in filtered_updates:
                        filtered_updates[field] = json.dumps(filtered_updates[field])
                
                # 상태가 completed로 변경되면 completed_at 설정
                if filtered_updates.get('status') == ExperimentStatus.COMPLETED.value:
                    filtered_updates['completed_at'] = datetime.now().isoformat()
                
                set_clause = ', '.join([f"{k} = ?" for k in filtered_updates.keys()])
                values = list(filtered_updates.values()) + [experiment_id]
                
                conn.execute(f"UPDATE experiments SET {set_clause} WHERE id = ?", values)
                
                # 활동 로그
                experiment = self.get_experiment(experiment_id)
                if experiment:
                    project = self.get_project(experiment.project_id)
                    if project:
                        self._log_activity(conn, project.user_id, "experiment_updated", 
                                         "experiments", experiment_id, 
                                         {"fields": list(filtered_updates.keys())})
                
                # 동기화 로그
                self._add_sync_log(conn, "experiments", experiment_id, "update")
                
                return True
    
    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Row를 Experiment 객체로 변환"""
        data = dict(row)
        
        # JSON 필드 파싱
        for field in ['factors', 'responses', 'design_matrix', 'results']:
            data[field] = json.loads(data.get(field, '{}' if field != 'design_matrix' else '[]'))
        
        # datetime 변환
        for field in ['created_at', 'updated_at', 'completed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return Experiment(**data)
    
    # ==================== API 키 관리 ====================
    
    def save_api_key(self, user_id: int, service: str, api_key: str) -> bool:
        """API 키 저장 (암호화)"""
        with self._lock:
            with self._get_connection() as conn:
                encrypted_key = self._cipher.encrypt(api_key.encode()).decode()
                
                try:
                    conn.execute('''
                        INSERT INTO api_keys (user_id, service, encrypted_key)
                        VALUES (?, ?, ?)
                    ''', (user_id, service, encrypted_key))
                except sqlite3.IntegrityError:
                    # 이미 존재하면 업데이트
                    conn.execute('''
                        UPDATE api_keys SET encrypted_key = ?
                        WHERE user_id = ? AND service = ?
                    ''', (encrypted_key, user_id, service))
                
                # 활동 로그
                self._log_activity(conn, user_id, "api_key_saved", "api_keys", None, 
                                 {"service": service})
                
                return True
    
    def get_api_key(self, user_id: int, service: str) -> Optional[str]:
        """API 키 조회 (복호화)"""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT encrypted_key FROM api_keys WHERE user_id = ? AND service = ?",
                (user_id, service)
            ).fetchone()
            
            if row:
                encrypted_key = row['encrypted_key']
                return self._cipher.decrypt(encrypted_key.encode()).decode()
            return None
    
    def list_api_keys(self, user_id: int) -> List[Dict[str, Any]]:
        """사용자의 API 키 목록 (키 값은 제외)"""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT service, created_at, updated_at FROM api_keys WHERE user_id = ?",
                (user_id,)
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_api_key(self, user_id: int, service: str) -> bool:
        """API 키 삭제"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM api_keys WHERE user_id = ? AND service = ?",
                    (user_id, service)
                )
                
                # 활동 로그
                self._log_activity(conn, user_id, "api_key_deleted", "api_keys", None, 
                                 {"service": service})
                
                return True
    
    # ==================== 캐시 관리 ====================
    
    def cache_set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """캐시 설정"""
        with self._lock:
            with self._get_connection() as conn:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                
                conn.execute('''
                    INSERT OR REPLACE INTO cache (key, value, expires_at)
                    VALUES (?, ?, ?)
                ''', (key, json.dumps(value), expires_at.isoformat()))
                
                # 메모리 캐시도 업데이트
                self._cache[key] = value
                self._cache_ttl[key] = expires_at
                
                return True
    
    def cache_get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        # 메모리 캐시 확인
        if key in self._cache:
            if self._cache_ttl[key] > datetime.now():
                return self._cache[key]
            else:
                # 만료된 캐시 삭제
                del self._cache[key]
                del self._cache_ttl[key]
        
        # DB 캐시 확인
        with self._get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            ).fetchone()
            
            if row:
                expires_at = datetime.fromisoformat(row['expires_at'])
                if expires_at > datetime.now():
                    value = json.loads(row['value'])
                    # 메모리 캐시 업데이트
                    self._cache[key] = value
                    self._cache_ttl[key] = expires_at
                    return value
                else:
                    # 만료된 캐시 삭제
                    self.cache_delete(key)
            
            return None
    
    def cache_delete(self, key: str) -> bool:
        """캐시 삭제"""
        with self._lock:
            # 메모리 캐시 삭제
            if key in self._cache:
                del self._cache[key]
                del self._cache_ttl[key]
            
            # DB 캐시 삭제
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return True
    
    def cache_clear(self) -> bool:
        """전체 캐시 삭제"""
        with self._lock:
            # 메모리 캐시 초기화
            self._cache.clear()
            self._cache_ttl.clear()
            
            # DB 캐시 삭제
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache")
                return True
    
    def cache_cleanup(self) -> int:
        """만료된 캐시 정리"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),)
                )
                return cursor.rowcount
    
    # ==================== 활동 로그 ====================
    
    def _log_activity(self, conn: sqlite3.Connection, user_id: int, action: str, 
                     resource_type: Optional[str] = None, resource_id: Optional[int] = None,
                     details: Optional[Dict[str, Any]] = None):
        """활동 로그 기록"""
        conn.execute('''
            INSERT INTO activity_log (user_id, action, resource_type, resource_id, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            action,
            resource_type,
            resource_id,
            json.dumps(details) if details else None
        ))
    
    def get_user_activities(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """사용자 활동 로그 조회"""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute('''
                SELECT * FROM activity_log 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit)).fetchall()
            
            activities = []
            for row in rows:
                activity = dict(row)
                if activity.get('details'):
                    activity['details'] = json.loads(activity['details'])
                activities.append(activity)
            
            return activities
    
    # ==================== 동기화 로그 ====================
    
    def _add_sync_log(self, conn: sqlite3.Connection, table_name: str, 
                     record_id: int, action: str):
        """동기화 로그 추가"""
        conn.execute('''
            INSERT INTO sync_log (table_name, record_id, action, sync_status)
            VALUES (?, ?, ?, ?)
        ''', (table_name, record_id, action, SyncStatus.PENDING.value))
    
    def get_pending_syncs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """대기 중인 동기화 목록"""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute('''
                SELECT * FROM sync_log 
                WHERE sync_status = ? 
                ORDER BY created_at 
                LIMIT ?
            ''', (SyncStatus.PENDING.value, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def update_sync_status(self, sync_id: int, status: SyncStatus, 
                          error_message: Optional[str] = None) -> bool:
        """동기화 상태 업데이트"""
        with self._lock:
            with self._get_connection() as conn:
                if status == SyncStatus.SYNCED:
                    conn.execute('''
                        UPDATE sync_log 
                        SET sync_status = ?, synced_at = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    ''', (status.value, sync_id))
                else:
                    conn.execute('''
                        UPDATE sync_log 
                        SET sync_status = ?, error_message = ? 
                        WHERE id = ?
                    ''', (status.value, error_message, sync_id))
                
                return True
    
    # ==================== 알림 관리 ====================
    
    def _create_notification(self, conn: sqlite3.Connection, user_id: int, 
                           notification_type: str, title: str, message: str,
                           data: Optional[Dict[str, Any]] = None):
        """알림 생성"""
        conn.execute('''
            INSERT INTO notifications (user_id, type, title, message, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            notification_type,
            title,
            message,
            json.dumps(data) if data else '{}'
        ))
    
    def get_user_notifications(self, user_id: int, unread_only: bool = False, 
                             limit: int = 50) -> List[Dict[str, Any]]:
        """사용자 알림 조회"""
        with self._get_connection(readonly=True) as conn:
            query = "SELECT * FROM notifications WHERE user_id = ?"
            params = [user_id]
            
            if unread_only:
                query += " AND is_read = 0"
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            notifications = []
            for row in rows:
                notification = dict(row)
                notification['data'] = json.loads(notification.get('data', '{}'))
                notifications.append(notification)
            
            return notifications
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """알림 읽음 표시"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE notifications 
                    SET is_read = 1, read_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (notification_id,))
                return True
    
    # ==================== 모듈 사용 기록 ====================
    
    def record_module_usage(self, user_id: int, module_id: str, 
                          project_id: Optional[int] = None, 
                          feedback: Optional[str] = None):
        """모듈 사용 기록"""
        with self._lock:
            with self._get_connection() as conn:
                # 기존 기록 확인
                row = conn.execute('''
                    SELECT id, usage_count FROM module_usage 
                    WHERE user_id = ? AND module_id = ?
                ''', (user_id, module_id)).fetchone()
                
                if row:
                    # 사용 횟수 증가
                    conn.execute('''
                        UPDATE module_usage 
                        SET usage_count = usage_count + 1, 
                            last_used = CURRENT_TIMESTAMP,
                            project_id = COALESCE(?, project_id),
                            feedback = COALESCE(?, feedback)
                        WHERE id = ?
                    ''', (project_id, feedback, row['id']))
                else:
                    # 새 기록 생성
                    conn.execute('''
                        INSERT INTO module_usage (user_id, module_id, project_id, feedback)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, module_id, project_id, feedback))
    
    def get_popular_modules(self, limit: int = 10) -> List[Dict[str, Any]]:
        """인기 모듈 목록"""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute('''
                SELECT module_id, SUM(usage_count) as total_usage, COUNT(DISTINCT user_id) as user_count
                FROM module_usage
                GROUP BY module_id
                ORDER BY total_usage DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            
            return [dict(row) for row in rows]
    
    # ==================== 유틸리티 메서드 ====================
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """직접 쿼리 실행"""
        with self._get_connection() as conn:
            return conn.execute(query, params)
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """단일 결과 조회"""
        with self._get_connection(readonly=True) as conn:
            return conn.execute(query, params).fetchone()
    
    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """전체 결과 조회"""
        with self._get_connection(readonly=True) as conn:
            return conn.execute(query, params).fetchall()
    
    def count(self, table: str, condition: str = "", params: tuple = ()) -> int:
        """레코드 수 조회"""
        query = f"SELECT COUNT(*) FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        
        with self._get_connection(readonly=True) as conn:
            return conn.execute(query, params).fetchone()[0]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계"""
        stats = {
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'tables': {}
        }
        
        # 각 테이블 레코드 수
        tables = ['users', 'projects', 'experiments', 'api_keys', 'sync_log', 'cache',
                 'activity_log', 'module_usage', 'notifications']
        for table in tables:
            stats['tables'][table] = self.count(table)
        
        # 백업 정보
        if self.backup_path.exists():
            backups = list(self.backup_path.glob("backup_*.db"))
            stats['backups'] = {
                'count': len(backups),
                'latest': max(backups, key=lambda p: p.stat().st_mtime).name if backups else None,
                'total_size': sum(b.stat().st_size for b in backups)
            }
        
        return stats
    
    def vacuum(self):
        """데이터베이스 최적화 (VACUUM)"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
        logger.info("Database optimized")
    
    def close(self):
        """데이터베이스 연결 종료"""
        # 백업 스레드 중지
        if self._backup_thread:
            self._backup_stop_event.set()
            self._backup_thread.join(timeout=5)
            self._backup_thread = None
        
        # 모든 연결 종료
        with self._lock:
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()
        
        logger.info("Database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 싱글톤 인스턴스 관리 ====================

_db_manager: Optional[DatabaseManager] = None
_db_lock = threading.Lock()

def get_database_manager(db_path: Optional[Path] = None) -> DatabaseManager:
    """DatabaseManager 싱글톤 인스턴스 반환"""
    global _db_manager
    
    with _db_lock:
        if _db_manager is None:
            if db_path is None:
                from config.local_config import LOCAL_CONFIG
                db_path = Path(LOCAL_CONFIG['database']['path'])
            
            config = {
                'wal_mode': True,
                'connection_pool_size': 5,
                'backup_enabled': True,
                'backup_interval': 3600,
                'max_backups': 5
            }
            
            _db_manager = DatabaseManager(db_path, config)
            logger.info(f"DatabaseManager initialized with {db_path}")
        
        return _db_manager
