"""로컬 SQLite 데이터베이스 관리"""
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

logger = logging.getLogger(__name__)

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
        
        # 캐시
        self._cache = {}
        self._cache_ttl = {}
        
        # 초기화
        self._init_database()
        
        # 백업 스레드 시작
        if self.backup_enabled:
            self._start_backup_thread()
        
        # 종료 시 정리
        atexit.register(self.close)
    
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
                is_deleted INTEGER DEFAULT 0,
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
                factors TEXT,
                responses TEXT,
                design_matrix TEXT,
                results TEXT,
                analysis TEXT,
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
                data TEXT,
                local_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sync_timestamp TIMESTAMP,
                sync_status TEXT DEFAULT 'pending',
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
        ''')
        
        # 캐시 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        ''')
        
        # 활동 로그 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        ''')
        
        # 모듈 사용 기록
        conn.execute('''
            CREATE TABLE IF NOT EXISTS module_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                module_id TEXT NOT NULL,
                project_id INTEGER,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rating INTEGER,
                feedback TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL
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
            "CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_log(created_at)"
        ]
        
        for index in indexes:
            conn.execute(index)
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
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
                self._connections[thread_id] = conn
            
            conn = self._connections[thread_id]
        
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
    
    def execute(self, query: str, params: Optional[tuple] = None) -> sqlite3.Cursor:
        """쿼리 실행"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
    
    def executemany(self, query: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """다중 쿼리 실행"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor
    
    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """단일 행 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def fetchall(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """전체 행 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    @contextmanager
    def transaction(self):
        """트랜잭션 컨텍스트 매니저"""
        with self._get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    # === CRUD 메서드 ===
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """데이터 삽입"""
        # 타임스탬프 자동 추가
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in data:
            data['updated_at'] = datetime.now().isoformat()
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        cursor = self.execute(query, tuple(data.values()))
        
        # 동기화 로그 추가
        self._log_sync_action(table, cursor.lastrowid, 'insert', data)
        
        return cursor.lastrowid
    
    def update(self, table: str, id: int, data: Dict[str, Any]) -> bool:
        """데이터 업데이트"""
        # 타임스탬프 업데이트
        data['updated_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE id = ?"
        
        cursor = self.execute(query, tuple(list(data.values()) + [id]))
        
        # 동기화 로그 추가
        if cursor.rowcount > 0:
            self._log_sync_action(table, id, 'update', data)
        
        return cursor.rowcount > 0
    
    def delete(self, table: str, id: int, soft: bool = True) -> bool:
        """데이터 삭제"""
        if soft and table in ['projects', 'experiments']:
            # 소프트 삭제
            return self.update(table, id, {'is_deleted': 1})
        else:
            # 하드 삭제
            query = f"DELETE FROM {table} WHERE id = ?"
            cursor = self.execute(query, (id,))
            
            # 동기화 로그 추가
            if cursor.rowcount > 0:
                self._log_sync_action(table, id, 'delete', {})
            
            return cursor.rowcount > 0
    
    def get_by_id(self, table: str, id: int) -> Optional[Dict[str, Any]]:
        """ID로 조회"""
        query = f"SELECT * FROM {table} WHERE id = ?"
        return self.fetchone(query, (id,))
    
    def get_all(self, table: str, 
                filters: Optional[Dict[str, Any]] = None,
                order_by: Optional[str] = None,
                limit: Optional[int] = None,
                offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """전체 조회"""
        query = f"SELECT * FROM {table}"
        params = []
        
        # 필터 적용
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is None:
                    conditions.append(f"{key} IS NULL")
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            query += " WHERE " + " AND ".join(conditions)
        
        # 정렬
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # 페이지네이션
        if limit:
            query += f" LIMIT {limit}"
            if offset:
                query += f" OFFSET {offset}"
        
        return self.fetchall(query, tuple(params) if params else None)
    
    # === 특수 메서드 ===
    
    def search(self, table: str, search_term: str, 
               search_fields: List[str]) -> List[Dict[str, Any]]:
        """텍스트 검색"""
        conditions = []
        params = []
        
        for field in search_fields:
            conditions.append(f"{field} LIKE ?")
            params.append(f"%{search_term}%")
        
        query = f"SELECT * FROM {table} WHERE " + " OR ".join(conditions)
        return self.fetchall(query, tuple(params))
    
    def count(self, table: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """레코드 수 조회"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is None:
                    conditions.append(f"{key} IS NULL")
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            query += " WHERE " + " AND ".join(conditions)
        
        result = self.fetchone(query, tuple(params) if params else None)
        return result['count'] if result else 0
    
    # === 캐시 메서드 ===
    
    def cache_get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        # 메모리 캐시 확인
        if key in self._cache:
            if key in self._cache_ttl and self._cache_ttl[key] > datetime.now():
                return self._cache[key]
            else:
                # 만료된 캐시 삭제
                del self._cache[key]
                if key in self._cache_ttl:
                    del self._cache_ttl[key]
        
        # DB 캐시 확인
        query = """
            SELECT value FROM cache 
            WHERE key = ? AND (expires_at IS NULL OR expires_at > datetime('now'))
        """
        result = self.fetchone(query, (key,))
        
        if result:
            # 히트 카운트 증가
            self.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
            
            try:
                value = json.loads(result['value'])
                # 메모리 캐시에 저장
                self._cache[key] = value
                return value
            except json.JSONDecodeError:
                return result['value']
        
        return None
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """캐시에 값 저장"""
        # 메모리 캐시 저장
        self._cache[key] = value
        if ttl:
            self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl)
        
        # DB 캐시 저장
        expires_at = None
        if ttl:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        
        value_str = json.dumps(value) if not isinstance(value, str) else value
        
        query = """
            INSERT OR REPLACE INTO cache (key, value, expires_at)
            VALUES (?, ?, ?)
        """
        self.execute(query, (key, value_str, expires_at))
    
    def cache_delete(self, key: str):
        """캐시에서 값 삭제"""
        # 메모리 캐시 삭제
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_ttl:
            del self._cache_ttl[key]
        
        # DB 캐시 삭제
        self.execute("DELETE FROM cache WHERE key = ?", (key,))
    
    def cache_clear(self):
        """전체 캐시 클리어"""
        self._cache.clear()
        self._cache_ttl.clear()
        self.execute("DELETE FROM cache")
    
    def cache_cleanup(self):
        """만료된 캐시 정리"""
        # 메모리 캐시 정리
        expired_keys = []
        now = datetime.now()
        for key, expires_at in self._cache_ttl.items():
            if expires_at <= now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        
        # DB 캐시 정리
        self.execute("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')")
    
    # === 동기화 관련 ===
    
    def _log_sync_action(self, table: str, record_id: int, 
                        action: str, data: Dict[str, Any]):
        """동기화 액션 로깅"""
        if table not in ['sync_log', 'cache', 'activity_log']:  # 로그 테이블은 제외
            query = """
                INSERT INTO sync_log (table_name, record_id, action, data)
                VALUES (?, ?, ?, ?)
            """
            self.execute(query, (table, record_id, action, json.dumps(data)))
    
    def get_pending_sync(self, limit: int = 100) -> List[Dict[str, Any]]:
        """대기 중인 동기화 항목 조회"""
        query = """
            SELECT * FROM sync_log 
            WHERE sync_status = 'pending' AND retry_count < 3
            ORDER BY local_timestamp ASC
            LIMIT ?
        """
        return self.fetchall(query, (limit,))
    
    def mark_synced(self, sync_id: int, success: bool = True, 
                   error_message: Optional[str] = None):
        """동기화 완료 표시"""
        if success:
            query = """
                UPDATE sync_log 
                SET sync_status = 'completed', sync_timestamp = ?
                WHERE id = ?
            """
            self.execute(query, (datetime.now().isoformat(), sync_id))
        else:
            query = """
                UPDATE sync_log 
                SET sync_status = 'failed', error_message = ?, retry_count = retry_count + 1
                WHERE id = ?
            """
            self.execute(query, (error_message, sync_id))
    
    # === 백업 관련 ===
    
    def _start_backup_thread(self):
        """백업 스레드 시작"""
        def backup_loop():
            while self._backup_thread:
                try:
                    time.sleep(self.backup_interval)
                    self.create_backup()
                except Exception as e:
                    logger.error(f"Backup failed: {str(e)}")
        
        self._backup_thread = threading.Thread(target=backup_loop, daemon=True)
        self._backup_thread.start()
    
    def create_backup(self) -> Optional[Path]:
        """데이터베이스 백업 생성"""
        if not self.backup_enabled:
            return None
        
        try:
            # 백업 디렉토리 생성
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # 백업 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"backup_{timestamp}.db"
            
            # 백업 실행
            with self._get_connection() as conn:
                backup = sqlite3.connect(str(backup_file))
                conn.backup(backup)
                backup.close()
            
            logger.info(f"Backup created: {backup_file}")
            
            # 오래된 백업 삭제
            self._cleanup_old_backups()
            
            return backup_file
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    def _cleanup_old_backups(self):
        """오래된 백업 파일 삭제"""
        try:
            backups = sorted(self.backup_path.glob("backup_*.db"), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
            
            # 최대 개수 초과 시 삭제
            for backup in backups[self.max_backups:]:
                backup.unlink()
                logger.info(f"Deleted old backup: {backup}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
    
    def restore_backup(self, backup_file: Path) -> bool:
        """백업에서 복원"""
        try:
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # 현재 DB 백업
            temp_backup = self.db_path.with_suffix('.db.temp')
            shutil.copy2(self.db_path, temp_backup)
            
            try:
                # 백업 파일로 교체
                shutil.copy2(backup_file, self.db_path)
                
                # 연결 재초기화
                self._connections.clear()
                self._init_database()
                
                logger.info(f"Database restored from: {backup_file}")
                temp_backup.unlink()
                return True
                
            except Exception as e:
                # 복원 실패 시 원복
                shutil.copy2(temp_backup, self.db_path)
                temp_backup.unlink()
                raise e
                
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
    
    # === 유틸리티 메서드 ===
    
    def get_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계"""
        stats = {
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'tables': {}
        }
        
        tables = ['users', 'projects', 'experiments', 'sync_log', 'cache']
        for table in tables:
            stats['tables'][table] = self.count(table)
        
        # 백업 정보
        if self.backup_path.exists():
            backups = list(self.backup_path.glob("backup_*.db"))
            stats['backups'] = {
                'count': len(backups),
                'latest': max(backups, key=lambda p: p.stat().st_mtime).name if backups else None
            }
        
        return stats
    
    def vacuum(self):
        """데이터베이스 최적화"""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        logger.info("Database optimized")
    
    def close(self):
        """데이터베이스 연결 종료"""
        # 백업 스레드 중지
        if self._backup_thread:
            backup_thread = self._backup_thread
            self._backup_thread = None
            backup_thread.join(timeout=5)
        
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


# 싱글톤 인스턴스 관리
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """DatabaseManager 싱글톤 인스턴스 반환"""
    global _db_manager
    
    if _db_manager is None:
        from config.local_config import LOCAL_CONFIG
        
        db_config = LOCAL_CONFIG.get('database', {})
        db_path = Path(db_config.get('path', './data/db/app.db'))
        
        _db_manager = DatabaseManager(db_path, db_config)
    
    return _db_manager
