"""데이터베이스 마이그레이션 관리"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import traceback

logger = logging.getLogger(__name__)

@dataclass
class Migration:
    """마이그레이션 정보"""
    version: int
    description: str
    up: Callable
    down: Callable
    checksum: Optional[str] = None
    dependencies: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        # 체크섬 생성 (마이그레이션 무결성 확인용)
        if not self.checksum:
            content = f"{self.version}:{self.description}"
            self.checksum = hashlib.md5(content.encode()).hexdigest()

class MigrationError(Exception):
    """마이그레이션 에러"""
    pass

class MigrationManager:
    """DB 스키마 마이그레이션 관리"""
    
    def __init__(self, db_manager):
        """
        마이그레이션 매니저 초기화
        
        Args:
            db_manager: DatabaseManager 인스턴스
        """
        self.db_manager = db_manager
        self.migrations: List[Migration] = []
        
        # 마이그레이션 정의
        self._define_migrations()
        
        # 마이그레이션 정렬 (버전 순)
        self.migrations.sort(key=lambda m: m.version)
    
    def _define_migrations(self):
        """마이그레이션 정의"""
        
        # 버전 1: 초기 스키마 (이미 database_manager에서 생성됨)
        self.add_migration(
            version=1,
            description="Initial schema",
            up=self._migration_001_initial,
            down=self._migration_001_rollback
        )
        
        # 버전 2: 모듈 메타데이터 추가
        self.add_migration(
            version=2,
            description="Add module metadata tables",
            up=self._migration_002_module_metadata,
            down=self._migration_002_rollback
        )
        
        # 버전 3: 협업 기능 테이블
        self.add_migration(
            version=3,
            description="Add collaboration tables",
            up=self._migration_003_collaboration,
            down=self._migration_003_rollback
        )
        
        # 버전 4: 분석 결과 캐싱
        self.add_migration(
            version=4,
            description="Add analysis cache tables",
            up=self._migration_004_analysis_cache,
            down=self._migration_004_rollback
        )
        
        # 버전 5: 알림 시스템
        self.add_migration(
            version=5,
            description="Add notification system",
            up=self._migration_005_notifications,
            down=self._migration_005_rollback
        )
    
    def add_migration(self, version: int, description: str, 
                     up: Callable, down: Callable,
                     dependencies: Optional[List[int]] = None):
        """마이그레이션 추가"""
        migration = Migration(
            version=version,
            description=description,
            up=up,
            down=down,
            dependencies=dependencies
        )
        self.migrations.append(migration)
    
    def get_current_version(self) -> int:
        """현재 DB 버전 확인"""
        try:
            result = self.db_manager.fetchone(
                "SELECT MAX(version) as version FROM schema_version WHERE status = 'applied'"
            )
            return result['version'] if result and result['version'] else 0
        except sqlite3.OperationalError:
            # schema_version 테이블이 없는 경우
            logger.info("No schema_version table found, assuming version 0")
            return 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """적용 대기 중인 마이그레이션 목록"""
        current_version = self.get_current_version()
        return [m for m in self.migrations if m.version > current_version]
    
    def migrate_to_latest(self, dry_run: bool = False) -> Tuple[bool, List[str]]:
        """최신 버전으로 마이그레이션"""
        current_version = self.get_current_version()
        target_version = self.migrations[-1].version if self.migrations else 0
        
        if current_version >= target_version:
            logger.info(f"Database is up to date (version {current_version})")
            return True, [f"Database is already at version {current_version}"]
        
        return self.migrate_to_version(target_version, dry_run)
    
    def migrate_to_version(self, target_version: int, 
                          dry_run: bool = False) -> Tuple[bool, List[str]]:
        """특정 버전으로 마이그레이션"""
        current_version = self.get_current_version()
        logs = []
        
        if current_version == target_version:
            msg = f"Already at version {target_version}"
            logs.append(msg)
            logger.info(msg)
            return True, logs
        
        # 백업 생성 (dry_run이 아닌 경우)
        backup_path = None
        if not dry_run:
            backup_path = self._create_backup()
            if backup_path:
                logs.append(f"Backup created: {backup_path}")
            else:
                return False, ["Failed to create backup"]
        
        try:
            if current_version < target_version:
                # 업그레이드
                success, migration_logs = self._upgrade(
                    current_version, target_version, dry_run
                )
            else:
                # 다운그레이드
                success, migration_logs = self._downgrade(
                    current_version, target_version, dry_run
                )
            
            logs.extend(migration_logs)
            
            if success and not dry_run:
                logs.append(f"Successfully migrated to version {target_version}")
                # 오래된 백업 정리
                self._cleanup_old_backups()
            
            return success, logs
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logs.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # 백업에서 복원 (dry_run이 아닌 경우)
            if not dry_run and backup_path:
                if self._restore_from_backup(backup_path):
                    logs.append(f"Restored from backup: {backup_path}")
                else:
                    logs.append("Failed to restore from backup!")
            
            return False, logs
    
    def _upgrade(self, from_version: int, to_version: int, 
                dry_run: bool) -> Tuple[bool, List[str]]:
        """업그레이드 실행"""
        logs = []
        
        # 적용할 마이그레이션 선택
        migrations_to_apply = [
            m for m in self.migrations 
            if from_version < m.version <= to_version
        ]
        
        # 의존성 순서대로 정렬
        migrations_to_apply = self._sort_by_dependencies(migrations_to_apply)
        
        for migration in migrations_to_apply:
            logs.append(f"Applying migration {migration.version}: {migration.description}")
            
            if dry_run:
                logs.append(f"[DRY RUN] Would apply migration {migration.version}")
            else:
                try:
                    # 마이그레이션 실행
                    self._run_migration(migration, 'up')
                    
                    # 버전 기록
                    self._record_migration(migration)
                    
                    logs.append(f"✓ Migration {migration.version} applied successfully")
                    
                except Exception as e:
                    logs.append(f"✗ Migration {migration.version} failed: {str(e)}")
                    raise
        
        return True, logs
    
    def _downgrade(self, from_version: int, to_version: int, 
                  dry_run: bool) -> Tuple[bool, List[str]]:
        """다운그레이드 실행"""
        logs = []
        
        # 롤백할 마이그레이션 선택 (역순)
        migrations_to_rollback = [
            m for m in reversed(self.migrations)
            if to_version < m.version <= from_version
        ]
        
        for migration in migrations_to_rollback:
            logs.append(f"Rolling back migration {migration.version}: {migration.description}")
            
            if dry_run:
                logs.append(f"[DRY RUN] Would rollback migration {migration.version}")
            else:
                try:
                    # 롤백 실행
                    self._run_migration(migration, 'down')
                    
                    # 버전 기록 제거
                    self._remove_migration_record(migration)
                    
                    logs.append(f"✓ Migration {migration.version} rolled back successfully")
                    
                except Exception as e:
                    logs.append(f"✗ Rollback of migration {migration.version} failed: {str(e)}")
                    raise
        
        return True, logs
    
    def _run_migration(self, migration: Migration, direction: str):
        """마이그레이션 실행"""
        with self.db_manager.transaction() as conn:
            if direction == 'up':
                migration.up(conn)
            else:
                migration.down(conn)
    
    def _record_migration(self, migration: Migration):
        """마이그레이션 기록"""
        self.db_manager.execute("""
            INSERT INTO schema_version (version, description, checksum, applied_at, status)
            VALUES (?, ?, ?, ?, 'applied')
        """, (
            migration.version,
            migration.description,
            migration.checksum,
            datetime.now().isoformat()
        ))
    
    def _remove_migration_record(self, migration: Migration):
        """마이그레이션 기록 제거"""
        self.db_manager.execute("""
            UPDATE schema_version 
            SET status = 'rolled_back', rolled_back_at = ?
            WHERE version = ? AND status = 'applied'
        """, (datetime.now().isoformat(), migration.version))
    
    def _sort_by_dependencies(self, migrations: List[Migration]) -> List[Migration]:
        """의존성에 따라 마이그레이션 정렬"""
        # 간단한 토폴로지 정렬
        sorted_migrations = []
        remaining = migrations.copy()
        
        while remaining:
            # 의존성이 모두 해결된 마이그레이션 찾기
            ready = []
            for m in remaining:
                deps_satisfied = all(
                    dep in [sm.version for sm in sorted_migrations]
                    or dep <= self.get_current_version()
                    for dep in m.dependencies
                )
                if deps_satisfied:
                    ready.append(m)
            
            if not ready:
                # 순환 의존성 또는 해결 불가능한 의존성
                raise MigrationError("Circular or unresolvable dependencies detected")
            
            # 버전 순으로 정렬하여 추가
            ready.sort(key=lambda m: m.version)
            sorted_migrations.extend(ready)
            
            # 처리된 마이그레이션 제거
            for m in ready:
                remaining.remove(m)
        
        return sorted_migrations
    
    def _create_backup(self) -> Optional[Path]:
        """마이그레이션 전 백업 생성"""
        try:
            backup_dir = self.db_manager.backup_path / 'migrations'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = self.get_current_version()
            backup_file = backup_dir / f"pre_migration_v{version}_{timestamp}.db"
            
            # 데이터베이스 백업
            shutil.copy2(self.db_manager.db_path, backup_file)
            logger.info(f"Migration backup created: {backup_file}")
            
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create migration backup: {str(e)}")
            return None
    
    def _restore_from_backup(self, backup_path: Path) -> bool:
        """백업에서 복원"""
        try:
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # 현재 파일을 임시로 이동
            temp_path = self.db_manager.db_path.with_suffix('.failed')
            shutil.move(self.db_manager.db_path, temp_path)
            
            # 백업 복원
            shutil.copy2(backup_path, self.db_manager.db_path)
            
            # 실패한 파일 삭제
            temp_path.unlink()
            
            logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {str(e)}")
            return False
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """오래된 마이그레이션 백업 정리"""
        try:
            backup_dir = self.db_manager.backup_path / 'migrations'
            if not backup_dir.exists():
                return
            
            backups = sorted(
                backup_dir.glob("pre_migration_*.db"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # 지정된 개수만 유지
            for backup in backups[keep_count:]:
                backup.unlink()
                logger.info(f"Deleted old migration backup: {backup}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup migration backups: {str(e)}")
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """마이그레이션 히스토리 조회"""
        return self.db_manager.fetchall("""
            SELECT version, description, applied_at, status, checksum
            FROM schema_version
            ORDER BY version DESC
        """)
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """데이터베이스 무결성 검증"""
        issues = []
        
        try:
            # 1. schema_version 테이블 체크
            current_version = self.get_current_version()
            
            # 2. 적용된 마이그레이션 체크섬 확인
            history = self.get_migration_history()
            for record in history:
                if record['status'] == 'applied':
                    version = record['version']
                    expected_migration = next(
                        (m for m in self.migrations if m.version == version), 
                        None
                    )
                    
                    if not expected_migration:
                        issues.append(f"Unknown migration version {version} in history")
                    elif expected_migration.checksum != record['checksum']:
                        issues.append(f"Checksum mismatch for migration {version}")
            
            # 3. 테이블 구조 확인
            tables = self._get_database_tables()
            expected_tables = self._get_expected_tables(current_version)
            
            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                issues.append(f"Missing tables: {', '.join(missing_tables)}")
            
            # 4. 외래 키 무결성
            fk_issues = self._check_foreign_keys()
            issues.extend(fk_issues)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Integrity check failed: {str(e)}")
            return False, issues
    
    def _get_database_tables(self) -> List[str]:
        """데이터베이스 테이블 목록 조회"""
        result = self.db_manager.fetchall("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        return [row['name'] for row in result]
    
    def _get_expected_tables(self, version: int) -> List[str]:
        """버전별 예상 테이블 목록"""
        tables = []
        
        if version >= 1:
            tables.extend([
                'users', 'projects', 'experiments', 'api_keys',
                'sync_log', 'cache', 'activity_log', 'module_usage',
                'schema_version'
            ])
        
        if version >= 2:
            tables.extend(['module_metadata', 'module_dependencies'])
        
        if version >= 3:
            tables.extend(['teams', 'team_members', 'comments', 'notifications'])
        
        if version >= 4:
            tables.extend(['analysis_cache', 'analysis_queue'])
        
        if version >= 5:
            tables.extend(['notification_preferences', 'notification_queue'])
        
        return tables
    
    def _check_foreign_keys(self) -> List[str]:
        """외래 키 무결성 검사"""
        issues = []
        
        # PRAGMA foreign_key_check 실행
        result = self.db_manager.fetchall("PRAGMA foreign_key_check")
        
        for row in result:
            issues.append(
                f"Foreign key violation in table {row['table']}: "
                f"row {row['rowid']} references {row['parent']}"
            )
        
        return issues
    
    # === 마이그레이션 정의 ===
    
    def _migration_001_initial(self, conn: sqlite3.Connection):
        """초기 스키마 (이미 생성됨)"""
        # DatabaseManager에서 이미 생성하므로 스킵
        pass
    
    def _migration_001_rollback(self, conn: sqlite3.Connection):
        """초기 스키마 롤백"""
        # 초기 스키마는 롤백하지 않음
        raise MigrationError("Cannot rollback initial schema")
    
    def _migration_002_module_metadata(self, conn: sqlite3.Connection):
        """모듈 메타데이터 테이블 추가"""
        # 모듈 메타데이터
        conn.execute('''
            CREATE TABLE IF NOT EXISTS module_metadata (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                author TEXT,
                description TEXT,
                category TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                downloads INTEGER DEFAULT 0,
                rating REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # 모듈 의존성
        conn.execute('''
            CREATE TABLE IF NOT EXISTS module_dependencies (
                module_id TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                version_spec TEXT,
                PRIMARY KEY (module_id, depends_on),
                FOREIGN KEY (module_id) REFERENCES module_metadata (id)
            )
        ''')
        
        # 인덱스 추가
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_module_category ON module_metadata(category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_module_active ON module_metadata(is_active)"
        )
    
    def _migration_002_rollback(self, conn: sqlite3.Connection):
        """모듈 메타데이터 롤백"""
        conn.execute("DROP TABLE IF EXISTS module_dependencies")
        conn.execute("DROP TABLE IF EXISTS module_metadata")
    
    def _migration_003_collaboration(self, conn: sqlite3.Connection):
        """협업 기능 테이블 추가"""
        # 팀 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                owner_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # 팀 멤버 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_members (
                team_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT DEFAULT 'member',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES teams (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 댓글 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                parent_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (parent_id) REFERENCES comments (id)
            )
        ''')
        
        # 인덱스 추가
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_teams_owner ON teams(owner_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comments_entity ON comments(entity_type, entity_id)"
        )
    
    def _migration_003_rollback(self, conn: sqlite3.Connection):
        """협업 기능 롤백"""
        conn.execute("DROP TABLE IF EXISTS comments")
        conn.execute("DROP TABLE IF EXISTS team_members")
        conn.execute("DROP TABLE IF EXISTS teams")
    
    def _migration_004_analysis_cache(self, conn: sqlite3.Connection):
        """분석 캐시 테이블 추가"""
        # 분석 캐시
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                parameters TEXT,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
                UNIQUE(experiment_id, analysis_type, parameters)
            )
        ''')
        
        # 분석 큐
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                parameters TEXT,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
        ''')
        
        # 인덱스 추가
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_cache_exp ON analysis_cache(experiment_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_queue_status ON analysis_queue(status)"
        )
    
    def _migration_004_rollback(self, conn: sqlite3.Connection):
        """분석 캐시 롤백"""
        conn.execute("DROP TABLE IF EXISTS analysis_queue")
        conn.execute("DROP TABLE IF EXISTS analysis_cache")
    
    def _migration_005_notifications(self, conn: sqlite3.Connection):
        """알림 시스템 테이블 추가"""
        # 알림 설정
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notification_preferences (
                user_id INTEGER PRIMARY KEY,
                email_enabled INTEGER DEFAULT 1,
                push_enabled INTEGER DEFAULT 1,
                in_app_enabled INTEGER DEFAULT 1,
                digest_frequency TEXT DEFAULT 'daily',
                quiet_hours_start TEXT,
                quiet_hours_end TEXT,
                categories TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 알림 큐
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notification_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sent_at TIMESTAMP,
                read_at TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # notifications 테이블이 없으면 생성
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                is_read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                read_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 인덱스 추가
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_notif_queue_user ON notification_queue(user_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_notif_user_read ON notifications(user_id, is_read)"
        )
    
    def _migration_005_rollback(self, conn: sqlite3.Connection):
        """알림 시스템 롤백"""
        conn.execute("DROP TABLE IF EXISTS notifications")
        conn.execute("DROP TABLE IF EXISTS notification_queue")
        conn.execute("DROP TABLE IF EXISTS notification_preferences")


# 유틸리티 함수
def check_and_migrate(db_manager) -> bool:
    """데이터베이스 확인 및 마이그레이션 실행"""
    try:
        migration_manager = MigrationManager(db_manager)
        
        # 무결성 검사
        is_valid, issues = migration_manager.verify_integrity()
        if not is_valid:
            logger.warning(f"Database integrity issues found: {issues}")
        
        # 마이그레이션 실행
        success, logs = migration_manager.migrate_to_latest()
        
        for log in logs:
            logger.info(log)
        
        return success
        
    except Exception as e:
        logger.error(f"Migration check failed: {str(e)}")
        return False
