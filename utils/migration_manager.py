"""
데이터베이스 마이그레이션 관리자

데이터베이스 스키마 버전 관리 및 마이그레이션을 담당합니다.
안전한 업그레이드/다운그레이드와 데이터 무결성을 보장합니다.
"""

import sqlite3
import json
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """마이그레이션 정의"""
    version: int
    name: str
    description: str
    up_func: Optional[Callable] = None
    down_func: Optional[Callable] = None
    up_sql: Optional[str] = None
    down_sql: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """체크섬 생성"""
        if not self.checksum:
            self.checksum = self._generate_checksum()
    
    def _generate_checksum(self) -> str:
        """마이그레이션 체크섬 생성"""
        content = f"{self.version}:{self.name}:{self.description}"
        
        # SQL 내용 포함
        if self.up_sql:
            content += f":{self.up_sql}"
        if self.down_sql:
            content += f":{self.down_sql}"
            
        # 함수 코드 포함
        if self.up_func:
            content += f":{self.up_func.__code__.co_code.hex()}"
        if self.down_func:
            content += f":{self.down_func.__code__.co_code.hex()}"
            
        # 의존성 포함
        content += f":{','.join(map(str, sorted(self.dependencies)))}"
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def up(self, conn: sqlite3.Connection):
        """업그레이드 실행"""
        if self.up_sql:
            for statement in self.up_sql.split(';'):
                if statement.strip():
                    conn.execute(statement)
        elif self.up_func:
            self.up_func(conn)
        else:
            raise ValueError(f"Migration {self.version} has no up operation")
    
    def down(self, conn: sqlite3.Connection):
        """다운그레이드 실행"""
        if self.down_sql:
            for statement in self.down_sql.split(';'):
                if statement.strip():
                    conn.execute(statement)
        elif self.down_func:
            self.down_func(conn)
        else:
            raise ValueError(f"Migration {self.version} has no down operation")


class MigrationManager:
    """데이터베이스 마이그레이션 관리자"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.migrations: List[Migration] = []
        self.migrations_path = Path("migrations")
        
        # 마이그레이션 테이블 생성
        self._create_migration_table()
        
        # 마이그레이션 로드
        self._load_builtin_migrations()
        self._load_migrations_from_files()
    
    def _create_migration_table(self):
        """마이그레이션 테이블 생성"""
        self.db_manager.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'applied',
                rolled_back_at TIMESTAMP,
                execution_time_ms INTEGER
            )
        """)
        
        # 인덱스 생성
        self.db_manager.execute("""
            CREATE INDEX IF NOT EXISTS idx_schema_version_status 
            ON schema_version(status)
        """)
    
    def register_migration(self, migration: Migration):
        """마이그레이션 등록"""
        # 중복 버전 체크
        if any(m.version == migration.version for m in self.migrations):
            raise ValueError(f"Migration version {migration.version} already exists")
        
        # 의존성 검증
        for dep in migration.dependencies:
            if not any(m.version == dep for m in self.migrations):
                logger.warning(f"Migration {migration.version} depends on missing version {dep}")
        
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def get_current_version(self) -> int:
        """현재 데이터베이스 버전 조회"""
        result = self.db_manager.fetchone("""
            SELECT MAX(version) 
            FROM schema_version 
            WHERE status = 'applied'
        """)
        return result[0] if result and result[0] else 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """대기 중인 마이그레이션 목록"""
        current = self.get_current_version()
        return [m for m in self.migrations if m.version > current]
    
    def migrate_to_latest(self, dry_run: bool = False) -> Tuple[bool, List[str]]:
        """최신 버전으로 마이그레이션"""
        latest_version = max((m.version for m in self.migrations), default=0)
        return self.migrate_to_version(latest_version, dry_run)
    
    def migrate_to_version(self, target_version: int, 
                          dry_run: bool = False) -> Tuple[bool, List[str]]:
        """특정 버전으로 마이그레이션"""
        logs = []
        current_version = self.get_current_version()
        
        if current_version == target_version:
            logs.append(f"Already at version {target_version}")
            return True, logs
        
        # 대상 버전 검증
        if not any(m.version == target_version for m in self.migrations):
            return False, [f"Unknown target version: {target_version}"]
        
        # dry_run 모드 알림
        if dry_run:
            logs.append("DRY RUN MODE - No changes will be made")
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
                    start_time = datetime.now()
                    
                    # 마이그레이션 실행
                    self._run_migration(migration, 'up')
                    
                    # 실행 시간 계산
                    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    # 버전 기록
                    self._record_migration(migration, execution_time)
                    
                    logs.append(f"✓ Migration {migration.version} applied successfully ({execution_time}ms)")
                    
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
    
    def _record_migration(self, migration: Migration, execution_time: int = 0):
        """마이그레이션 기록"""
        self.db_manager.execute("""
            INSERT INTO schema_version (version, name, description, checksum, applied_at, status, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, 'applied', ?)
        """, (
            migration.version,
            migration.name,
            migration.description,
            migration.checksum,
            datetime.now().isoformat(),
            execution_time
        ))
    
    def _remove_migration_record(self, migration: Migration):
        """마이그레이션 기록 제거"""
        self.db_manager.execute("""
            UPDATE schema_version 
            SET status = 'rolled_back', rolled_back_at = ?
            WHERE version = ?
        """, (datetime.now().isoformat(), migration.version))
    
    def _create_backup(self) -> Optional[Path]:
        """마이그레이션 전 백업 생성"""
        try:
            backup_dir = self.db_manager.backup_path / 'migrations'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"pre_migration_{timestamp}.db"
            
            return self.db_manager.backup(str(backup_file))
            
        except Exception as e:
            logger.error(f"Failed to create migration backup: {str(e)}")
            return None
    
    def _restore_from_backup(self, backup_path: Path) -> bool:
        """백업에서 복원"""
        try:
            return self.db_manager.restore(str(backup_path))
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
    
    def _sort_by_dependencies(self, migrations: List[Migration]) -> List[Migration]:
        """의존성에 따라 마이그레이션 정렬 (위상 정렬)"""
        # 각 마이그레이션의 진입 차수 계산
        in_degree = {m.version: 0 for m in migrations}
        adjacency = {m.version: [] for m in migrations}
        
        # 그래프 구성
        for migration in migrations:
            for dep in migration.dependencies:
                if dep in adjacency:  # 현재 실행할 마이그레이션 중에 있는 의존성만
                    adjacency[dep].append(migration.version)
                    in_degree[migration.version] += 1
        
        # 진입 차수가 0인 노드로 시작
        queue = [m for m in migrations if in_degree[m.version] == 0]
        result = []
        
        while queue:
            # 버전 순서로 정렬하여 결정적인 순서 보장
            queue.sort(key=lambda m: m.version)
            current = queue.pop(0)
            result.append(current)
            
            # 인접 노드의 진입 차수 감소
            for next_version in adjacency[current.version]:
                in_degree[next_version] -= 1
                if in_degree[next_version] == 0:
                    next_migration = next(m for m in migrations if m.version == next_version)
                    queue.append(next_migration)
        
        # 순환 의존성 체크
        if len(result) != len(migrations):
            raise ValueError("Circular dependency detected in migrations")
        
        return result
    
    def _load_migrations_from_files(self):
        """파일에서 마이그레이션 로드"""
        if not self.migrations_path.exists():
            return
        
        # SQL 파일 로드
        for sql_file in sorted(self.migrations_path.glob("*.sql")):
            try:
                self._load_sql_migration(sql_file)
            except Exception as e:
                logger.error(f"Failed to load SQL migration {sql_file}: {str(e)}")
        
        # Python 파일 로드
        for py_file in sorted(self.migrations_path.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            try:
                self._load_python_migration(py_file)
            except Exception as e:
                logger.error(f"Failed to load Python migration {py_file}: {str(e)}")
    
    def _load_sql_migration(self, file_path: Path):
        """SQL 마이그레이션 파일 로드"""
        content = file_path.read_text(encoding='utf-8')
        
        # 메타데이터 파싱
        lines = content.split('\n')
        metadata = {}
        sql_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('--'):
                sql_start = i
                break
            if line.startswith('-- '):
                key_value = line[3:].strip()
                if ':' in key_value:
                    key, value = key_value.split(':', 1)
                    metadata[key.strip().lower()] = value.strip()
        
        # 버전과 설명 추출
        version = int(metadata.get('version', '0'))
        if version == 0:
            # 파일명에서 버전 추출 시도
            if file_path.stem.startswith('V'):
                version = int(file_path.stem[1:].split('_')[0])
        
        description = metadata.get('description', file_path.stem)
        
        # UP/DOWN SQL 분리
        sql_content = '\n'.join(lines[sql_start:])
        up_sql = ""
        down_sql = ""
        current_section = None
        
        for line in sql_content.split('\n'):
            if line.strip().upper() == '-- UP':
                current_section = 'up'
            elif line.strip().upper() == '-- DOWN':
                current_section = 'down'
            elif current_section == 'up':
                up_sql += line + '\n'
            elif current_section == 'down':
                down_sql += line + '\n'
        
        # 의존성 파싱
        dependencies = []
        if 'dependencies' in metadata:
            dependencies = [int(d.strip()) for d in metadata['dependencies'].split(',')]
        
        # 마이그레이션 등록
        migration = Migration(
            version=version,
            name=file_path.stem,
            description=description,
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            dependencies=dependencies
        )
        
        self.register_migration(migration)
    
    def _load_python_migration(self, file_path: Path):
        """Python 마이그레이션 파일 로드"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 필수 속성 확인
        version = getattr(module, '__version__', 0)
        if version == 0:
            # 파일명에서 버전 추출
            if file_path.stem.startswith('V'):
                version = int(file_path.stem[1:].split('_')[0])
        
        description = getattr(module, '__description__', file_path.stem)
        dependencies = getattr(module, '__dependencies__', [])
        
        # UP/DOWN 함수 확인
        up_func = getattr(module, 'up', None)
        down_func = getattr(module, 'down', None)
        
        if not up_func:
            raise ValueError(f"Migration {file_path} missing 'up' function")
        
        # 마이그레이션 등록
        migration = Migration(
            version=version,
            name=file_path.stem,
            description=description,
            up_func=up_func,
            down_func=down_func,
            dependencies=dependencies
        )
        
        self.register_migration(migration)
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """마이그레이션 히스토리 조회"""
        return self.db_manager.fetchall("""
            SELECT version, name, description, applied_at, status, 
                   checksum, execution_time_ms, rolled_back_at
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
            
            # 4. 마이그레이션 연속성 확인
            applied_versions = sorted([
                r['version'] for r in history 
                if r['status'] == 'applied'
            ])
            
            for i, version in enumerate(applied_versions):
                if i > 0 and version != applied_versions[i-1] + 1:
                    # 버전 갭 확인 (의존성 때문에 허용될 수 있음)
                    gap_migrations = [
                        m for m in self.migrations
                        if applied_versions[i-1] < m.version < version
                    ]
                    if gap_migrations:
                        issues.append(f"Version gap detected: missing versions between {applied_versions[i-1]} and {version}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Integrity check failed: {str(e)}"]
    
    def _get_database_tables(self) -> List[str]:
        """데이터베이스 테이블 목록 조회"""
        tables = self.db_manager.fetchall("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        return [t['name'] for t in tables]
    
    def _get_expected_tables(self, version: int) -> List[str]:
        """특정 버전에서 예상되는 테이블 목록"""
        # 버전별 예상 테이블 정의
        expected = ['schema_version']  # 항상 존재
        
        if version >= 1:
            expected.extend(['users', 'projects', 'experiments', 'results'])
        
        if version >= 2:
            expected.append('module_metadata')
        
        if version >= 3:
            expected.append('experiment_templates')
        
        if version >= 4:
            expected.extend(['project_members', 'comments', 'activity_logs'])
        
        if version >= 5:
            expected.extend(['performance_metrics', 'analysis_cache'])
        
        return expected
    
    def _load_builtin_migrations(self):
        """내장 마이그레이션 정의"""
        # 버전 1: 초기 스키마
        self.register_migration(Migration(
            version=1,
            name="initial_schema",
            description="초기 데이터베이스 스키마",
            up_func=self._migration_001_up,
            down_func=self._migration_001_down
        ))
        
        # 버전 2: 모듈 메타데이터 추가
        self.register_migration(Migration(
            version=2,
            name="add_module_metadata",
            description="모듈 메타데이터 테이블 추가",
            up_sql="""
                CREATE TABLE IF NOT EXISTS module_metadata (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    author TEXT,
                    description TEXT,
                    category TEXT,
                    tags TEXT,
                    config TEXT,
                    is_core INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    install_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_module_metadata_category 
                ON module_metadata(category);
                
                CREATE INDEX IF NOT EXISTS idx_module_metadata_active 
                ON module_metadata(is_active);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_module_metadata_active;
                DROP INDEX IF EXISTS idx_module_metadata_category;
                DROP TABLE IF EXISTS module_metadata;
            """
        ))
        
        # 버전 3: 실험 템플릿 추가
        self.register_migration(Migration(
            version=3,
            name="add_experiment_templates",
            description="실험 템플릿 테이블 추가",
            up_sql="""
                CREATE TABLE IF NOT EXISTS experiment_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    description TEXT,
                    module_id TEXT NOT NULL,
                    design_config TEXT NOT NULL,
                    factors_template TEXT NOT NULL,
                    responses_template TEXT NOT NULL,
                    is_public INTEGER DEFAULT 1,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0,
                    FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_templates_category 
                ON experiment_templates(category);
                
                CREATE INDEX IF NOT EXISTS idx_templates_module 
                ON experiment_templates(module_id);
                
                CREATE INDEX IF NOT EXISTS idx_templates_public 
                ON experiment_templates(is_public);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_templates_public;
                DROP INDEX IF EXISTS idx_templates_module;
                DROP INDEX IF EXISTS idx_templates_category;
                DROP TABLE IF EXISTS experiment_templates;
            """,
            dependencies=[1]
        ))
        
        # 버전 4: 협업 기능 강화
        self.register_migration(Migration(
            version=4,
            name="enhance_collaboration",
            description="협업 기능 강화 - 댓글, 태그 등",
            up_func=self._migration_004_up,
            down_func=self._migration_004_down,
            dependencies=[1, 2, 3]
        ))
        
        # 버전 5: 성능 모니터링
        self.register_migration(Migration(
            version=5,
            name="add_performance_monitoring",
            description="성능 모니터링 테이블 추가",
            up_sql="""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL UNIQUE,
                    cache_value TEXT NOT NULL,
                    cache_type TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_type 
                ON performance_metrics(metric_type);
                
                CREATE INDEX IF NOT EXISTS idx_metrics_time 
                ON performance_metrics(recorded_at);
                
                CREATE INDEX IF NOT EXISTS idx_cache_key 
                ON analysis_cache(cache_key);
                
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON analysis_cache(expires_at);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_cache_expires;
                DROP INDEX IF EXISTS idx_cache_key;
                DROP INDEX IF EXISTS idx_metrics_time;
                DROP INDEX IF EXISTS idx_metrics_type;
                DROP TABLE IF EXISTS analysis_cache;
                DROP TABLE IF EXISTS performance_metrics;
            """
        ))
    
    def _migration_001_up(self, conn: sqlite3.Connection):
        """초기 스키마 생성"""
        # users 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                preferences TEXT
            )
        ''')
        
        # projects 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                owner_id INTEGER NOT NULL,
                category TEXT,
                tags TEXT,
                config TEXT,
                is_public INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # experiments 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                module_id TEXT NOT NULL,
                design_type TEXT NOT NULL,
                factors TEXT NOT NULL,
                responses TEXT NOT NULL,
                design_matrix TEXT NOT NULL,
                status TEXT DEFAULT 'planned',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
        ''')
        
        # results 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                run_number INTEGER NOT NULL,
                conditions TEXT NOT NULL,
                measurements TEXT NOT NULL,
                notes TEXT,
                performed_by INTEGER,
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
                FOREIGN KEY (performed_by) REFERENCES users (id) ON DELETE SET NULL,
                UNIQUE(experiment_id, run_number)
            )
        ''')
        
        # 인덱스 생성
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id)")
    
    def _migration_001_down(self, conn: sqlite3.Connection):
        """초기 스키마 롤백"""
        conn.execute("DROP TABLE IF EXISTS results")
        conn.execute("DROP TABLE IF EXISTS experiments")
        conn.execute("DROP TABLE IF EXISTS projects")
        conn.execute("DROP TABLE IF EXISTS users")
    
    def _migration_004_up(self, conn: sqlite3.Connection):
        """협업 기능 테이블 생성"""
        # project_members 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS project_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT DEFAULT 'viewer',
                invited_by INTEGER,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (invited_by) REFERENCES users (id) ON DELETE SET NULL,
                UNIQUE(project_id, user_id)
            )
        ''')
        
        # comments 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                parent_id INTEGER,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES comments (id) ON DELETE CASCADE
            )
        ''')
        
        # activity_logs 테이블
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id INTEGER,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # 인덱스 추가
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project_members_project ON project_members(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_entity ON comments(entity_type, entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_logs_user ON activity_logs(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_logs_time ON activity_logs(created_at)")
    
    def _migration_004_down(self, conn: sqlite3.Connection):
        """협업 기능 롤백"""
        conn.execute("DROP TABLE IF EXISTS activity_logs")
        conn.execute("DROP TABLE IF EXISTS comments")
        conn.execute("DROP TABLE IF EXISTS project_members")


# 싱글톤 패턴
_migration_manager = None

def get_migration_manager(db_manager) -> MigrationManager:
    """마이그레이션 매니저 싱글톤 인스턴스"""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager(db_manager)
    return _migration_manager


# 유틸리티 함수
def check_and_migrate(db_manager) -> bool:
    """데이터베이스 확인 및 마이그레이션 실행"""
    try:
        migration_manager = get_migration_manager(db_manager)
        
        # 무결성 검사
        is_valid, issues = migration_manager.verify_integrity()
        if not is_valid:
            logger.warning(f"Database integrity issues found: {issues}")
        
        # 대기 중인 마이그레이션 확인
        pending = migration_manager.get_pending_migrations()
        if pending:
            logger.info(f"Found {len(pending)} pending migrations")
            
            # 마이그레이션 실행
            success, logs = migration_manager.migrate_to_latest()
            
            for log in logs:
                logger.info(log)
            
            return success
        
        return True
        
    except Exception as e:
        logger.error(f"Migration check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False
