"""
자동 업데이트 시스템
데스크톱 애플리케이션의 버전 확인, 다운로드, 설치를 관리
"""
import os
import sys
import json
import shutil
import tempfile
import hashlib
import platform
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from datetime import datetime, timedelta
import requests
from packaging import version
import zipfile
import tarfile

from config.app_config import UPDATE_CONFIG, APP_INFO
from config.local_config import LOCAL_CONFIG

logger = logging.getLogger(__name__)

class UpdateError(Exception):
    """업데이트 관련 에러"""
    pass

class UpdateInfo:
    """업데이트 정보 모델"""
    def __init__(self, data: Dict[str, Any]):
        self.version = data['version']
        self.channel = data.get('channel', 'stable')
        self.download_url = data['download_url']
        self.file_size = data.get('file_size', 0)
        self.checksum = data['checksum']
        self.checksum_type = data.get('checksum_type', 'sha256')
        self.release_notes = data.get('release_notes', '')
        self.release_date = data.get('release_date', '')
        self.mandatory = data.get('mandatory', False)
        self.min_version = data.get('min_version')
        self.signature = data.get('signature')
        
    def is_newer_than(self, current_version: str) -> bool:
        """현재 버전보다 새로운지 확인"""
        return version.parse(self.version) > version.parse(current_version)
    
    def can_update_from(self, current_version: str) -> bool:
        """현재 버전에서 업데이트 가능한지 확인"""
        if not self.min_version:
            return True
        return version.parse(current_version) >= version.parse(self.min_version)

class AutoUpdater:
    """자동 업데이트 관리자"""
    
    def __init__(self):
        self.current_version = APP_INFO['version']
        self.update_channel = UPDATE_CONFIG.get('channel', 'stable')
        self.server_url = UPDATE_CONFIG.get('server_url', 'https://api.universaldoe.com/updates')
        self.check_interval = UPDATE_CONFIG.get('check_interval', timedelta(days=1))
        self.download_timeout = UPDATE_CONFIG.get('download_timeout', timedelta(minutes=30))
        
        # 경로 설정
        self.data_dir = LOCAL_CONFIG['app_data_dir']
        self.update_dir = self.data_dir / 'updates'
        self.backup_dir = self.data_dir / 'backups'
        self.temp_dir = self.data_dir / 'temp' / 'updates'
        
        # 디렉토리 생성
        for dir_path in [self.update_dir, self.backup_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 상태
        self.is_checking = False
        self.is_downloading = False
        self.download_progress = 0
        self.last_check = None
        self.available_update = None
        
        # 콜백
        self.progress_callback = None
        self.status_callback = None
        
    def set_callbacks(self, progress: Optional[Callable] = None, 
                     status: Optional[Callable] = None):
        """콜백 함수 설정"""
        self.progress_callback = progress
        self.status_callback = status
    
    def _notify_progress(self, progress: float, message: str = ""):
        """진행률 알림"""
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _notify_status(self, status: str, data: Dict[str, Any] = None):
        """상태 알림"""
        if self.status_callback:
            try:
                self.status_callback(status, data or {})
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def check_for_updates(self, force: bool = False) -> Optional[UpdateInfo]:
        """
        업데이트 확인
        
        Args:
            force: 강제 확인 여부
            
        Returns:
            UpdateInfo 또는 None
        """
        # 이미 확인 중이면 스킵
        if self.is_checking:
            logger.info("Already checking for updates")
            return self.available_update
        
        # 최근 확인 시간 체크 (force가 아닌 경우)
        if not force and self.last_check:
            time_since_check = datetime.now() - self.last_check
            if time_since_check < self.check_interval:
                logger.info(f"Skipping update check (last check: {time_since_check} ago)")
                return self.available_update
        
        self.is_checking = True
        self._notify_status("checking", {"message": "업데이트 확인 중..."})
        
        try:
            # API 호출
            params = {
                'current_version': self.current_version,
                'channel': self.update_channel,
                'platform': platform.system().lower(),
                'arch': platform.machine().lower()
            }
            
            response = requests.get(
                f"{self.server_url}/check",
                params=params,
                timeout=10,
                headers={'User-Agent': f"UniversalDOE/{self.current_version}"}
            )
            
            if response.status_code == 204:
                # 업데이트 없음
                logger.info("No updates available")
                self.available_update = None
                self._notify_status("no_update", {"message": "최신 버전입니다"})
                return None
            
            response.raise_for_status()
            update_data = response.json()
            
            # UpdateInfo 객체 생성
            update_info = UpdateInfo(update_data)
            
            # 버전 확인
            if not update_info.is_newer_than(self.current_version):
                logger.info(f"Server version {update_info.version} is not newer than {self.current_version}")
                self.available_update = None
                return None
            
            # 업데이트 가능 여부 확인
            if not update_info.can_update_from(self.current_version):
                logger.warning(f"Cannot update from {self.current_version} to {update_info.version}")
                self._notify_status("incompatible", {
                    "message": f"버전 {update_info.version}으로 직접 업데이트할 수 없습니다",
                    "min_version": update_info.min_version
                })
                return None
            
            self.available_update = update_info
            self.last_check = datetime.now()
            
            self._notify_status("update_available", {
                "version": update_info.version,
                "mandatory": update_info.mandatory,
                "release_notes": update_info.release_notes
            })
            
            logger.info(f"Update available: {update_info.version}")
            return update_info
            
        except requests.RequestException as e:
            logger.error(f"Update check failed: {e}")
            self._notify_status("check_failed", {"error": str(e)})
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during update check: {e}")
            self._notify_status("check_failed", {"error": "예기치 않은 오류"})
            return None
            
        finally:
            self.is_checking = False
    
    def download_update(self, update_info: Optional[UpdateInfo] = None) -> Optional[Path]:
        """
        업데이트 다운로드
        
        Args:
            update_info: 업데이트 정보 (None이면 available_update 사용)
            
        Returns:
            다운로드된 파일 경로 또는 None
        """
        if not update_info:
            update_info = self.available_update
            
        if not update_info:
            logger.error("No update info available")
            return None
        
        if self.is_downloading:
            logger.info("Already downloading")
            return None
        
        self.is_downloading = True
        self.download_progress = 0
        
        # 파일명 생성
        ext = self._get_package_extension()
        filename = f"UniversalDOE-{update_info.version}-{platform.system()}-{platform.machine()}{ext}"
        download_path = self.temp_dir / filename
        
        try:
            self._notify_status("downloading", {
                "version": update_info.version,
                "size": update_info.file_size
            })
            
            # 다운로드 시작
            response = requests.get(
                update_info.download_url,
                stream=True,
                timeout=30,
                headers={'User-Agent': f"UniversalDOE/{self.current_version}"}
            )
            response.raise_for_status()
            
            # 전체 크기 확인
            total_size = int(response.headers.get('content-length', update_info.file_size))
            
            # 파일 쓰기
            downloaded = 0
            chunk_size = 8192
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 진행률 업데이트
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.download_progress = progress
                            self._notify_progress(progress, f"{downloaded:,} / {total_size:,} bytes")
            
            logger.info(f"Download completed: {download_path}")
            
            # 체크섬 검증
            if not self._verify_checksum(download_path, update_info):
                raise UpdateError("체크섬 검증 실패")
            
            # 서명 검증 (옵션)
            if update_info.signature and not self._verify_signature(download_path, update_info):
                raise UpdateError("서명 검증 실패")
            
            # 다운로드 완료
            final_path = self.update_dir / filename
            shutil.move(str(download_path), str(final_path))
            
            self._notify_status("download_complete", {
                "version": update_info.version,
                "path": str(final_path)
            })
            
            return final_path
            
        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            self._notify_status("download_failed", {"error": str(e)})
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            self._notify_status("download_failed", {"error": str(e)})
            
            # 임시 파일 정리
            if download_path.exists():
                try:
                    download_path.unlink()
                except:
                    pass
            
            return None
            
        finally:
            self.is_downloading = False
    
    def install_update(self, update_file: Path, update_info: Optional[UpdateInfo] = None) -> bool:
        """
        업데이트 설치
        
        Args:
            update_file: 업데이트 파일 경로
            update_info: 업데이트 정보
            
        Returns:
            성공 여부
        """
        if not update_file.exists():
            logger.error(f"Update file not found: {update_file}")
            return False
        
        if not update_info:
            update_info = self.available_update
        
        try:
            self._notify_status("installing", {"version": update_info.version if update_info else "unknown"})
            
            # 플랫폼별 설치
            system = platform.system()
            
            if system == "Windows":
                return self._install_windows(update_file, update_info)
            elif system == "Darwin":  # macOS
                return self._install_macos(update_file, update_info)
            elif system == "Linux":
                return self._install_linux(update_file, update_info)
            else:
                raise UpdateError(f"Unsupported platform: {system}")
                
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            self._notify_status("install_failed", {"error": str(e)})
            return False
    
    def _verify_checksum(self, file_path: Path, update_info: UpdateInfo) -> bool:
        """체크섬 검증"""
        try:
            hash_algo = getattr(hashlib, update_info.checksum_type, hashlib.sha256)
            file_hash = hash_algo()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    file_hash.update(chunk)
            
            calculated = file_hash.hexdigest()
            expected = update_info.checksum.lower()
            
            if calculated != expected:
                logger.error(f"Checksum mismatch: {calculated} != {expected}")
                return False
            
            logger.info("Checksum verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Checksum verification error: {e}")
            return False
    
    def _verify_signature(self, file_path: Path, update_info: UpdateInfo) -> bool:
        """디지털 서명 검증 (향후 구현)"""
        # TODO: 공개키로 서명 검증
        logger.warning("Signature verification not implemented")
        return True
    
    def _get_package_extension(self) -> str:
        """플랫폼별 패키지 확장자"""
        system = platform.system()
        if system == "Windows":
            return ".exe"
        elif system == "Darwin":
            return ".dmg"
        elif system == "Linux":
            return ".AppImage"
        return ".zip"
    
    def _install_windows(self, update_file: Path, update_info: Optional[UpdateInfo]) -> bool:
        """Windows 업데이트 설치"""
        try:
            # 현재 실행 파일 경로
            if getattr(sys, 'frozen', False):
                current_exe = Path(sys.executable)
            else:
                logger.error("Cannot update in development mode")
                return False
            
            # 백업 생성
            backup_path = self._create_backup(current_exe)
            if not backup_path:
                return False
            
            # 업데이트 스크립트 생성
            script_path = self._create_windows_update_script(
                current_exe, update_file, backup_path
            )
            
            # 스크립트 실행 (새 프로세스)
            subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
            
            self._notify_status("restart_required", {
                "message": "업데이트를 완료하려면 프로그램을 다시 시작해야 합니다"
            })
            
            # 프로그램 종료
            if UPDATE_CONFIG.get('install_on_exit', True):
                logger.info("Exiting for update installation")
                sys.exit(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Windows installation error: {e}")
            return False
    
    def _install_macos(self, update_file: Path, update_info: Optional[UpdateInfo]) -> bool:
        """macOS 업데이트 설치"""
        try:
            # DMG 마운트
            mount_point = self.temp_dir / "mount"
            mount_point.mkdir(exist_ok=True)
            
            # hdiutil로 마운트
            subprocess.run([
                "hdiutil", "attach", str(update_file),
                "-mountpoint", str(mount_point),
                "-nobrowse", "-quiet"
            ], check=True)
            
            try:
                # 앱 번들 찾기
                app_bundle = None
                for item in mount_point.iterdir():
                    if item.suffix == ".app":
                        app_bundle = item
                        break
                
                if not app_bundle:
                    raise UpdateError("App bundle not found in DMG")
                
                # 현재 앱 경로
                current_app = Path(sys.executable).parent.parent
                if current_app.suffix != ".app":
                    raise UpdateError("Cannot determine current app bundle")
                
                # 백업 생성
                backup_path = self._create_backup(current_app)
                
                # 새 앱 복사
                temp_app = self.temp_dir / app_bundle.name
                shutil.copytree(app_bundle, temp_app)
                
                # 업데이트 스크립트 생성
                script_path = self._create_macos_update_script(
                    current_app, temp_app, backup_path
                )
                
                # 스크립트 실행
                subprocess.Popen(["bash", str(script_path)])
                
                self._notify_status("restart_required", {
                    "message": "업데이트를 완료하려면 앱을 다시 시작해야 합니다"
                })
                
                # 앱 종료
                if UPDATE_CONFIG.get('install_on_exit', True):
                    sys.exit(0)
                
                return True
                
            finally:
                # DMG 언마운트
                subprocess.run(["hdiutil", "detach", str(mount_point), "-quiet"])
                
        except Exception as e:
            logger.error(f"macOS installation error: {e}")
            return False
    
    def _install_linux(self, update_file: Path, update_info: Optional[UpdateInfo]) -> bool:
        """Linux 업데이트 설치"""
        try:
            if update_file.suffix == ".AppImage":
                # AppImage 직접 교체
                if getattr(sys, 'frozen', False):
                    current_exe = Path(sys.executable)
                else:
                    logger.error("Cannot update in development mode")
                    return False
                
                # 백업 생성
                backup_path = self._create_backup(current_exe)
                
                # 실행 권한 부여
                os.chmod(update_file, 0o755)
                
                # 업데이트 스크립트 생성
                script_path = self._create_linux_update_script(
                    current_exe, update_file, backup_path
                )
                
                # 스크립트 실행
                subprocess.Popen(["bash", str(script_path)])
                
                self._notify_status("restart_required", {
                    "message": "업데이트를 완료하려면 프로그램을 다시 시작해야 합니다"
                })
                
                # 프로그램 종료
                if UPDATE_CONFIG.get('install_on_exit', True):
                    sys.exit(0)
                
                return True
                
            else:
                raise UpdateError(f"Unsupported Linux package format: {update_file.suffix}")
                
        except Exception as e:
            logger.error(f"Linux installation error: {e}")
            return False
    
    def _create_backup(self, target: Path) -> Optional[Path]:
        """백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if target.is_file():
                # 파일 백업
                backup_name = f"{target.stem}_backup_{timestamp}{target.suffix}"
                backup_path = self.backup_dir / backup_name
                shutil.copy2(target, backup_path)
            else:
                # 디렉토리 백업 (macOS .app)
                backup_name = f"{target.name}_backup_{timestamp}"
                backup_path = self.backup_dir / backup_name
                shutil.copytree(target, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            
            # 오래된 백업 정리
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def _cleanup_old_backups(self, keep_count: int = 3):
        """오래된 백업 정리"""
        try:
            backups = sorted(self.backup_dir.glob("*_backup_*"), 
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
            
            for backup in backups[keep_count:]:
                if backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup}")
                
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")
    
    def _create_windows_update_script(self, current_exe: Path, 
                                    update_file: Path, backup_path: Path) -> Path:
        """Windows 업데이트 스크립트 생성"""
        script_content = f'''
import os
import sys
import time
import shutil
import subprocess

def main():
    print("Universal DOE Updater")
    
    # 현재 프로그램 종료 대기
    print("Waiting for application to close...")
    time.sleep(3)
    
    # 프로세스 확인 및 종료
    max_attempts = 10
    for i in range(max_attempts):
        try:
            # 기존 파일 이름 변경 시도
            temp_name = r"{current_exe}.old"
            if os.path.exists(temp_name):
                os.remove(temp_name)
            os.rename(r"{current_exe}", temp_name)
            break
        except:
            if i < max_attempts - 1:
                time.sleep(2)
            else:
                print("Failed to rename current executable")
                return 1
    
    try:
        # 새 파일 복사
        print("Installing update...")
        shutil.copy2(r"{update_file}", r"{current_exe}")
        
        # 임시 파일 제거
        try:
            os.remove(temp_name)
        except:
            pass
        
        # 업데이트 파일 제거
        try:
            os.remove(r"{update_file}")
        except:
            pass
        
        print("Update installed successfully!")
        
        # 새 프로그램 실행
        print("Starting updated application...")
        subprocess.Popen([r"{current_exe}"])
        
        return 0
        
    except Exception as e:
        print(f"Update failed: {{e}}")
        
        # 백업에서 복원
        try:
            shutil.copy2(r"{backup_path}", r"{current_exe}")
            print("Restored from backup")
        except:
            print("Failed to restore from backup!")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        script_path = self.temp_dir / "update_script.py"
        script_path.write_text(script_content, encoding='utf-8')
        return script_path
    
    def _create_macos_update_script(self, current_app: Path, 
                                   new_app: Path, backup_path: Path) -> Path:
        """macOS 업데이트 스크립트 생성"""
        script_content = f'''#!/bin/bash

echo "Universal DOE Updater"

# 앱 종료 대기
echo "Waiting for application to close..."
sleep 3

# 기존 앱 제거
echo "Removing old application..."
rm -rf "{current_app}"

# 새 앱 설치
echo "Installing update..."
cp -R "{new_app}" "{current_app}"

# 권한 설정
chmod -R 755 "{current_app}"

# 임시 파일 정리
rm -rf "{new_app}"

echo "Update installed successfully!"

# 새 앱 실행
echo "Starting updated application..."
open "{current_app}"

exit 0
'''
        
        script_path = self.temp_dir / "update_script.sh"
        script_path.write_text(script_content)
        os.chmod(script_path, 0o755)
        return script_path
    
    def _create_linux_update_script(self, current_exe: Path, 
                                   update_file: Path, backup_path: Path) -> Path:
        """Linux 업데이트 스크립트 생성"""
        script_content = f'''#!/bin/bash

echo "Universal DOE Updater"

# 프로그램 종료 대기
echo "Waiting for application to close..."
sleep 3

# 실행 파일 교체
echo "Installing update..."
mv "{update_file}" "{current_exe}"
chmod +x "{current_exe}"

echo "Update installed successfully!"

# 새 프로그램 실행
echo "Starting updated application..."
"{current_exe}" &

exit 0
'''
        
        script_path = self.temp_dir / "update_script.sh"
        script_path.write_text(script_content)
        os.chmod(script_path, 0o755)
        return script_path
    
    def check_and_apply_update(self, silent: bool = False) -> bool:
        """
        업데이트 확인 및 적용 (원스톱)
        
        Args:
            silent: 자동 모드 (사용자 확인 없음)
            
        Returns:
            업데이트 적용 여부
        """
        try:
            # 업데이트 확인
            update_info = self.check_for_updates()
            if not update_info:
                return False
            
            # 필수 업데이트이거나 자동 모드인 경우
            if update_info.mandatory or silent:
                # 다운로드
                update_file = self.download_update(update_info)
                if not update_file:
                    return False
                
                # 설치
                return self.install_update(update_file, update_info)
            
            # 사용자 확인 필요 (UI에서 처리)
            return False
            
        except Exception as e:
            logger.error(f"Auto update error: {e}")
            return False
    
    def get_update_status(self) -> Dict[str, Any]:
        """현재 업데이트 상태 반환"""
        return {
            'current_version': self.current_version,
            'update_channel': self.update_channel,
            'is_checking': self.is_checking,
            'is_downloading': self.is_downloading,
            'download_progress': self.download_progress,
            'available_update': {
                'version': self.available_update.version,
                'mandatory': self.available_update.mandatory,
                'release_notes': self.available_update.release_notes,
                'release_date': self.available_update.release_date
            } if self.available_update else None,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'auto_update_enabled': UPDATE_CONFIG.get('enabled', True)
        }
    
    def schedule_periodic_check(self):
        """주기적 업데이트 확인 스케줄링"""
        if not UPDATE_CONFIG.get('enabled', True):
            logger.info("Auto update is disabled")
            return
        
        def check_loop():
            while True:
                try:
                    # 대기
                    threading.Event().wait(self.check_interval.total_seconds())
                    
                    # 업데이트 확인
                    if UPDATE_CONFIG.get('enabled', True):
                        self.check_for_updates()
                        
                except Exception as e:
                    logger.error(f"Periodic check error: {e}")
        
        # 백그라운드 스레드 시작
        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()
        logger.info("Started periodic update check")

# 싱글톤 인스턴스
_updater = None

def get_updater() -> AutoUpdater:
    """AutoUpdater 싱글톤 인스턴스 반환"""
    global _updater
    if _updater is None:
        _updater = AutoUpdater()
    return _updater
