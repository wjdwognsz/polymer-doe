"""
utils/performance_monitor.py
시스템 성능 모니터링 및 최적화 관리자

앱의 성능을 실시간으로 추적하고 분석하여 최적화 제안을 제공합니다.
CPU, 메모리, API 호출, DB 쿼리 등 모든 성능 지표를 통합 관리합니다.
"""

import os
import sys
import time
import psutil
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import logging
from functools import wraps
import traceback
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# 로컬 모듈 임포트
try:
    from config.app_config import PERFORMANCE_CONFIG, DEBUG, APP_INFO
    from utils.database_manager import get_database_manager
except ImportError:
    # 기본값 설정
    PERFORMANCE_CONFIG = {
        'monitor_interval': timedelta(minutes=5),
        'gc_threshold': 80,
        'memory': {'monitor_interval': timedelta(minutes=5)}
    }
    DEBUG = True

logger = logging.getLogger(__name__)

# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class MetricPoint:
    """개별 메트릭 포인트"""
    timestamp: datetime
    name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APICallMetric:
    """API 호출 메트릭"""
    api_name: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    response_size: int = 0
    request_size: int = 0
    
    @property
    def duration(self) -> float:
        """호출 시간 (초)"""
        if self.end_time:
            return self.end_time - self.start_time
        return 0

@dataclass
class QueryMetric:
    """데이터베이스 쿼리 메트릭"""
    query: str
    start_time: float
    end_time: Optional[float] = None
    rows_affected: int = 0
    success: bool = False
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """쿼리 실행 시간 (초)"""
        if self.end_time:
            return self.end_time - self.start_time
        return 0

@dataclass
class PerformanceReport:
    """성능 리포트"""
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    metrics: List[MetricPoint]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]

# ============================================================================
# 성능 모니터 클래스
# ============================================================================

class PerformanceMonitor:
    """성능 모니터링 관리자"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """초기화"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.db_manager = None
        self.process = psutil.Process()
        
        # 메트릭 저장소
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.system_metrics = deque(maxlen=1000)  # 최근 1000개
        self.api_metrics = defaultdict(lambda: deque(maxlen=100))
        self.query_metrics = deque(maxlen=500)
        self.user_activity = defaultdict(int)
        
        # 통계 캐시
        self._stats_cache = {}
        self._cache_timestamp = None
        
        # 모니터링 설정
        self.monitor_interval = PERFORMANCE_CONFIG.get(
            'memory', {}
        ).get('monitor_interval', timedelta(seconds=60)).total_seconds()
        
        self.gc_threshold = PERFORMANCE_CONFIG.get('gc_threshold', 80)
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PerfMon")
        
        # 모니터링 스레드
        self.monitoring = False
        self.monitor_thread = None
        
        # 알림 임계값
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_usage_percent': 90,
            'response_time_ms': 1000,
            'error_rate_percent': 5
        }
        
        logger.info("PerformanceMonitor 초기화 완료")
    
    # ==================== 시스템 메트릭 ====================
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        try:
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # 네트워크 정보
            net_io = psutil.net_io_counters()
            
            # 프로세스 정보
            process_info = {
                'cpu_percent': self.process.cpu_percent(),
                'memory_info': self.process.memory_info()._asdict(),
                'num_threads': self.process.num_threads(),
                'open_files': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
            }
            
            metrics = {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq._asdict() if cpu_freq else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'swap_percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'process': process_info
            }
            
            # 메트릭 저장
            self.system_metrics.append(metrics)
            
            # 임계값 체크
            self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return {}
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """임계값 체크 및 알림"""
        alerts = []
        
        # CPU 체크
        if metrics['cpu']['percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"CPU 사용률이 {metrics['cpu']['percent']:.1f}%로 높습니다",
                'value': metrics['cpu']['percent']
            })
        
        # 메모리 체크
        if metrics['memory']['percent'] > self.thresholds['memory_percent']:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"메모리 사용률이 {metrics['memory']['percent']:.1f}%로 높습니다",
                'value': metrics['memory']['percent']
            })
        
        # 디스크 체크
        if metrics['disk']['percent'] > self.thresholds['disk_usage_percent']:
            alerts.append({
                'type': 'disk',
                'level': 'critical',
                'message': f"디스크 사용률이 {metrics['disk']['percent']:.1f}%로 매우 높습니다",
                'value': metrics['disk']['percent']
            })
        
        # 알림 처리
        for alert in alerts:
            logger.warning(f"성능 경고: {alert['message']}")
            # TODO: NotificationManager 연동
    
    # ==================== API 메트릭 ====================
    
    def track_api_call(self, api_name: str, endpoint: str) -> APICallMetric:
        """API 호출 추적 시작"""
        metric = APICallMetric(
            api_name=api_name,
            endpoint=endpoint,
            start_time=time.time()
        )
        return metric
    
    def complete_api_call(self, metric: APICallMetric, 
                         success: bool = True, 
                         error: Optional[str] = None,
                         response_size: int = 0):
        """API 호출 완료"""
        metric.end_time = time.time()
        metric.success = success
        metric.error = error
        metric.response_size = response_size
        
        # 메트릭 저장
        self.api_metrics[metric.api_name].append(metric)
        
        # 로깅
        if DEBUG:
            logger.debug(
                f"API 호출: {metric.api_name}/{metric.endpoint} - "
                f"{metric.duration*1000:.2f}ms, 성공: {success}"
            )
    
    def get_api_stats(self, api_name: Optional[str] = None) -> Dict[str, Any]:
        """API 통계 조회"""
        stats = {}
        
        if api_name:
            metrics = list(self.api_metrics.get(api_name, []))
        else:
            metrics = []
            for api_metrics in self.api_metrics.values():
                metrics.extend(list(api_metrics))
        
        if not metrics:
            return stats
        
        # 성공률
        success_count = sum(1 for m in metrics if m.success)
        stats['success_rate'] = (success_count / len(metrics)) * 100 if metrics else 0
        
        # 응답 시간
        durations = [m.duration * 1000 for m in metrics if m.duration > 0]
        if durations:
            stats['avg_response_time_ms'] = statistics.mean(durations)
            stats['min_response_time_ms'] = min(durations)
            stats['max_response_time_ms'] = max(durations)
            stats['p95_response_time_ms'] = statistics.quantiles(durations, n=20)[18]  # 95th percentile
        
        # 처리량
        stats['total_calls'] = len(metrics)
        stats['failed_calls'] = len(metrics) - success_count
        
        # 데이터 전송량
        stats['total_response_size'] = sum(m.response_size for m in metrics)
        stats['avg_response_size'] = stats['total_response_size'] / len(metrics) if metrics else 0
        
        return stats
    
    # ==================== 데이터베이스 메트릭 ====================
    
    def track_query(self, query: str) -> QueryMetric:
        """쿼리 추적 시작"""
        metric = QueryMetric(
            query=query[:100],  # 쿼리 앞부분만 저장
            start_time=time.time()
        )
        return metric
    
    def complete_query(self, metric: QueryMetric,
                      success: bool = True,
                      rows_affected: int = 0,
                      error: Optional[str] = None):
        """쿼리 완료"""
        metric.end_time = time.time()
        metric.success = success
        metric.rows_affected = rows_affected
        metric.error = error
        
        # 메트릭 저장
        self.query_metrics.append(metric)
        
        # 느린 쿼리 경고
        if metric.duration > 1.0:  # 1초 이상
            logger.warning(f"느린 쿼리 감지: {metric.duration:.2f}초 - {metric.query}")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """쿼리 통계 조회"""
        metrics = list(self.query_metrics)
        
        if not metrics:
            return {}
        
        stats = {
            'total_queries': len(metrics),
            'failed_queries': sum(1 for m in metrics if not m.success),
            'success_rate': sum(1 for m in metrics if m.success) / len(metrics) * 100
        }
        
        # 실행 시간 통계
        durations = [m.duration * 1000 for m in metrics if m.duration > 0]
        if durations:
            stats['avg_duration_ms'] = statistics.mean(durations)
            stats['max_duration_ms'] = max(durations)
            stats['slow_queries'] = sum(1 for d in durations if d > 1000)  # 1초 이상
        
        return stats
    
    # ==================== 사용자 활동 ====================
    
    def track_user_activity(self, user_id: str, action: str, metadata: Optional[Dict] = None):
        """사용자 활동 추적"""
        key = f"{user_id}:{action}"
        self.user_activity[key] += 1
        
        # 상세 메트릭
        metric = MetricPoint(
            timestamp=datetime.now(),
            name="user_activity",
            value=1,
            tags={
                'user_id': user_id,
                'action': action
            },
            metadata=metadata or {}
        )
        
        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            logger.warning("메트릭 큐가 가득 참")
    
    def track_page_view(self, page: str, load_time: float):
        """페이지 뷰 추적"""
        metric = MetricPoint(
            timestamp=datetime.now(),
            name="page_view",
            value=load_time,
            unit="seconds",
            tags={'page': page}
        )
        
        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            pass
    
    # ==================== 데코레이터 ====================
    
    def measure_time(self, name: str):
        """실행 시간 측정 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # 메트릭 기록
                    metric = MetricPoint(
                        timestamp=datetime.now(),
                        name=f"function_duration",
                        value=duration * 1000,  # ms로 변환
                        unit="ms",
                        tags={'function': name}
                    )
                    
                    try:
                        self.metrics_queue.put_nowait(metric)
                    except queue.Full:
                        pass
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"함수 실행 실패 {name}: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    # ==================== 모니터링 제어 ====================
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring:
            logger.warning("이미 모니터링 중입니다")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        # 메트릭 처리 스레드
        self.executor.submit(self._process_metrics_queue)
        
        logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.executor.shutdown(wait=False)
        
        logger.info("성능 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # 시스템 메트릭 수집
                self.collect_system_metrics()
                
                # 가비지 컬렉션 체크
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.gc_threshold:
                    import gc
                    gc.collect()
                    logger.info(f"가비지 컬렉션 실행 (메모리 사용률: {memory_percent:.1f}%)")
                
                # 대기
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 에러: {e}")
                time.sleep(10)  # 에러 시 잠시 대기
    
    def _process_metrics_queue(self):
        """메트릭 큐 처리"""
        batch = []
        last_flush = time.time()
        
        while self.monitoring:
            try:
                # 큐에서 메트릭 가져오기
                try:
                    metric = self.metrics_queue.get(timeout=1)
                    batch.append(metric)
                except queue.Empty:
                    pass
                
                # 배치 플러시 (100개 또는 10초마다)
                if len(batch) >= 100 or time.time() - last_flush > 10:
                    if batch:
                        self._save_metrics_batch(batch)
                        batch = []
                        last_flush = time.time()
                        
            except Exception as e:
                logger.error(f"메트릭 처리 에러: {e}")
    
    def _save_metrics_batch(self, metrics: List[MetricPoint]):
        """메트릭 배치 저장"""
        if not self.db_manager:
            return
        
        try:
            # TODO: 데이터베이스에 저장
            # performance_metrics 테이블 활용
            pass
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    # ==================== 리포트 생성 ====================
    
    def generate_report(self, period: timedelta = timedelta(hours=1)) -> PerformanceReport:
        """성능 리포트 생성"""
        end_time = datetime.now()
        start_time = end_time - period
        
        # 시스템 메트릭 요약
        system_summary = self._summarize_system_metrics(start_time, end_time)
        
        # API 메트릭 요약
        api_summary = self._summarize_api_metrics(start_time, end_time)
        
        # 쿼리 메트릭 요약
        query_summary = self._summarize_query_metrics(start_time, end_time)
        
        # 종합 요약
        summary = {
            'system': system_summary,
            'api': api_summary,
            'database': query_summary,
            'period_minutes': period.total_seconds() / 60
        }
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(summary)
        
        # 알림 수집
        alerts = []  # TODO: 알림 수집 구현
        
        report = PerformanceReport(
            period_start=start_time,
            period_end=end_time,
            summary=summary,
            metrics=[],  # TODO: 상세 메트릭 포함
            recommendations=recommendations,
            alerts=alerts
        )
        
        return report
    
    def _summarize_system_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """시스템 메트릭 요약"""
        recent_metrics = [m for m in self.system_metrics 
                         if start_time <= m['timestamp'] <= end_time]
        
        if not recent_metrics:
            return {}
        
        # CPU 통계
        cpu_values = [m['cpu']['percent'] for m in recent_metrics]
        
        # 메모리 통계
        memory_values = [m['memory']['percent'] for m in recent_metrics]
        
        return {
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'current': recent_metrics[-1]['memory']['percent']
            },
            'samples': len(recent_metrics)
        }
    
    def _summarize_api_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """API 메트릭 요약"""
        summary = {}
        
        for api_name, metrics in self.api_metrics.items():
            recent = [m for m in metrics 
                     if start_time.timestamp() <= m.start_time <= end_time.timestamp()]
            
            if recent:
                summary[api_name] = {
                    'total_calls': len(recent),
                    'success_rate': sum(1 for m in recent if m.success) / len(recent) * 100,
                    'avg_duration_ms': statistics.mean([m.duration * 1000 for m in recent])
                }
        
        return summary
    
    def _summarize_query_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """쿼리 메트릭 요약"""
        recent = [m for m in self.query_metrics 
                 if start_time.timestamp() <= m.start_time <= end_time.timestamp()]
        
        if not recent:
            return {}
        
        durations = [m.duration * 1000 for m in recent if m.duration > 0]
        
        return {
            'total_queries': len(recent),
            'success_rate': sum(1 for m in recent if m.success) / len(recent) * 100,
            'avg_duration_ms': statistics.mean(durations) if durations else 0,
            'slow_queries': sum(1 for d in durations if d > 1000)
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        # CPU 체크
        if summary.get('system', {}).get('cpu', {}).get('avg', 0) > 70:
            recommendations.append("CPU 사용률이 높습니다. 백그라운드 작업을 최적화하거나 하드웨어 업그레이드를 고려하세요.")
        
        # 메모리 체크
        if summary.get('system', {}).get('memory', {}).get('avg', 0) > 80:
            recommendations.append("메모리 사용률이 높습니다. 캐시 크기를 조정하거나 메모리 누수를 확인하세요.")
        
        # API 성능 체크
        for api_name, api_stats in summary.get('api', {}).items():
            if api_stats.get('avg_duration_ms', 0) > 2000:
                recommendations.append(f"{api_name} API의 응답 시간이 느립니다. API 사용을 최적화하거나 캐싱을 고려하세요.")
            
            if api_stats.get('success_rate', 100) < 95:
                recommendations.append(f"{api_name} API의 성공률이 낮습니다. 에러 처리를 개선하세요.")
        
        # 쿼리 성능 체크
        db_stats = summary.get('database', {})
        if db_stats.get('slow_queries', 0) > 10:
            recommendations.append("느린 쿼리가 많습니다. 인덱스를 추가하거나 쿼리를 최적화하세요.")
        
        return recommendations
    
    # ==================== 유틸리티 메서드 ====================
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 통계 조회"""
        # 캐시 확인
        if self._cache_timestamp and (datetime.now() - self._cache_timestamp).seconds < 10:
            return self._stats_cache
        
        stats = {
            'system': self._get_latest_system_stats(),
            'api': self.get_api_stats(),
            'database': self.get_query_stats(),
            'uptime': self._get_uptime()
        }
        
        # 캐시 업데이트
        self._stats_cache = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def _get_latest_system_stats(self) -> Dict[str, Any]:
        """최신 시스템 통계"""
        if not self.system_metrics:
            return {}
        
        latest = self.system_metrics[-1]
        return {
            'cpu_percent': latest['cpu']['percent'],
            'memory_percent': latest['memory']['percent'],
            'disk_percent': latest['disk']['percent'],
            'process_memory_mb': latest['process']['memory_info']['rss'] / 1024 / 1024
        }
    
    def _get_uptime(self) -> float:
        """프로세스 업타임 (초)"""
        try:
            create_time = self.process.create_time()
            return time.time() - create_time
        except:
            return 0
    
    def clear_old_metrics(self, days: int = 30):
        """오래된 메트릭 정리"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # TODO: 데이터베이스에서 오래된 메트릭 삭제
        
        logger.info(f"{days}일 이상 된 메트릭 정리 완료")

# ============================================================================
# 전역 인스턴스 및 헬퍼 함수
# ============================================================================

_monitor_instance = None

def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 인스턴스 반환"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance

# 편의 함수들
def track_api_call(api_name: str, endpoint: str) -> APICallMetric:
    """API 호출 추적"""
    return get_performance_monitor().track_api_call(api_name, endpoint)

def track_query(query: str) -> QueryMetric:
    """쿼리 추적"""
    return get_performance_monitor().track_query(query)

def measure_time(name: str):
    """실행 시간 측정 데코레이터"""
    return get_performance_monitor().measure_time(name)

def track_user_activity(user_id: str, action: str, metadata: Optional[Dict] = None):
    """사용자 활동 추적"""
    get_performance_monitor().track_user_activity(user_id, action, metadata)

# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    # 모니터 테스트
    monitor = get_performance_monitor()
    
    # 모니터링 시작
    monitor.start_monitoring()
    
    # 시스템 메트릭 테스트
    print("시스템 메트릭:", monitor.collect_system_metrics())
    
    # API 호출 테스트
    api_metric = monitor.track_api_call("test_api", "/test/endpoint")
    time.sleep(0.1)
    monitor.complete_api_call(api_metric, success=True, response_size=1024)
    
    # 쿼리 테스트
    query_metric = monitor.track_query("SELECT * FROM test")
    time.sleep(0.05)
    monitor.complete_query(query_metric, success=True, rows_affected=10)
    
    # 통계 확인
    print("\n현재 통계:", monitor.get_current_stats())
    
    # 리포트 생성
    report = monitor.generate_report(period=timedelta(minutes=5))
    print("\n성능 리포트:")
    print(f"기간: {report.period_start} ~ {report.period_end}")
    print(f"요약: {report.summary}")
    print(f"권장사항: {report.recommendations}")
    
    # 5초 대기
    time.sleep(5)
    
    # 모니터링 중지
    monitor.stop_monitoring()
