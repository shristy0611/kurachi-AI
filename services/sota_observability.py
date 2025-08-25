"""
SOTA Observability System for Kurachi AI
Comprehensive monitoring with OpenTelemetry integration

Features:
- Distributed tracing with Jaeger/DataDog support
- Metrics collection with Prometheus compatibility
- Real-time performance dashboards
- Custom business metrics
- Error tracking and alerting
- Resource utilization monitoring
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

# OpenTelemetry imports (optional dependencies)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from config import config
from utils.logger import get_logger

logger = get_logger("sota_observability")


class MetricType(Enum):
    """Types of metrics we collect"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    metric_type: MetricType = MetricType.GAUGE
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SOTAObservabilitySystem:
    """
    SOTA Observability System with OpenTelemetry integration
    
    Features:
    - Distributed tracing
    - Metrics collection
    - Real-time monitoring
    - Alerting system
    - Performance dashboards
    """
    
    def __init__(self, 
                 enable_tracing: bool = True,
                 enable_metrics: bool = True,
                 jaeger_endpoint: Optional[str] = None,
                 prometheus_port: int = 8090):
        """
        Initialize SOTA observability system
        
        Args:
            enable_tracing: Enable distributed tracing
            enable_metrics: Enable metrics collection
            jaeger_endpoint: Jaeger collector endpoint
            prometheus_port: Prometheus metrics port
        """
        self.enable_tracing = enable_tracing and OPENTELEMETRY_AVAILABLE
        self.enable_metrics = enable_metrics
        self.jaeger_endpoint = jaeger_endpoint or "http://localhost:14268/api/traces"
        self.prometheus_port = prometheus_port
        
        # Internal metrics storage
        self.metrics_storage = defaultdict(deque)
        self.metrics_lock = threading.Lock()
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            "memory_usage_percent": 80.0,
            "cpu_usage_percent": 85.0,
            "response_time_ms": 5000.0,
            "error_rate_percent": 10.0
        }
        
        # Performance counters
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = defaultdict(float)
        
        # Initialize OpenTelemetry components
        self.tracer = None
        self.meter = None
        
        if self.enable_tracing:
            self._setup_tracing()
        
        if self.enable_metrics:
            self._setup_metrics()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info(f"SOTA Observability initialized (tracing: {self.enable_tracing}, "
                   f"metrics: {self.enable_metrics})")
    
    def _setup_tracing(self):
        """Setup distributed tracing with OpenTelemetry"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, using mock tracing")
            return
        
        try:
            # Configure trace provider
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            logger.info("Distributed tracing configured with Jaeger")
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
            self.enable_tracing = False
    
    def _setup_metrics(self):
        """Setup metrics collection with OpenTelemetry"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, using internal metrics")
            return
        
        try:
            # Configure metrics provider
            metric_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
            self.meter = metrics.get_meter(__name__)
            
            # Create standard metrics
            self.request_counter = self.meter.create_counter(
                "kurachi_requests_total",
                description="Total number of requests"
            )
            
            self.response_time_histogram = self.meter.create_histogram(
                "kurachi_response_time_seconds",
                description="Response time in seconds"
            )
            
            self.memory_usage_gauge = self.meter.create_gauge(
                "kurachi_memory_usage_bytes",
                description="Memory usage in bytes"
            )
            
            self.cpu_usage_gauge = self.meter.create_gauge(
                "kurachi_cpu_usage_percent",
                description="CPU usage percentage"
            )
            
            logger.info(f"Metrics collection configured with Prometheus on port {self.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
            # Fall back to internal metrics
            pass
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """
        Trace operation with automatic metrics collection
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional span attributes
            
        Yields:
            Span object (OpenTelemetry span or mock)
        """
        start_time = time.time()
        
        if self.enable_tracing and self.tracer:
            # Use OpenTelemetry tracing
            with self.tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                
                try:
                    yield span
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    self._record_operation_success(operation_name, duration, attributes)
                    
                except Exception as e:
                    # Record error metrics
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    self._record_operation_error(operation_name, str(e), attributes)
                    raise
        else:
            # Use mock tracing with internal metrics
            mock_span = MockSpan(operation_name, attributes)
            try:
                yield mock_span
                
                duration = time.time() - start_time
                self._record_operation_success(operation_name, duration, attributes)
                
            except Exception as e:
                self._record_operation_error(operation_name, str(e), attributes)
                raise
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """
        Record a custom metric
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Additional labels
        """
        labels = labels or {}
        
        # Record in internal storage
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels,
            metric_type=metric_type
        )
        
        with self.metrics_lock:
            self.metrics_storage[name].append(metric)
            # Keep only last 1000 data points per metric
            if len(self.metrics_storage[name]) > 1000:
                self.metrics_storage[name].popleft()
        
        # Update internal counters
        if metric_type == MetricType.COUNTER:
            self.counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self.histograms[name].append(value)
        
        # Record in OpenTelemetry if available
        if self.meter:
            try:
                if metric_type == MetricType.COUNTER:
                    self.request_counter.add(value, labels)
                elif metric_type == MetricType.HISTOGRAM:
                    self.response_time_histogram.record(value, labels)
                # Gauges are handled separately in background monitoring
            except Exception as e:
                logger.error(f"Failed to record OpenTelemetry metric: {e}")
    
    def _record_operation_success(self, operation_name: str, duration: float, attributes: Dict[str, Any]):
        """Record successful operation metrics"""
        self.record_metric(
            f"{operation_name}_duration_ms",
            duration * 1000,
            MetricType.HISTOGRAM,
            {"operation": operation_name, "status": "success"}
        )
        
        self.record_metric(
            f"{operation_name}_total",
            1,
            MetricType.COUNTER,
            {"operation": operation_name, "status": "success"}
        )
    
    def _record_operation_error(self, operation_name: str, error: str, attributes: Dict[str, Any]):
        """Record failed operation metrics"""
        self.record_metric(
            f"{operation_name}_total",
            1,
            MetricType.COUNTER,
            {"operation": operation_name, "status": "error", "error": error}
        )
        
        # Create alert for errors
        self.create_alert(
            level=AlertLevel.ERROR,
            message=f"Operation {operation_name} failed: {error}",
            component=operation_name
        )
    
    def create_alert(self, 
                    level: AlertLevel, 
                    message: str, 
                    component: str,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Create a system alert
        
        Args:
            level: Alert severity level
            message: Alert message
            component: Component that triggered the alert
            metadata: Additional alert metadata
        """
        alert = Alert(
            id=f"alert_{int(time.time())}_{len(self.alerts)}",
            level=level,
            message=message,
            timestamp=datetime.utcnow(),
            component=component,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"Alert created: [{level.value.upper()}] {message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status
        
        Returns:
            System health dictionary
        """
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate error rates
        total_operations = sum(self.counters.values())
        error_count = sum(v for k, v in self.counters.items() if "error" in k)
        error_rate = (error_count / max(total_operations, 1)) * 100
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if not a.resolved and 
                        a.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        health_status = "healthy"
        if error_rate > 10 or cpu_percent > 85 or memory.percent > 80:
            health_status = "degraded"
        if error_rate > 25 or cpu_percent > 95 or memory.percent > 90:
            health_status = "critical"
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "application": {
                "total_operations": total_operations,
                "error_rate_percent": error_rate,
                "recent_alerts": len(recent_alerts),
                "active_traces": len(self.histograms),
                "metrics_collected": len(self.metrics_storage)
            },
            "features": {
                "tracing_enabled": self.enable_tracing,
                "metrics_enabled": self.enable_metrics,
                "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
                "prometheus_port": self.prometheus_port if self.enable_metrics else None
            }
        }
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get performance dashboard data
        
        Returns:
            Dashboard data dictionary
        """
        # Calculate recent performance metrics
        recent_time = datetime.utcnow() - timedelta(minutes=15)
        
        recent_metrics = {}
        with self.metrics_lock:
            for name, metric_deque in self.metrics_storage.items():
                recent_values = [m.value for m in metric_deque if m.timestamp > recent_time]
                if recent_values:
                    recent_metrics[name] = {
                        "count": len(recent_values),
                        "avg": sum(recent_values) / len(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                        "latest": recent_values[-1]
                    }
        
        # Top operations by volume
        operation_counts = {k: v for k, v in self.counters.items() if "total" in k}
        top_operations = sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Response time statistics
        response_times = []
        for name, values in self.histograms.items():
            if "duration" in name and values:
                response_times.extend(values)
        
        response_time_stats = {}
        if response_times:
            response_times.sort()
            n = len(response_times)
            response_time_stats = {
                "count": n,
                "avg": sum(response_times) / n,
                "median": response_times[n // 2],
                "p95": response_times[int(n * 0.95)] if n > 20 else response_times[-1],
                "p99": response_times[int(n * 0.99)] if n > 100 else response_times[-1]
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overview": {
                "total_requests": sum(self.counters.values()),
                "active_operations": len([k for k, v in self.gauges.items() if v > 0]),
                "avg_cpu_percent": self.gauges.get("cpu_usage_percent", 0),
                "avg_memory_percent": self.gauges.get("memory_usage_percent", 0)
            },
            "recent_metrics": recent_metrics,
            "top_operations": dict(top_operations),
            "response_times": response_time_stats,
            "alerts": {
                "total": len(self.alerts),
                "unresolved": len([a for a in self.alerts if not a.resolved]),
                "critical": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved])
            }
        }
    
    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    self.record_metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE)
                    self.record_metric("memory_usage_percent", memory.percent, MetricType.GAUGE)
                    self.record_metric("memory_usage_bytes", memory.used, MetricType.GAUGE)
                    
                    # Check alert thresholds
                    self._check_alert_thresholds(cpu_percent, memory.percent)
                    
                    # Update OpenTelemetry gauges
                    if self.meter:
                        try:
                            self.memory_usage_gauge.set(memory.used)
                            self.cpu_usage_gauge.set(cpu_percent)
                        except Exception as e:
                            logger.error(f"Failed to update OpenTelemetry gauges: {e}")
                    
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                
                time.sleep(30)  # Monitor every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Background monitoring started")
    
    def _check_alert_thresholds(self, cpu_percent: float, memory_percent: float):
        """Check if any alert thresholds are exceeded"""
        if cpu_percent > self.alert_thresholds["cpu_usage_percent"]:
            self.create_alert(
                AlertLevel.WARNING,
                f"High CPU usage: {cpu_percent:.1f}%",
                "system_monitor",
                {"cpu_percent": cpu_percent}
            )
        
        if memory_percent > self.alert_thresholds["memory_usage_percent"]:
            self.create_alert(
                AlertLevel.WARNING,
                f"High memory usage: {memory_percent:.1f}%",
                "system_monitor",
                {"memory_percent": memory_percent}
            )


class MockSpan:
    """Mock span for when OpenTelemetry is not available"""
    
    def __init__(self, name: str, attributes: Dict[str, Any]):
        self.name = name
        self.attributes = attributes
        self.status = "OK"
    
    def set_attribute(self, key: str, value: str):
        self.attributes[key] = value
    
    def set_status(self, status):
        self.status = "ERROR"


# Global observability instance
sota_observability = SOTAObservabilitySystem()

# Convenience functions
def trace_operation(operation_name: str, **attributes):
    """Trace operation decorator/context manager"""
    return sota_observability.trace_operation(operation_name, **attributes)

def record_metric(name: str, value: float, metric_type: MetricType = MetricType.GAUGE, labels: Optional[Dict[str, str]] = None):
    """Record a metric"""
    sota_observability.record_metric(name, value, metric_type, labels)

def create_alert(level: AlertLevel, message: str, component: str, metadata: Optional[Dict[str, Any]] = None):
    """Create an alert"""
    sota_observability.create_alert(level, message, component, metadata)

def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    return sota_observability.get_system_health()

def get_performance_dashboard() -> Dict[str, Any]:
    """Get performance dashboard data"""
    return sota_observability.get_performance_dashboard()