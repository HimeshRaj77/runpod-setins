"""Metrics collection for monitoring system performance."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)


@dataclass
class GpuBatchMetrics:
    """Metrics for a single GPU batch inference."""

    batch_size: int
    gpu_latency_seconds: float
    queue_wait_seconds: float
    total_latency_seconds: float
    audio_duration_seconds: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricsCollector:
    """Collect and aggregate metrics."""

    max_samples: int = 1000
    retention_seconds: int = 3600

    # Batch metrics
    batch_metrics: Deque[GpuBatchMetrics] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    # Counters
    total_batches_processed: int = 0
    total_transcripts_generated: int = 0
    total_audio_seconds_processed: float = 0.0
    total_bytes_processed: int = 0
    total_errors: int = 0

    # Connection events
    total_connections: int = 0
    total_disconnections: int = 0

    def record_batch(self, metrics: GpuBatchMetrics) -> None:
        """Record batch inference metrics."""
        self.batch_metrics.append(metrics)
        self.total_batches_processed += 1
        self.total_audio_seconds_processed += metrics.audio_duration_seconds

    def record_transcripts(self, count: int) -> None:
        """Record number of transcripts generated."""
        self.total_transcripts_generated += count

    def record_bytes(self, count: int) -> None:
        """Record bytes processed."""
        self.total_bytes_processed += count

    def record_error(self) -> None:
        """Record an error."""
        self.total_errors += 1

    def record_connection(self) -> None:
        """Record a new connection."""
        self.total_connections += 1

    def record_disconnection(self) -> None:
        """Record a disconnection."""
        self.total_disconnections += 1

    def get_avg_batch_size(self) -> float:
        """Get average batch size."""
        if not self.batch_metrics:
            return 0.0
        return sum(m.batch_size for m in self.batch_metrics) / len(self.batch_metrics)

    def get_avg_gpu_latency(self) -> float:
        """Get average GPU latency."""
        if not self.batch_metrics:
            return 0.0
        return sum(m.gpu_latency_seconds for m in self.batch_metrics) / len(
            self.batch_metrics
        )

    def get_avg_queue_wait(self) -> float:
        """Get average queue wait time."""
        if not self.batch_metrics:
            return 0.0
        return sum(m.queue_wait_seconds for m in self.batch_metrics) / len(
            self.batch_metrics
        )

    def get_avg_e2e_latency(self) -> float:
        """Get average end-to-end latency."""
        if not self.batch_metrics:
            return 0.0
        return sum(m.total_latency_seconds for m in self.batch_metrics) / len(
            self.batch_metrics
        )

    def get_p99_gpu_latency(self) -> float:
        """Get P99 GPU latency."""
        if not self.batch_metrics:
            return 0.0
        sorted_latencies = sorted(m.gpu_latency_seconds for m in self.batch_metrics)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def get_stats(self) -> dict:
        """Get current metrics snapshot."""
        return {
            "batches_processed": self.total_batches_processed,
            "transcripts_generated": self.total_transcripts_generated,
            "audio_seconds_processed": round(self.total_audio_seconds_processed, 2),
            "bytes_processed": self.total_bytes_processed,
            "errors": self.total_errors,
            "total_connections": self.total_connections,
            "total_disconnections": self.total_disconnections,
            "avg_batch_size": round(self.get_avg_batch_size(), 2),
            "avg_gpu_latency_ms": round(self.get_avg_gpu_latency() * 1000, 2),
            "avg_queue_wait_ms": round(self.get_avg_queue_wait() * 1000, 2),
            "avg_e2e_latency_ms": round(self.get_avg_e2e_latency() * 1000, 2),
            "p99_gpu_latency_ms": round(self.get_p99_gpu_latency() * 1000, 2),
            "recent_batches": len(self.batch_metrics),
        }
