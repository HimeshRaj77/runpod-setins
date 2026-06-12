"""Connection registry to track WebSocket connections per worker."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
from websockets.asyncio.server import ServerConnection

logger = logging.getLogger(__name__)


from llm_engine import LLMEngine
from tts_engine import TTSEngine

@dataclass
class LatencyStats:
    """Track latency statistics."""

    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def add(self, latency: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency)

    def get_average(self) -> float:
        """Get average latency."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def get_percentile(self, p: float) -> float:
        """Get percentile latency (p in 0-100)."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * (p / 100.0))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


@dataclass
class ConnectionState:
    """Track state for a single WebSocket connection."""

    connection_id: str
    websocket: ServerConnection
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    # Audio state
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Engines
    llm_engine: LLMEngine = field(default_factory=LLMEngine)
    tts_engine: TTSEngine = field(default_factory=TTSEngine)
    llm_task: Optional[asyncio.Task] = None
    llm_is_active: bool = False   # True while LLM+TTS are generating a response
    llm_enabled: bool = True      # Whether LLM processing is enabled
    llm_mode: str = "conversational" # Mode for LLM processing

    # Packet ordering / deduplication (driven by the 32-byte header seq_num)
    last_seq_num: int = -1          # last successfully processed sequence number
    client_session_id: str = ""     # UUID from header bytes 0-15
    expect_header: bool = True      # Whether we expect a 32-byte header
    audio_format: str = "float32"   # Audio format: "float32" or "int16"

    # Partial transcript deduplication — track the last partial we sent so we
    # can emit only the NEW suffix on each sliding-window fire.
    last_partial_text: str = ""

    # Statistics
    audio_chunks_received: int = 0
    audio_seconds_received: float = 0.0
    transcripts_sent: int = 0
    latency_stats: LatencyStats = field(default_factory=LatencyStats)
    bytes_received: int = 0

    async def append_audio(self, data: bytes) -> None:
        """Append audio data to buffer (thread-safe)."""
        async with self.buffer_lock:
            try:
                # Convert bytes to int16 array
                audio_data = np.frombuffer(data, dtype=np.int16)
                self.audio_buffer = np.append(self.audio_buffer, audio_data)

                self.audio_chunks_received += 1
                self.bytes_received += len(data)
                self.audio_seconds_received += len(audio_data) / 16000.0  # 16kHz

            except Exception as e:
                logger.error(
                    f"Error appending audio to buffer for connection {self.connection_id}: {e}"
                )

    async def get_and_clear_audio(self) -> np.ndarray:
        """Get audio buffer and clear it (thread-safe)."""
        async with self.buffer_lock:
            data = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.int16)
            return data

    def update_latency(self, latency_seconds: float) -> None:
        """Record a latency measurement."""
        self.latency_stats.add(latency_seconds)

    def get_avg_latency(self) -> float:
        """Get average latency."""
        return self.latency_stats.get_average()

    def is_alive(self, timeout: float) -> bool:
        """Check if connection is still alive."""
        return (time.time() - self.last_seen) < timeout

    def update_heartbeat(self) -> None:
        """Update last heartbeat time."""
        self.last_heartbeat = time.time()

    def update_seen(self) -> None:
        """Update last seen time."""
        self.last_seen = time.time()


class ConnectionRegistry:
    """Registry to manage all active WebSocket connections."""

    def __init__(self):
        """Initialize connection registry."""
        self.connections: Dict[str, ConnectionState] = {}
        self.lock = asyncio.Lock()

    async def register(self, connection_id: str, websocket: ServerConnection) -> ConnectionState:
        """Register a new connection."""
        async with self.lock:
            if connection_id in self.connections:
                logger.warning(f"Connection {connection_id} already registered, replacing")
                await self._cleanup_connection(self.connections[connection_id])

            conn_state = ConnectionState(connection_id=connection_id, websocket=websocket)
            self.connections[connection_id] = conn_state
            logger.info(f"Registered connection {connection_id}, total: {len(self.connections)}")
            return conn_state

    async def unregister(self, connection_id: str) -> None:
        """Unregister a connection."""
        async with self.lock:
            if connection_id in self.connections:
                conn_state = self.connections[connection_id]
                await self._cleanup_connection(conn_state)
                del self.connections[connection_id]
                logger.info(f"Unregistered connection {connection_id}, total: {len(self.connections)}")

    async def _cleanup_connection(self, conn_state: ConnectionState) -> None:
        """Clean up a connection's resources."""
        try:
            async with conn_state.buffer_lock:
                conn_state.audio_buffer = np.array([], dtype=np.int16)
        except Exception as e:
            logger.error(f"Error cleaning up connection {conn_state.connection_id}: {e}")

    def get_connection(self, connection_id: str) -> Optional[ConnectionState]:
        """Get a connection state."""
        return self.connections.get(connection_id)

    def get_all_connections(self) -> list[ConnectionState]:
        """Get all active connections."""
        return list(self.connections.values())

    def get_active_connections(self, timeout: float) -> list[ConnectionState]:
        """Get all connections that are still alive."""
        return [conn for conn in self.connections.values() if conn.is_alive(timeout)]

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)

    async def prune_dead_connections(self, timeout: float) -> int:
        """Remove connections that haven't been seen recently."""
        async with self.lock:
            dead_connections = [
                conn_id
                for conn_id, conn in self.connections.items()
                if not conn.is_alive(timeout)
            ]

            for conn_id in dead_connections:
                conn_state = self.connections[conn_id]
                await self._cleanup_connection(conn_state)
                del self.connections[conn_id]
                logger.info(
                    f"Pruned dead connection {conn_id}, total: {len(self.connections)}"
                )

            return len(dead_connections)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        connections = self.get_all_connections()
        total_transcripts = sum(c.transcripts_sent for c in connections)
        total_audio_seconds = sum(c.audio_seconds_received for c in connections)
        avg_latency = (
            sum(c.get_avg_latency() for c in connections) / len(connections)
            if connections
            else 0
        )

        return {
            "active_connections": len(connections),
            "total_transcripts": total_transcripts,
            "total_audio_seconds": total_audio_seconds,
            "avg_latency": avg_latency,
            "connections": [
                {
                    "id": c.connection_id,
                    "uptime_seconds": time.time() - c.created_at,
                    "audio_seconds_received": c.audio_seconds_received,
                    "transcripts_sent": c.transcripts_sent,
                    "avg_latency": c.get_avg_latency(),
                }
                for c in connections
            ],
        }
