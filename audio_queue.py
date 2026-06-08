"""Audio queue for managing incoming audio chunks."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioItem:
    """An audio chunk queued for processing."""

    connection_id: str
    audio_data: bytes
    timestamp: float
    sequence_number: int
    is_final: bool = False

    def age_seconds(self) -> float:
        """Get age of item in seconds."""
        return time.time() - self.timestamp


class AudioQueue:
    """Queue for managing audio chunks with backpressure."""

    def __init__(self, max_depth: int = 1000, timeout_seconds: float = 300):
        """Initialize audio queue."""
        self.max_depth = max_depth
        self.timeout_seconds = timeout_seconds
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_depth)
        self.sequence_counter = 0
        self.lock = asyncio.Lock()
        self.dropped_items = 0
        self.total_items = 0

    async def enqueue(self, connection_id: str, audio_data: bytes, is_final: bool = False) -> bool:
        """
        Enqueue audio data.

        Returns:
            True if successfully enqueued, False if queue is full.
        """
        try:
            async with self.lock:
                seq = self.sequence_counter
                self.sequence_counter += 1

            item = AudioItem(
                connection_id=connection_id,
                audio_data=audio_data,
                timestamp=time.time(),
                sequence_number=seq,
                is_final=is_final,
            )

            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(self.queue.put(item), timeout=1.0)
                self.total_items += 1
                return True
            except asyncio.TimeoutError:
                self.dropped_items += 1
                logger.warning(
                    f"Failed to enqueue audio from {connection_id}, dropped {self.dropped_items} items"
                )
                return False

        except Exception as e:
            logger.error(f"Error enqueuing audio: {e}")
            return False

    async def dequeue_batch(
        self,
        max_batch_size: int = 16,
        wait_ms: float = 50,
        timeout: float = 30,
    ) -> list[AudioItem]:
        """
        Dequeue a batch of audio items.

        Args:
            max_batch_size: Maximum number of items in batch
            wait_ms: Maximum time to wait for batch to fill (milliseconds)
            timeout: Overall timeout for operation

        Returns:
            List of audio items (may be smaller than max_batch_size)
        """
        batch = []
        wait_seconds = wait_ms / 1000.0

        try:
            # Get first item (blocking)
            first_item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            batch.append(first_item)

            # Try to collect more items up to max_batch_size or wait_ms
            start = time.time()
            while len(batch) < max_batch_size and (time.time() - start) < wait_seconds:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=(wait_seconds - (time.time() - start))
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            return batch

        except asyncio.TimeoutError:
            logger.debug(f"Queue timeout, returning batch of {len(batch)} items")
            return batch
        except Exception as e:
            logger.error(f"Error dequeuing batch: {e}")
            return batch

    def get_depth(self) -> int:
        """Get current queue depth."""
        return self.queue.qsize()

    def get_max_depth(self) -> int:
        """Get maximum queue depth."""
        return self.max_depth

    def is_full(self, threshold: float = 0.9) -> bool:
        """Check if queue is approaching capacity."""
        depth = self.queue.qsize()
        return depth >= (self.max_depth * threshold)

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "current_depth": self.queue.qsize(),
            "max_depth": self.max_depth,
            "total_items": self.total_items,
            "dropped_items": self.dropped_items,
            "utilization": self.queue.qsize() / self.max_depth,
        }
