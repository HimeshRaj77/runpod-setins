"""Batch manager for dynamic batching of audio samples."""

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from audio_queue import AudioItem

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """An item in a processing batch."""

    connection_id: str
    audio_data: np.ndarray
    sequence_number: int
    enqueued_at: float
    is_final: bool = False


class BatchManager:
    """Manage dynamic batching of audio for GPU inference."""

    def __init__(
        self,
        max_batch_size: int = 16,
        max_batch_wait_ms: float = 50,
        min_batch_wait_ms: float = 10,
    ):
        """
        Initialize batch manager.
        """
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms / 1000.0
        self.min_batch_wait_ms = min_batch_wait_ms / 1000.0
        self.batch_counter = 0

    def build_batch(self, items: List[AudioItem]) -> List[BatchItem]:
        """
        Build a batch from audio items.

        Args:
            items: List of AudioItem from queue

        Returns:
            List of BatchItem
        """
        batch_items = []

        for item in items:
            batch_item = BatchItem(
                connection_id=item.connection_id,
                audio_data=np.frombuffer(item.audio_data, dtype=np.int16),
                sequence_number=item.sequence_number,
                enqueued_at=item.timestamp,
                is_final=getattr(item, 'is_final', False),
            )
            batch_items.append(batch_item)

        self.batch_counter += 1
        return batch_items

    def get_batch_id(self) -> int:
        """Get current batch ID."""
        return self.batch_counter

    def get_stats(self) -> dict:
        """Get batch manager statistics."""
        return {
            "batches_created": self.batch_counter,
            "max_batch_size": self.max_batch_size,
            "max_batch_wait_ms": self.max_batch_wait_ms * 1000,
            "min_batch_wait_ms": self.min_batch_wait_ms * 1000,
        }
