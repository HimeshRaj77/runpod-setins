"""TTS Engine for streaming audio generation."""

import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self._abort_event = asyncio.Event()

    async def generate_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Mock TTS generation. Yields dummy audio bytes for the given text.
        In a real implementation, this would call Cartesia, ElevenLabs, or XTTS.
        """
        self._abort_event.clear()
        
        logger.info(f"TTS Start generating for: '{text}'")
        
        # Simulate TTS processing time
        await asyncio.sleep(0.1)
        
        # Mock audio generation: 5 chunks per sentence
        for i in range(5):
            if self._abort_event.is_set():
                logger.info("TTS generation aborted (barge-in).")
                break
                
            # Dummy PCM audio chunk (zeros)
            chunk = (b'\x00' * 3200) # 0.1s of 16kHz 16-bit audio
            yield chunk
            
            # Simulate real-time streaming delay
            await asyncio.sleep(0.1)

    def abort(self):
        """Abort current TTS generation (used for barge-in)."""
        self._abort_event.set()
