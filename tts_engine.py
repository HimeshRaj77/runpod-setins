"""TTS Engine for streaming audio generation."""

import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self._abort_event = asyncio.Event()

    async def generate_audio(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Mock TTS generation. Yields dummy audio bytes as text tokens are streamed.
        In a real implementation, this would stream text to Cartesia, ElevenLabs, or XTTS via WebSockets.
        """
        self._abort_event.clear()
        
        logger.info("TTS Start streaming audio from text stream...")
        
        # Simulate initial TTS connection/processing time
        await asyncio.sleep(0.1)
        
        async for token in text_stream:
            if self._abort_event.is_set():
                logger.info("TTS generation aborted (barge-in).")
                break
                
            # For each token received, yield some mock audio
            # Dummy PCM audio chunk (zeros)
            chunk = (b'\x00' * 3200) # 0.1s of 16kHz 16-bit audio
            yield chunk
            
            # Simulate real-time streaming delay per token
            await asyncio.sleep(0.05)

    def abort(self):
        """Abort current TTS generation (used for barge-in)."""
        self._abort_event.set()
