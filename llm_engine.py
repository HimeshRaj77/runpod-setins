"""LLM Engine for Orchestrating Ollama streaming and TTS chunking."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b"):
        self.host = host
        self.model = model
        self._current_task: Optional[asyncio.Task] = None
        self._abort_event = asyncio.Event()

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate response from Ollama, streaming tokens continuously.
        """
        self._abort_event.clear()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                    }
                }
                async with session.post(f"{self.host}/api/generate", json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama error: {resp.status}")
                        return

                    async for line in resp.content:
                        if self._abort_event.is_set():
                            logger.info("LLM generation aborted (barge-in).")
                            break

                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                token = data.get("response", "")
                                
                                if token:
                                    yield token
                                    
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                pass

        except asyncio.CancelledError:
            logger.info("LLM task was cancelled (barge-in).")
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")

    def abort(self):
        """Abort current generation (used for barge-in)."""
        self._abort_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
