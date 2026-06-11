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
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
            }
        }
        logger.info(f"[LLMEngine] Starting response generation. URL: {url}, Model: {self.model}, Prompt length: {len(prompt)} chars")
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"[LLMEngine] Sending POST request to Ollama...")
                async with session.post(url, json=payload) as resp:
                    logger.info(f"[LLMEngine] Ollama response status: {resp.status}")
                    if resp.status != 200:
                        logger.error(f"[LLMEngine] Ollama returned error status: {resp.status}")
                        return

                    token_count = 0
                    async for line in resp.content:
                        if self._abort_event.is_set():
                            logger.info("[LLMEngine] LLM generation aborted (barge-in event set).")
                            break

                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                token = data.get("response", "")
                                
                                if token:
                                    token_count += 1
                                    logger.info(f"[LLMEngine] Streamed token #{token_count}: '{token}'")
                                    yield token
                                    
                                if data.get("done", False):
                                    logger.info(f"[LLMEngine] Ollama generation completed. Total tokens: {token_count}")
                                    break
                            except json.JSONDecodeError as jde:
                                logger.warning(f"[LLMEngine] JSON decode error for line '{line}': {jde}")
                                pass

        except asyncio.CancelledError:
            logger.info("[LLMEngine] LLM task cancelled.")
            raise
        except Exception as e:
            logger.error(f"[LLMEngine] Error during Ollama streaming generation: {e}", exc_info=True)

    def abort(self):
        """Abort current generation (used for barge-in)."""
        self._abort_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
