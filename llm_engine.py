"""LLM Engine for Orchestrating Ollama streaming and TTS chunking."""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b"):
        self.host = host
        self.model = model
        self._current_task: Optional[asyncio.Task] = None
        self._abort_event = asyncio.Event()

    async def generate_response(self, prompt: str, on_sentence: Callable[[str], None], on_token: Optional[Callable[[str], None]] = None) -> str:
        """
        Generate response from Ollama, stream tokens, buffer into sentences,
        and trigger callbacks.
        """
        self._abort_event.clear()
        full_text = ""
        current_sentence = ""
        
        # Basic sentence boundary regex (periods, exclamation, question marks)
        # We also want to capture the boundary character.
        boundary_pattern = re.compile(r'([.!?]+)\s*')

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
                        return ""

                    async for line in resp.content:
                        if self._abort_event.is_set():
                            logger.info("LLM generation aborted (barge-in).")
                            break

                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                token = data.get("response", "")
                                
                                if on_token and token:
                                    on_token(token)
                                    
                                full_text += token
                                current_sentence += token
                                
                                # Check for sentence boundary
                                match = boundary_pattern.search(current_sentence)
                                if match:
                                    # Split at the boundary
                                    end_idx = match.end()
                                    sentence = current_sentence[:end_idx].strip()
                                    current_sentence = current_sentence[end_idx:]
                                    
                                    if sentence:
                                        # Trigger TTS for this sentence
                                        on_sentence(sentence)
                                        
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                pass

            # Flush remaining text
            if current_sentence.strip() and not self._abort_event.is_set():
                on_sentence(current_sentence.strip())

        except asyncio.CancelledError:
            logger.info("LLM task was cancelled (barge-in).")
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")

        return full_text

    def abort(self):
        """Abort current generation (used for barge-in)."""
        self._abort_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
