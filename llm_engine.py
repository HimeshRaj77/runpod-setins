"""LLM Engine for Orchestrating Ollama streaming and TTS chunking."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import aiohttp
from config import get_config

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b"):
        self.config = get_config()
        self.provider = self.config.LLM_PROVIDER.lower()
        self.host = host
        self.model = model
        self.groq_api_key = self.config.GROQ_API_KEY
        self.groq_model = self.config.GROQ_MODEL
        self._current_task: Optional[asyncio.Task] = None
        self._abort_event = asyncio.Event()

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate response from Ollama or Groq, streaming tokens continuously.
        """
        self._abort_event.clear()
        if self.provider == "groq":
            async for token in self._generate_groq(prompt, system_prompt):
                yield token
            return
        url = f"{self.host}/api/generate"
        system = system_prompt or "You are a helpful, conversational AI voice assistant. Please provide concise, natural-sounding spoken responses. Do not use markdown, emojis, or lists. Do not repeat yourself."
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": 0.6,
                "repeat_penalty": 1.15,
                "top_p": 0.9,
                "top_k": 40
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

    async def _generate_groq(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        system = system_prompt or "You are a helpful, conversational AI voice assistant. Please provide concise, natural-sounding spoken responses. Do not use markdown, emojis, or lists. Do not repeat yourself."
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": 0.6,
            "top_p": 0.9
        }
        logger.info(f"[LLMEngine] Starting Groq generation. Model: {self.groq_model}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"[LLMEngine] Groq returned error: {resp.status}")
                        return
                    async for line in resp.content:
                        if self._abort_event.is_set():
                            logger.info("[LLMEngine] Groq generation aborted (barge-in event set).")
                            break
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                if data_str == "[DONE]":
                                    logger.info("[LLMEngine] Groq generation completed.")
                                    break
                                try:
                                    data = json.loads(data_str)
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        token = delta["content"]
                                        if token:
                                            yield token
                                except Exception as e:
                                    pass
        except asyncio.CancelledError:
            logger.info("[LLMEngine] Groq LLM task cancelled.")
            raise
        except Exception as e:
            logger.error(f"[LLMEngine] Error during Groq streaming generation: {e}", exc_info=True)
