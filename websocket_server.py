"""WebSocket server handler and background processing for RunPod STT."""

import asyncio
import json
import logging
from typing import Dict, Any

from fastapi import WebSocket, WebSocketDisconnect

from config import Config
from connection_registry import ConnectionRegistry
from audio_queue import AudioQueue
from batch_manager import BatchManager
from vad_engine import VadEngine
from whisper_engine import WhisperEngine
import numpy as np

logger = logging.getLogger(__name__)

class STTServer:
    def __init__(self, config: Config):
        self.config = config
        self.registry = ConnectionRegistry()
        self.audio_queue = AudioQueue(
            max_depth=config.MAX_QUEUE_DEPTH, 
            timeout_seconds=config.QUEUE_TIMEOUT_SECONDS
        )
        self.batch_manager = BatchManager(
            max_batch_size=config.MAX_BATCH_SIZE,
            max_batch_wait_ms=config.MAX_BATCH_WAIT_MS,
            min_batch_wait_ms=config.MIN_BATCH_WAIT_MS
        )
        self.vad_engine = VadEngine(
            threshold=config.VAD_THRESHOLD, 
            sample_rate=config.SAMPLE_RATE
        )
        self.whisper_engine = WhisperEngine(
            model_id=config.WHISPER_MODEL,
            device=config.DEVICE,
            dtype=config.DTYPE,
            cache_dir=config.WHISPER_CACHE_DIR,
            language=config.WHISPER_LANGUAGE,
            task=config.WHISPER_TASK
        )
        self.worker_task = None
        self.is_running = False
        
        # Metrics
        self.total_transcripts = 0
        self.total_audio_seconds = 0.0

    async def start(self):
        self.is_running = True
        self.worker_task = asyncio.create_task(self.processing_loop())

    async def stop(self):
        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def processing_loop(self):
        logger.info("Started background processing loop.")
        while self.is_running:
            try:
                # Dequeue a batch based on dynamic batching settings
                items = await self.audio_queue.dequeue_batch(
                    max_batch_size=self.config.MAX_BATCH_SIZE,
                    wait_ms=self.config.MAX_BATCH_WAIT_MS,
                )
                
                if not items:
                    continue

                # Build batch correctly keeping separate connection contexts
                batch_items = self.batch_manager.build_batch(items)
                
                audio_arrays = [item.audio_data for item in batch_items]
                
                # Inference
                transcriptions, latency = self.whisper_engine.transcribe_batch(
                    audio_batch=audio_arrays,
                    sample_rate=self.config.SAMPLE_RATE
                )
                
                # Route responses back
                for batch_item, text in zip(batch_items, transcriptions):
                    conn_id = batch_item.connection_id
                    conn_state = self.registry.get_connection(conn_id)
                    
                    if conn_state and text:
                        response = {
                            "status": "transcribed",
                            "text": text,
                            "is_final": getattr(batch_item, 'is_final', False),
                            "latency_seconds": latency
                        }
                        
                        try:
                            # Send output back to exact connection
                            await conn_state.websocket.send_json(response)
                            conn_state.transcripts_sent += 1
                            conn_state.update_latency(latency)
                            self.total_transcripts += 1
                            self.total_audio_seconds += len(batch_item.audio_data) / self.config.SAMPLE_RATE
                            
                            # Start LLM pipeline
                            await self.run_llm_pipeline(conn_state, text, response["is_final"])

                        except Exception as e:
                            logger.error(f"Failed to send result to connection {conn_id}: {e}")
                            await self.registry.unregister(conn_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def run_llm_pipeline(self, conn_state, text: str, is_final: bool):
        """Run LLM and stream to TTS."""
        # Abort previous tasks for this connection
        conn_state.llm_engine.abort()
        conn_state.tts_engine.abort()
        
        if conn_state.llm_task and not conn_state.llm_task.done():
            conn_state.llm_task.cancel()
            try:
                await conn_state.llm_task
            except asyncio.CancelledError:
                pass

        def on_sentence(sentence):
            # Schedule TTS for this sentence
            asyncio.create_task(self.run_tts_for_sentence(conn_state, sentence))

        def on_token(token):
            # Stream LLM text token back to Node.js server
            async def _send_token():
                try:
                    await conn_state.websocket.send_json({
                        "status": "llm_token",
                        "text": token
                    })
                except Exception as e:
                    logger.error(f"Error sending token: {e}")
            asyncio.create_task(_send_token())

        async def _llm_job():
            try:
                conn_state.llm_engine._current_task = asyncio.current_task()
                # Run LLM generation
                full_text = await conn_state.llm_engine.generate_response(text, on_sentence, on_token)
                if is_final:
                    logger.info(f"Final LLM output: {full_text}")
                    # Optionally send an end of LLM message
                    await conn_state.websocket.send_json({
                        "status": "llm_final",
                        "text": full_text
                    })
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"LLM Error: {e}")

        conn_state.llm_task = asyncio.create_task(_llm_job())

    async def run_tts_for_sentence(self, conn_state, sentence: str):
        """Run TTS and stream audio back."""
        try:
            async for audio_chunk in conn_state.tts_engine.generate_audio(sentence):
                await conn_state.websocket.send_bytes(audio_chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        conn_id = str(id(websocket))
        
        # We need a dummy ServerConnection object with send_json support, but FastAPI WebSocket has send_json natively.
        # Let's adjust registry usage to store the FastAPI WebSocket directly.
        conn_state = await self.registry.register(conn_id, websocket)
        logger.info(f"Worker connected: {conn_id}")
        
        speech_buffer = []
        is_speaking = False
        silence_samples = 0

        try:
            while True:
                data = await websocket.receive_bytes()
                conn_state.update_seen()
                
                # Incoming Int16 PCM
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # Convert to float32 internally inside VAD engine
                # Silero VAD
                speech_int16, _ = self.vad_engine.detect_speech(audio_int16)
                
                if len(speech_int16) > 0:
                    silence_samples = 0
                    # User is speaking - detect barge-in
                    if not is_speaking:
                        is_speaking = True
                        # Abort any ongoing playback (barge-in)
                        conn_state.llm_engine.abort()
                        conn_state.tts_engine.abort()
                        logger.info(f"Barge-in detected for {conn_id}, stopping TTS/LLM")
                        
                    speech_buffer.append(speech_int16)
                    
                    # Sliding window: if buffer > 1 second, send partial transcript
                    total_samples = sum(len(x) for x in speech_buffer)
                    if total_samples >= self.config.SAMPLE_RATE: # 1 second
                        if not self.audio_queue.is_full():
                            await self.audio_queue.enqueue(
                                conn_id, 
                                np.concatenate(speech_buffer).tobytes(),
                                is_final=False
                            )
                else:
                    # Silence detected
                    if is_speaking:
                        silence_samples += len(audio_int16)
                        speech_buffer.append(audio_int16) # keep silence for natural pausing
                        
                        # Wait for VAD_MIN_SILENCE_DURATION before ending utterance
                        if silence_samples >= (self.config.VAD_MIN_SILENCE_DURATION * self.config.SAMPLE_RATE):
                            is_speaking = False
                            # User stopped speaking, send final transcript
                            if len(speech_buffer) > 0:
                                if not self.audio_queue.is_full():
                                    await self.audio_queue.enqueue(
                                        conn_id,
                                        np.concatenate(speech_buffer).tobytes(),
                                        is_final=True
                                    )
                                speech_buffer = [] # Reset buffer
                            silence_samples = 0

                # Always track stats
                conn_state.bytes_received += len(data)
                conn_state.audio_chunks_received += 1
                conn_state.audio_seconds_received += len(audio_int16) / self.config.SAMPLE_RATE

        except WebSocketDisconnect:
            logger.info(f"Worker disconnected: {conn_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {conn_id}: {e}", exc_info=True)
        finally:
            conn_state.llm_engine.abort()
            conn_state.tts_engine.abort()
            await self.registry.unregister(conn_id)

    def get_metrics(self) -> Dict[str, Any]:
        queue_stats = self.audio_queue.get_stats()
        batch_stats = self.batch_manager.get_stats()
        registry_stats = self.registry.get_stats()
        
        avg_batch_size = 0
        if batch_stats["batches_created"] > 0:
            avg_batch_size = queue_stats["total_items"] / batch_stats["batches_created"]
            
        return {
            "connections": registry_stats["active_connections"],
            "queue_depth": queue_stats["current_depth"],
            "batch_count": batch_stats["batches_created"],
            "avg_batch_size": avg_batch_size,
            "avg_gpu_latency": registry_stats["avg_latency"],
            "avg_e2e_latency": registry_stats["avg_latency"], # Rough approximation
            "transcripts_generated": self.total_transcripts,
            "audio_seconds_processed": self.total_audio_seconds
        }
