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
                            "latency_seconds": latency
                        }
                        
                        try:
                            # Send output back to exact connection
                            await conn_state.websocket.send_json(response)
                            conn_state.transcripts_sent += 1
                            conn_state.update_latency(latency)
                            self.total_transcripts += 1
                            self.total_audio_seconds += len(batch_item.audio_data) / self.config.SAMPLE_RATE
                        except Exception as e:
                            logger.error(f"Failed to send result to connection {conn_id}: {e}")
                            await self.registry.unregister(conn_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        conn_id = str(id(websocket))
        
        # We need a dummy ServerConnection object with send_json support, but FastAPI WebSocket has send_json natively.
        # Let's adjust registry usage to store the FastAPI WebSocket directly.
        conn_state = await self.registry.register(conn_id, websocket)
        logger.info(f"Worker connected: {conn_id}")
        
        try:
            while True:
                data = await websocket.receive_bytes()
                conn_state.update_seen()
                
                await conn_state.append_audio(data)
                
                # Process only when we have accumulated a full chunk (e.g. 1 second)
                if len(conn_state.audio_buffer) >= self.config.PCM_CHUNK_SIZE:
                    audio_int16 = await conn_state.get_and_clear_audio()
                    
                    # Silero VAD
                    speech_int16, _ = self.vad_engine.detect_speech(audio_int16)
                    
                    if len(speech_int16) > 0:
                        # Enqueue
                        if not self.audio_queue.is_full():
                            await self.audio_queue.enqueue(conn_id, speech_int16.tobytes())
                        else:
                            logger.warning(f"Backpressure: Queue full, dropping audio chunk from {conn_id}")
        except WebSocketDisconnect:
            logger.info(f"Worker disconnected: {conn_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {conn_id}: {e}", exc_info=True)
        finally:
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
