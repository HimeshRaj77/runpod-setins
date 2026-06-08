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

# ── tunables ─────────────────────────────────────────────────────────────────
# Client sends 256 ms chunks → 4 096 samples @ 16 kHz.
CHUNK_MS           = 256
# How many seconds of audio to keep in the rolling "partial" window.
# Whisper hallucinates badly on very long audio, so we cap here.
PARTIAL_WINDOW_SEC = 8          # 8 s → last 31 chunks
# Consecutive silence before we close the utterance.
# 640 ms = exactly 2.5 × 256 ms chunks — a clean boundary.
SILENCE_END_SEC    = 0.640      # ~2.5 chunks
# Minimum detected speech before we bother running Whisper.
# 512 ms = 2 full chunks — rejects single mic-blip fragments.
MIN_SPEECH_SEC     = 0.512      # 2 chunks
# Hard cap fed to Whisper (its context window is 30 s).
MAX_UTTERANCE_SEC  = 30
# ─────────────────────────────────────────────────────────────────────────────


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

    # ── lifecycle ────────────────────────────────────────────────────────────

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

    # ── processing loop ──────────────────────────────────────────────────────

    async def processing_loop(self):
        logger.info("Started background processing loop.")
        while self.is_running:
            try:
                items = await self.audio_queue.dequeue_batch(
                    max_batch_size=self.config.MAX_BATCH_SIZE,
                    wait_ms=self.config.MAX_BATCH_WAIT_MS,
                )
                if not items:
                    continue

                batch_items = self.batch_manager.build_batch(items)
                audio_arrays = [item.audio_data for item in batch_items]

                transcriptions, latency = self.whisper_engine.transcribe_batch(
                    audio_batch=audio_arrays,
                    sample_rate=self.config.SAMPLE_RATE
                )

                for batch_item, text in zip(batch_items, transcriptions):
                    conn_id  = batch_item.connection_id
                    is_final = getattr(batch_item, 'is_final', False)
                    conn_state = self.registry.get_connection(conn_id)

                    if not conn_state:
                        continue

                    if text:
                        response = {
                            "status": "transcribed",
                            "text": text,
                            "is_final": is_final,
                            "latency_seconds": latency
                        }
                        try:
                            await conn_state.websocket.send_json(response)
                            conn_state.transcripts_sent += 1
                            conn_state.update_latency(latency)
                            self.total_transcripts += 1
                            self.total_audio_seconds += (
                                len(batch_item.audio_data) / self.config.SAMPLE_RATE
                            )
                        except Exception as e:
                            logger.error(f"Failed to send transcript to {conn_id}: {e}")
                            await self.registry.unregister(conn_id)
                            continue

                    # ── LLM only fires on the FINAL utterance ────────────────
                    if is_final and text:
                        await self.run_llm_pipeline(conn_state, text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    # ── LLM pipeline ─────────────────────────────────────────────────────────

    async def run_llm_pipeline(self, conn_state, text: str):
        """Kick off LLM → TTS for the finalised user utterance."""
        # Cancel any leftover work from the previous turn.
        conn_state.llm_engine.abort()
        conn_state.tts_engine.abort()

        if conn_state.llm_task and not conn_state.llm_task.done():
            conn_state.llm_task.cancel()
            try:
                await conn_state.llm_task
            except asyncio.CancelledError:
                pass

        def on_sentence(sentence):
            asyncio.create_task(self.run_tts_for_sentence(conn_state, sentence))

        def on_token(token):
            async def _send():
                try:
                    await conn_state.websocket.send_json({
                        "status": "llm_token",
                        "text": token
                    })
                except Exception as e:
                    logger.error(f"Error sending LLM token: {e}")
            asyncio.create_task(_send())

        async def _llm_job():
            try:
                conn_state.llm_engine._current_task = asyncio.current_task()
                full_text = await conn_state.llm_engine.generate_response(
                    text, on_sentence, on_token
                )
                logger.info(f"LLM final output ({len(full_text)} chars)")
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
        """Stream TTS audio chunks back to the client."""
        try:
            async for audio_chunk in conn_state.tts_engine.generate_audio(sentence):
                await conn_state.websocket.send_bytes(audio_chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    # ── WebSocket handler ────────────────────────────────────────────────────

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        conn_id = str(id(websocket))
        conn_state = await self.registry.register(conn_id, websocket)
        logger.info(f"Worker connected: {conn_id}")

        sample_rate = self.config.SAMPLE_RATE

        # Per-connection VAD state
        speech_chunks: list[np.ndarray] = []   # accumulated *speech* samples only
        silence_samples: int = 0
        is_speaking: bool = False

        PARTIAL_WINDOW_SAMPLES = int(PARTIAL_WINDOW_SEC * sample_rate)
        SILENCE_END_SAMPLES    = int(SILENCE_END_SEC    * sample_rate)
        MIN_SPEECH_SAMPLES     = int(MIN_SPEECH_SEC     * sample_rate)
        MAX_UTTERANCE_SAMPLES  = int(MAX_UTTERANCE_SEC  * sample_rate)

        try:
            while True:
                data = await websocket.receive_bytes()
                conn_state.update_seen()

                audio_int16 = np.frombuffer(data, dtype=np.int16)

                # ── VAD ──────────────────────────────────────────────────────
                speech_int16, _ = self.vad_engine.detect_speech(audio_int16)
                has_speech = len(speech_int16) > 0

                if has_speech:
                    silence_samples = 0

                    # ── Barge-in detection ───────────────────────────────────
                    if not is_speaking:
                        is_speaking = True
                        conn_state.llm_engine.abort()
                        conn_state.tts_engine.abort()
                        logger.info(f"Speech start / barge-in: {conn_id}")

                    speech_chunks.append(speech_int16)

                    # ── Hard cap: prevent Whisper eating >MAX_UTTERANCE_SEC ──
                    total_samples = sum(len(c) for c in speech_chunks)
                    if total_samples > MAX_UTTERANCE_SAMPLES:
                        # Trim oldest chunks to stay within the window
                        while speech_chunks and sum(len(c) for c in speech_chunks) > PARTIAL_WINDOW_SAMPLES:
                            speech_chunks.pop(0)

                    # ── Sliding-window partial STT ───────────────────────────
                    # Send a read-only snapshot of the current window every time
                    # the buffer is at least PARTIAL_WINDOW_SAMPLES large.
                    total_samples = sum(len(c) for c in speech_chunks)
                    if total_samples >= PARTIAL_WINDOW_SAMPLES and not self.audio_queue.is_full():
                        # Only the LAST PARTIAL_WINDOW_SAMPLES of speech
                        window = np.concatenate(speech_chunks)[-PARTIAL_WINDOW_SAMPLES:]
                        await self.audio_queue.enqueue(
                            conn_id,
                            window.tobytes(),
                            is_final=False
                        )

                else:
                    # ── Silence ──────────────────────────────────────────────
                    if is_speaking:
                        silence_samples += len(audio_int16)

                        if silence_samples >= SILENCE_END_SAMPLES:
                            is_speaking = False
                            total_speech = sum(len(c) for c in speech_chunks)

                            if total_speech >= MIN_SPEECH_SAMPLES and not self.audio_queue.is_full():
                                # Cap to MAX_UTTERANCE_SAMPLES before submitting
                                full_audio = np.concatenate(speech_chunks)
                                if len(full_audio) > MAX_UTTERANCE_SAMPLES:
                                    full_audio = full_audio[-MAX_UTTERANCE_SAMPLES:]
                                await self.audio_queue.enqueue(
                                    conn_id,
                                    full_audio.tobytes(),
                                    is_final=True
                                )
                                logger.debug(
                                    f"Finalized utterance {total_speech/sample_rate:.2f}s → {conn_id}"
                                )
                            elif total_speech < MIN_SPEECH_SAMPLES:
                                logger.debug(
                                    f"Dropped short fragment {total_speech/sample_rate:.2f}s"
                                )

                            speech_chunks = []
                            silence_samples = 0

                # Stats
                conn_state.bytes_received += len(data)
                conn_state.audio_chunks_received += 1
                conn_state.audio_seconds_received += len(audio_int16) / sample_rate

        except WebSocketDisconnect:
            logger.info(f"Worker disconnected: {conn_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {conn_id}: {e}", exc_info=True)
        finally:
            conn_state.llm_engine.abort()
            conn_state.tts_engine.abort()
            await self.registry.unregister(conn_id)

    # ── metrics ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        queue_stats    = self.audio_queue.get_stats()
        batch_stats    = self.batch_manager.get_stats()
        registry_stats = self.registry.get_stats()

        avg_batch_size = 0
        if batch_stats["batches_created"] > 0:
            avg_batch_size = queue_stats["total_items"] / batch_stats["batches_created"]

        return {
            "connections":           registry_stats["active_connections"],
            "queue_depth":           queue_stats["current_depth"],
            "batch_count":           batch_stats["batches_created"],
            "avg_batch_size":        avg_batch_size,
            "avg_gpu_latency":       registry_stats["avg_latency"],
            "avg_e2e_latency":       registry_stats["avg_latency"],
            "transcripts_generated": self.total_transcripts,
            "audio_seconds_processed": self.total_audio_seconds
        }
