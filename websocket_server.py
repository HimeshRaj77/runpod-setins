"""WebSocket server handler and background processing for RunPod STT."""

import asyncio
import json
import logging
import struct
import time
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
CHUNK_MS              = 256

# How many seconds of audio to keep in each partial payload sent to Whisper.
# This is the CONTENT window — always the last N seconds of speech.
PARTIAL_WINDOW_SEC    = 8        # 8 s → last ~31 chunks

# How often to FIRE a partial STT job (measured in new speech samples).
# 1.5 s means Whisper gets its first result after ~1.5 s of speech,
# then a fresh result every further 1.5 s, giving early LLM warm-up.
PARTIAL_SEND_EVERY_SEC = 1.5     # ~6 chunks between each partial

# Consecutive silence before we close the utterance.
# 640 ms = 2.5 × 256 ms chunks — a clean boundary.
SILENCE_END_SEC       = 0.640    # ~2.5 chunks

# Minimum detected speech before we bother running Whisper.
# 512 ms = 2 full chunks — rejects single mic-blip fragments.
MIN_SPEECH_SEC        = 0.512    # 2 chunks

# Hard cap fed to Whisper (its 30 s context window).
MAX_UTTERANCE_SEC     = 30

# ── 32-byte binary header sent by the frontend ────────────────────────────────
# bytes  0-15  : session_id  (UUID as 16 raw Uint8 bytes)
# bytes 16-19  : seq_num     (Uint32, little-endian)
# bytes 20-27  : timestamp   (Float64, little-endian — Date.now() in ms)
# bytes 28-31  : padding     (0x00)
# bytes 32+    : Float32 PCM audio payload
HEADER_SIZE = 32
# ─────────────────────────────────────────────────────────────────────────────


def _parse_header(data: bytes) -> tuple[str, int, float, bytes]:
    """
    Strip the 32-byte binary header and return
    (session_id_hex, seq_num, client_timestamp_ms, audio_payload).
    Falls back gracefully if data is shorter than the header.
    """
    if len(data) < HEADER_SIZE:
        return "", 0, 0.0, data
    session_id = data[:16].hex()          # 16 bytes → 32-char hex string
    seq_num    = struct.unpack_from("<I", data, 16)[0]   # Uint32 LE
    timestamp  = struct.unpack_from("<d", data, 20)[0]   # Float64 LE
    return session_id, seq_num, timestamp, data[HEADER_SIZE:]


def _float32_to_int16(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw Float32 PCM bytes (values in [-1, 1]) to int16 samples.
    The rest of the pipeline (VAD, Whisper) expects int16 at 16 kHz.
    """
    f32 = np.frombuffer(audio_bytes, dtype=np.float32)
    f32 = np.clip(f32, -1.0, 1.0)
    return (f32 * 32767).astype(np.int16)


def _partial_delta(last_text: str, new_text: str) -> str:
    """
    Return only the words in `new_text` that extend beyond `last_text`.

    Because Whisper re-transcribes a sliding window it often prepends the
    same words from the previous partial. We walk forward through the new
    result word-by-word and skip any prefix that matches the tail of what
    we already sent, so the client only receives genuinely new content.

    Example:
        last = "Hello world"
        new  = "Hello world how are you"
        → delta = "how are you"
    """
    if not last_text:
        return new_text

    last_words = last_text.lower().split()
    new_words  = new_text.split()

    # Find the longest suffix of last_words that is a prefix of new_words
    best_skip = 0
    for length in range(1, min(len(last_words), len(new_words)) + 1):
        if last_words[-length:] == [w.lower() for w in new_words[:length]]:
            best_skip = length

    delta_words = new_words[best_skip:]
    return " ".join(delta_words)


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
            task=config.WHISPER_TASK,
            no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
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
                        if is_final:
                            # Final utterance — send the complete clean text and
                            # reset the partial delta baseline for the next turn.
                            send_text = text
                            conn_state.last_partial_text = ""
                        else:
                            # Partial — Whisper re-transcribed the full sliding window,
                            # so strip the prefix that was already sent to avoid the
                            # client seeing the same words repeated on every fire.
                            send_text = _partial_delta(conn_state.last_partial_text, text)
                            if send_text:
                                # Advance the baseline to include what we're about to send.
                                conn_state.last_partial_text = (
                                    (conn_state.last_partial_text + " " + send_text).strip()
                                )
                            else:
                                # No new words — skip this partial entirely.
                                continue

                        response = {
                            "status": "transcribed",
                            "text": send_text,
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

        async def _llm_job():
            try:
                conn_state.llm_engine._current_task = asyncio.current_task()
                
                token_queue = asyncio.Queue()
                tts_task = None
                
                async def token_stream():
                    while True:
                        token = await token_queue.get()
                        if token is None:
                            break
                        yield token

                async def _tts_worker():
                    try:
                        async for audio_chunk in conn_state.tts_engine.generate_audio(token_stream()):
                            await conn_state.websocket.send_bytes(audio_chunk)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"TTS Error: {e}")
                        
                tts_task = asyncio.create_task(_tts_worker())
                conn_state.llm_is_active = True
                
                full_text = ""
                async for token in conn_state.llm_engine.generate_response(text):
                    full_text += token
                    
                    try:
                        await conn_state.websocket.send_json({
                            "status": "llm_token",
                            "text": token
                        })
                    except Exception as e:
                        logger.error(f"Error sending LLM token: {e}")
                        
                    await token_queue.put(token)
                    
                await token_queue.put(None)
                if tts_task:
                    await tts_task
                
                logger.info(f"LLM final output ({len(full_text)} chars)")
                await conn_state.websocket.send_json({
                    "status": "llm_final",
                    "text": full_text
                })
            except asyncio.CancelledError:
                # Cancel TTS too so it doesn't keep sending stale audio
                if tts_task and not tts_task.done():
                    tts_task.cancel()
            except Exception as e:
                logger.error(f"LLM Error: {e}")
            finally:
                conn_state.llm_is_active = False

        conn_state.llm_task = asyncio.create_task(_llm_job())

    # ── WebSocket handler ────────────────────────────────────────────────────

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()

        # We use a placeholder conn_id until we receive the first chunk with
        # a valid header (which gives us the real client session UUID).
        conn_id = str(id(websocket))
        conn_state = await self.registry.register(conn_id, websocket)
        logger.info(f"Worker connected: {conn_id}")

        sample_rate = self.config.SAMPLE_RATE

        # Per-connection VAD state
        speech_chunks: list[np.ndarray] = []  # accumulated speech samples
        silence_samples: int  = 0
        is_speaking: bool     = False
        samples_since_partial: int = 0        # new-speech samples since last partial

        PARTIAL_WINDOW_SAMPLES    = int(PARTIAL_WINDOW_SEC     * sample_rate)
        PARTIAL_SEND_EVERY        = int(PARTIAL_SEND_EVERY_SEC * sample_rate)
        SILENCE_END_SAMPLES       = int(SILENCE_END_SEC        * sample_rate)
        MIN_SPEECH_SAMPLES        = int(MIN_SPEECH_SEC         * sample_rate)
        MAX_UTTERANCE_SAMPLES     = int(MAX_UTTERANCE_SEC      * sample_rate)

        try:
            while True:
                raw = await websocket.receive_bytes()
                conn_state.update_seen()

                # ── Parse 32-byte binary header ───────────────────────────────
                session_id, seq_num, client_ts_ms, audio_bytes = _parse_header(raw)

                # Store session ID on first packet
                if session_id and not conn_state.client_session_id:
                    conn_state.client_session_id = session_id
                    logger.info(f"Session ID bound: {session_id} → {conn_id}")

                # Drop duplicate or out-of-order packets using seq_num.
                # seq_num wraps at 2^32; allow rollover by checking distance.
                if seq_num != 0 and conn_state.last_seq_num >= 0:
                    gap = (seq_num - conn_state.last_seq_num) & 0xFFFFFFFF
                    if gap == 0:
                        logger.debug(f"Duplicate packet seq={seq_num} dropped")
                        continue
                    if gap > 0x7FFFFFFF:  # wrapped backwards = out of order
                        logger.warning(f"Out-of-order packet seq={seq_num} (last={conn_state.last_seq_num}) dropped")
                        continue
                conn_state.last_seq_num = seq_num

                # ── Convert Float32 PCM → int16 ───────────────────────────────
                # The frontend sends Float32Array (values in [-1, 1]).
                # VAD and WhisperEngine both expect int16 @ 16 kHz.
                audio_int16 = _float32_to_int16(audio_bytes)

                if len(audio_int16) == 0:
                    continue

                # Measure end-to-end latency from client capture to server receipt
                if client_ts_ms > 0:
                    e2e_ms = (time.time() * 1000) - client_ts_ms
                    logger.debug(f"E2E recv latency: {e2e_ms:.1f} ms (seq={seq_num})")

                # ── VAD ───────────────────────────────────────────────────────
                speech_int16, _ = self.vad_engine.detect_speech(audio_int16)
                has_speech = len(speech_int16) > 0

                if has_speech:
                    silence_samples = 0

                    if not is_speaking:
                        is_speaking = True
                        samples_since_partial = 0
                        conn_state.last_partial_text = ""  # reset delta baseline

                        # Only abort LLM/TTS if they are actually running
                        if conn_state.llm_is_active:
                            conn_state.llm_engine.abort()
                            conn_state.tts_engine.abort()
                            logger.info(f"Barge-in: {conn_id} — interrupted active response")
                        else:
                            logger.info(f"Speech start: {conn_id}")

                    speech_chunks.append(speech_int16)
                    samples_since_partial += len(speech_int16)

                    # ── Hard cap: keep buffer within Whisper's context ────────
                    total_samples = sum(len(c) for c in speech_chunks)
                    if total_samples > MAX_UTTERANCE_SAMPLES:
                        while speech_chunks and sum(len(c) for c in speech_chunks) > PARTIAL_WINDOW_SAMPLES:
                            speech_chunks.pop(0)

                    # ── Sliding-window partial STT ────────────────────────────
                    total_samples = sum(len(c) for c in speech_chunks)
                    if (
                        samples_since_partial >= PARTIAL_SEND_EVERY
                        and total_samples     >= MIN_SPEECH_SAMPLES
                        and not self.audio_queue.is_full()
                    ):
                        window = np.concatenate(speech_chunks)[-PARTIAL_WINDOW_SAMPLES:]
                        await self.audio_queue.enqueue(
                            conn_id,
                            window.tobytes(),
                            is_final=False
                        )
                        samples_since_partial = 0

                else:
                    # ── Silence ───────────────────────────────────────────────
                    if is_speaking:
                        silence_samples += len(audio_int16)

                        if silence_samples >= SILENCE_END_SAMPLES:
                            is_speaking = False
                            total_speech = sum(len(c) for c in speech_chunks)

                            if total_speech >= MIN_SPEECH_SAMPLES and not self.audio_queue.is_full():
                                full_audio = np.concatenate(speech_chunks)
                                if len(full_audio) > MAX_UTTERANCE_SAMPLES:
                                    full_audio = full_audio[-MAX_UTTERANCE_SAMPLES:]
                                await self.audio_queue.enqueue(
                                    conn_id,
                                    full_audio.tobytes(),
                                    is_final=True
                                )
                                logger.info(
                                    f"Utterance finalised {total_speech/sample_rate:.2f}s → {conn_id}"
                                )
                            elif total_speech < MIN_SPEECH_SAMPLES:
                                logger.debug(
                                    f"Short fragment {total_speech/sample_rate:.2f}s dropped"
                                )

                            speech_chunks = []
                            silence_samples = 0
                            samples_since_partial = 0
                            conn_state.last_partial_text = ""  # reset delta baseline

                # Stats (count audio samples, not raw bytes, for accuracy)
                conn_state.bytes_received += len(raw)
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
