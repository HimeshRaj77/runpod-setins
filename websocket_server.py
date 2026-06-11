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

# Pre-inference: minimum RMS energy below which we reject audio as silence/noise
# without running Whisper at all. Saves GPU time and eliminates the most common
# hallucination trigger. Tunable via env var.
# 0.001 ~= -60 dBFS — captures genuine speech above background hiss.
import os
MIN_AUDIO_ENERGY = float(os.getenv("MIN_AUDIO_ENERGY", "0.001"))

def _audio_energy(audio_float: np.ndarray) -> float:
    """Return RMS energy of a float32 audio array normalised to [-1, 1]."""
    if len(audio_float) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_float ** 2)))

# ── 32-byte binary header sent by the frontend ────────────────────────────────
# bytes  0-15  : session_id  (UUID as 16 raw Uint8 bytes)
# bytes 16-19  : seq_num     (Uint32, little-endian)
# bytes 20-27  : timestamp   (Float64, little-endian — Date.now() in ms)
# bytes 28-31  : padding     (0x00)
# bytes 32+    : Float32 PCM audio payload
HEADER_SIZE = 32
# ─────────────────────────────────────────────────────────────────────────────



def _float32_to_int16(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw Float32 PCM bytes (values in [-1, 1]) to int16 samples.
    The rest of the pipeline (VAD, Whisper) expects int16 at 16 kHz.

    Sanitizes NaN / Inf before converting — these appear when the microphone
    buffer hasn't fully initialised or when the OS returns a malformed chunk.
    """
    if len(audio_bytes) == 0 or len(audio_bytes) % 4 != 0:
        return np.array([], dtype=np.int16)
    f32 = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    # Replace NaN and Inf with 0 / ±1 before any arithmetic
    f32 = np.nan_to_num(f32, nan=0.0, posinf=1.0, neginf=-1.0)
    f32 = np.clip(f32, -1.0, 1.0)
    return (f32 * 32767).astype(np.int16)


def _is_utf16_session_id(header_bytes: bytes) -> bool:
    """
    Detect if the session ID field (bytes 0-15) looks like a UTF-16 LE
    encoded string — i.e. every odd byte is 0x00.  This happens when the
    frontend encodes the UUID as a JS string instead of a Uint8Array.
    """
    if len(header_bytes) < 16:
        return False
    # If the session ID is completely all zeros, it is a valid binary zero-UUID,
    # not a UTF-16 encoded string.
    if all(b == 0 for b in header_bytes):
        return False
    odd_bytes = header_bytes[1:16:2]   # bytes at positions 1,3,5,...,15
    return all(b == 0 for b in odd_bytes)


def _parse_header(data: bytes) -> tuple[str, int, float, bytes]:
    """
    Strip the 32-byte binary header and return
    (session_id_hex, seq_num, client_timestamp_ms, audio_payload).

    Handles three cases:
      1. Correct binary header  → parse normally
      2. UTF-16 session ID      → log warning, still parse seq/ts correctly
      3. No header / too short  → treat entire payload as legacy int16 audio
    """
    if len(data) < HEADER_SIZE:
        # Legacy client — no header, entire payload is audio
        return "", 0, 0.0, data

    if _is_utf16_session_id(data[:16]):
        # Frontend encoded the UUID as a UTF-16 string — extract whatever we
        # can and log once so the frontend team can fix it.
        logger.warning(
            "Header session_id appears UTF-16 encoded. "
            "Frontend should send UUID as a Uint8Array of raw bytes, not a string."
        )
        # The seq_num and timestamp fields start at byte 16 so they are
        # unaffected by the UTF-16 encoding of bytes 0–15.
        session_id = data[:16].hex() + "_utf16"
        # Parse fields as Big Endian as specified in runpod_api_implementation_details.md
        seq_num   = struct.unpack_from(">I", data, 16)[0]
        timestamp = struct.unpack_from(">d", data, 20)[0]
        return session_id, seq_num, timestamp, data[HEADER_SIZE:]

    session_id = data[:16].hex()
    # Parse fields as Big Endian as specified in runpod_api_implementation_details.md
    seq_num    = struct.unpack_from(">I", data, 16)[0]
    timestamp  = struct.unpack_from(">d", data, 20)[0]
    return session_id, seq_num, timestamp, data[HEADER_SIZE:]


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
        self.vad_engine = None
        if config.VAD_ENABLED:
            logger.info("VAD is enabled. Initializing Silero VadEngine...")
            self.vad_engine = VadEngine(
                threshold=config.VAD_THRESHOLD,
                sample_rate=config.SAMPLE_RATE
            )
        else:
            logger.info("VAD is disabled. Skipping VadEngine initialization.")
            
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
        logger.info("[processing_loop] Started background processing loop.")
        while self.is_running:
            try:
                logger.info("[processing_loop] Waiting for batch from audio_queue...")
                items = await self.audio_queue.dequeue_batch(
                    max_batch_size=self.config.MAX_BATCH_SIZE,
                    wait_ms=self.config.MAX_BATCH_WAIT_MS,
                )
                if not items:
                    logger.info("[processing_loop] Dequeued empty batch. Continuing...")
                    continue

                logger.info(f"[processing_loop] Dequeued batch of {len(items)} items. Building batch...")
                batch_items = self.batch_manager.build_batch(items)
                audio_arrays = [item.audio_data for item in batch_items]

                logger.info(f"[processing_loop] Calling Whisper engine to transcribe batch of {len(audio_arrays)} items...")
                transcriptions, latency = self.whisper_engine.transcribe_batch(
                    audio_batch=audio_arrays,
                    sample_rate=self.config.SAMPLE_RATE
                )
                logger.info(f"[processing_loop] Whisper completed transcription in {latency:.3f}s. Results: {transcriptions}")

                for batch_item, text in zip(batch_items, transcriptions):
                    conn_id  = batch_item.connection_id
                    is_final = getattr(batch_item, 'is_final', False)
                    conn_state = self.registry.get_connection(conn_id)

                    logger.info(f"[processing_loop] Processing result for connection {conn_id} (is_final={is_final}, raw_text='{text}')")

                    if not conn_state:
                        logger.warning(f"[processing_loop] Connection {conn_id} not found in registry. Skipping.")
                        continue

                    if text:
                        if is_final:
                            # Final utterance — send the complete clean text and
                            # reset the partial delta baseline for the next turn.
                            send_text = text
                            conn_state.last_partial_text = ""
                            logger.info(f"[processing_loop] Connection {conn_id} (FINAL): sending full text '{send_text}'")
                        else:
                            # Partial — Whisper re-transcribed the full sliding window,
                            # so strip the prefix that was already sent to avoid the
                            # client seeing the same words repeated on every fire.
                            send_text = _partial_delta(conn_state.last_partial_text, text)
                            logger.info(f"[processing_loop] Connection {conn_id} (PARTIAL): last='{conn_state.last_partial_text}', new='{text}' -> delta='{send_text}'")
                            if send_text:
                                # Advance the baseline to include what we're about to send.
                                conn_state.last_partial_text = (
                                    (conn_state.last_partial_text + " " + send_text).strip()
                                )
                            else:
                                # No new words — skip this partial entirely.
                                logger.info(f"[processing_loop] Connection {conn_id} (PARTIAL): No new words. Skipping.")
                                continue

                        response = {
                            "status": "transcribed",
                            "text": send_text,
                            "is_final": is_final,
                            "latency_seconds": latency
                        }
                        try:
                            logger.info(f"[processing_loop] Sending JSON response to connection {conn_id}: {response}")
                            await conn_state.websocket.send_json(response)
                            conn_state.transcripts_sent += 1
                            conn_state.update_latency(latency)
                            self.total_transcripts += 1
                            self.total_audio_seconds += (
                                len(batch_item.audio_data) / self.config.SAMPLE_RATE
                            )
                            logger.info(f"Transcribe Hit #{self.total_transcripts} | Conn: {conn_id} | Latency: {latency:.3f}s | Final: {is_final} | Text Length: {len(send_text)}")
                        except Exception as e:
                            logger.error(f"[processing_loop] Failed to send transcript to {conn_id}: {e}", exc_info=True)
                            await self.registry.unregister(conn_id)
                            continue
                    else:
                        logger.info(f"[processing_loop] Connection {conn_id}: text is empty. Skipping send.")

                    # ── LLM only fires on the FINAL utterance ────────────────
                    if is_final and text:
                        logger.info(f"[processing_loop] Connection {conn_id} is FINAL with text. Triggering LLM pipeline...")
                        await self.run_llm_pipeline(conn_state, text)
                    elif is_final and not text:
                        logger.warning(f"[processing_loop] Connection {conn_id} is FINAL but text is empty. LLM pipeline NOT triggered.")

            except asyncio.CancelledError:
                logger.info("[processing_loop] Background processing loop cancelled.")
                break
            except Exception as e:
                logger.error(f"[processing_loop] Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    # ── LLM pipeline ─────────────────────────────────────────────────────────

    async def run_llm_pipeline(self, conn_state, text: str):
        """Kick off LLM → TTS for the finalised user utterance."""
        conn_id = conn_state.connection_id
        logger.info(f"[run_llm_pipeline] [{conn_id}] Initializing LLM pipeline for input text: '{text}'")
        
        # Cancel any leftover work from the previous turn.
        logger.info(f"[run_llm_pipeline] [{conn_id}] Aborting any active LLM/TTS engine generations...")
        conn_state.llm_engine.abort()
        conn_state.tts_engine.abort()

        if conn_state.llm_task and not conn_state.llm_task.done():
            logger.info(f"[run_llm_pipeline] [{conn_id}] Cancelling active LLM task...")
            conn_state.llm_task.cancel()
            try:
                await conn_state.llm_task
                logger.info(f"[run_llm_pipeline] [{conn_id}] Active LLM task successfully cancelled.")
            except asyncio.CancelledError:
                pass

        async def _llm_job():
            try:
                llm_start_time = time.time()
                conn_state.llm_engine._current_task = asyncio.current_task()
                
                token_queue = asyncio.Queue()
                tts_task = None
                
                async def token_stream():
                    logger.info(f"[run_llm_pipeline] [{conn_id}] token_stream reader started")
                    while True:
                        token = await token_queue.get()
                        if token is None:
                            logger.info(f"[run_llm_pipeline] [{conn_id}] token_stream reader received None (EOF)")
                            break
                        yield token

                async def _tts_worker():
                    try:
                        logger.info(f"[run_llm_pipeline] [{conn_id}] TTS worker task started. Waiting for tokens...")
                        chunk_count = 0
                        async for audio_chunk in conn_state.tts_engine.generate_audio(token_stream()):
                            chunk_count += 1
                            logger.info(f"[run_llm_pipeline] [{conn_id}] TTS worker generated chunk #{chunk_count} ({len(audio_chunk)} bytes). Sending bytes to client...")
                            await conn_state.websocket.send_bytes(audio_chunk)
                        logger.info(f"[run_llm_pipeline] [{conn_id}] TTS worker finished generating all audio chunks.")
                    except asyncio.CancelledError:
                        logger.info(f"[run_llm_pipeline] [{conn_id}] TTS worker task cancelled.")
                    except Exception as e:
                        logger.error(f"[run_llm_pipeline] [{conn_id}] TTS Error: {e}", exc_info=True)
                        
                tts_task = asyncio.create_task(_tts_worker())
                conn_state.llm_is_active = True
                
                full_text = ""
                first_token = True
                logger.info(f"[run_llm_pipeline] [{conn_id}] Requesting response from LLM engine...")
                async for token in conn_state.llm_engine.generate_response(text):
                    if first_token:
                        ttft = time.time() - llm_start_time
                        logger.info(f"[run_llm_pipeline] [{conn_id}] First LLM token received in {ttft:.3f}s: '{token}'")
                        first_token = False
                        
                    full_text += token
                    logger.info(f"[run_llm_pipeline] [{conn_id}] LLM generated token: '{token}'")
                    
                    try:
                        await conn_state.websocket.send_json({
                            "status": "llm_token",
                            "text": token
                        })
                    except Exception as e:
                        logger.error(f"[run_llm_pipeline] [{conn_id}] Error sending LLM token to client: {e}", exc_info=True)
                        
                    await token_queue.put(token)
                    
                await token_queue.put(None)
                logger.info(f"[run_llm_pipeline] [{conn_id}] LLM finished response generation. Waiting for TTS worker...")
                if tts_task:
                    await tts_task
                
                reply_latency = time.time() - llm_start_time
                logger.info(f"[run_llm_pipeline] [{conn_id}] LLM Reply completed | Final output length: {len(full_text)} chars | Reply Latency: {reply_latency:.3f}s")
                
                await conn_state.websocket.send_json({
                    "status": "llm_final",
                    "text": full_text
                })
            except asyncio.CancelledError:
                logger.info(f"[run_llm_pipeline] [{conn_id}] LLM job task cancelled.")
                # Cancel TTS too so it doesn't keep sending stale audio
                if tts_task and not tts_task.done():
                    logger.info(f"[run_llm_pipeline] [{conn_id}] Cancelling TTS worker task...")
                    tts_task.cancel()
            except Exception as e:
                logger.error(f"[run_llm_pipeline] [{conn_id}] LLM Error in _llm_job: {e}", exc_info=True)
            finally:
                conn_state.llm_is_active = False
                logger.info(f"[run_llm_pipeline] [{conn_id}] LLM pipeline completed and marked inactive.")

        conn_state.llm_task = asyncio.create_task(_llm_job())
        logger.info(f"[run_llm_pipeline] [{conn_id}] LLM job task created and running in background.")

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
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                
                if "bytes" not in message:
                    if "text" in message:
                        logger.warning(f"[{conn_id}] Received text message instead of bytes: {message['text'][:500]}")
                    else:
                        logger.warning(f"[{conn_id}] Received unknown message structure: {message}")
                    continue
                
                raw = message["bytes"]
                conn_state.update_seen()

                # ── Parse 32-byte binary header ───────────────────────────────
                session_id, seq_num, client_ts_ms, audio_bytes = _parse_header(raw)
                logger.info(f"[{conn_id}] Recv raw={len(raw)} bytes, parsed audio_bytes={len(audio_bytes)}, seq={seq_num}, client_ts={client_ts_ms}")

                # Store session ID on first packet
                if session_id and not conn_state.client_session_id:
                    conn_state.client_session_id = session_id
                    logger.info(f"Session ID bound: {session_id} → {conn_id}")

                # Drop duplicate or out-of-order packets using seq_num.
                # seq_num wraps at 2^32; allow rollover by checking distance.
                if seq_num != 0 and conn_state.last_seq_num >= 0:
                    gap = (seq_num - conn_state.last_seq_num) & 0xFFFFFFFF
                    if gap == 0:
                        logger.info(f"[{conn_id}] Duplicate packet seq={seq_num} dropped")
                        continue
                    if gap > 0x7FFFFFFF:  # wrapped backwards = out of order
                        logger.warning(f"[{conn_id}] Out-of-order packet seq={seq_num} (last={conn_state.last_seq_num}) dropped")
                        continue
                conn_state.last_seq_num = seq_num

                # ── Convert Float32 PCM → int16 ───────────────────────────────
                # The frontend sends Float32Array (values in [-1, 1]).
                # VAD and WhisperEngine both expect int16 @ 16 kHz.
                audio_int16 = _float32_to_int16(audio_bytes)

                if len(audio_int16) == 0:
                    logger.info(f"[{conn_id}] Converted int16 audio array is empty. Skipping.")
                    continue

                # Measure end-to-end latency from client capture to server receipt
                if client_ts_ms > 0:
                    e2e_ms = (time.time() * 1000) - client_ts_ms
                    logger.info(f"[{conn_id}] E2E recv latency: {e2e_ms:.1f} ms (seq={seq_num})")

                # ── VAD ───────────────────────────────────────────────────────
                if self.config.VAD_ENABLED:
                    speech_int16, _ = self.vad_engine.detect_speech(audio_int16)
                    has_speech = len(speech_int16) > 0
                    logger.info(f"[{conn_id}] VAD check: has_speech={has_speech} (speech_samples={len(speech_int16)}/{len(audio_int16)})")
                else:
                    # If VAD is disabled, we use a simple energy-based threshold gate.
                    # This completely bypasses the deep learning VAD model to save CPU/GPU.
                    f = audio_int16.astype(np.float32) / 32768.0
                    energy = _audio_energy(f)
                    has_speech = energy >= MIN_AUDIO_ENERGY
                    speech_int16 = audio_int16 if has_speech else np.array([], dtype=np.int16)
                    logger.info(f"[{conn_id}] VAD (Energy-based fallback, Silero disabled) check: has_speech={has_speech} (energy={energy:.6f}, threshold={MIN_AUDIO_ENERGY})")

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
                    logger.info(f"[{conn_id}] Speech buffer size: {total_samples} samples ({total_samples/sample_rate:.2f}s)")
                    if total_samples > MAX_UTTERANCE_SAMPLES:
                        logger.info(f"[{conn_id}] Speech buffer exceeds hard cap ({MAX_UTTERANCE_SAMPLES}). Truncating oldest chunks...")
                        while speech_chunks and sum(len(c) for c in speech_chunks) > PARTIAL_WINDOW_SAMPLES:
                            speech_chunks.pop(0)

                    # ── Sliding-window partial STT ────────────────────────────
                    total_samples = sum(len(c) for c in speech_chunks)
                    logger.info(f"[{conn_id}] Partial trigger check: samples_since_partial={samples_since_partial}/{PARTIAL_SEND_EVERY}, total_samples={total_samples}/{MIN_SPEECH_SAMPLES}")
                    if (
                        samples_since_partial >= PARTIAL_SEND_EVERY
                        and total_samples     >= MIN_SPEECH_SAMPLES
                        and not self.audio_queue.is_full()
                    ):
                        window = np.concatenate(speech_chunks)[-PARTIAL_WINDOW_SAMPLES:]
                        logger.info(f"[{conn_id}] Triggering PARTIAL STT. Enqueuing {len(window)} samples ({len(window)/sample_rate:.2f}s)")
                        await self.audio_queue.enqueue(
                            conn_id,
                            window.tobytes(),
                            is_final=False
                        )
                        samples_since_partial = 0
                    elif self.audio_queue.is_full():
                        logger.warning(f"[{conn_id}] Cannot enqueue PARTIAL STT: Audio queue is FULL!")

                else:
                    # ── Silence ───────────────────────────────────────────────
                    if is_speaking:
                        silence_samples += len(audio_int16)
                        logger.info(f"[{conn_id}] Silence check: accumulated silence_samples={silence_samples}/{SILENCE_END_SAMPLES}")

                        if silence_samples >= SILENCE_END_SAMPLES:
                            is_speaking = False
                            total_speech = sum(len(c) for c in speech_chunks)
                            logger.info(f"[{conn_id}] Speech ended. Total speaking time: {total_speech/sample_rate:.2f}s ({total_speech} samples)")

                            if total_speech >= MIN_SPEECH_SAMPLES and not self.audio_queue.is_full():
                                full_audio = np.concatenate(speech_chunks)
                                if len(full_audio) > MAX_UTTERANCE_SAMPLES:
                                    logger.info(f"[{conn_id}] Utterance exceeds MAX_UTTERANCE_SAMPLES ({MAX_UTTERANCE_SAMPLES}). Truncating...")
                                    full_audio = full_audio[-MAX_UTTERANCE_SAMPLES:]
                                logger.info(f"[{conn_id}] Triggering FINAL STT. Enqueuing {len(full_audio)} samples ({len(full_audio)/sample_rate:.2f}s)")
                                await self.audio_queue.enqueue(
                                    conn_id,
                                    full_audio.tobytes(),
                                    is_final=True
                                )
                                logger.info(
                                    f"Utterance finalised {total_speech/sample_rate:.2f}s → {conn_id}"
                                )
                            elif total_speech < MIN_SPEECH_SAMPLES:
                                logger.info(
                                    f"[{conn_id}] Short fragment {total_speech/sample_rate:.2f}s dropped"
                                )
                            elif self.audio_queue.is_full():
                                logger.error(f"[{conn_id}] Cannot enqueue FINAL STT: Audio queue is FULL!")

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
            logger.info(f"[{conn_id}] Closing websocket. Cleaning up active tasks...")
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
