"""Whisper inference engine for speech-to-text transcription."""

import logging
import os
import re
import time
from typing import List, Optional

import numpy as np
import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Minimum number of word-level repetitions before we truncate.
# e.g. "What do you mean? " repeated 3+ times → keep 1.
MIN_REPS_TO_TRUNCATE = 2

# Minimum n-gram window size (words) to scan for loops.
MIN_NGRAM = 3

# Maximum n-gram window size; beyond this the pattern is too long to be a loop.
MAX_NGRAM = 40

# If the normalised text is shorter than this many chars, treat as too short.
MIN_CHARS = 2

# Pre-inference: minimum RMS energy below which we reject audio as silence/noise
# without running Whisper at all. Saves GPU time and eliminates the most common
# hallucination trigger. Tunable via env var.
# 0.001 ~= -60 dBFS — captures genuine speech above background hiss.
MIN_AUDIO_ENERGY = float(os.getenv("MIN_AUDIO_ENERGY", "0.001"))

# Post-dedup: if the cleaned text has fewer words than this after loop truncation,
# treat the whole thing as noise (e.g., orphaned "is" after deduplication).
MIN_WORDS_AFTER_DEDUP = int(os.getenv("MIN_WORDS_AFTER_DEDUP", "2"))
# ─────────────────────────────────────────────────────────────────────────────


def _audio_energy(audio_float: np.ndarray) -> float:
    """Return RMS energy of a float32 audio array normalised to [-1, 1]."""
    if len(audio_float) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_float ** 2)))


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(
        "".join(c for c in text.lower() if c.isalpha() or c.isspace()).split()
    )


def _is_noise_hallucination(text: str) -> bool:
    """
    Detect pure-silence / background-noise hallucinations.

    These are extremely short outputs that carry no real content.
    We intentionally keep this list minimal — real repetition
    suppression is handled generically by _deduplicate.
    """
    norm = _normalize(text)
    if len(norm) < MIN_CHARS:
        return True
    # Only filter the absolute zero-content tokens Whisper emits on dead air.
    NOISE_TOKENS = {
        "thank you", "thanks", "you", "bye", "okay", "amara.org",
        "subscribe", "bye bye", "see you",
    }
    return norm in NOISE_TOKENS


def _find_repeating_unit(words: list[str]) -> tuple[int, int]:
    """
    Generic n-gram loop detector.

    Scans for the earliest position where a word window of size `w`
    starts repeating consecutively, at ANY starting offset.
    Returns (start_of_second_rep, window_size) or (-1, -1) if no loop found.
    """
    n = len(words)
    # The max window size we can check is n // MIN_REPS_TO_TRUNCATE
    hi = min(MAX_NGRAM, n // MIN_REPS_TO_TRUNCATE)

    for w in range(MIN_NGRAM, hi + 1):
        # Slide a window across the text to find a repeating pattern
        for offset in range(n - w * MIN_REPS_TO_TRUNCATE + 1):
            pattern = words[offset : offset + w]
            
            # Check for consecutive repetitions
            reps = 1
            pos = offset + w
            while pos + w <= n and words[pos : pos + w] == pattern:
                reps += 1
                pos += w
                
            if reps >= MIN_REPS_TO_TRUNCATE:
                # We found a loop. Return the index where the SECOND repetition begins,
                # so we can truncate everything from that point onwards.
                return offset + w, w
                
    return -1, -1


def _deduplicate(text: str) -> str:
    """
    Two-pass generic repetition suppressor — no hardcoded phrases.

    Pass 1 — sentence-level:
        Walk through sentences.  The first time a (normalised) sentence
        appears twice, stop.

    Pass 2 — word n-gram level:
        If the remaining text still contains a repeating word-window,
        truncate at the first full repetition.
    """
    # ── pass 1: sentence dedup ────────────────────────────────────────────────
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen_sentences: list[str] = []
    kept_sentences: list[str] = []

    for s in sentences:
        norm = _normalize(s)
        if not norm:
            continue
        if norm in seen_sentences:
            break                    # discard from second occurrence onwards
        seen_sentences.append(norm)
        kept_sentences.append(s.strip())

    dedupe1 = " ".join(kept_sentences)

    # ── pass 2: word n-gram dedup ─────────────────────────────────────────────
    words = dedupe1.split()
    cut_at, _ = _find_repeating_unit(words)
    if cut_at != -1:
        dedupe2 = " ".join(words[:cut_at])
    else:
        dedupe2 = dedupe1

    if dedupe2 != text:
        logger.debug(f"[whisper] deduped: '{text[:100]}…' → '{dedupe2[:100]}'")

    return dedupe2.strip()


class WhisperEngine:
    """Whisper model inference engine."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        dtype: str = "float16",
        cache_dir: Optional[str] = None,
        language: str = "en",
        task: str = "transcribe",
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 1.35,
    ):
        self.model_id   = model_id
        self.device     = device
        self.dtype_str  = dtype
        self.language   = language
        self.task       = task
        self.cache_dir  = cache_dir
        self.no_speech_threshold       = no_speech_threshold
        self.compression_ratio_threshold = compression_ratio_threshold

        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            logger.warning(f"Unknown dtype {dtype}, defaulting to float16")
            self.torch_dtype = torch.float16

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        try:
            logger.info(f"Loading Whisper model {model_id}...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=cache_dir,
            )
            if device == "cuda":
                self.model = self.model.to(device)
            self.model.eval()

            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, cache_dir=cache_dir
                )
            except OSError as e:
                logger.error(f"Failed to load processor for {model_id}. Error: {e}")
                raise e

            # ── Pipeline ─────────────────────────────────────────────────────
            # We manage chunking ourselves; do NOT set chunk_length_s here.
            # return_timestamps=False avoids the extra overhead.
            # We also pass the attention implementation for Flash Attention 2
            # if available on the current CUDA device.
            pipe_kwargs = dict(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=device if device == "cuda" else -1,
                return_timestamps=False,
            )

            self.pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)

            logger.info("Whisper engine initialised successfully")
        except Exception as e:
            logger.error(f"Failed to initialise Whisper engine: {e}")
            raise

    # ── warmup ────────────────────────────────────────────────────────────────

    def warmup(self):
        """Warm up the GPU by running a dummy inference to compile the PyTorch execution graph."""
        logger.info("[whisper] Warming up the GPU with a dummy inference...")
        start_time = time.time()
        # 1 second of silent float32 audio
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            with torch.no_grad():
                self.pipe(
                    [dummy_audio],
                    batch_size=1,
                    generate_kwargs={
                        "language": self.language if self.language != "auto" else None,
                        "task": self.task,
                        "condition_on_prev_tokens": False,
                        "temperature": 0.0,
                    },
                )
            logger.info(f"[whisper] GPU warm-up completed in {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"[whisper] Failed to warm up GPU: {e}")

    # ── inference ─────────────────────────────────────────────────────────────

    def transcribe_batch(
        self,
        audio_batch: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> tuple[List[str], float]:
        start_time = time.time()
        try:
            if not audio_batch:
                logger.info("[whisper] empty audio batch received")
                return [], 0.0

            logger.info(f"[whisper] transcribing batch of {len(audio_batch)} audio chunk(s)")

            # ── Layer 1: Pre-inference audio energy gate ──────────────────────
            # If an audio chunk is below the minimum RMS energy threshold it is
            # silence or background noise — skip Whisper entirely for that chunk.
            # This is the primary hallucination prevention: never feed bad audio
            # to the model in the first place.
            audio_floats: list[np.ndarray] = []
            energy_mask: list[bool] = []   # True = passed energy gate

            for idx, a in enumerate(audio_batch):
                f = a.astype(np.float32) / 32768.0
                energy = _audio_energy(f)
                if energy < MIN_AUDIO_ENERGY:
                    logger.info(f"[whisper] chunk {idx} below energy threshold (energy={energy:.6f} < {MIN_AUDIO_ENERGY}) — skipped")
                    energy_mask.append(False)
                else:
                    logger.info(f"[whisper] chunk {idx} passed energy gate (energy={energy:.6f} >= {MIN_AUDIO_ENERGY})")
                    audio_floats.append(f)
                    energy_mask.append(True)

            # If every chunk was silence, return early without touching the GPU.
            if not audio_floats:
                latency = time.time() - start_time
                logger.info(f"[whisper] all chunks skipped by energy gate. returning empty. latency: {latency:.3f}s")
                return ["" for _ in audio_batch], latency

            # ── Layer 2: Inference-time anti-hallucination ────────────────────
            # Note: We do NOT pass no_speech_threshold and compression_ratio_threshold
            # here because they cause a known bug in certain transformers versions
            # (UnboundLocalError: cannot access local variable 'logprobs').
            # Instead, we rely on our multi-layer pre-and-post-filtering.
            logger.info(f"[whisper] Running pipeline inference on {len(audio_floats)} active chunks...")
            infer_start = time.time()
            with torch.no_grad():
                results = self.pipe(
                    audio_floats,
                    batch_size=len(audio_floats),
                    generate_kwargs={
                        "language": self.language if self.language != "auto" else None,
                        "task": self.task,
                        # condition_on_prev_tokens=False: prevent the decoder from
                        # conditioning on its own prior output — the #1 cause of loops.
                        "condition_on_prev_tokens": False,
                        # temperature=0 → greedy decoding, most deterministic output.
                        "temperature": 0.0,
                    },
                )
            infer_time = time.time() - infer_start
            logger.info(f"[whisper] pipeline inference completed in {infer_time:.3f}s")

            if isinstance(results, dict):
                results = [results]

            # ── Layer 3: Post-inference cleanup ───────────────────────────────
            transcriptions_active: list[str] = []
            for idx, res in enumerate(results):
                raw = res.get("text", "").strip()
                logger.info(f"[whisper] active chunk {idx} raw text: '{raw}'")

                # 3a. Pure-noise / silence hallucination token gate
                if _is_noise_hallucination(raw):
                    logger.info(f"[whisper] noise-hallucination dropped: '{raw}'")
                    transcriptions_active.append("")
                    continue

                # 3b. Generic repetition deduplication (sliding window n-gram)
                cleaned = _deduplicate(raw)
                logger.info(f"[whisper] active chunk {idx} deduplicated text: '{cleaned}'")

                # 3c. Post-dedup orphan word check.
                # When a large loop is removed (e.g. "is It's a day-to-day × 49")
                # the leftover prefix ("is") is too short to be real content.
                # Only drop if the original was much longer — avoids penalising
                # genuine short responses.
                if (len(cleaned.split()) < MIN_WORDS_AFTER_DEDUP
                        and len(raw.split()) >= MIN_WORDS_AFTER_DEDUP):
                    logger.info(
                        f"[whisper] orphan after dedup dropped: '{cleaned}' "
                        f"(original: '{raw[:80]}')"
                    )
                    transcriptions_active.append("")
                    continue

                transcriptions_active.append(cleaned)

            # Reconstruct full-length list aligned with original audio_batch,
            # inserting "" for any chunks that were skipped by the energy gate.
            transcriptions: list[str] = []
            active_iter = iter(transcriptions_active)
            for passed in energy_mask:
                transcriptions.append(next(active_iter) if passed else "")

            latency = time.time() - start_time
            logger.info(f"[whisper] transcription complete for {len(audio_batch)} item(s) in {latency:.3f}s. results: {transcriptions}")
            return transcriptions, latency

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[whisper] transcription error: {e}", exc_info=True)
            return ["" for _ in audio_batch], latency

    def get_model_info(self) -> dict:
        return {
            "model_id":   self.model_id,
            "device":     self.device,
            "dtype":      self.dtype_str,
            "language":   self.language,
            "task":       self.task,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
