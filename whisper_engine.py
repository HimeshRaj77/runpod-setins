"""Whisper inference engine for speech-to-text transcription."""

import logging
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
# ─────────────────────────────────────────────────────────────────────────────


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

    Scans for the earliest position where a word window `words[0:w]`
    starts repeating consecutively.  Returns (start_of_second_rep, window_size)
    or (-1, -1) if no loop found.

    Example:
        words = ["What", "do", "you", "mean", "What", "do", "you", "mean", "What", ...]
        → returns (4, 4)   # second occurrence starts at index 4, window=4 words
    """
    n = len(words)
    hi = min(MAX_NGRAM, n // (MIN_REPS_TO_TRUNCATE + 1))

    for w in range(MIN_NGRAM, hi + 1):
        pattern = words[:w]
        # Find where the next occurrence begins
        for start in range(w, n - w + 1):
            if words[start : start + w] == pattern:
                # Verify at least MIN_REPS_TO_TRUNCATE consecutive repetitions
                reps = 1
                pos  = start + w
                while pos + w <= n and words[pos : pos + w] == pattern:
                    reps += 1
                    pos  += w
                if reps >= MIN_REPS_TO_TRUNCATE:
                    return start, w   # cut before second occurrence
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

            self.processor = AutoProcessor.from_pretrained(
                model_id, cache_dir=cache_dir
            )

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

    # ── inference ─────────────────────────────────────────────────────────────

    def transcribe_batch(
        self,
        audio_batch: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> tuple[List[str], float]:
        """
        Transcribe a batch of int16 PCM arrays.

        Returns:
            (transcriptions, latency_seconds)
        """
        start_time = time.time()
        try:
            if not audio_batch:
                return [], 0.0

            # int16 PCM → float32 [-1, 1]
            audio_floats = [a.astype(np.float32) / 32768.0 for a in audio_batch]

            with torch.no_grad():
                results = self.pipe(
                    audio_floats,
                    batch_size=len(audio_floats),
                    generate_kwargs={
                        # Pin language/task
                        "language": self.language if self.language != "auto" else None,
                        "task": self.task,
                        # ── Anti-hallucination knobs ──────────────────────────
                        # In HuggingFace Transformers, this is called condition_on_prev_tokens
                        "condition_on_prev_tokens": False,
                        # Greedy decoding — temperature=0 is deterministic & most accurate.
                        "temperature": 0.0,
                    },
                )

            if isinstance(results, dict):
                results = [results]

            transcriptions: list[str] = []
            for res in results:
                raw = res.get("text", "").strip()

                # ── 1. pure-noise / silence hallucination gate ────────────────
                if _is_noise_hallucination(raw):
                    logger.debug(f"[whisper] noise-hallucination dropped: '{raw}'")
                    transcriptions.append("")
                    continue

                # ── 2. generic repetition deduplication ──────────────────────
                cleaned = _deduplicate(raw)
                transcriptions.append(cleaned)

            latency = time.time() - start_time
            logger.debug(
                f"[whisper] {len(audio_batch)} item(s) → {latency:.3f}s"
            )
            return transcriptions, latency

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[whisper] transcription error: {e}")
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
