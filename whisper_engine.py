"""Whisper inference engine for speech-to-text transcription."""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize Whisper engine.
        """
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype
        self.language = language
        self.task = task
        self.cache_dir = cache_dir

        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            logger.warning(f"Unknown dtype {dtype}, using float16")
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

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=device if device == "cuda" else -1,
                chunk_length_s=30,
                stride_length_s=(4, 2),
                return_timestamps=False,
            )

            logger.info(f"Whisper engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper engine: {e}")
            raise

    def transcribe_batch(
        self,
        audio_batch: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> tuple[List[str], float]:
        """
        Transcribe a batch of audio.
        """
        start_time = time.time()
        try:
            if not audio_batch:
                return [], 0.0

            audio_floats = []
            for audio in audio_batch:
                audio_float = audio.astype(np.float32) / 32768.0
                audio_floats.append(audio_float)

            with torch.no_grad():
                results = self.pipe(
                    audio_floats,
                    batch_size=len(audio_floats),
                    generate_kwargs={
                        "language": self.language if self.language != "auto" else None,
                        "task": self.task,
                    },
                )
            
            # The pipeline can return a dict if length is 1, or a list of dicts.
            if isinstance(results, dict):
                results = [results]
                
            transcriptions = []
            
            # Common Whisper v3 hallucinations on background noise / silence
            hallucination_phrases = {
                "thank you", "thanks for watching", "please subscribe", 
                "thanks for subscribing", "thank you very much", "thank you so much",
                "subscribe to the channel", "thanks", "you", "i'm sorry", "amara.org",
                "subscribe", "bye", "okay", "bye bye"
            }
            
            for res in results:
                text = res.get("text", "").strip()
                
                # Normalize text for checking: lowercase and remove punctuation
                norm_text = "".join(c.lower() for c in text if c.isalpha() or c.isspace()).strip()
                
                if norm_text in hallucination_phrases or norm_text.replace(" ", "") == "thankyou":
                    logger.debug(f"Filtered hallucination: '{text}'")
                    text = ""
                    
                transcriptions.append(text)
                
            latency = time.time() - start_time
            logger.debug(f"Transcribed {len(audio_batch)} samples in {latency:.3f}s")
            return transcriptions, latency

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Error during transcription: {e}")
            return ["" for _ in audio_batch], latency

    def get_model_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype_str,
            "language": self.language,
            "task": self.task,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
