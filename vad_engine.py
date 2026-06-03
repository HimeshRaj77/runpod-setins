"""VAD (Voice Activity Detection) engine using Silero VAD."""

import logging
from typing import Optional

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class VadEngine:
    """Silero VAD engine for speech detection."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize VAD engine.

        Args:
            threshold: VAD confidence threshold (0-1)
            sample_rate: Audio sample rate (must be 16000)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate

        # Silero VAD only supports 16kHz
        assert sample_rate == 16000, "Silero VAD requires 16kHz sample rate"

        try:
            # Load Silero VAD model
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,  # Use ONNX for faster inference
                verbose=False,
            )
            self.get_speech_timestamps = utils[0]

            self.model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"

            logger.info(f"Silero VAD loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def detect_speech(self, audio: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Detect speech in audio using VAD.

        Args:
            audio: Audio samples as int16 numpy array (16kHz, mono)

        Returns:
            Tuple of (speech_segments, timestamps)
            - speech_segments: Concatenated audio of detected speech
            - timestamps: List of dicts with 'start' and 'end' in samples
        """
        try:
            if len(audio) == 0:
                return np.array([], dtype=np.int16), []

            # Convert int16 to float32
            audio_float = audio.astype(np.float32) / 32768.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()

            # Detect speech timestamps
            speeches = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                return_seconds=False,
                visualize_probs=False,
            )

            if not speeches:
                return np.array([], dtype=np.int16), speeches

            # Extract speech segments
            speech_segments = []
            for speech in speeches:
                start = speech["start"]
                end = speech["end"]
                speech_segments.append(audio[start:end])

            # Concatenate all speech segments
            if speech_segments:
                speech_audio = np.concatenate(speech_segments)
            else:
                speech_audio = np.array([], dtype=np.int16)

            return speech_audio, speeches

        except Exception as e:
            logger.error(f"Error in VAD detection: {e}")
            # Return original audio on error (fail open)
            return audio, []

    def get_speech_probability(self, audio: np.ndarray, chunk_size: int = 512) -> float:
        """
        Get speech probability for audio chunk.

        Args:
            audio: Audio samples as int16 numpy array
            chunk_size: Size of analysis chunk

        Returns:
            Speech probability (0-1)
        """
        try:
            if len(audio) == 0:
                return 0.0

            # Take last chunk if audio is longer
            if len(audio) > chunk_size:
                audio = audio[-chunk_size:]

            # Convert to float32
            audio_float = audio.astype(np.float32) / 32768.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()

            # Get speech probability (requires ONNX model with prob output)
            with torch.no_grad():
                prob = self.model(audio_tensor).item()

            return prob

        except Exception as e:
            logger.error(f"Error getting speech probability: {e}")
            return 0.0

    def is_speech(self, audio: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Determine if audio contains speech.

        Args:
            audio: Audio samples as int16 numpy array
            threshold: Optional threshold override

        Returns:
            True if speech detected above threshold
        """
        prob = self.get_speech_probability(audio)
        threshold = threshold or self.threshold
        return prob > threshold
