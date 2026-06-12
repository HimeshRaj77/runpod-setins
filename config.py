"""Configuration management for RunPod Whisper STT service."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))

    # WebSocket
    WEBSOCKET_PING_INTERVAL: float = float(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))
    WEBSOCKET_TIMEOUT: float = float(os.getenv("WEBSOCKET_TIMEOUT", "60"))
    MAX_CONCURRENT_CONNECTIONS: int = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "32"))

    # Audio Processing
    SAMPLE_RATE: int = 16000  # Fixed, required by Whisper and Silero
    PCM_CHUNK_SIZE: int = int(os.getenv("PCM_CHUNK_SIZE", "16000"))  # 1 second of audio
    AUDIO_BUFFER_MAX_SIZE: int = int(os.getenv("AUDIO_BUFFER_MAX_SIZE", "320000"))  # ~20 seconds

    # VAD (Voice Activity Detection)
    VAD_ENABLED: bool = os.getenv("VAD_ENABLED", "true").lower() == "true"
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_MIN_SPEECH_DURATION: float = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.1"))  # seconds
    VAD_MIN_SILENCE_DURATION: float = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5"))  # seconds

    # Dynamic Batching
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "16"))
    MAX_BATCH_WAIT_MS: float = float(os.getenv("MAX_BATCH_WAIT_MS", "50"))
    MIN_BATCH_WAIT_MS: float = float(os.getenv("MIN_BATCH_WAIT_MS", "10"))

    # Queue Management
    MAX_QUEUE_DEPTH: int = int(os.getenv("MAX_QUEUE_DEPTH", "1000"))
    QUEUE_TIMEOUT_SECONDS: float = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "300"))  # 5 minutes

    # GPU
    DEVICE: str = os.getenv("DEVICE", "cuda")
    DTYPE: str = os.getenv("DTYPE", "float16")
    GPU_INDEX: int = int(os.getenv("GPU_INDEX", "0"))

    # Whisper Model
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
    WHISPER_CACHE_DIR: str = os.getenv("WHISPER_CACHE_DIR", "/tmp/whisper_cache")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")
    WHISPER_TASK: str = os.getenv("WHISPER_TASK", "transcribe")  # transcribe or translate

    # Whisper Accuracy / Anti-hallucination Knobs
    # no_speech_threshold: suppress output when Whisper's own no-speech prob exceeds this.
    WHISPER_NO_SPEECH_THRESHOLD: float = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
    # compression_ratio_threshold: suppress output that is overly repetitive/compressed.
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "1.35"))

    # Backpressure
    ENABLE_BACKPRESSURE: bool = os.getenv("ENABLE_BACKPRESSURE", "true").lower() == "true"
    BACKPRESSURE_THRESHOLD: float = float(os.getenv("BACKPRESSURE_THRESHOLD", "0.9"))

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").lower() == "true"
    LLM_MODE: str = os.getenv("LLM_MODE", "conversational")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")

    # Metrics
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    METRICS_RETENTION_SECONDS: int = int(os.getenv("METRICS_RETENTION_SECONDS", "3600"))

    # Health Check
    HEALTH_CHECK_INTERVAL: float = float(os.getenv("HEALTH_CHECK_INTERVAL", "10"))

    # Debug
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


def get_config() -> Config:
    """Get global configuration instance."""
    return Config()
