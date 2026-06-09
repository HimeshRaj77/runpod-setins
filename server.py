"""FastAPI Server Entry Point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from config import get_config
from websocket_server import STTServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

config = get_config()
stt_server = STTServer(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stt_server.start()
    yield
    # Shutdown
    await stt_server.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def health_check():
    """Health check endpoint required by architecture."""
    return {
        "status": "running",
        "model": config.WHISPER_MODEL,
        "device": config.DEVICE,
        "active_connections": stt_server.registry.get_connection_count(),
        "queue_depth": stt_server.audio_queue.get_depth()
    }

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint required by architecture."""
    return JSONResponse(content=stt_server.get_metrics())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving audio streams."""
    await stt_server.handle_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        # Send a WebSocket ping frame every 20 s.
        # This keeps the RunPod proxy tunnel alive and prevents the
        # proxy from treating the connection as idle and closing it.
        ws_ping_interval=20,   # seconds between pings
        ws_ping_timeout=30,    # seconds to wait for pong before closing
    )
