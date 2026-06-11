# RunPod STT API Integration - Technical Specification

This document provides a comprehensive technical breakdown of how the RunPod WebSocket API is currently integrated into the VoiceFlow AI backend. It details the connection lifecycle, audio processing pipeline, worker pool architecture, distributed queueing, and billing logic.

## 1. System Architecture Overview

The backend uses a decoupled **L3/L4 Worker-Pool and FIFO Queue Architecture** for interacting with RunPod. This isolates the client's browser WebSocket connection from the actual RunPod Whisper GPU WebSocket connection, allowing thousands of transient client streams to share a fixed pool of persistent GPU connections.

### High-Level Data Flow
1. **Client $\rightarrow$ Server (Proxy)**: Browser streams raw Float32 microphone data (binary) via WebSockets.
2. **Server Processing**: Audio is cleaned (RNNoise), VAD-gated, and converted to Int16.
3. **Queueing**: Cleaned audio chunks are Base64 encoded and pushed to a Redis FIFO queue.
4. **Dispatcher**: A recursive event-loop pulls chunks and assigns them to available RunPod Workers.
5. **Server $\rightarrow$ RunPod**: Raw Int16 PCM buffers are streamed directly to the GPU.
6. **RunPod $\rightarrow$ Server**: RunPod returns JSON transcriptions (partial/final).
7. **Server $\rightarrow$ Client**: Backend relays `partial` and `final` JSON events back to the client UI.

---

## 2. Authentication & Client Session Management

### WebSocket Initialization (`runpodSocket.js`)
- **Endpoint**: Clients connect via query parameters (e.g., `ws://<server>/ws/runpod?api_key=...&device_session_id=...`).
- **Authentication**: Validates standard JWTs for internal users or `vf_live_...` API Keys for developers. Checks for disabled keys or insufficient balances (`<= 0.01`).

### Distributed Device Locking (`deviceSessionManager.js`)
To prevent users from opening multiple browser tabs and draining API credits with duplicate audio streams:
- **Distributed Locks**: Uses Redis `SETNX` (`lock:device:{id}`) to lock the hardware device session to a specific active WebSocket.
- **Resilience**: Features 60-second TTLs on Redis keys. Clients emit `{"type": "heartbeat"}` JSON pings every 15 seconds to refresh the TTL.
- **Reconnect Recovery**: If a client temporarily drops connection, they can reconnect within the 60s grace period using a secure `resume_token`. The backend hot-swaps the underlying active RunPod session without duplicating the GPU stream.

---

## 3. Client Audio Packet Specification

The client must stream binary data using a strict 32-byte header, followed by raw Float32 PCM audio (typically 16kHz).

**Binary Packet Structure**:
- `Bytes 0-15`: **Session ID** (16-byte Hex UUID).
- `Bytes 16-19`: **Sequence Number** (Uint32, Big Endian). Prevents out-of-order processing.
- `Bytes 20-27`: **Timestamp** (Float64, Big Endian) for latency tracking.
- `Bytes 28-31`: **Padding/Reserved** (4 bytes).
- `Bytes 32+`: **Payload** (Float32 PCM Audio Samples).

---

## 4. The Processing Pipeline (`runpodPipeline.js`)

Before audio is sent to RunPod, it undergoes a deep cleaning pipeline to maximize Whisper's transcription accuracy and prevent hallucinations.

1. **RMS Silence Pre-Gate**: Drop completely silent chunks instantly using an RMS threshold (`0.003`) to save CPU and GPU cost.
2. **Format Conversion**: Float32 is normalized to Int16.
3. **RNNoise WASM**: Int16 (upsampled to 48kHz temporarily) is processed through a neural noise suppressor to remove background static.
4. **Silero VAD**: The downsampled 16kHz audio is aggregated into 320ms frames and scored by the Silero Voice Activity Detector.
5. **Filtration**: Frames with a speech probability $< 0.35$ are discarded.

---

## 5. Worker Pool & FIFO Queue Engine (`runpodTranscriptionService.js`)

This is the core orchestration engine interfacing with RunPod.

### A. The Persistent Worker Pool
- On server startup, exactly `5` persistent WebSocket connections (`RUNPOD_WS_URL`) are spawned.
- **Self-Healing**: If a worker disconnects, it attempts to reconnect with an exponential backoff (3s $\rightarrow$ 6s $\rightarrow$ 12s $\rightarrow$ 24s $\rightarrow$ capped at 30s).
- **Sticky Sessions**: Once a worker pulls a chunk for `Session A`, it remains bound to `Session A` until the session completes or times out.

### B. Redis FIFO Queue
- Cleaned Int16 audio is Base64 encoded and pushed (`RPUSH`) to `runpod:audio:queue`.
- **Backpressure**: Capped at `200` chunks. If exceeded, the server selectively drops new chunks to prevent memory bloat.
- **Fallback**: If Redis is offline, it seamlessly degrades to an in-memory local JavaScript array queue.

### C. The Dispatcher Loop
- A recursive, non-blocking asynchronous `setTimeout` loop.
- **Polling Strategy**: 
  - `1000ms` sleep if the server has 0 active sessions (saves Redis request limits).
  - `100ms` sleep if sessions are active but the queue is temporarily empty.
  - `50ms` sleep if the queue has chunks but all 5 GPU workers are busy.
  - `0ms` sleep when rapidly dequeuing chunks to free workers.
- **Dispatch**: Uses `LINDEX 0` to peak at the queue. If the corresponding client session is still alive, it allocates a free worker, pops the chunk (`LPOP`), decodes the Base64, and sends the raw Int16 PCM buffer to the RunPod Worker.

---

## 6. RunPod GPU Responses & State Management

### Handling Worker Messages
- The RunPod WebSocket sends back JSON containing `{ status, text/transcript, latency_seconds }`.
- The worker updates the centralized in-memory session map with the new `lastText` and tracks the GPU latency.
- It immediately routes a `partial` JSON event back to the client's browser WebSocket to update the "Typewriter UI".

### Silence & Turn Committing
- If the RMS pre-gate or Silero VAD detects continuous silence (~800ms) after speech, the system assumes the user has finished a sentence.
- `commitFinalRunpod()` is triggered.
- A `final` JSON event is sent to the client. The chunk is marked as completed, unlocking the transcription block in the UI.

---

## 7. Billing & Database Persistence

To maintain ultra-low latency, the database is only updated minimally during the real-time stream.

1. **In-Memory Telemetry**: Throughout the stream, the `sessionManager.js` aggregates total audio duration, total GPU latency, and total characters transcribed in memory.
2. **Chunk Persistence**: After every finalized turn, the transcript string and its latency metrics are appended to the PostgreSQL database (`Transcript` table) asynchronously.
3. **Session Finalization**: When the client disconnects or times out:
   - `finalizeRealtimeSession()` executes.
   - Computes total cost dynamically based on constants:
     - Audio Length: `$0.0001 / second`
     - GPU Execution: `$0.0005 / second`
     - Characters: `$0.001 / 1000 characters`
4. **Atomic Ledger Commit**: Executes a strict Prisma `$transaction` that simultaneously:
   - Updates the `Session` row with `totalCost` and marks it `COMPLETED`.
   - Decrements the `balance` field on the corresponding `ApiKey` row.
   - If one query fails, the entire billing transaction rolls back automatically.
