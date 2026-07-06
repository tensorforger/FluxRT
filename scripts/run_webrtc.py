"""
FluxRT WebRTC streaming server.

Streams the StreamProcessor output to any modern browser over WebRTC.
Includes an inline minimal client (HTML+JS) served at `/`.

Usage:
    python scripts/run_webrtc.py
    python scripts/run_webrtc.py --int8
    python scripts/run_webrtc.py --config configs/config_with_reference.json --port 8765

Then open `http://<linux-lan-ip>:8765/` on any LAN client.

Control:
    - Prompt / seed / steps updates: sent over a DataChannel.
    - Reference image upload: HTTP POST /reference with raw image bytes.
      Requires the active config to have `use_reference_image: true`
      (e.g. `--config configs/config_with_reference.json`).
"""

import argparse
import asyncio
import contextlib
import fractions
import io
import itertools
import json
import logging
import os
import threading
import time
from typing import Optional

import av
import cv2
import httpx
import numpy as np
import uvicorn
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

# ComfyUI API-format workflow templates patched + queued by /comfy/edit.
WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_workflows")
QWEN_EDIT_TEMPLATE = os.path.join(WORKFLOWS_DIR, "qwen_edit_2509.api.json")

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle
from fluxrt.webrtc.input_ownership import InputOwnership, consume_peer_input
from fluxrt.webrtc.stats import connection_pool_stats


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fluxrt.webrtc")


# ──────────────────────────────────────────────────────────────────────────────
# Globals — populated in main() so module import stays cheap.
# ──────────────────────────────────────────────────────────────────────────────
sp: Optional[StreamProcessor] = None
input_tensor = None
output_tensor = None
resolution = None
out_resolution = None

latest_rgb: Optional[np.ndarray] = None
# Latest pipeline INPUT frame (cropped BGR), so a recvonly/listening client can
# preview what's being sent in. Guarded by latest_lock alongside latest_rgb.
latest_input_bgr: Optional[np.ndarray] = None
latest_lock = threading.Lock()
producer_stop = threading.Event()

# Serializes the actual pipeline drive (shared-tensor write + read). Both the
# local producer thread and per-peer executor calls go through push_input_frame;
# without this they can race during the camera->peer ownership handoff and drive
# the pipeline from two threads at once.
pipeline_lock = threading.Lock()

# Cached reference image as PNG bytes for GET /reference preview.
latest_reference_png: Optional[bytes] = None
reference_lock = threading.Lock()

pcs: set[RTCPeerConnection] = set()

# Open control DataChannels for cross-client broadcast (e.g. reference image sync).
ctrl_channels: set = set()
ref_version: int = 0

# Pipeline input ownership — the oldest-connected peer with a live video
# track steers the input; every other peer just views the output. Waiting
# peers keep draining their tracks (aiortc decodes inbound RTP regardless),
# and the oldest waiter takes over when the current steerer disconnects.
# When a peer owns the input, the local camera producer thread pauses.
# The input-steering state machine (owner / waiters / active flag / seq counter)
# lives in InputOwnership so the ownership transitions and the recv-eviction
# policy are unit-tested off-GPU (tests/webrtc/). has_server_camera is set from
# --no-server-camera in main(); it gates whether the local camera resumes when
# the last peer leaves.
ownership = InputOwnership(has_server_camera=True)

# Strong refs to peer-input consumer tasks. asyncio only holds weak refs to
# tasks, so without this a consumer can be GC'd mid-run; if its `finally`
# never executes, peer_input_active stays set and the producer never resumes.
peer_input_tasks: set = set()

MAX_REFERENCE_BYTES = 10 * 1024 * 1024  # 10 MB cap for uploaded reference images.

# Configured ComfyUI servers (populated in main() from --comfy-server flags).
# Maps display name → base URL (no trailing slash).
comfy_servers: dict[str, str] = {}

# Lip transfer (LivePortrait postprocessor) runtime state. The pipeline
# always loads the model when `lip_transfer.enable` is true in config, but
# the per-frame postprocess is gated by `set_lip_transfer(bool)`.
lip_active: bool = False

# Mirror the pipeline state so /healthz and late-joining peers can see it.
# Updated from any path that changes the underlying value (API, DataChannel).
current_prompt: str = ""
current_seed: int = 0
current_steps: int = 0

# Liked prompts, persisted as JSON (no DB). Each entry carries the PROMPTING.md
# live-testing scores: style / tracking / stability, 1-5, 0 = unrated.
SAVED_PROMPTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_prompts.json"
)
saved_prompts: list = []
saved_prompts_lock = threading.Lock()


def _load_saved_prompts() -> None:
    global saved_prompts
    try:
        with open(SAVED_PROMPTS_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            saved_prompts = [e for e in data if isinstance(e, dict) and e.get("prompt")]
    except FileNotFoundError:
        saved_prompts = []
    except Exception as exc:
        log.warning("Could not load %s: %s", SAVED_PROMPTS_PATH, exc)
        saved_prompts = []


def _write_saved_prompts() -> None:
    """Atomic write — call with saved_prompts_lock held."""
    tmp = SAVED_PROMPTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(saved_prompts, f, indent=2)
    os.replace(tmp, SAVED_PROMPTS_PATH)


def _clamp_rating(v) -> int:
    try:
        return max(0, min(5, int(v)))
    except (TypeError, ValueError):
        return 0


def broadcast_ctrl(msg: str) -> None:
    """Send a string message to every open control DataChannel."""
    for ch in list(ctrl_channels):
        safe_send(ch, msg)


def safe_send(channel, msg: str) -> None:
    """Send on a DataChannel, dropping it from the broadcast set on failure.
    aiortc raises if the channel closed between the event and the send; that
    must not propagate out of an event callback."""
    try:
        channel.send(msg)
    except Exception as exc:
        log.debug("ctrl send failed, dropping channel: %s", exc)
        ctrl_channels.discard(channel)


def send_to_pc(pc: RTCPeerConnection, msg: str) -> None:
    """Send a control message only to the channels of one peer connection.
    Used for role messages (input:you) that must not reach other clients."""
    for ch in list(getattr(pc, "_fluxrt_channels", ()) or ()):
        safe_send(ch, msg)


def _drive_prompt(prompt: str) -> None:
    """Apply a prompt to whichever processor is active: the live pipeline and/or a
    running batch render (live steering). Both are no-ops when absent."""
    if sp is not None:
        sp.set_prompt(prompt)
    if _batch_manager is not None:
        _batch_manager.set_prompt(prompt)


def _drive_prompt_travel(prompt: str, frames: int, mode: str) -> None:
    """Like _drive_prompt but a slerp/lerp morph over `frames` generated frames."""
    if sp is not None:
        sp.start_prompt_travel(prompt, frames=frames, mode=mode)
    if _batch_manager is not None:
        _batch_manager.start_prompt_travel(prompt, frames=frames, mode=mode)


# ──────────────────────────────────────────────────────────────────────────────
# Shared frame push — runs the BGR input through the pipeline and updates
# `latest_rgb` for any active video track to send out. Used by both the local
# camera producer thread and per-peer track consumers.
# ──────────────────────────────────────────────────────────────────────────────
def push_input_frame(frame_bgr: np.ndarray) -> None:
    global latest_rgb, latest_input_bgr
    h, w = resolution["height"], resolution["width"]
    cropped = crop_maximal_rectangle(frame_bgr, h, w)
    # Hold pipeline_lock across write+read so the producer thread and a peer's
    # executor call can't interleave shared-tensor access during handoff.
    with pipeline_lock:
        input_tensor.copy_from(cropped)
        out_bgr = output_tensor.to_numpy()
    rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    with latest_lock:
        latest_rgb = rgb
        latest_input_bgr = cropped  # served to listeners via GET /input.jpg


def _decode_and_push(frame) -> None:
    """Decode an av.VideoFrame to BGR and run it through the pipeline. Kept
    separate so the CPU-bound decode (to_ndarray) runs in an executor thread,
    never on the asyncio event loop."""
    push_input_frame(frame.to_ndarray(format="bgr24"))


# ──────────────────────────────────────────────────────────────────────────────
# Producer — local webcam frames. Pauses while a peer owns the input.
# ──────────────────────────────────────────────────────────────────────────────
def producer_loop(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Camera %d failed to open", camera_index)
        return

    log.info("Waiting for StreamProcessor to be ready...")
    while not sp.is_ready():
        if producer_stop.is_set():
            cap.release()
            return
        time.sleep(0.1)
    log.info("StreamProcessor ready, producing frames from local camera.")

    while not producer_stop.is_set():
        if ownership.is_active():
            # A peer is driving input — skip local camera frames.
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            log.warning("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        push_input_frame(frame)

    cap.release()
    log.info("Producer stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Per-peer input consumer lives in fluxrt.webrtc.input_ownership.consume_peer_input
# (unit-tested off-GPU). Here we only supply the frame sink (decode + pipeline
# drive, offloaded to an executor) and the role-broadcast hook. The recv-eviction
# POLICY — never evict a live owner on a frame gap; only drop a never-connected
# waiter — also lives there. (c855950 regressed by evicting healthy owners on a
# blind 5s timeout, which froze output under --no-server-camera.)
# ──────────────────────────────────────────────────────────────────────────────
async def _frame_sink(frame) -> None:
    """Owner-pump sink: offload decode (to_ndarray) + the pipeline write/read to
    an executor so the asyncio event loop never blocks."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _decode_and_push, frame)


def _input_notify(event: str, pc, outcome=None) -> None:
    """Role broadcasts for consume_peer_input ownership transitions."""
    if event == "claimed":
        broadcast_ctrl("input:peer")
        send_to_pc(pc, "input:you")
    elif event == "released" and outcome is not None and outcome.became_idle:
        if outcome.server_camera_resumes:
            broadcast_ctrl("input:server")
            log.info("No peer inputs left — server camera resumes")
        else:
            log.info("No peer inputs left — output holds the last frame")


# ──────────────────────────────────────────────────────────────────────────────
# WebRTC video track — emits latest_rgb at a fixed frame rate.
# ──────────────────────────────────────────────────────────────────────────────
class FluxRTTrack(VideoStreamTrack):
    """Yields the latest FluxRT output frame as `av.VideoFrame`s at `fps` Hz."""

    kind = "video"

    def __init__(self, fps: int = 30):
        super().__init__()
        self.fps = fps
        self._t0 = time.time()
        self._n = 0
        # Output frames are at the (possibly upscaled) output resolution.
        h, w = out_resolution["height"], out_resolution["width"]
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self) -> av.VideoFrame:
        # Pace ourselves; aiortc will pull frames as fast as recv() returns.
        self._n += 1
        target_t = self._t0 + self._n / self.fps
        delay = target_t - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Behind schedule — skip ahead so we don't accumulate debt.
            self._t0 = time.time() - self._n / self.fps

        with latest_lock:
            rgb = latest_rgb if latest_rgb is not None else self._blank

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._n
        frame.time_base = fractions.Fraction(1, self.fps)
        return frame


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI signaling app.
# ──────────────────────────────────────────────────────────────────────────────
_cleanup_done = threading.Event()


async def _graceful_cleanup() -> None:
    """Idempotent teardown: stop producer, close peers, stop the pipeline
    (which now joins/terminates its subprocesses with timeouts). Safe to call
    from the lifespan shutdown and from a signal handler."""
    if _cleanup_done.is_set():
        return
    _cleanup_done.set()

    log.info("Shutting down — stopping producer, peers, pipeline...")
    producer_stop.set()

    # Hard-deadline backstop. On Ctrl+C the spawned children also receive SIGINT
    # (shared process group) and die immediately, and reap_process detaches any
    # SIGKILL survivor from multiprocessing._children — so the normal path exits
    # fast. This timer only fires if a teardown step itself wedges; os._exit
    # bypasses the atexit no-timeout child join. (No killpg: no blast radius onto
    # the operator's shell when the server isn't its own process-group leader.)
    def _hard_exit():
        os._exit(1)

    watchdog = threading.Timer(15.0, _hard_exit)
    watchdog.daemon = True
    watchdog.start()
    try:
        # Close all peers concurrently and bounded: a half-open peer's
        # pc.close() can await a transport teardown that never completes
        # (contextlib.suppress does NOT bound a hang), which would otherwise
        # block shutdown before sp.stop() ever runs.
        async def _close(pc):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pc.close(), timeout=3.0)

        if pcs:
            await asyncio.gather(*[_close(pc) for pc in list(pcs)])
        pcs.clear()

        # Tear down the live processor AND any in-flight batch render CONCURRENTLY
        # (independent processes) so neither eats the other's slice of the 15s
        # watchdog budget. Both do blocking joins → run off the event loop. The
        # batch join is bounded short so the live reap keeps headroom.
        loop = asyncio.get_running_loop()
        teardowns = []
        if _batch_manager is not None and _batch_manager.active_job_id() is not None:
            teardowns.append(loop.run_in_executor(None, lambda: _batch_manager.shutdown(4.0)))
        if sp is not None:
            teardowns.append(loop.run_in_executor(None, sp.stop))
        if teardowns:
            await asyncio.gather(*teardowns, return_exceptions=True)
        log.info("Shutdown complete.")
    finally:
        watchdog.cancel()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    await _graceful_cleanup()


app = FastAPI(lifespan=_lifespan)

# Cross-origin: the web client can run anywhere now (a local `vite dev`, a static
# host) and connect to this backend directly. Allow its origin(s) — set
# FLUXRT_CORS_ORIGINS (comma-separated) to lock down; default "*" (no credentials
# are used, so "*" is valid).
_cors_origins = [o.strip() for o in os.environ.get("FLUXRT_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _rtc_config() -> RTCConfiguration:
    """ICE servers for NAT traversal so a remote client can reach this backend.
    STUN by default; add TURN via FLUXRT_TURN_URL (+ _USER/_PASS) when the backend
    is behind NAT — host candidates alone aren't reachable across networks."""
    servers = [RTCIceServer(urls=os.environ.get("FLUXRT_STUN", "stun:stun.l.google.com:19302"))]
    turn = os.environ.get("FLUXRT_TURN_URL")
    if turn:
        servers.append(
            RTCIceServer(
                urls=turn,
                username=os.environ.get("FLUXRT_TURN_USER"),
                credential=os.environ.get("FLUXRT_TURN_PASS"),
            )
        )
    return RTCConfiguration(iceServers=servers)


@app.post("/offer")
async def offer(request: Request):
    if sp is None:
        raise HTTPException(status_code=503, detail="live pipeline not running (batch-only server)")
    body = await request.json()
    sdp = body.get("sdp")
    sdp_type = body.get("type")
    if not sdp or not sdp_type:
        raise HTTPException(status_code=400, detail="offer requires 'sdp' and 'type'")
    offer_sdp = RTCSessionDescription(sdp=sdp, type=sdp_type)

    pc = RTCPeerConnection(_rtc_config())
    pcs.add(pc)
    # Channels opened by this PC — pruned from the broadcast set when the
    # connection dies, since the channel "close" event may never fire then.
    # Also reachable via the pc for targeted role messages (send_to_pc).
    pc_channels: set = set()
    pc._fluxrt_channels = pc_channels

    @pc.on("connectionstatechange")
    async def _on_state():
        log.info("PC state: %s", pc.connectionState)
        # NOTE: aiortc's connectionState is only new/connecting/connected/failed/
        # closed — it never emits "disconnected" (unlike the browser). "disconnected"
        # is kept here as a harmless no-op guard; a dead owner whose pc lingers in
        # "connected" until ICE consent expiry (~30s) is handed off far sooner by
        # the frame-gap policy in consume_peer_input (owner_gap_should_release),
        # not by this handler.
        if pc.connectionState in ("failed", "closed", "disconnected"):
            # Cancel the input consumer directly — do NOT rely solely on
            # pc.close() ending the track to make recv() raise. aiortc's
            # receiver.stop() is a no-op when the receiver never started (DTLS
            # never connected on a failed reconnect), so the consumer's
            # `await track.recv()` would never raise and the task would leak as
            # a phantom waiter that pins peer_input_active and freezes output.
            t = getattr(pc, "_fluxrt_consume_task", None)
            if t is not None:
                t.cancel()
            await pc.close()
            pcs.discard(pc)
            ctrl_channels.difference_update(pc_channels)
            pc_channels.clear()

    @pc.on("track")
    def _on_track(track):
        log.info("Inbound track from peer: kind=%s id=%s", track.kind, track.id)
        if track.kind == "video":
            task = asyncio.ensure_future(
                consume_peer_input(
                    track, pc, ownership, _frame_sink, notify=_input_notify, log=log
                )
            )
            peer_input_tasks.add(task)
            task.add_done_callback(peer_input_tasks.discard)
            # Let _on_state cancel this consumer on disconnect (the pump blocks on
            # recv() — death is detected via connectionState, not a frame gap).
            pc._fluxrt_consume_task = task

    @pc.on("datachannel")
    def _on_dc(channel):
        log.info("DataChannel opened: %s", channel.label)
        ctrl_channels.add(channel)
        pc_channels.add(channel)

        # Send the current reference version to a fresh peer so it can
        # decide whether to refresh its preview.
        if latest_reference_png is not None:
            safe_send(channel, f"ref:set:{ref_version}")

        # Sync lip transfer state on connect.
        if _lip_enabled():
            safe_send(channel, "lip:on" if lip_active else "lip:off")

        # Send current pipeline parameter snapshot to a fresh peer.
        if current_prompt:
            safe_send(channel, f"state:prompt:{current_prompt}")
        safe_send(channel, f"state:seed:{current_seed}")
        safe_send(channel, f"state:steps:{current_steps}")

        # Input-source role sync. The DataChannel usually opens after the
        # video track was claimed, so the targeted input:you from the claim
        # can be missed — resend it here.
        if ownership.is_active():
            safe_send(channel, "input:peer")
            if ownership.owner_is(pc):
                safe_send(channel, "input:you")

        @channel.on("close")
        def _on_close():
            ctrl_channels.discard(channel)
            log.info("DataChannel closed: %s", channel.label)

        @channel.on("message")
        def _on_msg(msg):
            if not isinstance(msg, str):
                return
            global current_prompt, current_seed, current_steps
            if msg.startswith("prompt-travel:"):
                # Wire format (ctrlProtocol.ts): "prompt-travel:<frames>:<mode>:<text>".
                # text may contain colons, so peel off exactly the two leading
                # fields with maxsplit=2. Reject anything that doesn't match the
                # contract (same frames>=1 / mode checks as POST /prompt-travel)
                # rather than silently treating the prefix as prompt text.
                parts = msg[len("prompt-travel:"):].split(":", 2)
                if (
                    len(parts) == 3
                    and parts[0].isdigit()
                    and int(parts[0]) >= 1
                    and parts[1] in ("slerp", "lerp")
                    and parts[2].strip()
                ):
                    frames, mode = int(parts[0]), parts[1]
                    new_prompt = parts[2].strip()
                    log.info(
                        "Prompt travel: %r (frames=%d, mode=%s)",
                        new_prompt,
                        frames,
                        mode,
                    )
                    current_prompt = new_prompt
                    _drive_prompt_travel(new_prompt, frames, mode)
                    safe_send(channel, "ack:prompt")
                    broadcast_ctrl(f"state:prompt:{new_prompt}")
                else:
                    safe_send(channel, "err:prompt-travel")
            elif msg.startswith("prompt:"):
                new_prompt = msg[len("prompt:"):].strip()
                if new_prompt:
                    log.info("Prompt update: %r", new_prompt)
                    current_prompt = new_prompt
                    _drive_prompt(new_prompt)
                    safe_send(channel, "ack:prompt")
                    broadcast_ctrl(f"state:prompt:{new_prompt}")
            elif msg.startswith("seed:"):
                try:
                    seed = int(msg[len("seed:"):])
                    current_seed = seed
                    sp.set_seed(seed)
                    safe_send(channel, f"ack:seed:{seed}")
                    broadcast_ctrl(f"state:seed:{seed}")
                except ValueError:
                    safe_send(channel, "err:seed")
            elif msg.startswith("steps:"):
                try:
                    steps = int(msg[len("steps:"):])
                    if steps < 1 or steps > 8:
                        raise ValueError("steps out of range")
                    current_steps = steps
                    sp.set_steps(steps)
                    safe_send(channel, f"ack:steps:{steps}")
                    broadcast_ctrl(f"state:steps:{steps}")
                except ValueError:
                    safe_send(channel, "err:steps")

    pc.addTrack(FluxRTTrack(fps=30))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reference image upload — requires `use_reference_image: true` in config.
# Accepts raw image bytes (PNG / JPEG / WebP) as the request body.
# Browser posts with `Content-Type: application/octet-stream` (or any).
# ──────────────────────────────────────────────────────────────────────────────
def _reference_enabled() -> bool:
    return bool(sp and sp.config.get("use_reference_image", False))


def _lip_enabled() -> bool:
    """Whether lip transfer is configured + the LivePortrait model loaded."""
    if not sp:
        return False
    return bool(sp.config.get("lip_transfer", {}).get("enable", False))


@app.post("/prompt")
async def post_prompt(request: Request):
    """Set the generation prompt.
    Body: JSON {"prompt": "..."} OR raw text/plain.
    Query: ?prompt=... (URL-encoded) as a third option.
    Works against the live pipeline AND/OR a running batch render (live steering)."""
    if sp is None and (_batch_manager is None or _batch_manager.active_job_id() is None):
        raise HTTPException(status_code=503, detail="no live pipeline or batch render to set a prompt on")

    prompt: str | None = None
    q = request.query_params.get("prompt")
    if q is not None:
        prompt = q
    else:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            try:
                payload = await request.json()
                prompt = payload.get("prompt") if isinstance(payload, dict) else None
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Bad JSON: {exc}")
        else:
            prompt = (await request.body()).decode("utf-8", errors="replace")

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    prompt = prompt.strip()
    global current_prompt
    current_prompt = prompt
    _drive_prompt(prompt)
    log.info("Prompt set via API: %r", prompt)
    broadcast_ctrl(f"state:prompt:{prompt}")
    return JSONResponse({"ok": True, "prompt": prompt})


@app.post("/prompt-travel")
async def post_prompt_travel(request: Request):
    """Smoothly morph from the current prompt to a target prompt.
    Body: JSON {"prompt": "...", "frames": 48, "mode": "slerp"} OR raw
    text/plain (the target prompt, with default frames/mode).
    Query: ?prompt=...&frames=...&mode=... as an alternative.
    Works against the live pipeline AND/OR a running batch render (live steering)."""
    if sp is None and (_batch_manager is None or _batch_manager.active_job_id() is None):
        raise HTTPException(status_code=503, detail="no live pipeline or batch render to morph")

    prompt: str | None = None
    frames_raw = 48
    mode = "slerp"

    q = request.query_params.get("prompt")
    if q is not None:
        prompt = q
        frames_raw = request.query_params.get("frames", frames_raw)
        mode = request.query_params.get("mode", mode)
    else:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            try:
                payload = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Bad JSON: {exc}")
            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="Expected JSON object")
            prompt = payload.get("prompt")
            frames_raw = payload.get("frames", frames_raw)
            mode = payload.get("mode", mode)
        else:
            prompt = (await request.body()).decode("utf-8", errors="replace")

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")
    try:
        frames = int(frames_raw)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="frames must be an integer")
    if frames < 1:
        raise HTTPException(status_code=400, detail="frames must be >= 1")
    if mode not in ("slerp", "lerp"):
        raise HTTPException(status_code=400, detail="mode must be 'slerp' or 'lerp'")

    prompt = prompt.strip()
    global current_prompt
    current_prompt = prompt
    _drive_prompt_travel(prompt, frames, mode)
    log.info("Prompt travel via API: %r (frames=%d, mode=%s)", prompt, frames, mode)
    broadcast_ctrl(f"state:prompt:{prompt}")
    return JSONResponse(
        {"ok": True, "prompt": prompt, "frames": frames, "mode": mode}
    )


@app.post("/seed")
async def post_seed(request: Request):
    raw = request.query_params.get("value")
    if raw is None:
        try:
            body = await request.json()
            raw = body.get("value") if isinstance(body, dict) else None
        except Exception:
            raw = None
    if raw is None:
        raise HTTPException(status_code=400, detail="Missing ?value= (int)")
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Not an int: {raw!r}")

    global current_seed
    current_seed = seed
    sp.set_seed(seed)
    log.info("Seed set via API: %d", seed)
    broadcast_ctrl(f"state:seed:{seed}")
    return JSONResponse({"ok": True, "seed": seed})


@app.post("/steps")
async def post_steps(request: Request):
    raw = request.query_params.get("value")
    if raw is None:
        try:
            body = await request.json()
            raw = body.get("value") if isinstance(body, dict) else None
        except Exception:
            raw = None
    if raw is None:
        raise HTTPException(status_code=400, detail="Missing ?value= (int)")
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Not an int: {raw!r}")
    if steps < 1 or steps > 8:
        raise HTTPException(status_code=400, detail="steps must be 1..8")

    global current_steps
    current_steps = steps
    sp.set_steps(steps)
    log.info("Steps set via API: %d", steps)
    broadcast_ctrl(f"state:steps:{steps}")
    return JSONResponse({"ok": True, "steps": steps})


@app.post("/lip-transfer")
async def post_lip_transfer(request: Request):
    """Toggle LivePortrait lip transfer on/off.
    Query: ?on=true|false (also accepts 1/0, yes/no)."""
    if not _lip_enabled():
        raise HTTPException(
            status_code=400,
            detail=(
                "Lip transfer not enabled in config. Restart with a config "
                "containing lip_transfer.enable = true and the LivePortrait "
                "directory present."
            ),
        )
    raw = request.query_params.get("on", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        on = True
    elif raw in ("0", "false", "no", "off"):
        on = False
    else:
        raise HTTPException(status_code=400, detail="?on= must be true/false")

    global lip_active
    lip_active = on
    sp.set_lip_transfer(on)
    log.info("Lip transfer: %s", "on" if on else "off")
    broadcast_ctrl("lip:on" if on else "lip:off")
    return JSONResponse({"ok": True, "lip_active": on})


def _apply_reference_bytes(body: bytes, source_label: str) -> dict:
    """Decode image bytes, set as reference, cache preview, bump version,
    broadcast. Returns the JSON-shaped result dict. Raises HTTPException
    on decode failure. Used by both /reference upload and /comfy/pull."""
    try:
        img = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decode error: {exc}")

    rgb = np.array(img)
    sp.set_reference_image(rgb)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    global latest_reference_png, ref_version
    with reference_lock:
        latest_reference_png = buf.getvalue()
        ref_version += 1
        version = ref_version

    log.info(
        "Reference image set from %s: %dx%d (%d bytes, v%d)",
        source_label, *rgb.shape[1::-1], len(body), version,
    )
    broadcast_ctrl(f"ref:set:{version}")

    return {
        "ok": True,
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "bytes": len(body),
        "version": version,
        "source": source_label,
    }


@app.post("/reference")
async def post_reference(request: Request):
    if not _reference_enabled():
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference image conditioning is disabled. "
                "Restart with --config configs/config_with_reference.json"
            ),
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_REFERENCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (>{MAX_REFERENCE_BYTES // (1024 * 1024)} MB)",
        )

    return JSONResponse(_apply_reference_bytes(body, "upload"))


# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI integration — pull latest output image from a configured server
# and use it as the reference image. See the SKILL.md in
# `~/Downloads/qwen-edit-test/` for the canonical comfy endpoints.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/comfy/servers")
async def list_comfy_servers():
    return {
        "servers": [{"name": n, "url": u} for n, u in comfy_servers.items()],
    }


@app.post("/comfy/pull")
async def pull_comfy(request: Request):
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")

    name = request.query_params.get("server", "")
    if not name or name not in comfy_servers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown comfy server: {name!r}. "
            f"Known: {list(comfy_servers.keys())}",
        )
    base = comfy_servers[name].rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            hr = await client.get(f"{base}/history", params={"max_items": 20})
            hr.raise_for_status()
            history = hr.json()

            # `history` is dict insertion-ordered; newest entries last.
            chosen = None
            for prompt_id, entry in reversed(list(history.items())):
                outputs = entry.get("outputs", {}) or {}
                for node_id, out in outputs.items():
                    for image in out.get("images", []) or []:
                        if image.get("type") != "output":
                            continue
                        chosen = (prompt_id, node_id, image)
                        break
                    if chosen:
                        break
                if chosen:
                    break

            if not chosen:
                raise HTTPException(
                    status_code=404,
                    detail="No output images in recent comfy history",
                )

            prompt_id, node_id, image = chosen
            params = {
                "filename": image["filename"],
                "type": image["type"],
                "subfolder": image.get("subfolder", ""),
            }
            vr = await client.get(f"{base}/view", params=params)
            vr.raise_for_status()
            body = vr.content

    except httpx.HTTPError as exc:
        log.warning("Comfy pull HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Comfy server error: {exc}")

    label = f"comfy:{name}:{image['filename']}"
    result = _apply_reference_bytes(body, label)
    result["prompt_id"] = prompt_id
    result["node_id"] = node_id
    result["filename"] = image["filename"]
    return JSONResponse(result)


@app.post("/comfy/edit")
async def comfy_edit(request: Request):
    """Snap an image (raw bytes body), run it through the Qwen-Image-Edit
    2509 workflow on the selected comfy server, wait for the result, and
    install it as the reference image.
    Query: ?server=NAME&prompt=... (prompt falls back to the live prompt)."""
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")

    name = request.query_params.get("server", "")
    if not name or name not in comfy_servers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown comfy server: {name!r}. Known: {list(comfy_servers.keys())}",
        )
    base = comfy_servers[name].rstrip("/")

    prompt_text = request.query_params.get("prompt") or current_prompt or "enhance this person"

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        with open(QWEN_EDIT_TEMPLATE, "r") as f:
            wf = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow template error: {exc}")

    # The patch targets specific node ids from the exported graph — fail clearly
    # if the template was re-exported/renumbered rather than KeyError-ing mid-run
    # (after the snap was already uploaded to the comfy server).
    for node_id in ("78", "111", "3"):
        if node_id not in wf or "inputs" not in wf[node_id]:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow template missing node {node_id} — re-export qwen_edit_2509.api.json",
            )

    seed = int.from_bytes(os.urandom(4), "big")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 1. upload the snapped frame to the comfy input/ folder
            files = {"image": ("fluxrt_snap.png", body, "image/png")}
            ur = await client.post(
                f"{base}/upload/image", files=files, data={"overwrite": "true"}
            )
            ur.raise_for_status()
            up = ur.json()
            uploaded_name = up["name"]

            # 2. patch the workflow: input image, prompt, seed
            wf["78"]["inputs"]["image"] = uploaded_name
            wf["111"]["inputs"]["prompt"] = prompt_text
            wf["3"]["inputs"]["seed"] = seed

            # 3. queue it
            pr = await client.post(f"{base}/prompt", json={"prompt": wf})
            pr.raise_for_status()
            prompt_id = pr.json()["prompt_id"]
            log.info("Qwen edit queued on %s: prompt_id=%s seed=%d", name, prompt_id, seed)

            # 4. poll history until the output image appears (~4-step, fast)
            out_image = None
            for _ in range(90):
                await asyncio.sleep(1.0)
                hr = await client.get(f"{base}/history/{prompt_id}")
                if hr.status_code != 200:
                    continue
                hist = hr.json()
                entry = hist.get(prompt_id)
                if not entry:
                    continue
                for _node, out in (entry.get("outputs", {}) or {}).items():
                    for img in out.get("images", []) or []:
                        if img.get("type") == "output":
                            out_image = img
                            break
                    if out_image:
                        break
                if out_image:
                    break

            if not out_image:
                raise HTTPException(status_code=504, detail="Qwen edit timed out")

            # 5. fetch the result
            vr = await client.get(
                f"{base}/view",
                params={
                    "filename": out_image["filename"],
                    "type": out_image["type"],
                    "subfolder": out_image.get("subfolder", ""),
                },
            )
            vr.raise_for_status()
            result_bytes = vr.content

    except httpx.HTTPError as exc:
        log.warning("Comfy edit HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Comfy server error: {exc}")

    # 6. install the edited image as the reference
    result = _apply_reference_bytes(result_bytes, f"qwen-edit:{name}:{out_image['filename']}")
    result["prompt_id"] = prompt_id
    result["filename"] = out_image["filename"]
    return JSONResponse(result)


@app.delete("/reference")
async def delete_reference():
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")
    sp.set_reference_image(None)
    global latest_reference_png, ref_version
    with reference_lock:
        latest_reference_png = None
        ref_version += 1
        version = ref_version
    log.info("Reference image cleared (v%d)", version)
    broadcast_ctrl(f"ref:clear:{version}")
    return JSONResponse({"ok": True, "cleared": True, "version": version})


@app.get("/reference")
async def get_reference():
    with reference_lock:
        png = latest_reference_png
    if png is None:
        raise HTTPException(status_code=404, detail="No reference image set")
    return Response(content=png, media_type="image/png")


@app.get("/input.jpg")
async def get_input_jpg():
    # Current pipeline INPUT frame as JPEG — what's actually feeding the model
    # (a peer's stream or the server camera). Lets a recvonly/listening client
    # preview the input it isn't sending. 204 when nothing is driving input.
    if ownership.is_active() or ownership.has_server_camera:
        with latest_lock:
            frame = None if latest_input_bgr is None else latest_input_bgr.copy()
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                return Response(
                    content=buf.tobytes(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"},
                )
    return Response(status_code=204)


# ──────────────────────────────────────────────────────────────────────────────
# Saved prompts — like/rate prompts, shared across clients, JSON-file backed.
# Ratings follow the PROMPTING.md scale: style / tracking / stability, 1-5.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/prompts")
async def list_saved_prompts():
    with saved_prompts_lock:
        return {"prompts": list(saved_prompts)}


@app.post("/prompts")
async def save_saved_prompt(request: Request):
    """Upsert a prompt. Body: JSON {"prompt": "...", "style": 0-5,
    "tracking": 0-5, "stability": 0-5} — ratings optional, 0 = unrated."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad JSON: {exc}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    entry = {
        "prompt": prompt,
        "style": _clamp_rating(payload.get("style")),
        "tracking": _clamp_rating(payload.get("tracking")),
        "stability": _clamp_rating(payload.get("stability")),
    }
    with saved_prompts_lock:
        for e in saved_prompts:
            if e["prompt"] == prompt:
                e.update(entry)
                break
        else:
            saved_prompts.append(entry)
        _write_saved_prompts()
        count = len(saved_prompts)

    log.info("Prompt saved (%d/%d/%d): %r",
             entry["style"], entry["tracking"], entry["stability"], prompt)
    broadcast_ctrl("prompts:changed")
    return JSONResponse({"ok": True, "count": count, "entry": entry})


@app.delete("/prompts")
async def delete_saved_prompt(request: Request):
    prompt = (request.query_params.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing ?prompt=")
    with saved_prompts_lock:
        before = len(saved_prompts)
        saved_prompts[:] = [e for e in saved_prompts if e["prompt"] != prompt]
        removed = before - len(saved_prompts)
        if removed:
            _write_saved_prompts()
    if not removed:
        raise HTTPException(status_code=404, detail="Prompt not in saved list")
    log.info("Prompt deleted: %r", prompt)
    broadcast_ctrl("prompts:changed")
    return JSONResponse({"ok": True, "count": before - removed})


@app.get("/")
async def _index():
    # Backend-only now (the web client lives in the realtime-client repo and is
    # hosted separately). Root is a liveness/identity ping; see /healthz for stats.
    return JSONResponse({"service": "fluxrt-webrtc", "ok": True})


@app.get("/test")
async def _test_client():
    """Minimal standalone WebRTC test client (same-origin), with an ICE
    diagnostics panel — for comparing the FluxRT connection candidate-by-candidate
    against the StreamDiffusion (sd-webrtc) client. Open http://<server>/test."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webrtc_test_client.html")
    try:
        with open(path, encoding="utf-8") as f:
            return Response(f.read(), media_type="text/html")
    except OSError:
        raise HTTPException(status_code=404, detail="webrtc_test_client.html not found")


@app.get("/healthz")
async def _health():
    # Batch-only server has no live pipeline — return a minimal payload so a client
    # polling /healthz (e.g. the clip header) doesn't error on the missing sp.
    if sp is None:
        return {"ready": False, "peers": 0, "batch_only": True}
    return {
        "ready": bool(sp and sp.is_ready()),
        "peers": len(pcs),
        # Active connection-pool breakdown (total / connected / per-state).
        "connections": connection_pool_stats(
            [getattr(pc, "connectionState", "unknown") for pc in list(pcs)]
        ),
        "resolution": resolution,
        "reference_enabled": _reference_enabled(),
        "reference_set": latest_reference_png is not None,
        "reference_version": ref_version,
        "input_source": (
            "peer"
            if ownership.is_active()
            else "server"
            if ownership.has_server_camera
            else "none"
        ),
        "input_waiters": ownership.num_waiters(),
        "lip_enabled": _lip_enabled(),
        "lip_active": lip_active,
        "prompt": current_prompt,
        "seed": current_seed,
        "steps": current_steps,
        **_perf_metrics(),
    }


def _perf_metrics() -> dict:
    """Pipeline FPS + VRAM snapshot for /healthz polling."""
    if not sp:
        return {
            "fps_pipeline": 0.0,
            "fps_interpolated": 0.0,
            "proc_time_ms": 0.0,
            "vram_mb": 0,
        }
    pt = sp.get_last_processing_time() or 0.0
    base = (1.0 / pt) if pt > 0 else 0.0
    exp = int(sp.config.get("interpolation_exp", 0))
    try:
        vram_mb = int(sp.get_reserved_memory())
    except Exception:
        vram_mb = 0
    return {
        "fps_pipeline": round(base, 2),
        "fps_interpolated": round(base * (2 ** exp), 2),
        "proc_time_ms": round(pt * 1000.0, 2),
        "vram_mb": vram_mb,
    }



# ──────────────────────────────────────────────────────────────────────────────
# Batch frame-for-frame render (offline). A dedicated, parked StreamProcessor
# renders a whole video 1:1 (scripts/batch_render.py + docs/batch-render-spec.md).
# Gated behind an explicit opt-in: a second full FLUX.2 model does NOT co-fit with
# the live stream on a 24 GB GPU, so the operator enables this only on a host with
# the capacity. The batch StreamProcessor is created on submit and torn down on
# completion, so an idle server pays nothing.
# ──────────────────────────────────────────────────────────────────────────────
_batch_enabled = os.environ.get("FLUXRT_BATCH_RENDER") == "1"
# Batch-only: serve the batch API WITHOUT starting the live pipeline, so the batch
# model is the only one on the GPU (the live model otherwise stays resident and a
# second won't co-fit on a 24 GB card). Set via FLUXRT_BATCH_ONLY=1 or --batch-only.
_batch_only = os.environ.get("FLUXRT_BATCH_ONLY") == "1"
_batch_cfg_path: Optional[str] = None  # config to clone for batch when sp is absent
_batch_manager = None


def _free_vram_mb() -> Optional[int]:
    """Free VRAM (MB) of the most-free GPU via nvidia-smi, or None if unavailable.
    Uses the CLI on purpose — importing torch / calling mem_get_info() here would
    create a CUDA context in the web process and waste VRAM that the live model needs."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
    except Exception:
        return None
    vals = [int(x) for x in out.split() if x.strip().isdigit()]
    return max(vals) if vals else None


def _batch_vram_preflight() -> None:
    """Best-effort gate: refuse a batch job when a second full model is unlikely to
    fit alongside the live one. Skips silently if VRAM can't be measured — the
    FLUXRT_BATCH_RENDER opt-in remains the primary guard. Raises RuntimeError (→ 409)
    when there isn't headroom."""
    free_mb = _free_vram_mb()
    if free_mb is None:
        return  # cannot measure → rely on the operator opt-in
    need_mb = 0
    with contextlib.suppress(Exception):
        if sp is not None:
            need_mb = int(sp.get_reserved_memory())  # live model footprint ≈ batch model footprint
    if need_mb and free_mb < need_mb * 1.1:
        raise RuntimeError("stop the live stream to run a batch render (insufficient free VRAM for a second model)")


def _batch_base_config() -> dict:
    """Config the batch instance clones: the live processor's when it exists,
    otherwise (batch-only) the config file loaded from disk."""
    if sp is not None:
        return sp.config
    if _batch_cfg_path:
        with open(_batch_cfg_path) as f:
            return json.load(f)
    raise RuntimeError("no config available for batch render")


def _get_batch_manager():
    global _batch_manager
    if _batch_manager is None:
        from batch_render import BatchJobManager  # local import: pulls PyAV only when used

        _batch_manager = BatchJobManager(
            base_config=_batch_base_config(),
            make_processor=lambda cfg: StreamProcessor(cfg),
            preflight=_batch_vram_preflight,
        )
    return _batch_manager


from batch_routes import make_batch_router  # noqa: E402

app.include_router(
    make_batch_router(
        get_manager=_get_batch_manager,
        # Enabled when opted in with a live processor present, OR in batch-only mode
        # (no live model — the batch model has the GPU to itself).
        is_enabled=lambda: (_batch_enabled or _batch_only) and (sp is not None or _batch_only),
        default_prompt=lambda: current_prompt or (sp.config.get("default_prompt", "") if sp else ""),
    )
)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point.
# ──────────────────────────────────────────────────────────────────────────────
def _run_server(args) -> None:
    use_tls = bool(args.ssl_certfile and args.ssl_keyfile)
    scheme = "https" if use_tls else "http"
    log.info("Open %s://<this-host>:%d/ in a LAN browser", scheme, args.port)
    if not use_tls:
        log.warning(
            "Running plain HTTP — browsers will block getUserMedia on non-"
            "localhost origins. Pass --ssl-certfile + --ssl-keyfile for HTTPS."
        )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


def main() -> None:
    global sp, input_tensor, output_tensor, resolution, out_resolution

    parser = argparse.ArgumentParser(description="FluxRT WebRTC server")
    parser.add_argument("--config", default="configs/stream_processor_config.json")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    parser.add_argument(
        "--tiny-vae", action="store_true", help="Enable TAEF2 tiny VAE (enable_tiny_vae)"
    )
    parser.add_argument(
        "--flow-upscaler",
        action="store_true",
        help="Enable 2x latent flow upscaler (enable_flow_upscaler); pairs well with --tiny-vae",
    )
    parser.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Serve ONLY the /batch/* render API; do not start the live pipeline so "
        "the batch model has the GPU to itself. Implies batch render enabled.",
    )
    parser.add_argument(
        "--interp",
        type=int,
        default=None,
        metavar="EXP",
        help=(
            "Override interpolation_exp from config. RIFE emits 2^EXP output "
            "frames per generated frame (0=off, 1=2x, 2=4x, 3=8x). Boot-time "
            "only — sizes the shared frame buffer, cannot change at runtime."
        ),
    )
    parser.add_argument(
        "--no-server-camera",
        action="store_true",
        help=(
            "Do not open the local OpenCV camera. Pipeline input then comes "
            "from the first WebRTC peer that sends a video track."
        ),
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS cert (enables HTTPS, required by browsers for "
        "getUserMedia on non-localhost origins).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key. Use together with --ssl-certfile.",
    )
    parser.add_argument(
        "--comfy-server",
        action="append",
        default=[],
        metavar="NAME=URL",
        help=(
            "Register a ComfyUI server the client can pull reference images "
            "from. Repeatable. Example: "
            "--comfy-server A=http://comfy-host:12040"
        ),
    )
    args = parser.parse_args()

    # Populate comfy_servers dict. Servers come from --comfy-server flags, or
    # fall back to the FLUXRT_COMFY_SERVERS env var (comma-separated NAME=URL
    # entries, e.g. "A=http://host:12040,B=http://host:12041"). No addresses
    # are baked into the source. If neither is set, no servers are registered.
    raw_entries = args.comfy_server or [
        e.strip()
        for e in os.environ.get("FLUXRT_COMFY_SERVERS", "").split(",")
        if e.strip()
    ]
    for entry in raw_entries:
        name, _, url = entry.partition("=")
        name = name.strip()
        url = url.strip().rstrip("/")
        if not name or not url:
            log.warning("Ignoring bad --comfy-server entry: %r", entry)
            continue
        comfy_servers[name] = url
    log.info("Configured comfy servers: %s", comfy_servers)

    config_path = args.config

    # Some flags must be baked into the config the StreamProcessor constructor
    # reads (it sizes buffers / selects models from it in __init__), so collect
    # any overrides and write them to a patched temp config the constructor uses.
    if args.interp is not None and (args.interp < 0 or args.interp > 4):
        parser.error("--interp must be in 0..4")
    overrides: dict = {}
    if args.interp is not None:
        overrides["interpolation_exp"] = args.interp
    if args.tiny_vae:
        overrides["enable_tiny_vae"] = True
    if args.flow_upscaler:
        overrides["enable_flow_upscaler"] = True
    if overrides:
        import atexit
        import json as _json
        import tempfile

        with open(args.config) as f:
            cfg = _json.load(f)
        cfg.update(overrides)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="fluxrt_cfg_", delete=False
        )
        _json.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name
        # Don't leave the patched temp config behind in /tmp on exit.
        atexit.register(lambda p=config_path: os.path.exists(p) and os.unlink(p))
        log.info("config overrides from flags: %s", overrides)

    _load_saved_prompts()
    log.info("Loaded %d saved prompts from %s", len(saved_prompts), SAVED_PROMPTS_PATH)

    global _batch_only, _batch_cfg_path
    _batch_only = _batch_only or args.batch_only
    if _batch_only:
        # No live model — the batch instance (created per job) owns the GPU.
        _batch_cfg_path = config_path
        log.info("Batch-only mode: live pipeline NOT started; serving /batch/* only (config %s)", config_path)
        _run_server(args)
        return

    log.info("Loading StreamProcessor from %s", config_path)
    sp = StreamProcessor(config_path)

    # Lip transfer only loads under int8 — bf16 + LivePortrait exceeds the
    # 4090's 24 GB. Mutate the in-memory config before sp.start() so the
    # inference subprocess never instantiates the postprocessor.
    # ModelInferenceSubprocess holds the same dict reference and the spawn
    # pickle happens during sp.start(), so this propagates correctly.
    lp_cfg = sp.config.get("lip_transfer")
    if lp_cfg and lp_cfg.get("enable"):
        if args.int8:
            log.info("Lip transfer: enabled (int8 mode)")
        else:
            log.warning(
                "Lip transfer disabled: bf16 mode + LivePortrait would exceed "
                "24 GB VRAM. Re-run with --int8 to enable lipsync."
            )
            lp_cfg["enable"] = False

    if args.int8:
        sp.enable_quantization()
    sp.start()

    # Initialize parameter mirror from config + CLI overrides.
    global current_prompt, current_seed, current_steps
    current_prompt = args.initial_prompt or sp.config.get("default_prompt", "")
    current_seed = int(sp.config.get("default_seed", 0))
    current_steps = int(sp.config.get("default_steps", 2))

    if args.initial_prompt:
        sp.set_prompt(args.initial_prompt)

    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()
    resolution = sp.get_resolution()
    out_resolution = sp.get_out_resolution()
    log.info(
        "Resolution: in %dx%d  out %dx%d",
        resolution["width"],
        resolution["height"],
        out_resolution["width"],
        out_resolution["height"],
    )

    if args.no_server_camera:
        log.info("--no-server-camera: skipping local camera; waiting for peer input.")
        # No local producer exists — /healthz reports 'none' until a peer claims,
        # and the ownership object will not pretend a server camera resumes.
        ownership.has_server_camera = False
    else:
        producer_thread = threading.Thread(
            target=producer_loop, args=(args.camera,), daemon=True
        )
        producer_thread.start()

    _run_server(args)


if __name__ == "__main__":
    main()
