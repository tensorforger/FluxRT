"""
Input-ownership state machine + per-peer input consumer for the FluxRT WebRTC
server.

The oldest-connected peer with a live video track steers the pipeline input;
every other peer views the output. When the steering peer leaves, the oldest
waiter takes over; if none remain, the local server camera resumes (or the
output holds its last frame under --no-server-camera).

This module is torch/aiortc/FastAPI-free *to import* (aiortc's MediaStreamError
is imported with a fallback) so the ownership transitions and the recv policy
are unit-testable with fake tracks and fake peer-connection objects.

recv POLICY (this is the bug class that regressed in c855950):
- A LONE OWNER is NEVER evicted because of a frame gap. A healthy owner
  legitimately produces no frame for >5s (first keyframe after claiming,
  ICE/DTLS/TURN settle, a brief stall, a paused camera). Death is detected
  out-of-band: the caller's connectionstatechange handler cancels this task when
  the pc goes terminal.
- An OWNER WITH A WAITER READY does yield on a gap >= OWNER_GAP_WITH_WAITER
  (owner_gap_should_release). This is NOT the c855950 blind evict — it fires
  only when a takeover candidate exists, bounding a dead owner's freeze (abrupt
  network loss keeps the pc 'connected' until ICE consent expiry ~30s) instead
  of frozen output while a ready client sits idle. The yielded peer rejoins as
  the NEWEST waiter: its track keeps being drained (aiortc decodes inbound RTP
  into an unbounded queue whether or not anyone recv()s) and it can reclaim
  when the slot frees.
- A WAITER is bounded only on its FIRST frame: a peer that never delivers a frame
  AND is not in the 'connected' ICE state is a dead reconnect and is dropped after
  WAITER_FIRST_FRAME_DEADLINE. Once it has delivered a frame, it is never evicted
  on a gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

try:  # aiortc is present on the server, absent in unit tests.
    from aiortc.mediastreams import MediaStreamError
except Exception:  # pragma: no cover - exercised only when aiortc is installed

    class MediaStreamError(Exception):
        """Fallback so this module imports without aiortc (tests)."""


# ── recv policy ───────────────────────────────────────────────────────────────
# Terminal connection states: the pc is dead, ownership/waiter slot must release.
TERMINAL_STATES = ("failed", "closed", "disconnected")
# How long a waiter that has NEVER delivered a frame may stay before it is treated
# as a dead reconnect (only if it is also not 'connected').
WAITER_FIRST_FRAME_DEADLINE = 25.0

# How long an OWNER may receive no frame before it yields the input — but ONLY
# when another publisher is waiting to take over (see owner_gap_should_release).
# A dead peer's pc stays 'connected' in aiortc until ICE consent expiry (aioice
# CONSENT_INTERVAL=5 * CONSENT_FAILURES=6 ≈ 25-35s), which would freeze the input
# that whole time even though a ready client is queued. This bounds the freeze to
# ~this gap when a takeover candidate exists.
#
# Accepted limitation: a waiter whose network just died still counts as a
# candidate (its state stays 'connected' until consent expiry), so a paused
# owner can yield to a corpse. Self-healing: the yielded owner rejoins as a
# waiter, so the corpse — silent, with a waiter present — gap-yields right back
# within this same threshold (or consent-expires), returning the input.
OWNER_GAP_WITH_WAITER = 8.0


def owner_should_release(connection_state: str) -> bool:
    """An owner is released ONLY on a terminal connection state, never on a frame
    gap. (c855950 evicted on a blind 5s recv timeout regardless of state — that
    froze healthy owners; this is the fix.)"""
    return connection_state in TERMINAL_STATES


def owner_gap_should_release(
    gap_s: float, other_waiters: int, threshold: float = OWNER_GAP_WITH_WAITER
) -> bool:
    """An owner yields the input on a receive gap ONLY when (a) the gap is at
    least `threshold` seconds AND (b) another publisher is waiting to drive. No
    waiter → the owner keeps the slot regardless of the gap: this is NOT a blind
    gap-evict (the c855950 regression), it is a handoff to a ready candidate. It
    bounds a dead owner's freeze (abrupt network loss, pc still 'connected' until
    ICE consent expiry) to ~threshold when someone can take over."""
    return other_waiters >= 1 and gap_s >= threshold


def waiter_should_evict(
    connection_state: str, got_first_frame: bool, deadline_passed: bool
) -> bool:
    """A waiter is evicted only if it has never delivered a frame, its first-frame
    deadline passed, and it is not currently 'connected'. A connected-but-slow
    waiter keeps waiting; a waiter that delivered a frame is never gap-evicted."""
    return (not got_first_frame) and deadline_passed and connection_state != "connected"


def _conn_state(pc) -> str:
    return getattr(pc, "connectionState", "connected")


@dataclass(frozen=True)
class ReleaseOutcome:
    """Result of release(): the I/O (broadcasts) is the caller's job, the decision
    lives here."""

    had_owner: bool
    became_idle: bool  # ownership went None AND no waiters remain
    server_camera_resumes: bool


class InputOwnership:
    """Single owner of the input-steering state. All of input_owner / waiters /
    the active flag / the seq counter are mutated only here, under one lock — so
    a phantom consumer can never pin the active flag against a dead pc."""

    def __init__(self, has_server_camera: bool = True):
        self._lock = threading.Lock()
        self._owner = None  # pc identity, or None
        self._waiters: dict[int, object] = {}  # seq -> pc
        self._seq = itertools.count()
        self._active = threading.Event()  # the producer thread waits on this
        self.has_server_camera = has_server_camera

    def register_waiter(self, pc) -> int:
        with self._lock:
            seq = next(self._seq)
            self._waiters[seq] = pc
            return seq

    def try_claim(self, seq: int, pc) -> bool:
        """Become owner iff no one owns and this seq is the oldest waiter.
        Idempotent if already owner."""
        with self._lock:
            if self._owner is pc:
                return True
            if self._owner is None and self._waiters and seq == min(self._waiters):
                self._owner = pc
                self._active.set()
                return True
            return False

    def release(self, seq: int, pc) -> ReleaseOutcome:
        """The ONLY path that clears ownership / the active flag."""
        with self._lock:
            self._waiters.pop(seq, None)
            had_owner = self._owner is pc
            if had_owner:
                self._owner = None
            became_idle = False
            server_camera_resumes = False
            if self._owner is None and not self._waiters and self._active.is_set():
                self._active.clear()
                became_idle = True
                server_camera_resumes = self.has_server_camera
            return ReleaseOutcome(had_owner, became_idle, server_camera_resumes)

    def owner_is(self, pc) -> bool:
        with self._lock:
            return self._owner is pc

    def is_active(self) -> bool:
        return self._active.is_set()

    def active_event(self) -> threading.Event:
        """The same Event the local camera producer thread waits on."""
        return self._active

    def num_waiters(self) -> int:
        with self._lock:
            return len(self._waiters)

    def num_other_waiters(self, pc) -> int:
        """Count of registered waiters that are NOT `pc` — peers ready to take
        over if `pc` (the current owner) releases. The owner stays registered in
        _waiters until release, so this excludes the owner's own slot."""
        with self._lock:
            return sum(1 for w in self._waiters.values() if w is not pc)


# Type of the frame sink: an async callable that consumes one decoded VideoFrame
# (the caller offloads decode + pipeline drive to an executor inside it).
FrameSink = Callable[[object], Awaitable[None]]
# Role-broadcast hook: notify(event, pc, outcome=None); event in {"claimed","released"}.
NotifyHook = Callable[..., None]


async def _pump_owner_frames(
    track,
    pc,
    sink: FrameSink,
    ownership: InputOwnership,
    *,
    log=None,
    gap_release_s: float = OWNER_GAP_WITH_WAITER,
) -> bool:
    """Drain-to-latest: a reader task always overwrites `latest`, the processing
    loop drives only the newest frame. Bounds round-trip lag and memory when the
    pipeline is slower than the peer camera.

    The owner reader BLOCKS on recv() with no timeout — a frame gap never evicts
    a *lone* owner (a paused camera / slow first keyframe is legitimate; the
    caller cancels this task when the pc goes terminal). BUT when another
    publisher is waiting to take over, an owner that stops delivering frames for
    >= gap_release_s yields the input (owner_gap_should_release): a dead owner's
    pc stays 'connected' until ICE consent expiry (~30s), which would otherwise
    freeze the input while a ready client sits idle.

    Returns True on a gap-yield (pc may still be alive — the caller MUST keep
    draining the track and may re-register as a waiter), False when the track
    ended/errored."""
    loop = asyncio.get_running_loop()
    latest = [None]
    last_recv_t = [loop.time()]  # claim time is t0; a real gap counts from now
    new_frame = asyncio.Event()
    stopped = asyncio.Event()

    async def _reader():
        try:
            while True:
                latest[0] = await track.recv()
                last_recv_t[0] = loop.time()
                new_frame.set()
        except MediaStreamError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            if log:
                log.warning("Owner track recv error: %s", exc)
        finally:
            stopped.set()
            new_frame.set()  # wake the processing loop so it can exit

    reader = asyncio.ensure_future(_reader())
    # Poll cadence: wake often enough that a crossed gap is caught within ~1s of
    # the threshold. A healthy owner wakes on every frame and never hits this
    # timeout — the poll only matters once frames stop.
    poll = min(1.0, gap_release_s)
    try:
        while not (stopped.is_set() and latest[0] is None):
            try:
                await asyncio.wait_for(new_frame.wait(), timeout=poll)
            except asyncio.TimeoutError:
                gap = loop.time() - last_recv_t[0]
                if owner_gap_should_release(
                    gap, ownership.num_other_waiters(pc), gap_release_s
                ):
                    if log:
                        log.info(
                            "Owner %x silent %.1fs with a waiter ready — yielding input",
                            id(pc),
                            gap,
                        )
                    return True  # caller releases → handoff, then rejoins as waiter
                continue
            new_frame.clear()
            frame, latest[0] = latest[0], None
            if frame is None:
                continue
            await sink(frame)
        return False  # track ended
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader


async def consume_peer_input(
    track,
    pc,
    ownership: InputOwnership,
    sink: FrameSink,
    *,
    notify: Optional[NotifyHook] = None,
    log=None,
    first_frame_deadline: float = WAITER_FIRST_FRAME_DEADLINE,
    owner_gap_release: float = OWNER_GAP_WITH_WAITER,
) -> None:
    """Pull VideoFrames from a remote track and (when this peer owns the input)
    feed them into the pipeline via `sink`. Waits for ownership while draining
    its track view-only; the oldest waiter takes over on release. A gap-yielded
    owner (silent with a waiter ready) rejoins as the NEWEST waiter and keeps
    draining — never leave an alive track unconsumed (aiortc decodes inbound RTP
    into an unbounded queue regardless), and it can reclaim when the slot frees."""
    notify = notify or (lambda *a, **k: None)
    loop = asyncio.get_running_loop()

    while True:
        seq = ownership.register_waiter(pc)
        yielded = False
        try:
            # ── wait for ownership, draining frames so the inbound queue can't grow ──
            got_first_frame = False
            deadline = loop.time() + first_frame_deadline
            announced_waiting = False
            while not ownership.try_claim(seq, pc):
                if not announced_waiting:
                    announced_waiting = True
                    if log:
                        log.info("Peer %x (seq %d) waiting — view-only", id(pc), seq)
                if got_first_frame:
                    # Delivered a frame already: a real, connected view-only peer.
                    # Never gap-evicted; the 1s timeout only re-checks the claim so
                    # a stalled-track waiter still takes over when the slot frees.
                    # Death is handled by the caller's connectionstatechange ->
                    # task cancel.
                    try:
                        await asyncio.wait_for(track.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except MediaStreamError:
                        return
                    continue
                remaining = deadline - loop.time()
                if waiter_should_evict(_conn_state(pc), got_first_frame, remaining <= 0):
                    if log:
                        log.info("Peer %x (seq %d) never connected — dropping", id(pc), seq)
                    return
                try:
                    # 1s cap = claim re-check cadence for a silent waiter; the
                    # first-frame DEADLINE (eviction) is enforced by `remaining`
                    # above, not by this timeout.
                    await asyncio.wait_for(track.recv(), timeout=1.0)
                    got_first_frame = True
                except asyncio.TimeoutError:
                    continue  # re-check claim + liveness; do NOT evict a connected peer
                except MediaStreamError:
                    return

            # ── now the owner ──
            if log:
                log.info("Peer %x (seq %d) now drives input", id(pc), seq)
            notify("claimed", pc)
            yielded = await _pump_owner_frames(
                track, pc, sink, ownership, log=log, gap_release_s=owner_gap_release
            )
        finally:
            outcome = ownership.release(seq, pc)
            if log and outcome.had_owner:
                log.info("Peer %x (seq %d) released input", id(pc), seq)
            notify("released", pc, outcome)
        if not yielded:
            return
        if log:
            log.info("Peer %x rejoining as waiter after gap-yield", id(pc))
