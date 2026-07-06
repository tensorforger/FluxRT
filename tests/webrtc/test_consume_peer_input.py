"""Async integration tests for consume_peer_input / the owner pump.

Drives the real coroutine with a fake track + a fake peer-connection (just a
`connectionState` attribute) and a recording sink. No aiortc transport, no GPU.
"""

import asyncio

import pytest

from fluxrt.webrtc.input_ownership import (
    InputOwnership,
    MediaStreamError,
    consume_peer_input,
)


class FakeTrack:
    """recv() yields queued frames (optionally with a gap), then ends; or blocks
    forever (`never=True`) to simulate a registered-but-silent track."""

    def __init__(self, frames=(), gap=0.0, ends=True, never=False):
        self._frames = list(frames)
        self._gap = gap
        self._ends = ends
        self._never = never
        self.recv_calls = 0

    async def recv(self):
        self.recv_calls += 1
        if self._never:
            await asyncio.Event().wait()  # never returns
        if self._gap:
            await asyncio.sleep(self._gap)
        if self._frames:
            return self._frames.pop(0)
        if self._ends:
            raise MediaStreamError("ended")
        await asyncio.Event().wait()


class FakePC:
    def __init__(self, state="connected"):
        self.connectionState = state


def _sink_recorder():
    got = []

    async def sink(frame):
        got.append(frame)

    return got, sink


async def test_owner_drives_frames_then_releases_on_track_end():
    own = InputOwnership(has_server_camera=True)
    track = FakeTrack(frames=["f1", "f2", "f3"], gap=0.01, ends=True)
    pc = FakePC("connected")
    got, sink = _sink_recorder()

    await asyncio.wait_for(consume_peer_input(track, pc, own, sink), timeout=3)

    assert got == ["f1", "f2", "f3"]
    assert own.is_active() is False
    assert own.num_waiters() == 0


async def test_owner_survives_long_frame_gap_no_false_evict():
    # The behaviour c855950 broke: a connected owner with a gap is NOT evicted.
    own = InputOwnership(has_server_camera=False)
    track = FakeTrack(frames=["f1", "f2"], gap=0.2, ends=True)
    pc = FakePC("connected")
    got, sink = _sink_recorder()

    await asyncio.wait_for(consume_peer_input(track, pc, own, sink), timeout=3)
    assert got == ["f1", "f2"]  # both delivered across the gaps, owner never dropped


async def test_silent_dead_waiter_is_evicted():
    own = InputOwnership()
    # An existing owner so the new peer must wait.
    a = FakePC("connected")
    sa = own.register_waiter(a)
    assert own.try_claim(sa, a)

    silent = FakeTrack(never=True)
    pc_b = FakePC("failed")  # never connected
    _, sink = _sink_recorder()

    # Should evict within ~the deadline and return (not hang).
    await asyncio.wait_for(
        consume_peer_input(silent, pc_b, own, sink, first_frame_deadline=0.2),
        timeout=2,
    )
    assert own.num_waiters() == 1  # only A remains
    assert own.owner_is(a) is True


async def test_silent_but_connected_waiter_is_not_evicted():
    own = InputOwnership()
    a = FakePC("connected")
    sa = own.register_waiter(a)
    own.try_claim(sa, a)

    silent = FakeTrack(never=True)
    pc_b = FakePC("connected")  # connected, just slow — must NOT be dropped
    _, sink = _sink_recorder()

    task = asyncio.ensure_future(
        consume_peer_input(silent, pc_b, own, sink, first_frame_deadline=0.2)
    )
    await asyncio.sleep(0.5)  # well past the deadline
    assert not task.done(), "a connected waiter must not be gap-evicted"
    assert own.num_waiters() == 2

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_oldest_waiter_takes_over_when_owner_releases():
    own = InputOwnership(has_server_camera=False)
    a = FakePC("connected")
    sa = own.register_waiter(a)
    own.try_claim(sa, a)  # A owns

    # Long-running stream so B is still alive (not drained out) when A leaves.
    track_b = FakeTrack(frames=[f"b{i}" for i in range(200)], gap=0.02, ends=False)
    pc_b = FakePC("connected")
    got, sink = _sink_recorder()
    task = asyncio.ensure_future(consume_peer_input(track_b, pc_b, own, sink))

    await asyncio.sleep(0.1)  # B drains view-only, cannot claim while A owns
    assert own.owner_is(a) is True
    assert not own.owner_is(pc_b)

    own.release(sa, a)  # A leaves
    await asyncio.wait_for(_wait(lambda: own.owner_is(pc_b)), timeout=2)
    await asyncio.wait_for(_wait(lambda: len(got) > 0), timeout=2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert own.num_waiters() == 0  # B's finally released its slot on cancel


async def test_owner_gap_with_waiter_yields_input():
    # Owner A stops delivering frames while publisher B waits. A must yield within
    # ~the gap so B takes over — NOT wait out ICE consent expiry (~30s). This is
    # the fast dead-owner path (V3).
    own = InputOwnership(has_server_camera=False)
    a_track = FakeTrack(frames=["a1"], ends=False)  # one frame, then silent forever
    pc_a = FakePC("connected")
    got, sink = _sink_recorder()
    task_a = asyncio.ensure_future(
        consume_peer_input(a_track, pc_a, own, sink, owner_gap_release=0.2)
    )
    await asyncio.wait_for(_wait(lambda: own.owner_is(pc_a)), timeout=2)

    # B registers as a ready takeover candidate.
    pc_b = FakePC("connected")
    sb = own.register_waiter(pc_b)

    # A silent > gap with B waiting → A yields; its task finishes and releases.
    await asyncio.wait_for(task_a, timeout=2)
    assert not own.owner_is(pc_a)
    # B is now the oldest remaining waiter and can claim.
    assert own.try_claim(sb, pc_b) is True


async def test_owner_gap_no_waiter_holds():
    # A lone owner is NEVER gap-evicted, no matter how long the gap (V2 / c855950).
    own = InputOwnership(has_server_camera=False)
    a_track = FakeTrack(frames=["a1"], ends=False)  # one frame, then silent forever
    pc_a = FakePC("connected")
    got, sink = _sink_recorder()
    task = asyncio.ensure_future(
        consume_peer_input(a_track, pc_a, own, sink, owner_gap_release=0.2)
    )
    await asyncio.wait_for(_wait(lambda: own.owner_is(pc_a)), timeout=2)

    await asyncio.sleep(0.6)  # 3x the gap, but no other waiter exists
    assert own.owner_is(pc_a) is True
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_owner_delivering_frames_with_waiter_not_evicted():
    # An owner still delivering frames is never yielded, even with a waiter present
    # — the gap policy keys on a receive GAP, not on the mere presence of a waiter
    # (V5).
    own = InputOwnership(has_server_camera=False)
    a_track = FakeTrack(frames=[f"a{i}" for i in range(200)], gap=0.02, ends=False)
    pc_a = FakePC("connected")
    got, sink = _sink_recorder()
    task = asyncio.ensure_future(
        consume_peer_input(a_track, pc_a, own, sink, owner_gap_release=0.2)
    )
    await asyncio.wait_for(_wait(lambda: own.owner_is(pc_a)), timeout=2)

    pc_b = FakePC("connected")
    own.register_waiter(pc_b)  # waiter present the whole time

    await asyncio.sleep(0.6)  # 3x the gap, but A keeps delivering (0.02s << 0.2s)
    assert own.owner_is(pc_a) is True
    assert len(got) > 5

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _wait(pred, interval=0.01):
    while not pred():
        await asyncio.sleep(interval)
