# SPEC — input connect/reconnect: single owner, seamless handoff

## §G GOAL
∀ time ≤1 peer drives pipeline input; owner death/leave → oldest waiter takes over seamless (≤~9s worst case, next-frame when graceful); transient blip ⊥ force full renegotiation (client grace); no-waiter owner ⊥ evicted on gap.

## §C CONSTRAINTS
- C1: InputOwnership sole mutator of owner/waiter/active state; policy fns pure, unit-testable w/o aiortc/GPU (existing module ethos).
- C2: c855950 guard FROZEN: ⊥ blind gap-evict owner. Gap-evict ONLY w/ takeover candidate present (§R1).
- C3: aiortc connectionState ∈ {new,connecting,connected,failed,closed}; ⊥ 'disconnected' server-side (§R2). Server logic ⊥ depend on 'disconnected' firing.
- C4: wire protocol `ctrlProtocol.ts` frozen — no new msg types (frame-gap needs none).
- C5: client scope = `engineSession.ts` only; `sessionStore.ts` untouched.
- C6: existing backoff `min(800·1.7^n, 6000)` & `waitServerReady` gating stay.
- C7: old test client `scripts/webrtc_test_client.html` ! keep working unmodified.

## §I INTERFACES
- fn: `owner_gap_should_release(gap_s: float, other_waiters: int) -> bool`  // pure, input_ownership.py; True iff gap_s ≥ OWNER_GAP_WITH_WAITER & other_waiters ≥ 1
- const: `OWNER_GAP_WITH_WAITER = 8.0`  // input_ownership.py
- method: `InputOwnership.num_other_waiters(pc) -> int`  // waiters excl. pc's own seq, under lock
- behavior: `_pump_owner_frames` gains gap supervision (recv wrapped in wait_for vs last-frame ts; claim time = initial ts) → policy fire → pump returns → existing `finally: release()` handoff path
- client: `engineSession.ts` — ice 'disconnected' → 3s grace timer → `scheduleRetry`; 'connected|completed' cancels; `pc.onconnectionstatechange === 'failed'` → `scheduleRetry`; `ch.onmessage` → `decodeCtrl` → per-clip inputRole via onStatus/store
- api: `/healthz` unchanged (`input_waiters`, `input_source` already exposed — e2e observability)

## §R RESEARCH
id|topic|finding|src
R1|prior regression|c855950 blind 5s owner gap-evict froze healthy owners (keyframe/TURN settle, paused cam) → fixed by never-gap-evict; new policy conditions evict on waiter presence, not gap alone|FluxRT git history + input_ownership.py docstring
R2|aiortc states|connectionState ⊥ 'disconnected' — explicit `# NOTE: we do not have a 'disconnected' state`; states new/connecting/connected/failed/closed|github.com/aiortc/aiortc rtcpeerconnection.py __updateConnectionState
R3|dead-peer latency|aioice RFC7675 consent: CONSENT_INTERVAL=5, CONSENT_FAILURES=6, interval ×(0.8–1.2) → dead peer → close ≈25–35s|github.com/aiortc/aioice ice.py query_consent
R4|browser side|'disconnected' transient, may self-heal; escalation to 'failed' browser-dependent (~10s); engineSession.ts:136 comment relies on it|MDN RTCPeerConnection.connectionState

## §V INVARIANTS
V1: ∀ time ≤1 owner (existing machine + tests)
V2: owner w/ 0 other waiters ⊥ gap-evicted — holds slot until terminal state / consent expiry backstop
V3: owner silent ≥ OWNER_GAP_WITH_WAITER & ≥1 other waiter → released; oldest waiter claims on its next frame
V4: handoff msgs: `input:you` → new owner only; `input:peer` broadcast (existing _input_notify path)
V5: waiter first-frame policy unchanged (WAITER_FIRST_FRAME_DEADLINE 25s, connected-waiter never evicted)
V6: client engine: 'disconnected' recovers ≤3s → same pc kept, ⊥ renegotiation; expires → teardown+retry w/ backoff; `closed=true` halts all retry
V7: client engine: inbound ctrl decoded; unknown msg → ignored, ⊥ throw; inputRole surfaced per clip
V8: `webrtc_test_client.html` connects & functions unmodified

## §T TASKS
id|status|task|cites
T1|x|`input_ownership.py`: add OWNER_GAP_WITH_WAITER + pure `owner_gap_should_release`|V2,V3
T2|x|`InputOwnership.num_other_waiters(pc)` under lock|V3,C1
T3|x|`_pump_owner_frames`: gap supervision → policy fire → return (release via existing finally)|V3,C2
T4|x|server tests: gap+waiter evicts & handoff; gap no-waiter holds; frame-delivering owner + waiter never evicted|V2,V3,V5
T5|x|`run_webrtc.py` comment fix re aiortc states (no behavior change)|R2
T6|x|`engineSession.ts`: 3s disconnected-grace + connectionstatechange 'failed' → retry|V6
T7|x|`engineSession.ts`: `ch.onmessage` → decodeCtrl → per-clip role|V7,C4
T8|x|client vitest: grace cancel/expiry/closed-halt; ctrl dispatch (fake RTCPeerConnection + fake timers)|V6,V7

## §B BUGS
id|date|cause|fix
