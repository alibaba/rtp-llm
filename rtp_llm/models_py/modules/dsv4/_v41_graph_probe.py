"""R27/C (KDA-IDLE-20261030 item 1 first check) + R28/B: CUDA-graph replay
for the mHC FFN boundary (fixed shapes, no NCCL).

Flag ``DSV41_GRAPH_PROBE_ROUTER`` (default "0" = fully inert). Under "1",
each layer's FFN-side mHC pre boundary + ffn norm becomes a
capture-per-address-set / replay region (the router gate stays OUT — its
CP-local path refuses capture by contract, _require_supported raises).

R28 hit-rate fix: the R27 single-graph probe captured ONE input address
set (~76% hit — the caching allocator cycles the boundary input through a
few buffers). GraphReplay now keeps up to MAX_GRAPHS=4 graphs keyed by
(shape, dtype, ptr) per input — copy-in/static-buffer staging was priced
out (~0.25 ms per 167 MB boundary input x per layer ≈ +35 ms/request,
erasing the whole win; per-graph pools measured at ~1.3 MB each, so
multi-graph memory is negligible). Capture on the SECOND sighting of a
key (JIT warm + confirmed recurring); any capture failure blacklists only
that key. Side effects (unit.pre_mix_out) are per-key via
side_get/side_set.

Scope ``DSV41_GRAPH_PROBE_ROUTER_SCOPE``: "one" (default) = first block
per device; "all" = every layer.

Byte-exactness: replay runs the SAME kernels on the SAME weights with the
current inputs in the same addresses => identical bytes; the parity sha
gate proves it end-to-end.

Engagement evidence: [GRAPHPROBE] engaged/captured+replaying key#N /
capture-failed lines + [GRAPHPROBE-STATS] name=... replays=N fallbacks=M
every 512 calls.
"""

import os
import sys

import torch

GRAPH_PROBE_FLAG = "DSV41_GRAPH_PROBE_ROUTER"
GRAPH_PROBE_SCOPE = "DSV41_GRAPH_PROBE_ROUTER_SCOPE"
GRAPH_ATTN_FLAG = "DSV41_GRAPH_ATTN"
GRAPH_ATTN_PARTS = "DSV41_GRAPH_ATTN_PARTS"

_ENABLED = os.environ.get(GRAPH_PROBE_FLAG, "1") == "1"
_SCOPE = os.environ.get(GRAPH_PROBE_SCOPE, "all")
_ATTN_ENABLED = os.environ.get(GRAPH_ATTN_FLAG, "1") == "1"
_ATTN_PARTS = frozenset(
    p for p in os.environ.get(GRAPH_ATTN_PARTS, "boundary,qkv").split(",") if p
)
_STATS = {}
_ANNOUNCED = [False]

def graph_probe_enabled() -> bool:
    return _ENABLED


def graph_attn_enabled() -> bool:
    return _ATTN_ENABLED


def graph_attn_parts() -> frozenset:
    """R29/A bisection aid: which attention stretches may graph
    ("boundary" = attn mHC boundary + norm; "qkv" = the qkv prefix)."""
    return _ATTN_PARTS


def premix_pin_enabled() -> bool:
    """R29/A: pin every delayed-HC unit's pre_mix_out to a per-unit static
    buffer while ANY boundary graphing is on.

    Why: every non-entry boundary's ``_pre_impl`` collapses the
    PREDECESSOR's ``pre_mix_out`` (delayed chain, collapse-mode
    independent). A captured boundary graph bakes that buffer's ADDRESS
    in at capture time; without pinning, the producer's buffer moves
    between captures (multi-graph pools / allocator churn) and replays
    read stale content — the observed wrong-sha mechanism. The pin copy
    is [T,hc] fp32 (~64 KB), stream-safe (runs at post-end, after
    join_stats).
    """
    return _ENABLED or _ATTN_ENABLED


def probe_scope() -> str:
    return _SCOPE


_CLAIM_COUNTS = {}


def claim_one_layer(device_key) -> bool:
    """Scope 'one': first claimant per key+device; 'all': every layer;
    a decimal string N: first N claimants (R29/A bisection)."""
    if _SCOPE == "all":
        return True
    limit = int(_SCOPE) if _SCOPE.isdigit() else 1
    n = _CLAIM_COUNTS.get(device_key, 0)
    if n >= limit:
        return False
    _CLAIM_COUNTS[device_key] = n + 1
    return True


def _note(name, kind):
    st = _STATS.setdefault(name, [0, 0])
    st[0 if kind == "replay" else 1] += 1
    if (st[0] + st[1]) % 512 == 0:
        print(
            f"[GRAPHPROBE-STATS] {name} replays={st[0]} fallbacks={st[1]}",
            file=sys.stderr,
            flush=True,
        )


def _announce(msg):
    print(f"[GRAPHPROBE] {msg}", file=sys.stderr, flush=True)


class GraphReplay:
    """Multi-graph capture/replay: one graph per stable input address set.

    R28 productionizes the R27 probe: instead of a single captured address
    set (~76% hit — the allocator cycles x through a few buffers), keep up
    to MAX_GRAPHS graphs keyed by (shape, dtype, ptr) per input. Capture on
    the SECOND sighting of a key (JIT warm + confirmed recurring). Any
    capture failure blacklists that key only; a key that never recurs just
    never captures. Side effects (e.g. unit.pre_mix_out) are per-key: the
    caller passes side_get()/side_set() and the probe stores/restores the
    capture-time value on every replay.
    """

    MAX_GRAPHS = 4

    def __init__(self, name):
        self.name = name
        self.graphs = {}       # key -> (graph, outputs, side_effect)
        self.pending = {}      # key -> sightings
        self.failed_keys = set()
        self.static_bufs = {}  # copy_in index -> static staging buffer
        self.permanently_off = False

    def run(self, inputs, fn, side_get=None, side_set=None, copy_in=()):
        """Returns (outputs, kind) with kind in replay|captured|fallback.

        ``copy_in``: indices of inputs that are small, per-chunk-rebuilt
        tensors (e.g. freqs_cis) whose allocator addresses churn and whose
        CONTENT must be current at replay. They are excluded from the
        address key and staged through one static buffer per probe (copied
        in before every capture and every replay); ``fn`` receives the
        staged input list.
        """
        if self.permanently_off:
            _note(self.name, "fallback")
            return fn(inputs), "fallback"
        key = tuple(
            (tuple(t.shape), str(t.dtype), t.data_ptr())
            for i, t in enumerate(inputs)
            if i not in copy_in
        )
        staged = list(inputs)
        entry = self.graphs.get(key)
        if entry is not None:
            graph, outputs, side = entry
            for i in copy_in:
                self.static_bufs[i].copy_(inputs[i])
            graph.replay()
            if side_set is not None:
                side_set(side)
            _note(self.name, "replay")
            return outputs, "replay"
        if key in self.failed_keys or len(self.graphs) >= self.MAX_GRAPHS:
            _note(self.name, "fallback")
            return fn(inputs), "fallback"
        seen = self.pending.get(key, 0) + 1
        self.pending[key] = seen
        if seen < 2:
            _note(self.name, "fallback")
            return fn(inputs), "fallback"
        try:
            if copy_in and not self.static_bufs:
                self.static_bufs = {
                    i: inputs[i].clone() for i in copy_in
                }
            for i in copy_in:
                self.static_bufs[i].copy_(inputs[i])
                staged[i] = self.static_bufs[i]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = fn(staged)
            graph.replay()
            self.graphs[key] = (graph, outputs, side_get() if side_get else None)
            _announce(f"captured+replaying {self.name} key#{len(self.graphs)}")
            _note(self.name, "replay")
            return outputs, "captured"
        except Exception as exc:
            self.failed_keys.add(key)
            if isinstance(exc, NotImplementedError):
                # Contract-level refusal (e.g. the CP-local router's capture
                # guard): retriable per key is pointless — one attempt is
                # enough, and churning capture attempts costs a device
                # synchronize each (R28/B b1: 13k gate retries = +240 ms).
                self.permanently_off = True
            _note(self.name, "fallback")
            _announce(f"capture-failed {self.name}: {type(exc).__name__}: {exc}")
            return fn(inputs), "fallback"
