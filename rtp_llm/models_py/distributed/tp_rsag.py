"""Token-sharded TP decode for EP: reduce-scatter of the attention output and all-gather of the attention input over
the FlashInfer PCIe IPC kernels (upstream main, vendored as ``fi_up``), NCCL fallback.

Opt-in with DSV4_TP_RSAG=1 (EP decode with the routed rail MoE, MOE_STRATEGY=sm120_rail_routed). Inside each decode
forward the hyper-connection residual stream stays token-sharded in the TP group. With n_pad = max(T, ROW_FLOOR) and
cap = ceil(n_pad / tp), TP rank r owns rows [r*cap, min(T, (r+1)*cap)) -- the routed rail's owner rows -- so per layer

  hc_pre + attn_norm (cap rows) --AG--> attention input [T, D] -> attention (heads split, no all-reduce)
  --RS--> owned rows [cap, D] -> hc_post / hc_pre / ffn_norm / gate / shared expert (cap rows)
  -> routed rail dispatch of the owned rows -> experts -> combine straight into the owned rows (no y all-gather)
  -> hc_post (cap rows)

which replaces the attention all-reduce and the combine's intra-group y all-gather, and runs every per-token op on
1/tp of the rows. The residual is gathered once at the end of the forward (hc head, MTP hidden, lm_head need it).
Rows past T are padding: their routed ids are -1 (never dispatched) and they are dropped by the final gather.

Contract (as fi_pcie_ar.py): the FI workspaces are built lazily on the first eager call, which every rank of a TP
group reaches in lockstep (decode warm-up); launch configurations come from the library's deterministic seed policy
(a pure function of the shape) and are resolved in eager mode, so a captured graph only launches kernels. Shapes the
workspace does not admit, or eager calls on a second stream, take NCCL on every rank alike.
Environment:
  DSV4_TP_RSAG            1 = token-sharded decode layer
  DSV4_TP_RSAG_BACKEND    fi (default) | nccl
  DSV4_TP_RSAG_MAX_ROWS   largest per-rank shard row count of one collective (default 256; the final residual gather
                          moves cap * hc_mult rows)
  DSV4_FI_PCIE_PATH       directory containing the vendored ``fi_up`` package (see comm/largeep_a2a/README.md "FlashInfer PCIe IPC"; required)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)

ENABLED = os.environ.get("DSV4_TP_RSAG", "0") == "1"
_BACKEND = os.environ.get("DSV4_TP_RSAG_BACKEND", "fi")
# the routed rail strategy pads every call to at least this many rows (Sm120DecodeStrategy pad_floor=4); the shard
# geometry must be computed from the same n_pad so the owner rows match the rail's
ROW_FLOOR = 4


@dataclass(frozen=True)
class ShardCtx:
    T: int  # real token rows of the TP group (= B * q_len)
    n_pad: int  # rail call size, max(T, ROW_FLOOR); identical on every EP rank
    tp: int
    tp_rank: int
    cap: int  # rows per TP rank, ceil(n_pad / tp)
    b: int  # first owned row
    e: int  # one past the last real owned row (e - b may be < cap)


# set by decode.forward.forward_layers for the duration of one sharded decode forward
CTX: Optional[ShardCtx] = None
# token ids of the owned rows [cap] for that forward (hash-routed MoE layers need them), filled once per forward
CTX_IDS: Optional[torch.Tensor] = None


def make_ctx(T: int, tp: int, tp_rank: int, n_pad: Optional[int] = None) -> ShardCtx:
    n_pad = max(int(T), ROW_FLOOR) if n_pad is None else int(n_pad)
    cap = (n_pad + tp - 1) // tp
    b = min(int(T), tp_rank * cap)
    return ShardCtx(int(T), n_pad, int(tp), int(tp_rank), cap, b, min(int(T), b + cap))


class _FiRsAg:
    def __init__(self, group: dist.ProcessGroup, hidden: int, dtype: torch.dtype):
        from rtp_llm.models_py.distributed.fi_pcie_ar import _fi_env

        path = os.environ.get("DSV4_FI_PCIE_PATH", "")
        if not path:
            raise RuntimeError(
                "DSV4_TP_RSAG_BACKEND=fi needs DSV4_FI_PCIE_PATH (directory of the vendored fi_up package)"
            )
        if path not in sys.path:
            sys.path.insert(0, path)
        self.hidden = int(hidden)
        self.dtype = dtype
        self.max_rows = int(os.environ.get("DSV4_TP_RSAG_MAX_ROWS", "256"))
        with _fi_env(path):
            from fi_up.comm.pcie_ipc_collectives.all_gather import (
                PcieIpcAllGatherWorkspace,
            )
            from fi_up.comm.pcie_ipc_collectives.reduce_scatter import (
                PcieIpcReduceScatterWorkspace,
            )

            n = self.max_rows * self.hidden
            self.rs = PcieIpcReduceScatterWorkspace(group, n, dtype=dtype)
            self.ag = PcieIpcAllGatherWorkspace(group, n, dtype=dtype)
        self._rs_cfg: Dict[int, object] = {}
        self._ag_cfg: Dict[int, object] = {}
        self._stream = None
        _logger.warning(
            "FI PCIe IPC RS/AG up: ranks=%s hidden=%d max_rows=%d",
            dist.get_process_group_ranks(group),
            self.hidden,
            self.max_rows,
        )

    def _usable(self) -> bool:
        if torch.cuda.is_current_stream_capturing():
            return True
        # eager calls must stay on one stream (the workspaces' arrival state is stream-ordered); the first eager
        # stream wins, any other one takes NCCL -- the same code path on every rank, so the choice is rank-uniform
        cur = torch.cuda.current_stream()
        if self._stream is None:
            self._stream = cur
        return cur == self._stream

    def _cfg(self, cache: dict, ws, probe_rows: int, key: int):
        cfg = cache.get(key)
        if cfg is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            probe = torch.empty(
                probe_rows, self.hidden, dtype=self.dtype, device="cuda"
            )
            cfg = ws.launch_config(probe)
            cache[key] = cfg if cfg is not None else False
        return cfg or None

    def reduce_scatter(self, inp: torch.Tensor, out: torch.Tensor) -> bool:
        rows = out.size(0)
        if rows > self.max_rows or inp.size(1) != self.hidden or not self._usable():
            return False
        cfg = self._cfg(self._rs_cfg, self.rs, inp.size(0), rows)
        if cfg is None:
            return False
        self.rs.reduce_scatter(inp, out=out, config=cfg)
        return True

    def all_gather(self, inp: torch.Tensor, out: torch.Tensor) -> bool:
        rows = inp.size(0)
        if rows > self.max_rows or inp.size(1) != self.hidden or not self._usable():
            return False
        cfg = self._cfg(self._ag_cfg, self.ag, rows, rows)
        if cfg is None:
            return False
        self.ag.all_gather(inp, out=out, config=cfg)
        return True


_IMPL: Dict[int, Optional[_FiRsAg]] = {}
# persistent buffers: every generation stays alive (graphs captured earlier keep pointing at the old ones)
_BUF: Dict[tuple, List[torch.Tensor]] = {}


def _tp_group():
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

    return _get_group(Group.TP)


def _impl(hidden: int, dtype: torch.dtype) -> Optional[_FiRsAg]:
    dev = torch.cuda.current_device()
    if dev in _IMPL:
        return _IMPL[dev]
    if _BACKEND != "fi":
        _IMPL[dev] = None
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        _IMPL[dev] = _FiRsAg(_tp_group(), hidden, dtype)
    except (
        Exception
    ) as e:  # noqa: BLE001  (construction is collective, so the outcome is the same on every rank)
        _logger.warning("FI PCIe IPC RS/AG unavailable, NCCL fallback: %s", e)
        _IMPL[dev] = None
    return _IMPL[dev]


def buf(
    tag: str, rows: int, cols: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (tag, int(cols), dtype, device.index)
    gens = _BUF.setdefault(key, [])
    if not gens or gens[-1].size(0) < rows:
        if torch.cuda.is_current_stream_capturing() and gens:
            raise RuntimeError(
                f"tp_rsag buffer {tag} must grow outside graph capture ({gens[-1].size(0)} < {rows})"
            )
        gens.append(
            torch.zeros(max(int(rows), 64), int(cols), dtype=dtype, device=device)
        )
    return gens[-1][:rows]


def reduce_scatter_rows(x: torch.Tensor, ctx: ShardCtx) -> torch.Tensor:
    """SUM over the TP group of the bf16 partials x [T, D]; returns this rank's owned rows [cap, D]."""
    T, D = x.shape
    rows = ctx.tp * ctx.cap
    full = x if x.is_contiguous() else x.contiguous()
    if rows != T:
        full = buf("rs_in", rows, D, x.dtype, x.device)
        full[:T].copy_(x)
        full[T:].zero_()
    out = buf("rs_out", ctx.cap, D, x.dtype, x.device)
    impl = _impl(D, x.dtype)
    if impl is None or not impl.reduce_scatter(full, out):
        dist.reduce_scatter_tensor(out, full, group=_tp_group())
    return out


def all_gather_rows(
    x_local: torch.Tensor, ctx: ShardCtx, tag: str = "ag_out"
) -> torch.Tensor:
    """Gather the owned rows [k*cap, D] of every TP rank into [tp*k*cap, D] (rank-major)."""
    rows, D = x_local.shape
    out = buf(tag, ctx.tp * rows, D, x_local.dtype, x_local.device)
    src = x_local if x_local.is_contiguous() else x_local.contiguous()
    impl = _impl(D, x_local.dtype)
    if impl is None or not impl.all_gather(src, out):
        dist.all_gather_into_tensor(out, src, group=_tp_group())
    return out


def all_gather_nccl(x_local: torch.Tensor, ctx: ShardCtx) -> torch.Tensor:
    """Eager-only NCCL gather of any dtype ([cap, ...] -> [tp*cap, ...])."""
    out = torch.empty(
        (ctx.tp * x_local.size(0),) + tuple(x_local.shape[1:]),
        dtype=x_local.dtype,
        device=x_local.device,
    )
    dist.all_gather_into_tensor(out, x_local.contiguous(), group=_tp_group())
    return out


def owned_ids(input_ids_2d: torch.Tensor, ctx: ShardCtx) -> torch.Tensor:
    """Token ids of the owned rows [cap] (hash-routed MoE layers need them); pad rows get id 0."""
    flat = input_ids_2d.reshape(-1)
    out = buf("ids", ctx.cap, 1, flat.dtype, flat.device).view(-1)
    n = max(0, ctx.e - ctx.b)
    if n:
        out[:n].copy_(flat[ctx.b : ctx.e])
    if n < ctx.cap:
        out[n:].zero_()
    return out


def sharded_decode_ctx(v4, T: int, capture_ids, record_on: bool) -> Optional[ShardCtx]:
    """The shard context for this decode forward, or None to run the ordinary (replicated) layers.
    Only the routed rail EP decode with TP > 1 is sharded; DSpARK aux capture and tensor recording stay replicated.
    """
    if not ENABLED or capture_ids or record_on:
        return None
    args = v4.args
    tp = int(getattr(args, "tp_size", 1))
    if tp <= 1 or not v4.layers:
        return None
    moe = getattr(v4.layers[0], "ffn", None)
    if moe is None or not hasattr(
        getattr(moe, "_strategy", None), "forward_owned_rows"
    ):
        return None
    n_pad = max(int(T), ROW_FLOOR)
    if not torch.cuda.is_current_stream_capturing():
        # every EP rank must issue the rail call with the same n_pad (the routed strategy's eager rule); a captured
        # graph is keyed by the decode batch, which the DP batch equalizer keeps identical across DP groups
        ext = torch.tensor(
            [n_pad],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(ext, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        n_pad = int(ext.item())
    return make_ctx(T, tp, int(getattr(args, "tp_rank", 0)), n_pad)


def run_sharded_layers(
    v4,
    h: torch.Tensor,
    attn_metadata,
    input_ids_2d: torch.Tensor,
    kv_cache,
    ctx: ShardCtx,
):
    """Layer loop of one token-sharded decode forward: h [B, q_len, hc, dim] (replicated) in, the same shape out."""
    global CTX, CTX_IDS
    B, q_len, hc, dim = h.shape
    flat = h.reshape(ctx.T, hc, dim)
    x = buf("res_in", ctx.cap, hc * dim, h.dtype, h.device).view(ctx.cap, hc, dim)
    n = max(0, ctx.e - ctx.b)
    if n:
        x[:n].copy_(flat[ctx.b : ctx.e])
    if n < ctx.cap:
        x[n:].zero_()
    CTX, CTX_IDS = ctx, owned_ids(input_ids_2d, ctx)
    try:
        for layer in v4.layers:
            x = layer.forward_decode(x, attn_metadata, input_ids_2d, kv_cache=kv_cache)
    finally:
        CTX, CTX_IDS = None, None
    # one gather of the residual stream ([cap, hc*dim] per rank) for the HC head / MTP hidden / lm_head
    full = all_gather_rows(x.reshape(ctx.cap * hc, dim), ctx, tag="res_out").view(
        ctx.tp * ctx.cap, hc, dim
    )
    return full[: ctx.T].view(B, q_len, hc, dim)
