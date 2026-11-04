#!/usr/bin/env python3
# routed2_test.py - correctness + performance harness for librail_routed2 (rail_routed2.RailRouted2).
#
# One process per GPU; world = 2*local_world (2x4 emulated on one box: node0 = GPUs 0-3 + erdma_0, node1 = GPUs 4-7
# + erdma_1; 2x8 on two hosts). Control plane: gloo (wire exchange, result gather only).
#
# Inputs are generated deterministically from (seed, call t, TP group g), so every rank can compute exactly what it
# must receive (gathered rows, byte-exact) and the expected y of its TP group (bit-exact: same op order as the
# library's fixed-order fp32 reduction). The "expert" is a stub computed by torch on the gathered rows:
#   partial[row] = bf16( (sum_k [id_k local] * w_k * f(id_k)) * fp8(x_row) ),  f(e) = 1 + (e % 7) / 8
# Phases per n_pad: eager (sync + check every call, every routing mode, fp32/bf16/None out), graph-verify (43-layer
# capture with in-graph stub expert + per-layer gathered snapshots, all checked), graph-perf (43-layer capture,
# precomputed partials, y checked every layer; timed), graph-perf with a fake expert (sleep) between dispatch and
# combine.
import argparse
import json
import os
import signal
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rail_routed2 import TS_NAMES, RailRouted2  # noqa: E402

M32 = 0xFFFFFFFF
MODES_ALL = [
    "uniform",
    "zipf",
    "one_rank",
    "remote",
    "local",
    "hot",
    "pad",
    "spread",
    "pad40",
    "allpad",
]


def mix(h):
    h = h & M32
    h = h ^ (h >> 16)
    h = (h * 0x7FEB352D) & M32
    h = h ^ (h >> 15)
    h = (h * 0x846CA68B) & M32
    h = h ^ (h >> 16)
    return h


def mix_int(*vals):
    h = 0x12345
    for v in vals:
        h = int(mix(torch.tensor([(h * 0x9E3779B1 + v + 0x632BE5AB) & M32])).item())
    return h


def pct(v, q):
    v = sorted(v)
    return 0.0 if not v else v[min(len(v) - 1, int(len(v) * q))]


class Gen:
    def __init__(self, a, dev):
        self.a = a
        self.dev = dev
        self.W = a.world
        self.lw = a.local_world
        self.tp = a.tp
        self.G = self.W // self.tp
        self.E = self.W * a.epr
        self.P = a.payload
        self.dim = a.dim
        self.topk = a.topk
        self.scale_bytes = self.P - self.dim - 8 * self.topk
        self.w_off = self.dim + self.scale_bytes
        self.i_off = self.w_off + 4 * self.topk
        self.fexp = (
            1.0 + (torch.arange(self.E, device=dev) % 7).float() / 8.0
        ).contiguous()
        g = torch.Generator().manual_seed(a.seed)
        self.zipf_perm = torch.randperm(self.E, generator=g)
        p = 1.0 / (torch.arange(self.E, dtype=torch.float64) + 1.0) ** 1.1
        self.zipf_p = torch.empty(self.E, dtype=torch.float64)
        self.zipf_p[self.zipf_perm] = p
        self.zipf_p = self.zipf_p.float()
        self.modes = a.modes
        self.padding = False

    def mode_for(self, t, g):
        if (
            self.padding
        ):  # one TP group fully padded per call, the others 5/6 valid (40 of 48)
            return "allpad" if g == t % self.G else "pad40"
        return self.modes[(t * 7 + g * 3) % len(self.modes)]

    def routing(self, t, g, n_pad, mode):
        a = self.a
        gen = torch.Generator().manual_seed(mix_int(a.seed, t, g, 77))
        E, W, epr, lw, k = self.E, self.W, a.epr, self.lw, self.topk
        node_g = (g * self.tp) // lw
        n_valid = n_pad
        if mode in ("uniform", "pad", "pad40", "allpad"):
            ids = torch.rand(n_pad, E, generator=gen).argsort(1)[:, :k]
            if mode == "pad":
                n_valid = int(torch.randint(0, n_pad + 1, (1,), generator=gen).item())
            elif mode == "pad40":
                n_valid = (n_pad * 5) // 6
            elif mode == "allpad":
                n_valid = 0
        elif mode == "zipf":
            ids = torch.multinomial(
                self.zipf_p.expand(n_pad, E), k, replacement=False, generator=gen
            )
        elif mode == "one_rank":
            d = torch.randint(0, W, (n_pad, 1), generator=gen)
            ids = d * epr + torch.rand(n_pad, epr, generator=gen).argsort(1)[:, :k]
        elif mode in ("remote", "local"):
            nd = (1 - node_g) if mode == "remote" else node_g
            ids = (
                nd * lw * epr
                + torch.rand(n_pad, lw * epr, generator=gen).argsort(1)[:, :k]
            )
        elif mode == "hot":
            h = t % W
            ids = h * epr + torch.rand(n_pad, epr, generator=gen).argsort(1)[:, :k]
        elif mode == "spread":
            d = torch.rand(n_pad, W, generator=gen).argsort(1)[:, :k]
            ids = d * epr + torch.randint(0, epr, (n_pad, k), generator=gen)
        else:
            raise ValueError(mode)
        ids = ids.long()
        if n_valid < n_pad:
            ids[n_valid:] = -1
        return ids, n_valid

    def group_payload(self, t, g, n_pad):
        mode = self.mode_for(t, g)
        ids, n_valid = self.routing(t, g, n_pad, mode)
        ids = ids.to(self.dev)
        key = mix_int(self.a.seed, t, g, 11)
        dev = self.dev
        idx = torch.arange(n_pad * self.dim, device=dev, dtype=torch.int64)
        h = mix(idx * 0x9E3779B1 + key)
        xb = h & 0xFF
        xb = (
            torch.where((xb & 0x7F) == 0x7F, xb ^ 1, xb)
            .to(torch.uint8)
            .view(n_pad, self.dim)
        )
        sidx = torch.arange(n_pad * self.scale_bytes, device=dev, dtype=torch.int64)
        sb = (
            (mix(sidx * 0x85EBCA77 + key + 1) & 0xFF)
            .to(torch.uint8)
            .view(n_pad, self.scale_bytes)
        )
        widx = torch.arange(n_pad * self.topk, device=dev, dtype=torch.int64)
        w = ((mix(widx * 0xC2B2AE3D + key + 2) % 997) + 1).float().view(
            n_pad, self.topk
        ) / 1000.0
        w = torch.where(ids >= 0, w, torch.zeros_like(w))
        if n_valid < n_pad:  # what the pack kernel writes for rows >= n_valid
            xb[n_valid:] = 0
            sb[n_valid:] = 0
            w[n_valid:] = 0
        row = torch.cat(
            [
                xb,
                sb,
                w.contiguous().view(torch.uint8),
                ids.to(torch.int32).contiguous().view(torch.uint8),
            ],
            dim=1,
        ).contiguous()
        assert row.shape == (n_pad, self.P)
        return row, ids, mode, n_valid

    def stub(self, rows, d):
        """Stub expert of receiver rank d over payload rows [R, P] -> partial [R, dim] bf16."""
        R = rows.size(0)
        x = rows[:, : self.dim].contiguous().view(torch.float8_e4m3fn).float()
        w = (
            rows[:, self.w_off : self.w_off + 4 * self.topk]
            .contiguous()
            .view(torch.float32)
        )
        ids = (
            rows[:, self.i_off : self.i_off + 4 * self.topk]
            .contiguous()
            .view(torch.int32)
            .long()
        )
        lo = d * self.a.epr
        coef = torch.zeros(R, device=rows.device, dtype=torch.float32)
        for k in range(self.topk):
            e = ids[:, k]
            m = (e >= lo) & (e < lo + self.a.epr)
            term = torch.where(
                m, w[:, k] * self.fexp[e.clamp(0, self.E - 1)], torch.zeros_like(coef)
            )
            coef = coef + term
        return (coef[:, None] * x).to(torch.bfloat16)


class Call:
    """Everything one rank needs for one call t: its group's payload, expected gathered, expected y."""

    def __init__(self, gen, me, t, n_pad, y_fp32):
        W, tp, lw, P = gen.W, gen.tp, gen.lw, gen.P
        self.t, self.n_pad = t, n_pad
        cap = -(-n_pad // tp)
        self.cap = cap
        pays, idss, self.modes = [], [], []
        for g in range(gen.G):
            row, ids, mode, _ = gen.group_payload(t, g, n_pad)
            pays.append(row)
            idss.append(ids)
            self.modes.append(mode)
        pays = torch.stack(pays)
        ids_all = torch.stack(idss)
        dest = torch.where(
            ids_all >= 0, ids_all // gen.a.epr, torch.full_like(ids_all, -1)
        )
        hit = (dest[..., None] == torch.arange(W, device=gen.dev)).any(
            2
        )  # [G, n_pad, W]
        gm = me // tp
        self.payload = pays[gm].contiguous()
        src_idx = torch.full((W * cap,), -1, dtype=torch.long, device=gen.dev)
        for s in range(W):
            g, tr = s // tp, s % tp
            b = min(n_pad, tr * cap)
            e = min(n_pad, b + cap)
            rows = b + torch.nonzero(hit[g, b:e, me]).flatten()
            src_idx[s * cap : s * cap + rows.numel()] = g * n_pad + rows
        self.src_idx = src_idx
        self.valid = src_idx >= 0
        flat = pays.view(-1, P)
        self.exp_gath = torch.where(
            self.valid[:, None], flat[src_idx.clamp(min=0)], torch.zeros_like(flat[:1])
        )
        self.partial = gen.stub(self.exp_gath, me)
        # expected y of my group (bit-exact order: same-node dests ascending, + bf16(remote dests ascending))
        node_g = me // lw
        loc = torch.zeros(n_pad, gen.dim, device=gen.dev, dtype=torch.float32)
        rem = torch.zeros_like(loc)
        for l in range(lw):
            loc = loc + gen.stub(self.payload, node_g * lw + l).float()
        for l in range(lw):
            rem = rem + gen.stub(self.payload, (1 - node_g) * lw + l).float()
        y = loc + rem.to(torch.bfloat16).float()
        self.y = y.to(torch.bfloat16).float() if not y_fp32 else y
        self.hits_me = int(self.valid.sum().item())

        # expected RDMA payload rows: dispatch = my owned rows with >=1 other-node dest; combine = my rail
        # partner's owned rows with >=1 dest on my node (I pre-reduce and return them)
        def remote_rows(rank_x):
            gx, trx = rank_x // tp, rank_x % tp
            b = min(n_pad, trx * cap)
            e = min(n_pad, b + cap)
            other = (rank_x // lw) ^ 1
            return int(hit[gx, b:e, other * lw : (other + 1) * lw].any(1).sum().item())

        self.rdma_disp_rows = remote_rows(me)
        self.rdma_comb_rows = remote_rows((1 - me // lw) * lw + me % lw)


def check_gath(gen, call, actual):
    io = gen.i_off
    rows = actual.size(0)
    a = actual[:rows]
    bad_v = (a != call.exp_gath).any(1) & call.valid
    ids = a[:, io : io + 4 * gen.topk].contiguous().view(torch.int32)
    bad_i = (ids != -1).any(1) & ~call.valid
    return (bad_v | bad_i).sum()


def check_y(call, out):
    o = out[: call.n_pad].float()
    return (
        (o.view(torch.int32) != call.y.view(torch.int32)).any(1).sum()
        if out.dtype == torch.float32
        else ((out[: call.n_pad] != call.y.to(torch.bfloat16)).any(1).sum())
    )


def check_y_any(rr, call, out, n_pad, y_local):
    """Full-y check, or for y_local only this rank's owned rows against the same rows of the expected y."""
    if not y_local:
        return check_y(call, out)
    b, e = rr.owned_rows(n_pad)
    if e <= b:
        return torch.zeros((), dtype=torch.int64)
    got, exp = out[: e - b], call.y[b:e]
    if out.dtype == torch.float32:
        return (got.view(torch.int32) != exp.float().view(torch.int32)).any(1).sum()
    return (got != exp.to(torch.bfloat16)).any(1).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npads", default="48,3,6,12,24,96,144,192")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--verify-replays", type=int, default=2)
    ap.add_argument("--eager-calls", type=int, default=16)
    ap.add_argument("--expert-us", default="0,200")
    ap.add_argument("--modes", default=",".join(MODES_ALL))
    ap.add_argument(
        "--perf-modes",
        default="",
        help="routing modes for perf phases (default: --modes)",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--payload", type=int, default=4176)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--epr", type=int, default=16)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument(
        "--flag-mode", type=int, default=int(os.environ.get("RR2_FLAG_MODE", "2"))
    )
    ap.add_argument(
        "--blocks", type=int, default=int(os.environ.get("RR2_BLOCKS", "16"))
    )
    ap.add_argument("--y-fp32", type=int, default=0)
    ap.add_argument("--out-fp32", type=int, default=1)
    ap.add_argument("--n-pad-max", type=int, default=192)
    ap.add_argument("--phases", default="eager,verify,perf")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--y-local-test",
        type=int,
        default=0,
        help="1: eager phase also exercises combine_local (ABI 6)",
    )
    ap.add_argument(
        "--y-local",
        type=int,
        default=0,
        help="1: graph verify/perf phases use combine_local (ABI 6)",
    )
    a = ap.parse_args()
    signal.alarm(a.timeout)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    lw = int(os.environ.get("LOCAL_WORLD", str(world // 2)))
    gpu = int(os.environ.get("GPU", str(rank % lw)))
    nic = os.environ.get("NIC", f"erdma_{0 if (rank % lw) < lw // 2 else 1}")
    a.world, a.local_world = world, lw
    a.modes = a.modes.split(",")
    perf_modes = a.perf_modes.split(",") if a.perf_modes else a.modes
    dev = torch.device("cuda", gpu)
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}][r{rank}] {msg}", flush=True)

    rr = RailRouted2(
        rank,
        world=world,
        local_world=lw,
        gpu=gpu,
        nic=nic,
        n_pad_max=a.n_pad_max,
        payload_bytes=a.payload,
        dim=a.dim,
        topk=a.topk,
        experts_per_rank=a.epr,
        tp=a.tp,
        flag_mode=a.flag_mode,
        blocks=a.blocks,
        y_fp32=bool(a.y_fp32),
        proxy_cpu=int(os.environ.get("RR2_PROXY_CPU", "-1")),
        spin_ns=int(os.environ.get("RR2_SPIN_NS", "2000000000")),
    )
    log(
        f"up gpu={gpu} nic={nic} flag_mode={rr.flag_mode} blocks={a.blocks} y_fp32={a.y_fp32}"
    )
    gen = Gen(a, dev)
    s = torch.cuda.Stream(device=dev)
    t_next = [1]
    res = {"rank": rank, "phases": {}, "tag": a.tag}
    out_dt = torch.float32 if a.out_fp32 else torch.bfloat16

    def eager(n_pad, padding=False):
        cap = rr.cap(n_pad)
        bad_g = bad_y = 0
        bad_tr = [0]
        rows_tot = [0, 0, 0]
        gen.padding = padding
        partial = torch.empty(world * cap, a.dim, dtype=torch.bfloat16, device=dev)
        with torch.cuda.stream(s):
            for c in range(a.eager_calls):
                t = t_next[0]
                t_next[0] += 1
                call = Call(gen, rank, t, n_pad, a.y_fp32)
                torch.cuda.current_stream().synchronize()
                tr0 = rr.traffic()
                gath = rr.dispatch(call.payload, None, n_pad)
                torch.cuda.current_stream().synchronize()
                bg = int(check_gath(gen, call, gath).item())
                partial.copy_(gen.stub(gath, rank))
                kind = c % (5 if a.y_local_test else 3)
                if kind == 0:
                    out = torch.full(
                        (n_pad, a.dim), float("nan"), dtype=torch.float32, device=dev
                    )
                elif kind == 1:
                    out = torch.full(
                        (n_pad, a.dim), float("nan"), dtype=torch.bfloat16, device=dev
                    )
                elif kind == 2:
                    out = None
                else:
                    # y_local: only this rank's owned rows; compare with the same rows of the full expected y
                    b, e = rr.owned_rows(n_pad)
                    dt = torch.float32 if kind == 3 else torch.bfloat16
                    out = torch.full((cap, a.dim), float("nan"), dtype=dt, device=dev)
                    rr.combine_local(partial, out, n_pad)
                    torch.cuda.current_stream().synchronize()
                    exp = call.y[b:e]
                    got = out[: e - b]
                    if dt == torch.float32:
                        by = (
                            int(
                                (got.view(torch.int32) != exp.float().view(torch.int32))
                                .any(1)
                                .sum()
                                .item()
                            )
                            if e > b
                            else 0
                        )
                    else:
                        by = (
                            int((got != exp.to(torch.bfloat16)).any(1).sum().item())
                            if e > b
                            else 0
                        )
                    y = None
                if kind < 3:
                    y = rr.combine(partial, out, n_pad)
                    torch.cuda.current_stream().synchronize()
                    if out is None:
                        y = y.clone()
                    by = int(check_y(call, y).item())
                tr1 = rr.traffic()
                dr, cr = (
                    tr1["disp_rows"] - tr0["disp_rows"],
                    tr1["comb_rows"] - tr0["comb_rows"],
                )
                if (dr, cr) != (call.rdma_disp_rows, call.rdma_comb_rows):
                    bad_tr[0] += 1
                    if bad_tr[0] < 4:
                        log(
                            f"traffic np={n_pad} t={t}: rdma rows disp {dr} (exp {call.rdma_disp_rows}) "
                            f"comb {cr} (exp {call.rdma_comb_rows})"
                        )
                rows_tot[0] += dr
                rows_tot[1] += cr
                rows_tot[2] += call.hits_me
                if (bg or by) and (bad_g + bad_y) < 8:
                    log(
                        f"eager np={n_pad} t={t} modes={call.modes} bad_gath={bg} bad_y={by} "
                        f"stats={rr.stats()}"
                    )
                bad_g += bg
                bad_y += by
        gen.padding = False
        return {
            "calls": a.eager_calls,
            "bad_gath_rows": bad_g,
            "bad_y_rows": bad_y,
            "bad_traffic_calls": bad_tr[0],
            "rdma_disp_rows": rows_tot[0],
            "rdma_comb_rows": rows_tot[1],
            "gathered_rows": rows_tot[2],
        }

    def negative(n_pad):
        """Protocol misuse: node 1 calls with n_pad-1 while node 0 uses n_pad. Must count npad_mismatch (dev_err[3]),
        not hang; the following correctly-sized call must be exact again."""
        before = rr.stats()["npad_mismatch"]
        my_np = n_pad if rank < lw else n_pad - 1
        cap = rr.cap(n_pad)
        partial = torch.zeros(world * cap, a.dim, dtype=torch.bfloat16, device=dev)
        out = torch.zeros(n_pad, a.dim, dtype=torch.float32, device=dev)
        with torch.cuda.stream(s):
            t = t_next[0]
            t_next[0] += 1
            call = Call(gen, rank, t, my_np, a.y_fp32)
            rr.dispatch(call.payload, None, my_np)
            rr.combine(partial, out, my_np)
            torch.cuda.current_stream().synchronize()
        dist.barrier()
        after = rr.stats()["npad_mismatch"]
        rec = eager_one(n_pad)
        return {"npad_mismatch_counted": int(after - before), "followup_bad_rows": rec}

    def eager_one(n_pad):
        cap = rr.cap(n_pad)
        partial = torch.empty(world * cap, a.dim, dtype=torch.bfloat16, device=dev)
        out = torch.full((n_pad, a.dim), float("nan"), dtype=torch.float32, device=dev)
        with torch.cuda.stream(s):
            t = t_next[0]
            t_next[0] += 1
            call = Call(gen, rank, t, n_pad, a.y_fp32)
            gath = rr.dispatch(call.payload, None, n_pad)
            torch.cuda.current_stream().synchronize()
            bg = int(check_gath(gen, call, gath).item())
            partial.copy_(gen.stub(gath, rank))
            rr.combine(partial, out, n_pad)
            torch.cuda.current_stream().synchronize()
            return bg + int(check_y(call, out).item())

    def graph_verify(n_pad):
        L = a.layers
        cap = rr.cap(n_pad)
        pays = [
            torch.zeros(n_pad, a.payload, dtype=torch.uint8, device=dev)
            for _ in range(L)
        ]
        snaps = [
            torch.zeros(world * cap, a.payload, dtype=torch.uint8, device=dev)
            for _ in range(L)
        ]
        yl = bool(a.y_local)
        outs = [
            torch.zeros(cap if yl else n_pad, a.dim, dtype=out_dt, device=dev)
            for _ in range(L)
        ]
        partial = torch.zeros(world * cap, a.dim, dtype=torch.bfloat16, device=dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for l in range(L):
                gath = rr.dispatch(pays[l], None, n_pad)
                snaps[l].copy_(gath)
                partial.copy_(gen.stub(gath, rank))
                if yl:
                    rr.combine_local(partial, outs[l], n_pad)
                else:
                    rr.combine(partial, outs[l], n_pad)
        dist.barrier()
        bad_g = bad_y = 0
        with torch.cuda.stream(s):
            for rep in range(a.verify_replays):
                calls = []
                for l in range(L):
                    c = Call(gen, rank, t_next[0] + l, n_pad, a.y_fp32)
                    pays[l].copy_(c.payload)
                    calls.append(c)
                t_next[0] += L
                g.replay()
                torch.cuda.current_stream().synchronize()
                for l in range(L):
                    bg = int(check_gath(gen, calls[l], snaps[l]).item())
                    by = int(check_y_any(rr, calls[l], outs[l], n_pad, yl).item())
                    if (bg or by) and (bad_g + bad_y) < 8:
                        log(
                            f"verify np={n_pad} t={calls[l].t} l={l} modes={calls[l].modes} bad_gath={bg} bad_y={by}"
                        )
                    bad_g += bg
                    bad_y += by
        del g
        return {
            "replays": a.verify_replays,
            "L": L,
            "bad_gath_rows": bad_g,
            "bad_y_rows": bad_y,
        }

    def graph_perf(n_pad, expert_us):
        L = a.layers
        cap = rr.cap(n_pad)
        pays = [
            torch.zeros(n_pad, a.payload, dtype=torch.uint8, device=dev)
            for _ in range(L)
        ]
        parts = [
            torch.zeros(world * cap, a.dim, dtype=torch.bfloat16, device=dev)
            for _ in range(L)
        ]
        yl = bool(a.y_local)
        outs = [
            torch.zeros(cap if yl else n_pad, a.dim, dtype=out_dt, device=dev)
            for _ in range(L)
        ]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for l in range(L):
                rr.dispatch(pays[l], None, n_pad)
                if expert_us > 0:
                    rr.sleep_ns(expert_us * 1000)
                if yl:
                    rr.combine_local(parts[l], outs[l], n_pad)
                else:
                    rr.combine(parts[l], outs[l], n_pad)
        dist.barrier()
        bad_y = 0
        bad_last_gath = 0
        step_ms = []
        t_first = None
        saved = gen.modes
        gen.modes = perf_modes
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s):
            for st in range(a.warm + a.steps):
                calls = []
                for l in range(L):
                    c = Call(gen, rank, t_next[0] + l, n_pad, a.y_fp32)
                    pays[l].copy_(c.payload)
                    parts[l].copy_(c.partial)
                    calls.append(c)
                if st == a.warm:
                    t_first = t_next[0]
                t_next[0] += L
                torch.cuda.current_stream().synchronize()
                dist.barrier()
                ev0.record()
                g.replay()
                ev1.record()
                ev1.synchronize()
                if st >= a.warm:
                    step_ms.append(ev0.elapsed_time(ev1))
                for l in range(L):
                    by = int(check_y_any(rr, calls[l], outs[l], n_pad, yl).item())
                    if by and bad_y < 8:
                        log(
                            f"perf np={n_pad} t={calls[l].t} l={l} modes={calls[l].modes} bad_y={by}"
                        )
                    bad_y += by
                bad_last_gath += int(
                    check_gath(gen, calls[-1], rr.gathered_buffers(n_pad)).item()
                )
        gen.modes = saved
        del g
        ts = rr.timestamps()
        n_ts = len(ts)
        split = {
            k: []
            for k in ("disp", "comb", "d_wait_x", "c_wait_x", "c_local_in", "period")
        }
        names = [
            "d_db",
            "d1_end",
            "d2_start",
            "d_x_lat",
            "d2_tail",
            "c1_end",
            "c_local_in",
            "c_prered",
            "c_x_lat",
            "c2_tail",
            "gap_c2d",
        ]
        det = {k: [] for k in names}
        for t in range(t_first, t_next[0]):
            q = ts[(t - 1) % n_ts]
            if not all(q):
                continue
            split["disp"].append((q[5] - q[0]) / 1e3)
            split["comb"].append((q[11] - q[6]) / 1e3)
            split["d_wait_x"].append((q[4] - q[3]) / 1e3)
            split["c_wait_x"].append((q[10] - q[9]) / 1e3)
            split["c_local_in"].append((q[8] - q[6]) / 1e3)
            vals = [
                q[1] - q[0],
                q[2] - q[0],
                q[3] - q[0],
                q[4] - q[1],
                q[5] - q[4],
                q[7] - q[6],
                q[8] - q[6],
                q[9] - q[8],
                q[10] - q[9],
                q[11] - q[10],
            ]
            for k, v in zip(names, vals):
                det[k].append(v / 1e3)
            if (t - t_first) % L:
                qp = ts[(t - 2) % n_ts]
                split["period"].append((q[0] - qp[0]) / 1e3)
                det["gap_c2d"].append((q[0] - qp[11]) / 1e3)
        per_layer = [ms * 1000.0 / L - expert_us for ms in step_ms]
        r = {
            "L": L,
            "steps": a.steps,
            "expert_us": expert_us,
            "bad_y_rows": bad_y,
            "bad_last_gath": bad_last_gath,
            "layer_p50": round(pct(per_layer, 0.5), 1),
            "layer_p90": round(pct(per_layer, 0.9), 1),
            "layer_min": round(min(per_layer), 1) if per_layer else 0,
        }
        for k, v in split.items():
            r[k + "_p50"] = round(pct(v, 0.5), 1)
            r[k + "_p90"] = round(pct(v, 0.9), 1)
        r["detail_p50"] = {k: round(pct(v, 0.5), 1) for k, v in det.items()}
        return r

    phases = a.phases.split(",")
    for n_pad in [int(x) for x in a.npads.split(",")]:
        if "eager" in phases:
            res["phases"][f"eager_np{n_pad}"] = eager(n_pad)
            log(f"eager np={n_pad}: {res['phases'][f'eager_np{n_pad}']}")
        if "padding" in phases:
            res["phases"][f"padding_np{n_pad}"] = eager(n_pad, padding=True)
            log(f"padding np={n_pad}: {res['phases'][f'padding_np{n_pad}']}")
        if "verify" in phases:
            res["phases"][f"verify_np{n_pad}"] = graph_verify(n_pad)
            log(f"verify np={n_pad}: {res['phases'][f'verify_np{n_pad}']}")
        if "perf" in phases:
            for eus in [int(x) for x in a.expert_us.split(",")]:
                k = f"perf_np{n_pad}_e{eus}"
                res["phases"][k] = graph_perf(n_pad, eus)
                log(f"{k}: {res['phases'][k]}")
        torch.cuda.empty_cache()
    if "negative" in phases:
        res["phases"]["negative"] = negative(int(a.npads.split(",")[0]))
        log(f"negative: {res['phases']['negative']}")
    st = rr.stats()
    res["stats"] = st
    res["t_last"] = t_next[0] - 1
    log(f"stats {st}")
    allres = [None] * world
    dist.all_gather_object(allres, res)
    if rank == 0:
        summary = {
            "tag": a.tag,
            "world": world,
            "lw": lw,
            "flag_mode": rr.flag_mode,
            "blocks": a.blocks,
            "y_fp32": a.y_fp32,
            "phases": {},
        }
        for k in res["phases"]:
            ph = [r["phases"][k] for r in allres]
            agg = {}
            for f in ph[0]:
                vals = [p[f] for p in ph]
                if f.startswith(
                    ("bad", "rdma_", "gathered_rows", "npad_mismatch", "followup")
                ):
                    agg[f] = sum(vals)
                elif isinstance(vals[0], (int, float)):
                    agg[f] = max(vals) if f.endswith(("p50", "p90", "min")) else vals[0]
            summary["phases"][k] = agg
        errs = {
            k: sum(r["stats"][k] for r in allres)
            for k in (
                "host_errors",
                "spin_timeouts",
                "bad_flags",
                "row_mismatch",
                "npad_mismatch",
            )
        }
        summary["errors"] = errs
        summary["ep_consistent"] = len({r["stats"]["ep"] for r in allres}) == 1
        print("SUMMARY " + json.dumps(summary), flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    dist.barrier()
    rr.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
