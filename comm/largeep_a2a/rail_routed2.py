# rail_routed2.py - Python wrapper for librail_routed2.so (routed EP all-to-all, 2 nodes x local_world GPUs).
#
#   rr = RailRouted2(rank, world=16, local_world=8, gpu=local, nic="erdma_0"|"erdma_1", n_pad_max=192,
#                    payload_bytes=4176, dim=4096, topk=6, experts_per_rank=16, group=dist.group.WORLD)
#   gathered = rr.gathered_buffers(n_pad)            # [world*cap, payload_bytes] uint8, cap = ceil(n_pad/tp)
#   rr.dispatch(payload, topk_ids, n_pad)            # fills gathered (+ ids=-1 on unfilled rows)
#   ... masked grouped GEMM over gathered -> partial[world*cap, dim] bf16 (row-aligned with gathered) ...
#   rr.combine(partial, out_y, n_pad)                # out_y[n_pad, dim] fp32 or bf16: full y of the TP group
#
# All three emit only kernel launches into torch.cuda.current_stream() (CUDA-graph capturable). Construction
# is a collective over `group` (all_gather_object of the wire info + barrier) and must run eagerly.
from __future__ import annotations

import base64
import ctypes
import os

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_CANDIDATES = [
    os.environ.get("RR2_LIB", ""),
    os.path.join(_HERE, "librail_routed2.so"),
]


class RailRouted2Error(RuntimeError):
    pass


class _Config(ctypes.Structure):
    _fields_ = [
        ("gpu_index", ctypes.c_int),
        ("rank", ctypes.c_int),
        ("world", ctypes.c_int),
        ("local_world", ctypes.c_int),
        ("tp", ctypes.c_int),
        ("gid_index", ctypes.c_int),
        ("n_pad_max", ctypes.c_int),
        ("payload_bytes", ctypes.c_int),
        ("dim", ctypes.c_int),
        ("topk", ctypes.c_int),
        ("experts_per_rank", ctypes.c_int),
        ("ids_offset", ctypes.c_int),
        ("y_fp32", ctypes.c_int),
        ("flag_mode", ctypes.c_int),
        ("blocks", ctypes.c_int),
        ("proxy_cpu", ctypes.c_int),
        ("ts_calls", ctypes.c_int),
        ("ib_port", ctypes.c_int),
        ("spin_ns", ctypes.c_uint64),
        ("nic_name", ctypes.c_char * 64),
    ]


class _Wire(ctypes.Structure):
    _fields_ = [
        ("slab_base", ctypes.c_uint64),
        ("ipc", ctypes.c_uint8 * 64),
        ("gid", ctypes.c_uint8 * 16),
        ("qpn", ctypes.c_uint32),
        ("psn", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
        ("rank", ctypes.c_int32),
        ("proto", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("disp_posted", ctypes.c_uint64),
        ("comb_posted", ctypes.c_uint64),
        ("flags_posted", ctypes.c_uint64),
        ("send_cqe", ctypes.c_uint64),
        ("recv_imm", ctypes.c_uint64),
        ("host_errors", ctypes.c_uint64),
        ("dev_err", ctypes.c_uint64 * 4),
        ("last_d", ctypes.c_uint64),
        ("last_c", ctypes.c_uint64),
        ("ep", ctypes.c_uint64),
    ]


TS_NAMES = [
    "d1_start",
    "d_doorbell",
    "d1_end",
    "d2_start",
    "d_x_arrive",
    "d2_end",
    "c1_start",
    "c1_end",
    "c2_local_in",
    "c_doorbell",
    "c_x_arrive",
    "c2_end",
    "c2_y_written",
    "c2_y_mates_in",
]


def _load(path_hint=None):
    errs = []
    for p in ([path_hint] if path_hint else []) + _LIB_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        try:
            lib = ctypes.CDLL(p)
            break
        except OSError as e:  # noqa: PERF203
            errs.append(f"{p}: {e}")
    else:
        raise RailRouted2Error(
            f"librail_routed2.so not loadable: {errs or 'no candidates'}"
        )
    vp, ci, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    lib.rr2_version.restype = ci
    lib.rr2_init.restype = vp
    lib.rr2_init.argtypes = [ctypes.POINTER(_Config)]
    lib.rr2_last_error.restype = ctypes.c_char_p
    lib.rr2_last_error.argtypes = [vp]
    lib.rr2_get_wire.restype = ci
    lib.rr2_get_wire.argtypes = [vp, ctypes.POINTER(_Wire)]
    lib.rr2_connect.restype = ci
    lib.rr2_connect.argtypes = [vp, ctypes.POINTER(_Wire), ci]
    lib.rr2_gathered_ptr.restype = vp
    lib.rr2_gathered_ptr.argtypes = [vp]
    lib.rr2_y_ptr.restype = vp
    lib.rr2_y_ptr.argtypes = [vp]
    lib.rr2_dispatch.restype = ci
    lib.rr2_dispatch.argtypes = [vp, vp, ci, vp]
    lib.rr2_combine.restype = ci
    lib.rr2_combine.argtypes = [vp, vp, vp, ci, ci, vp]
    if hasattr(lib, "rr2_combine2"):
        lib.rr2_combine2.restype = ci
        lib.rr2_combine2.argtypes = [vp, vp, vp, ci, ci, ci, vp]
    lib.rr2_get_stats.restype = ci
    lib.rr2_get_stats.argtypes = [vp, ctypes.POINTER(_Stats)]
    lib.rr2_fetch_ts.restype = ci
    lib.rr2_fetch_ts.argtypes = [vp, ctypes.POINTER(ctypes.c_uint64), u64]
    lib.rr2_get_traffic.restype = ci
    lib.rr2_get_traffic.argtypes = [vp, ctypes.POINTER(ctypes.c_uint64)]
    lib.rr2_sleep_ns.restype = ci
    lib.rr2_sleep_ns.argtypes = [u64, vp]
    lib.rr2_destroy.restype = None
    lib.rr2_destroy.argtypes = [vp]
    return lib


class _DevMem:
    """Minimal __cuda_array_interface__ exporter so torch can view library-owned VRAM without copying."""

    def __init__(self, ptr, shape, typestr, owner):
        self._owner = (
            owner  # keeps the RailRouted2 (and its slab) alive while any view exists
        )
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
        }


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


class RailRouted2:
    """Routed EP16 dispatch/combine (one instance per process / GPU)."""

    def __init__(
        self,
        rank: int,
        world: int = 16,
        local_world: int = 8,
        gpu: int | None = None,
        nic: str | None = None,
        n_pad_max: int = 192,
        payload_bytes: int = 4176,
        dim: int = 4096,
        topk: int = 6,
        experts_per_rank: int = 16,
        *,
        tp: int = 4,
        group=None,
        ids_offset: int = -1,
        y_fp32: bool | None = None,
        flag_mode: int | None = None,
        blocks: int | None = None,
        proxy_cpu: int | None = None,
        spin_ns: int | None = None,
        gid_index: int | None = None,
        ts_calls: int | None = None,
        lib_path: str | None = None,
    ):
        if world != 2 * local_world:
            raise RailRouted2Error(f"world {world} must be 2*local_world {local_world}")
        self.rank, self.world, self.local_world, self.tp = (
            int(rank),
            int(world),
            int(local_world),
            int(tp),
        )
        self.node, self.local = divmod(self.rank, self.local_world)
        self.gpu = self.local if gpu is None else int(gpu)
        self.nic = nic or (
            "erdma_0" if self.local < max(1, self.local_world // 2) else "erdma_1"
        )
        self.n_pad_max, self.payload_bytes, self.dim, self.topk = (
            int(n_pad_max),
            int(payload_bytes),
            int(dim),
            int(topk),
        )
        self.experts_per_rank = int(experts_per_rank)
        self.y_fp32 = (
            bool(_env_int("RR2_Y_FP32", 0)) if y_fp32 is None else bool(y_fp32)
        )
        self.y_dtype = torch.float32 if self.y_fp32 else torch.bfloat16
        self.device = torch.device("cuda", self.gpu)
        self._lib = _load(lib_path)
        cfg = _Config(
            gpu_index=self.gpu,
            rank=self.rank,
            world=self.world,
            local_world=self.local_world,
            tp=self.tp,
            gid_index=_env_int("RR2_GID", 1) if gid_index is None else int(gid_index),
            n_pad_max=self.n_pad_max,
            payload_bytes=self.payload_bytes,
            dim=self.dim,
            topk=self.topk,
            experts_per_rank=self.experts_per_rank,
            ids_offset=int(ids_offset),
            y_fp32=int(self.y_fp32),
            flag_mode=(
                _env_int("RR2_FLAG_MODE", 2) if flag_mode is None else int(flag_mode)
            ),
            blocks=_env_int("RR2_BLOCKS", 16) if blocks is None else int(blocks),
            proxy_cpu=(
                _env_int("RR2_PROXY_CPU", -1) if proxy_cpu is None else int(proxy_cpu)
            ),
            ts_calls=(
                _env_int("RR2_TS_CALLS", 4096) if ts_calls is None else int(ts_calls)
            ),
            ib_port=1,
            spin_ns=(
                _env_int("RR2_SPIN_NS", 5_000_000_000)
                if spin_ns is None
                else int(spin_ns)
            ),
            nic_name=self.nic.encode(),
        )
        self.flag_mode = cfg.flag_mode
        self.ts_calls = cfg.ts_calls
        with torch.cuda.device(self.device):
            torch.cuda.current_stream().synchronize()
            self._h = self._lib.rr2_init(ctypes.byref(cfg))
        if not self._h:
            raise RailRouted2Error("rr2_init returned NULL")
        err = self._err()
        if err:
            self.close()
            raise RailRouted2Error(f"rr2_init: {err}")
        w = _Wire()
        if self._lib.rr2_get_wire(self._h, ctypes.byref(w)):
            raise RailRouted2Error(f"rr2_get_wire: {self._err()}")
        group = group if group is not None else dist.group.WORLD
        gathered = [None] * self.world
        dist.all_gather_object(
            gathered, base64.b64encode(bytes(w)).decode(), group=group
        )
        wires = (_Wire * self.world)()
        for i, b in enumerate(gathered):
            wires[i] = _Wire.from_buffer_copy(base64.b64decode(b))
        rc = self._lib.rr2_connect(self._h, wires, self.world)
        if rc:
            raise RailRouted2Error(f"rr2_connect rc={rc}: {self._err()}")
        dist.barrier(group=group)
        self._gath_ptr = int(self._lib.rr2_gathered_ptr(self._h))
        self._y_ptr = int(self._lib.rr2_y_ptr(self._h))
        self._views = {}

    # ------------------------------------------------------------------ helpers
    def _err(self) -> str:
        p = self._lib.rr2_last_error(self._h) if getattr(self, "_h", None) else b""
        return p.decode() if p else ""

    def cap(self, n_pad: int) -> int:
        return -(-int(n_pad) // self.tp)

    def owned_rows(self, n_pad: int) -> tuple[int, int]:
        """[begin, end) rows of the TP group's n_pad rows owned (dispatched + reduced) by this rank."""
        c = self.cap(n_pad)
        b = min(n_pad, (self.rank % self.tp) * c)
        return b, min(n_pad, b + c)

    def gathered_buffers(self, n_pad: int) -> torch.Tensor:
        """Fixed [world*cap, payload_bytes] uint8 view of the receive buffer for this n_pad (cached).

        Row s*cap + j holds the j-th row this rank received from source rank s; rows past the per-source count
        have every topk id = -1. Valid between dispatch(t) and combine(t) of the same call.
        """
        v = self._views.get(("g", n_pad))
        if v is None:
            rows = self.world * self.cap(n_pad)
            v = torch.as_tensor(
                _DevMem(self._gath_ptr, (rows, self.payload_bytes), "|u1", self),
                device=self.device,
            )
            self._views[("g", n_pad)] = v
        return v

    def y_buffer(self, n_pad: int) -> torch.Tensor:
        """[n_pad, dim] view of the library y buffer (dtype self.y_dtype), filled by combine(); valid until the
        next combine of this process is issued."""
        v = self._views.get(("y", n_pad))
        if v is None:
            v = torch.as_tensor(
                _DevMem(
                    self._y_ptr,
                    (n_pad, self.dim),
                    "<f4" if self.y_fp32 else "<f2",
                    self,
                ),
                device=self.device,
            )
            if not self.y_fp32:
                v = v.view(torch.bfloat16)
            self._views[("y", n_pad)] = v
        return v

    # -------------------------------------------------------------------- emits
    def dispatch(
        self, payload: torch.Tensor, topk_ids: torch.Tensor | None, n_pad: int
    ) -> torch.Tensor:
        """Emit the routed dispatch into the current stream; returns gathered_buffers(n_pad).

        Routing is read from the int32 topk ids embedded in each payload row (the copy the receiver's GEMM
        uses); `topk_ids` is accepted for interface symmetry and not read."""
        if (
            payload.dtype != torch.uint8
            or not payload.is_contiguous()
            or payload.dim() != 2
        ):
            raise RailRouted2Error(
                "payload must be a contiguous [n_pad, payload_bytes] uint8 tensor"
            )
        if payload.size(0) < n_pad or payload.size(1) != self.payload_bytes:
            raise RailRouted2Error(
                f"payload shape {tuple(payload.shape)} vs n_pad {n_pad} / P {self.payload_bytes}"
            )
        rc = self._lib.rr2_dispatch(
            self._h,
            payload.data_ptr(),
            int(n_pad),
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if rc:
            raise RailRouted2Error(f"rr2_dispatch rc={rc}: {self._err()}")
        return self.gathered_buffers(n_pad)

    def combine(
        self, partial: torch.Tensor, out_y: torch.Tensor | None, n_pad: int
    ) -> torch.Tensor:
        """Emit the routed combine; out_y[n_pad, dim] (fp32 or bf16) receives the TP group's full y.

        out_y=None leaves the result in y_buffer(n_pad) and returns that view."""
        cap = self.cap(n_pad)
        if (
            partial.dtype != torch.bfloat16
            or not partial.is_contiguous()
            or partial.size(0) < self.world * cap
        ):
            raise RailRouted2Error(
                f"partial must be contiguous bf16 [{self.world * cap}, {self.dim}]"
            )
        if partial.size(1) != self.dim:
            raise RailRouted2Error(f"partial dim {partial.size(1)} != {self.dim}")
        if out_y is not None:
            if (
                out_y.dtype not in (torch.float32, torch.bfloat16)
                or not out_y.is_contiguous()
            ):
                raise RailRouted2Error("out_y must be contiguous fp32/bf16")
            if out_y.size(0) < n_pad or out_y.size(1) != self.dim:
                raise RailRouted2Error(
                    f"out_y shape {tuple(out_y.shape)} vs [{n_pad}, {self.dim}]"
                )
            optr, of32 = out_y.data_ptr(), int(out_y.dtype == torch.float32)
        else:
            optr, of32 = 0, 0
        rc = self._lib.rr2_combine(
            self._h,
            partial.data_ptr(),
            optr,
            of32,
            int(n_pad),
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if rc:
            raise RailRouted2Error(f"rr2_combine rc={rc}: {self._err()}")
        return out_y if out_y is not None else self.y_buffer(n_pad)

    def combine_local(
        self, partial: torch.Tensor, out_local: torch.Tensor, n_pad: int
    ) -> torch.Tensor:
        """Combine with y_local=1 (library ABI >= 6): out_local[cap, dim] (fp32/bf16) receives only this rank's owned
        rows (local row i = TP-group row own_base + i); no intra-group y all-gather. All ranks of a TP group must
        use the same variant for a call."""
        cap = self.cap(n_pad)
        if not hasattr(self._lib, "rr2_combine2"):
            raise RailRouted2Error(
                "library has no rr2_combine2 (need routed2 ABI >= 6)"
            )
        if (
            partial.dtype != torch.bfloat16
            or not partial.is_contiguous()
            or partial.size(0) < self.world * cap
        ):
            raise RailRouted2Error(
                f"partial must be contiguous bf16 [{self.world * cap}, {self.dim}]"
            )
        if (
            out_local.dtype not in (torch.float32, torch.bfloat16)
            or not out_local.is_contiguous()
        ):
            raise RailRouted2Error("out_local must be contiguous fp32/bf16")
        if out_local.size(0) < cap or out_local.size(1) != self.dim:
            raise RailRouted2Error(
                f"out_local shape {tuple(out_local.shape)} vs [{cap}, {self.dim}]"
            )
        rc = self._lib.rr2_combine2(
            self._h,
            partial.data_ptr(),
            out_local.data_ptr(),
            int(out_local.dtype == torch.float32),
            int(n_pad),
            1,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if rc:
            raise RailRouted2Error(f"rr2_combine2 rc={rc}: {self._err()}")
        return out_local

    def sleep_ns(self, ns: int):
        """Fake compute (spin kernel) on the current stream; for benchmarks."""
        self._lib.rr2_sleep_ns(
            int(ns), torch.cuda.current_stream(self.device).cuda_stream
        )

    # --------------------------------------------------------------- telemetry
    def stats(self) -> dict:
        """Host-syncing counters (never call during capture)."""
        s = _Stats()
        self._lib.rr2_get_stats(self._h, ctypes.byref(s))
        return {
            "disp_posted": s.disp_posted,
            "comb_posted": s.comb_posted,
            "flags_posted": s.flags_posted,
            "send_cqe": s.send_cqe,
            "recv_imm": s.recv_imm,
            "host_errors": s.host_errors,
            "spin_timeouts": s.dev_err[0],
            "bad_flags": s.dev_err[1],
            "row_mismatch": s.dev_err[2],
            "npad_mismatch": s.dev_err[3],
            "last_d": s.last_d,
            "last_c": s.last_c,
            "ep": s.ep,
            "last_error": self._err(),
        }

    def traffic(self) -> dict:
        """RDMA payload rows/bytes this rank has posted so far (dispatch / combine legs)."""
        b = (ctypes.c_uint64 * 4)()
        self._lib.rr2_get_traffic(self._h, b)
        return {
            "disp_rows": b[0],
            "comb_rows": b[1],
            "disp_bytes": b[2],
            "comb_bytes": b[3],
        }

    def timestamps(self):
        """[ts_calls, len(TS_NAMES)] globaltimer ns per call slot ((t-1) % ts_calls); columns = TS_NAMES."""
        n = self.ts_calls * len(TS_NAMES)
        buf = (ctypes.c_uint64 * n)()
        k = self._lib.rr2_fetch_ts(self._h, buf, n)
        nc = len(TS_NAMES)
        return [list(buf[i * nc : (i + 1) * nc]) for i in range(max(0, k) // nc)]

    def close(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.rr2_destroy(h)
            self._h = None
        self._views = {}

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
