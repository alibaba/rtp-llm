# Routed rail transport context for the SM120 routed rail MoE strategy (librail_routed2.so).
#
# Process-wide singleton per GPU: one rail RC QP to the rank with the same local id on the other
# node, a CPU proxy thread inside the library (ibverbs only, no CUDA), a VRAM slab registered with
# the NIC and IPC-shared with the same-node ranks. Wire info is exchanged with one WORLD
# all_gather_object at model build time (eager), before any CUDA graph capture.
#
# Graph contract: ``dispatch``/``combine`` only launch kernels into the current torch stream; the
# library reads the call sequence from a device counter, so a captured graph replays correctly.

from __future__ import annotations

import ctypes
import logging
import os
import threading

import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


class RoutedRailUnavailable(RuntimeError):
    pass


class _Rr2Config(ctypes.Structure):
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


class _Rr2Wire(ctypes.Structure):
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


class _Rr2Stats(ctypes.Structure):
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


def _load_lib():
    # build with comm/largeep_a2a/build.sh; default = the copy next to this module
    path = os.environ.get(
        "DSV4_SM120_RAIL_ROUTED_LIB",
        os.path.join(os.path.dirname(__file__), "librail_routed2.so"),
    )
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        raise RoutedRailUnavailable(f"{path}: {e}") from e
    lib.rr2_init.restype = ctypes.c_void_p
    lib.rr2_init.argtypes = [ctypes.POINTER(_Rr2Config)]
    lib.rr2_last_error.restype = ctypes.c_char_p
    lib.rr2_last_error.argtypes = [ctypes.c_void_p]
    lib.rr2_get_wire.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rr2Wire)]
    lib.rr2_connect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rr2Wire), ctypes.c_int]
    lib.rr2_gathered_ptr.restype = ctypes.c_void_p
    lib.rr2_gathered_ptr.argtypes = [ctypes.c_void_p]
    lib.rr2_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.rr2_combine.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.rr2_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rr2Stats)]
    if hasattr(
        lib, "rr2_combine2"
    ):  # ABI >= 6 (y_local combine, used by the token-sharded decode / tp_rsag)
        lib.rr2_combine2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
    lib.rr2_destroy.argtypes = [ctypes.c_void_p]
    return lib


class RoutedRailTransport:
    """One per process: rail QP + proxy + slab for this rank."""

    def __init__(self, cfg, device: torch.device, payload_bytes: int, ids_offset: int):
        if not dist.is_initialized():
            raise RoutedRailUnavailable(
                "routed rail requires an initialized WORLD group"
            )
        world = dist.get_world_size(dist.group.WORLD)
        rank = dist.get_rank(dist.group.WORLD)
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
        if cfg.ep_size != world or world != 2 * local_world:
            raise RoutedRailUnavailable(
                f"routed rail expects ep_size==world==2*local_world, got ep={cfg.ep_size} world={world} local={local_world}"
            )
        tp = int(cfg.tp_size)
        local = rank % local_world
        # DSV4_SM120_RAIL_CPU: "c0,c1,..,c7" (per local rank) or a single base core (proxy = base + local);
        # one shared core for all 8 proxies of a node would serialize them
        cpu_env = os.environ.get("DSV4_SM120_RAIL_CPU", "-1").strip()
        if "," in cpu_env:
            proxy_cpu = int(cpu_env.split(",")[local])
        else:
            proxy_cpu = int(cpu_env) + local if int(cpu_env) >= 0 else -1
        nic_map = os.environ.get("DSV4_SM120_RAIL_NIC_MAP", "0,0,0,0,1,1,1,1")
        nic_idx = int(nic_map.split(",")[local])
        # DSV4_SM120_RAIL_NIC_PREFIX: RDMA device name prefix (erdma_ on the development hosts; mlx5_ etc. elsewhere)
        nic_name = os.environ.get("DSV4_SM120_RAIL_NIC_PREFIX", "erdma_") + str(nic_idx)
        self.world, self.rank, self.tp = world, rank, tp
        self.n_pad_max = int(os.environ.get("DSV4_SM120_RAIL_NPAD_MAX", "192"))
        self.payload_bytes = int(payload_bytes)
        self.dim = int(cfg.dim)
        self.device = device
        self._lib = _load_lib()
        rcfg = _Rr2Config(
            gpu_index=device.index,
            rank=rank,
            world=world,
            local_world=local_world,
            tp=tp,
            gid_index=int(os.environ.get("DSV4_SM120_RAIL_GID", "1")),
            n_pad_max=self.n_pad_max,
            payload_bytes=self.payload_bytes,
            dim=self.dim,
            topk=int(cfg.n_activated_experts),
            experts_per_rank=int(cfg.n_local_experts),
            ids_offset=int(ids_offset),
            # y all-gather dtype between TP mates: bf16 (0) measured 133us/layer at n_pad=48 on 2x8 vs 200us with
            # fp32 (1). out stays fp32 either way (bf16-rounded sum widened); see routed2/README.md
            y_fp32=int(os.environ.get("DSV4_SM120_RAIL_Y_FP32", "0")),
            flag_mode=int(os.environ.get("DSV4_SM120_RAIL_FLAG_MODE", "2")),
            blocks=int(os.environ.get("DSV4_SM120_RAIL_BLOCKS", "16")),
            proxy_cpu=proxy_cpu,
            ts_calls=int(os.environ.get("DSV4_SM120_RAIL_TS_CALLS", "256")),
            ib_port=1,
            spin_ns=int(os.environ.get("DSV4_SM120_RAIL_SPIN_NS", "5000000000")),
            nic_name=nic_name.encode(),
        )
        self._h = self._lib.rr2_init(ctypes.byref(rcfg))
        err = self._last_error()
        if not self._h or err:
            raise RoutedRailUnavailable(f"rr2_init failed: {err}")
        self._exchange_and_connect()
        self._gathered = None
        self._start_stats_thread()
        self._start_ts_dump_thread()
        _logger.warning(
            "routed rail up: rank=%d tp=%d nic=%s n_pad_max=%d",
            rank,
            tp,
            nic_name,
            self.n_pad_max,
        )

    def _last_error(self) -> str:
        p = self._lib.rr2_last_error(ctypes.c_void_p(self._h)) if self._h else None
        return p.decode() if p else ""

    def _exchange_and_connect(self):
        import base64

        wire = _Rr2Wire()
        rc = self._lib.rr2_get_wire(ctypes.c_void_p(self._h), ctypes.byref(wire))
        if rc != 0:
            raise RoutedRailUnavailable(f"rr2_get_wire failed: {self._last_error()}")
        mine = base64.b64encode(bytes(wire)).decode()
        gathered = [None] * self.world
        dist.all_gather_object(gathered, mine, group=dist.group.WORLD)
        wires = (_Rr2Wire * self.world)()
        for i, g in enumerate(gathered):
            wires[i] = _Rr2Wire.from_buffer_copy(base64.b64decode(g))
        rc = self._lib.rr2_connect(ctypes.c_void_p(self._h), wires, self.world)
        if rc != 0:
            raise RoutedRailUnavailable(
                f"rr2_connect failed rc={rc}: {self._last_error()}"
            )
        dist.barrier(group=dist.group.WORLD)

    def cap(self, n_pad: int) -> int:
        return (int(n_pad) + self.tp - 1) // self.tp

    def gathered_view(self, rows: int) -> torch.Tensor:
        """Library-owned receive buffer viewed as ``[rows, payload_bytes]`` uint8."""
        cap_max = self.cap(self.n_pad_max)
        if self._gathered is None:
            ptr = self._lib.rr2_gathered_ptr(ctypes.c_void_p(self._h))
            n = self.world * cap_max * self.payload_bytes
            self._gathered = _wrap_device_ptr(ptr, n, self.device)
        return self._gathered[: rows * self.payload_bytes].view(
            rows, self.payload_bytes
        )

    def dispatch(self, payload: torch.Tensor, n_pad: int):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        rc = self._lib.rr2_dispatch(
            ctypes.c_void_p(self._h),
            ctypes.c_void_p(payload.data_ptr()),
            int(n_pad),
            ctypes.c_void_p(stream),
        )
        if rc != 0:
            raise RoutedRailUnavailable(f"rr2_dispatch rc={rc}: {self._last_error()}")

    def combine(self, partial: torch.Tensor, out: torch.Tensor, n_pad: int):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        rc = self._lib.rr2_combine(
            ctypes.c_void_p(self._h),
            ctypes.c_void_p(partial.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            1 if out.dtype == torch.float32 else 0,
            int(n_pad),
            ctypes.c_void_p(stream),
        )
        if rc != 0:
            raise RoutedRailUnavailable(f"rr2_combine rc={rc}: {self._last_error()}")

    def combine_local(self, partial: torch.Tensor, out_local: torch.Tensor, n_pad: int):
        """Combine into this rank's owned rows only (out_local [cap, dim], local row i = TP-group row own_base + i);
        skips the intra-group y all-gather. Every rank of a TP group must use the same variant for a call.
        """
        if not hasattr(self._lib, "rr2_combine2"):
            raise RoutedRailUnavailable(
                "librail_routed2 has no rr2_combine2 (need ABI >= 6 for DSV4_TP_RSAG=1)"
            )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        rc = self._lib.rr2_combine2(
            ctypes.c_void_p(self._h),
            ctypes.c_void_p(partial.data_ptr()),
            ctypes.c_void_p(out_local.data_ptr()),
            1 if out_local.dtype == torch.float32 else 0,
            int(n_pad),
            1,
            ctypes.c_void_p(stream),
        )
        if rc != 0:
            raise RoutedRailUnavailable(f"rr2_combine2 rc={rc}: {self._last_error()}")

    def _start_stats_thread(self):
        interval = float(os.environ.get("DSV4_SM120_RAIL_STATS_S", "0"))
        if interval <= 0:
            return

        def loop():
            import time

            while getattr(self, "_h", None):
                s = _Rr2Stats()
                if (
                    self._lib.rr2_get_stats(ctypes.c_void_p(self._h), ctypes.byref(s))
                    == 0
                ):
                    _logger.warning(
                        "routed rail stats r%d: disp=%d comb=%d ep=%d host_err=%d dev_err=%s",
                        self.rank,
                        s.disp_posted,
                        s.comb_posted,
                        s.ep,
                        s.host_errors,
                        list(s.dev_err),
                    )
                time.sleep(interval)

        threading.Thread(
            target=loop, daemon=True, name=f"rr2-stats-{self.rank}"
        ).start()

    def _start_ts_dump_thread(self):
        """Dump the library's per-call device timestamp ring when the trigger file appears (benchmark-only).

        DSV4_SM120_RAIL_TS_DUMP=<trigger file>; each time its mtime changes this rank writes
        ``<trigger>.r<rank>.json`` = {"rank", "ts": [[12 globaltimer ns] per ring slot]}. Columns: 0 d1 start,
        1 d1 doorbell, 2 d1 end, 3 d2 start, 4 d2 rail rows in, 5 d2 end, 6 c1 start, 7 c1 end, 8 c2 local
        partials in, 9 c2 pre-reduce sent, 10 c2 remote row in, 11 c2 end, 12 c2 owner reduce + y all-gather
        writes done, 13 c2 TP mates' y slices in (library protocol >= 5; older builds return 12 columns).
        """
        trigger = os.environ.get("DSV4_SM120_RAIL_TS_DUMP", "")
        if not trigger:
            return
        n_ts = int(os.environ.get("DSV4_SM120_RAIL_TS_CALLS", "256"))

        def loop():
            import json
            import time

            seen = None
            while getattr(self, "_h", None):
                try:
                    m = os.stat(trigger).st_mtime
                except OSError:
                    m = None
                if m is not None and m != seen:
                    seen = m
                    nc = int(os.environ.get("DSV4_SM120_RAIL_TS_COLS", "14"))
                    buf = (ctypes.c_uint64 * (n_ts * nc))()
                    k = self._lib.rr2_fetch_ts(ctypes.c_void_p(self._h), buf, n_ts * nc)
                    rows = [
                        list(buf[i * nc : (i + 1) * nc]) for i in range(max(0, k) // nc)
                    ]
                    with open(f"{trigger}.r{self.rank}.json", "w") as f:
                        json.dump({"rank": self.rank, "tp": self.tp, "ts": rows}, f)
                time.sleep(0.5)

        threading.Thread(target=loop, daemon=True, name=f"rr2-ts-{self.rank}").start()

    def shutdown(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.rr2_destroy(ctypes.c_void_p(h))
            self._h = None


def _wrap_device_ptr(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    """Zero-copy uint8 tensor over library-owned device memory (lifetime = transport)."""

    class _Cai:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
        }

    with torch.cuda.device(device):
        return torch.as_tensor(_Cai(), device=device)


_TRANSPORTS: dict = {}
_LOCK = threading.Lock()


def get_routed2_transport(cfg, device: torch.device) -> RoutedRailTransport:
    key = device.index
    t = _TRANSPORTS.get(key)
    if t is None:
        with _LOCK:
            t = _TRANSPORTS.get(key)
            if t is None:
                from ._dispatch_quant_pack_triton import dispatch_payload_layout

                payload_bytes, packed, x_bytes, scale_bytes, w_bytes = (
                    dispatch_payload_layout(cfg.dim, cfg.n_activated_experts)
                )
                ids_offset = x_bytes + scale_bytes + w_bytes
                t = RoutedRailTransport(cfg, device, payload_bytes, ids_offset)
                _TRANSPORTS[key] = t
    return t


def _shutdown_all():
    for t in list(_TRANSPORTS.values()):
        try:
            t.shutdown()
        except Exception:  # noqa: BLE001
            pass
    _TRANSPORTS.clear()


import atexit

atexit.register(_shutdown_all)
