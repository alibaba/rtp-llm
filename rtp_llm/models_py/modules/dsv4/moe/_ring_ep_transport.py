"""Copy-engine ring all-gather dispatch and ring reduce-scatter combine.

Stage-local transport for ``fork_nccl_mxfp8`` on PCIe-only hosts (vLLM
cp4ep4pp2 ``pcie_dma`` protocol).  Rank r writes only into rank r+1's CUDA-IPC
slab with host-launched ``cuMemcpyAsync`` on one comm stream; arrival and
credit flags are 32-bit monotonic counters written by ``cuStreamWriteValue32``
and polled by ``cuStreamWaitValue32``, so there is no host sync per call.

Reduce-scatter over world W; block b = rows [b*n, (b+1)*n) of the partial and
is owned by rank b:
  step 0      send partial[r-1]                       -> stage[0] of r+1
  step k>0    wait stage[k-1]; acc = stage[k-1] + partial[r-1-k]; send -> stage[k]
  final       wait stage[W-2]; out = stage[W-2] + partial[r]   (FP32 out)
MXFP8 mode carries [n, H] e4m3 + [n, H/32] UE8M0 per hop; every hop
dequantizes, adds the BF16-rounded local FP32 partial and requantizes with the
same per-32 rule as ``flashinfer.mxfp8_quantize`` (bit-identical on BF16
input).  BF16 mode carries BF16 running sums.

Two slabs alternate by call parity so call e+1 can start while peers still
consume call e.  Every rank of the stage must issue the same sequence of
calls with the same row count.
"""

from __future__ import annotations

import ctypes
import logging

import torch
import triton
import triton.language as tl

SCALE_BLOCK = 32
_FLAG_STRIDE = 128
_CU_IPC_HANDLE_SIZE = 64
_MAX_CHUNKS = 8


def _align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _CU_IPC_HANDLE_SIZE)]


class _Cu:
    """Minimal CUDA driver bindings: IPC slabs, async copies, stream memory ops."""

    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        u64, sz, vp, u32, ui = (
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
        )
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), sz]
        lib.cuMemsetD8_v2.argtypes = [u64, ctypes.c_ubyte, sz]
        lib.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), u64]
        lib.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(u64), _IpcHandle, ui]
        lib.cuMemcpyAsync.argtypes = [u64, u64, sz, vp]
        lib.cuStreamWaitValue32_v2.argtypes = [vp, u64, u32, ui]
        lib.cuStreamWriteValue32_v2.argtypes = [vp, u64, u32, ui]
        lib.cuCtxSynchronize.argtypes = []
        self.lib = lib

    def ok(self, rc: int, what: str) -> None:
        if rc != 0:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(s))
            raise RuntimeError(
                f"{what} failed: {rc} {s.value.decode() if s.value else ''}"
            )

    def malloc(self, n: int) -> int:
        p = ctypes.c_uint64()
        self.ok(self.lib.cuMemAlloc_v2(ctypes.byref(p), n), "cuMemAlloc")
        self.ok(self.lib.cuMemsetD8_v2(p.value, 0, n), "cuMemsetD8")
        self.ok(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
        return int(p.value)

    def ipc_handle(self, ptr: int) -> bytes:
        h = _IpcHandle()
        self.ok(self.lib.cuIpcGetMemHandle(ctypes.byref(h), ptr), "cuIpcGetMemHandle")
        return bytes(h)

    def ipc_open(self, handle: bytes) -> int:
        h = _IpcHandle.from_buffer_copy(handle)
        p = ctypes.c_uint64()
        # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        self.ok(
            self.lib.cuIpcOpenMemHandle_v2(ctypes.byref(p), h, 1), "cuIpcOpenMemHandle"
        )
        return int(p.value)

    def copy(self, dst: int, src: int, n: int, stream: int) -> None:
        self.ok(self.lib.cuMemcpyAsync(dst, src, n, stream), "cuMemcpyAsync")

    def write(self, stream: int, ptr: int, value: int) -> None:
        # flags=0: the write is ordered after all prior work on the stream.
        self.ok(
            self.lib.cuStreamWriteValue32_v2(stream, ptr, value, 0),
            "cuStreamWriteValue32",
        )

    def wait(self, stream: int, ptr: int, value: int) -> None:
        # flags=0: CU_STREAM_WAIT_VALUE_GEQ
        self.ok(
            self.lib.cuStreamWaitValue32_v2(stream, ptr, value, 0),
            "cuStreamWaitValue32",
        )


class _CAI:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _view(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(_CAI(ptr, nbytes), device=device)
    assert t.data_ptr() == ptr and t.numel() == nbytes
    return t


@triton.jit
def _mx_quant_store(v, q_ptr, s_ptr, row, cols, blk, H: tl.constexpr, SB: tl.constexpr):
    """Per-32 MXFP8 quantization matching TRT-LLM's cvt_warp_fp16_to_mxfp8:
    SF = amax * rcp.approx(448); e8m0 = round-toward-+inf(SF) (exponent + 1 if any
    mantissa bit is set, saturating at 254); x_q = fp8(x * rcp.approx(2^(e-127)))."""
    v = v.to(tl.float32)
    amax = tl.max(tl.abs(v), axis=1).to(tl.float32)
    rcp448 = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [tl.full(amax.shape, 448.0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sf = amax * rcp448
    bits = sf.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    byte = tl.where(mant != 0, exp + 1, exp)
    byte = tl.where(exp == 0, tl.where(mant != 0, 1, 0), byte)
    byte = tl.minimum(byte, 254)
    sfq = (byte << 23).to(tl.float32, bitcast=True)
    inv = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [sfq],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    inv = tl.where(byte != 0, inv, 0.0)
    q = (v * inv[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * H + cols, q.to(tl.uint8, bitcast=True))
    tl.store(s_ptr + row * (H // SB) + blk, byte.to(tl.uint8))


@triton.jit
def _partial_quant_kernel(
    p_ptr, q_ptr, s_ptr, H: tl.constexpr, SB: tl.constexpr, BLOCK_G: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    blk = g * BLOCK_G + tl.arange(0, BLOCK_G)
    cols = blk[:, None] * SB + tl.arange(0, SB)[None, :]
    v = tl.load(p_ptr + row * H + cols).to(tl.bfloat16).to(tl.float32)
    _mx_quant_store(v, q_ptr, s_ptr, row, cols, blk, H, SB)


@triton.jit
def _hop_add_requant_kernel(
    inq_ptr,
    ins_ptr,
    p_ptr,
    outq_ptr,
    outs_ptr,
    H: tl.constexpr,
    SB: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    blk = g * BLOCK_G + tl.arange(0, BLOCK_G)
    cols = blk[:, None] * SB + tl.arange(0, SB)[None, :]
    qa = (
        tl.load(inq_ptr + row * H + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    sa = tl.exp2(tl.load(ins_ptr + row * (H // SB) + blk).to(tl.float32) - 127.0)
    v = qa * sa[:, None] + tl.load(p_ptr + row * H + cols).to(tl.bfloat16).to(
        tl.float32
    )
    _mx_quant_store(v, outq_ptr, outs_ptr, row, cols, blk, H, SB)


@triton.jit
def _final_add_kernel(
    inq_ptr,
    ins_ptr,
    p_ptr,
    out_ptr,
    H: tl.constexpr,
    SB: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    blk = g * BLOCK_G + tl.arange(0, BLOCK_G)
    cols = blk[:, None] * SB + tl.arange(0, SB)[None, :]
    qa = (
        tl.load(inq_ptr + row * H + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    sa = tl.exp2(tl.load(ins_ptr + row * (H // SB) + blk).to(tl.float32) - 127.0)
    v = qa * sa[:, None] + tl.load(p_ptr + row * H + cols).to(tl.bfloat16).to(
        tl.float32
    )
    tl.store(out_ptr + row * H + cols, v)


@triton.jit
def _bf16_hop_add_kernel(in_ptr, p_ptr, out_ptr, n_elems, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n_elems
    a = tl.load(in_ptr + i, mask=m).to(tl.float32)
    b = tl.load(p_ptr + i, mask=m).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + i, (a + b).to(tl.bfloat16), mask=m)


@triton.jit
def _bf16_final_add_kernel(in_ptr, p_ptr, out_ptr, n_elems, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n_elems
    a = tl.load(in_ptr + i, mask=m).to(tl.float32)
    b = tl.load(p_ptr + i, mask=m).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + i, a + b, mask=m)


def _grid_rows(n: int, hidden: int, block_g: int = 8):
    return (n, hidden // SCALE_BLOCK // block_g)


class RingEpTransport:
    """W-rank ring of CUDA-IPC slabs: AG for dispatch, MXFP8/BF16 RS for combine.

    Slab layout (two slabs, parity = call index & 1):
      flags : ARRIVE[step * MAXC + chunk] (written by r-1), CREDIT (written by r+1)
      ag    : W blocks of the dispatch packet            [W * n_max, ag_cols] u8
      stage : W-1 landing blocks of the reduce-scatter   [(W-1) * n_max, rs_row] u8
    """

    MAXC = _MAX_CHUNKS

    def __init__(
        self, group, n_max: int, hidden: int, ag_cols: int, device: torch.device
    ):
        dist = torch.distributed
        if hidden % (SCALE_BLOCK * 8):
            raise ValueError("ring transport requires hidden % 256 == 0")
        self.group = group
        self.device = device
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        w, r = self.world, self.rank
        self.nxt, self.prv = (r + 1) % w, (r - 1) % w
        self.n_max, self.H, self.ag_cols = int(n_max), int(hidden), int(ag_cols)
        self.cu = _Cu()
        self.rs_row = max(hidden * 2, hidden + hidden // SCALE_BLOCK)
        n_flags = w * self.MAXC + 2
        self.flags_bytes = _align(n_flags * _FLAG_STRIDE, 1 << 16)
        self.ag_off = self.flags_bytes
        self.stage_off = self.ag_off + _align(w * self.n_max * ag_cols, 1 << 16)
        self.slab_bytes = self.stage_off + _align(
            (w - 1) * self.n_max * self.rs_row, 1 << 16
        )
        self.local = [self.cu.malloc(self.slab_bytes) for _ in range(2)]
        handles = [self.cu.ipc_handle(p) for p in self.local]
        allh = [None] * w
        dist.all_gather_object(allh, handles, group=group)
        # Data flows r -> r+1; slab-reuse credits flow back r -> r-1.
        self.peer = [self.cu.ipc_open(h) for h in allh[self.nxt]]
        self.peer_prv = (
            self.peer
            if self.prv == self.nxt
            else [self.cu.ipc_open(h) for h in allh[self.prv]]
        )
        self.stream = torch.cuda.Stream(device=device)
        self.calls = 0
        self.ag_view = [
            _view(p + self.ag_off, w * self.n_max * ag_cols, device) for p in self.local
        ]
        self.stage_view = [
            _view(p + self.stage_off, (w - 1) * self.n_max * self.rs_row, device)
            for p in self.local
        ]
        self.q_scratch = torch.empty(
            (self.n_max, hidden), dtype=torch.uint8, device=device
        )
        self.s_scratch = torch.empty(
            (self.n_max, hidden // SCALE_BLOCK), dtype=torch.uint8, device=device
        )
        self.bf_scratch = torch.empty(
            (self.n_max, hidden), dtype=torch.bfloat16, device=device
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        logging.info(
            "[DSV4 MoE] ring EP transport ready: world=%d rank=%d n_max=%d hidden=%d "
            "ag_cols=%d slab=%.1f MiB x2",
            w,
            r,
            self.n_max,
            hidden,
            ag_cols,
            self.slab_bytes / 2**20,
        )

    def _arrive(self, base: int, step: int, chunk: int) -> int:
        return base + (step * self.MAXC + chunk) * _FLAG_STRIDE

    def _credit(self, base: int) -> int:
        return base + (self.world * self.MAXC) * _FLAG_STRIDE

    def _begin(self):
        """Open call e: grant r-1 the credit for my call e-1, wait r+1's credit for e-2.

        Credit value v lives in the parity(v) slab of the writer's predecessor:
        r-1 writes into my slabs, so r-1 needs my consumption credit, and I
        need r+1's before overwriting its parity(e) slab."""
        self.calls += 1
        e = self.calls
        par, prev_par = (e - 1) & 1, e & 1
        main = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(main)
        ss = self.stream.cuda_stream
        if e >= 2:
            # The comm stream has passed every consumer of call e-1's slab
            # (they were issued on main before this call), so r-1 may reuse it.
            self.cu.write(ss, self._credit(self.peer_prv[prev_par]), e - 1)
        if e > 2:
            self.cu.wait(ss, self._credit(self.local[par]), e - 2)
        return e, par, main, ss

    def allgather(self, packet: torch.Tensor) -> torch.Tensor:
        """packet [n, ag_cols] u8 -> [W*n, ag_cols] (block s = rank s's packet).

        The result views the slab and stays valid until this transport's
        second-next call; consume it on the current stream before then."""
        w, r, cu = self.world, self.rank, self.cu
        n, cols = packet.shape
        if cols != self.ag_cols or n > self.n_max or not packet.is_contiguous():
            raise ValueError(
                "ring allgather packet %s exceeds [<=%d, %d]"
                % (tuple(packet.shape), self.n_max, self.ag_cols)
            )
        nbytes = n * cols
        e, par, main, ss = self._begin()
        base, peer = self.local[par], self.peer[par]
        ag, pag = base + self.ag_off, peer + self.ag_off
        cu.copy(ag + r * nbytes, packet.data_ptr(), nbytes, ss)
        for k in range(w - 1):
            blk = (r - k) % w
            if k > 0:
                cu.wait(ss, self._arrive(base, k - 1, 0), e)
            cu.copy(pag + blk * nbytes, ag + blk * nbytes, nbytes, ss)
            cu.write(ss, self._arrive(peer, k, 0), e)
        cu.wait(ss, self._arrive(base, w - 2, 0), e)
        main.wait_stream(self.stream)
        packet.record_stream(self.stream)
        return self.ag_view[par][: w * nbytes].view(w * n, cols)

    def reduce_scatter(
        self, partial: torch.Tensor, n: int, mode: str = "mxfp8"
    ) -> torch.Tensor:
        """partial [W*n, H] fp32 (block b is owned by rank b) -> [n, H] fp32 on rank r."""
        if mode not in ("mxfp8", "bf16"):
            raise ValueError("ring reduce_scatter mode must be mxfp8 or bf16")
        w, r, cu, H = self.world, self.rank, self.cu, self.H
        if (
            partial.dtype != torch.float32
            or tuple(partial.shape) != (w * n, H)
            or not partial.is_contiguous()
            or n > self.n_max
        ):
            raise ValueError(
                "ring reduce_scatter expects contiguous fp32 [%d, %d], got %s %s"
                % (w * n, H, partial.dtype, tuple(partial.shape))
            )
        e, par, main, ss = self._begin()
        base, peer = self.local[par], self.peer[par]
        out = torch.empty((n, H), dtype=torch.float32, device=partial.device)
        S = H // SCALE_BLOCK
        blk_bytes = self.n_max * self.rs_row

        def stage_t(k):
            return self.stage_view[par][k * blk_bytes : (k + 1) * blk_bytes]

        with torch.cuda.stream(self.stream):
            for k in range(w - 1):
                blk = (r - 1 - k) % w
                p_blk = partial[blk * n : (blk + 1) * n]
                if k > 0:
                    cu.wait(ss, self._arrive(base, k - 1, 0), e)
                dst = peer + self.stage_off + k * blk_bytes
                if mode == "mxfp8":
                    q_src, s_src = self.q_scratch[:n], self.s_scratch[:n]
                    if k == 0:
                        _partial_quant_kernel[_grid_rows(n, H)](
                            p_blk, q_src, s_src, H=H, SB=SCALE_BLOCK, BLOCK_G=8
                        )
                    else:
                        v = stage_t(k - 1)
                        iq = v[: n * H].view(n, H)
                        isc = v[n * H : n * H + n * S].view(n, S)
                        _hop_add_requant_kernel[_grid_rows(n, H)](
                            iq, isc, p_blk, q_src, s_src, H=H, SB=SCALE_BLOCK, BLOCK_G=8
                        )
                    cu.copy(dst, q_src.data_ptr(), n * H, ss)
                    cu.copy(dst + n * H, s_src.data_ptr(), n * S, ss)
                else:
                    src = self.bf_scratch[:n]
                    if k == 0:
                        src.copy_(p_blk)
                    else:
                        ib = stage_t(k - 1)[: n * H * 2].view(torch.bfloat16).view(n, H)
                        ne = n * H
                        _bf16_hop_add_kernel[(triton.cdiv(ne, 2048),)](
                            ib, p_blk, src, ne, BLOCK=2048
                        )
                    cu.copy(dst, src.data_ptr(), n * H * 2, ss)
                cu.write(ss, self._arrive(peer, k, 0), e)
            p_own = partial[r * n : (r + 1) * n]
            cu.wait(ss, self._arrive(base, w - 2, 0), e)
            v = stage_t(w - 2)
            if mode == "mxfp8":
                iq = v[: n * H].view(n, H)
                isc = v[n * H : n * H + n * S].view(n, S)
                _final_add_kernel[_grid_rows(n, H)](
                    iq, isc, p_own, out, H=H, SB=SCALE_BLOCK, BLOCK_G=8
                )
            else:
                ne = n * H
                _bf16_final_add_kernel[(triton.cdiv(ne, 2048),)](
                    v[: n * H * 2].view(torch.bfloat16).view(n, H),
                    p_own,
                    out,
                    ne,
                    BLOCK=2048,
                )
        main.wait_stream(self.stream)
        partial.record_stream(self.stream)
        out.record_stream(self.stream)
        return out


_TRANSPORTS: dict = {}


def get_ring_transport(
    group, n_max: int, hidden: int, ag_cols: int, device: torch.device
):
    """One transport per (stage group, device, shape); creation is a stage collective."""
    key = (id(group), device.index, int(n_max), int(hidden), int(ag_cols))
    t = _TRANSPORTS.get(key)
    if t is None:
        t = RingEpTransport(group, n_max, hidden, ag_cols, device)
        _TRANSPORTS[key] = t
    return t


def mx_requant_selftest(device=None) -> tuple[bool, bool]:
    """The Triton requant must match flashinfer.mxfp8_quantize bit for bit on BF16 input."""
    from flashinfer import mxfp8_quantize

    device = device or torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.randn(1024, 4096, device=device, generator=g)
        * torch.logspace(-3, 2, 4096, device=device)
    ).to(torch.bfloat16)
    q_ref, s_ref = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
    q = torch.empty(1024, 4096, dtype=torch.uint8, device=device)
    s = torch.empty(1024, 128, dtype=torch.uint8, device=device)
    _partial_quant_kernel[_grid_rows(1024, 4096)](
        x.float(), q, s, H=4096, SB=32, BLOCK_G=8
    )
    torch.cuda.synchronize(device)
    q_ok = torch.equal(q, q_ref.view(torch.uint8).reshape(1024, 4096))
    s_ok = torch.equal(s, s_ref.view(torch.uint8).reshape(1024, 128))
    return q_ok, s_ok


__all__ = [
    "RingEpTransport",
    "get_ring_transport",
    "mx_requant_selftest",
    "SCALE_BLOCK",
]
