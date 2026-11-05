"""One-shot P2P all-reduce for TP=2 decode (CUDA-graph safe).

Enabled by RTP_LLM_P2P_AR=1 (default off). TP2-only, bf16 tensors up to
_p2p_ar_max_bytes. Uses CUDA IPC symmetric staging buffers and a per-block
parity double-buffered flag protocol; see kernel comments for the argument
why no buffer-free pre-wait is needed.

The handshake pairs the k-th call on both ranks via per-block call counters
and needs each rank's call k-1 to finish before its call k starts, so
consecutive TP collectives must be issued on the same stream on each rank.
"""

import logging
import os
from typing import Optional

import torch

_ENABLED = os.environ.get("RTP_LLM_P2P_AR", "0") == "1"
_AG_ENABLED = os.environ.get("RTP_LLM_P2P_AR_AG", "1") == "1"
_MAX_BYTES = int(os.environ.get("RTP_LLM_P2P_AR_MAX_BYTES", str(64 * 5120 * 2)))

_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include <cstring>
#include <string>

#define AR_MAX_BLOCKS 16
#define AR_HEAD_BYTES 4096

// Per rank: header = [0..64) per-block call counters, [64..192) ready[2][16];
// two stage buffers at base+AR_HEAD_BYTES + parity*stage_stride_bytes.
// Call k, block b: p=k&1, expected=k+1 (uint32).
//   1. write my input slice into peer's stage[p] (P2P)
//   2. threadfence_system; peer.ready[p][b] = expected (release)
//   3. spin until (int)(my.ready[p][b] - expected) >= 0 (acquire)
//   4. out = in + my_stage[p]
// No buffer-free pre-wait: being at call k means the peer reached k-1, so its
// read of stage[p] for call k-2 finished before its call k-1 completed.
// expected uses k+1 rather than (k>>1)+1 so a uint32 wrap cannot make the
// signed wait treat a stale ready value as already satisfied. Same-parity
// stores step by 2, and (int)((expected-2u)-expected) == -2 until the peer
// publishes expected.
__global__ void p2p_ar_kernel(const uint4* __restrict__ in,
                              uint4* __restrict__ out,
                              uint4* __restrict__ peer_stage0,
                              uint4* __restrict__ my_stage0,
                              unsigned int* __restrict__ my_head,
                              unsigned int* __restrict__ peer_head,
                              int nv, int nblocks, size_t stage_stride_v) {
    int b = blockIdx.x;
    if (b >= nblocks) return;
    __shared__ unsigned int s_call;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> c(*(my_head + b));
        s_call = c.fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();
    unsigned int k = s_call;
    unsigned int p = k & 1u;
    unsigned int expected = k + 1u;
    int chunk = (nv + nblocks - 1) / nblocks;
    int beg = b * chunk;
    int end = min(nv, beg + chunk);
    uint4* peer_stage = peer_stage0 + p * stage_stride_v;
    const uint4* my_stage = my_stage0 + p * stage_stride_v;
    unsigned int* peer_ready = peer_head + 64 + p * AR_MAX_BLOCKS;
    unsigned int* my_ready = my_head + 64 + p * AR_MAX_BLOCKS;
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
        peer_stage[i] = in[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> f(*(peer_ready + b));
        f.store(expected, cuda::memory_order_release);
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> f(*(my_ready + b));
        while ((int)(f.load(cuda::memory_order_acquire) - expected) < 0) {
        }
    }
    __syncthreads();
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
        uint4 a = in[i];
        uint4 c = my_stage[i];
        const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* cv = reinterpret_cast<const __nv_bfloat16*>(&c);
        uint4 r;
        __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
        for (int j = 0; j < 8; j++) rv[j] = __hadd(av[j], cv[j]);
        out[i] = r;
    }
}

// One-shot all-gather for TP=2, same parity double-buffered protocol as the AR
// above but on a dedicated allocation (independent counters/flags/stages).
// out is [2*nv] uint4s: rank r's input goes to out[r*nv..], the peer's input
// (read back from my stage) goes to out[(1-r)*nv..]. Same no-pre-wait argument.
__global__ void p2p_ag_kernel(const uint4* __restrict__ in,
                              uint4* __restrict__ out,
                              uint4* __restrict__ peer_stage0,
                              uint4* __restrict__ my_stage0,
                              unsigned int* __restrict__ my_head,
                              unsigned int* __restrict__ peer_head,
                              int nv, int nblocks, size_t stage_stride_v, int rank) {
    int b = blockIdx.x;
    if (b >= nblocks) return;
    __shared__ unsigned int s_call;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> c(*(my_head + b));
        s_call = c.fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();
    unsigned int k = s_call;
    unsigned int p = k & 1u;
    unsigned int expected = k + 1u;
    int chunk = (nv + nblocks - 1) / nblocks;
    int beg = b * chunk;
    int end = min(nv, beg + chunk);
    uint4* peer_stage = peer_stage0 + p * stage_stride_v;
    const uint4* my_stage = my_stage0 + p * stage_stride_v;
    unsigned int* peer_ready = peer_head + 64 + p * AR_MAX_BLOCKS;
    unsigned int* my_ready = my_head + 64 + p * AR_MAX_BLOCKS;
    uint4* my_out = out + (size_t)rank * nv;
    uint4* peer_out = out + (size_t)(1 - rank) * nv;
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
        uint4 v = in[i];
        peer_stage[i] = v;
        my_out[i] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> f(*(peer_ready + b));
        f.store(expected, cuda::memory_order_release);
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> f(*(my_ready + b));
        while ((int)(f.load(cuda::memory_order_acquire) - expected) < 0) {
        }
    }
    __syncthreads();
    for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
        peer_out[i] = my_stage[i];
    }
}

uintptr_t p2p_alloc(size_t stage_bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, AR_HEAD_BYTES + 2 * stage_bytes) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    cudaMemset(p, 0, AR_HEAD_BYTES);
    return (uintptr_t)p;
}

void p2p_free(uintptr_t base) {
    if (base && cudaFree((void*)base) != cudaSuccess) cudaGetLastError();
}

std::string p2p_get_handle(uintptr_t base) {
    cudaIpcMemHandle_t h;
    cudaError_t e = cudaIpcGetMemHandle(&h, (void*)base);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return std::string();
    }
    return std::string(h.reserved, sizeof(h.reserved));
}

uintptr_t p2p_open_handle(const std::string& bytes) {
    if (bytes.size() != 64) return 0;
    cudaIpcMemHandle_t h;
    memcpy(h.reserved, bytes.data(), 64);
    void* p = nullptr;
    if (cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (uintptr_t)p;
}

void p2p_close_handle(uintptr_t peer_base) {
    if (peer_base && cudaIpcCloseMemHandle((void*)peer_base) != cudaSuccess) cudaGetLastError();
}

void p2p_ar_launch(uintptr_t in, uintptr_t out, uintptr_t peer_base, uintptr_t my_base,
                   long n, int nblocks, uintptr_t stream, long stage_bytes) {
    uintptr_t my_stage0 = my_base + AR_HEAD_BYTES;
    uintptr_t peer_stage0 = peer_base + AR_HEAD_BYTES;
    int nv = (int)(n / 8);
    p2p_ar_kernel<<<nblocks, 512, 0, (cudaStream_t)stream>>>(
        (const uint4*)in, (uint4*)out, (uint4*)peer_stage0, (uint4*)my_stage0,
        (unsigned int*)my_base, (unsigned int*)peer_base, nv, nblocks, (size_t)stage_bytes / 16);
}

void p2p_ag_launch(uintptr_t in, uintptr_t out, uintptr_t peer_base, uintptr_t my_base,
                   long n, int nblocks, uintptr_t stream, long stage_bytes, long rank) {
    uintptr_t my_stage0 = my_base + AR_HEAD_BYTES;
    uintptr_t peer_stage0 = peer_base + AR_HEAD_BYTES;
    int nv = (int)(n / 8);
    p2p_ag_kernel<<<nblocks, 512, 0, (cudaStream_t)stream>>>(
        (const uint4*)in, (uint4*)out, (uint4*)peer_stage0, (uint4*)my_stage0,
        (unsigned int*)my_base, (unsigned int*)peer_base, nv, nblocks, (size_t)stage_bytes / 16,
        (int)rank);
}

int p2p_enable_peer(int peer) {
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    if (e != cudaSuccess) cudaGetLastError();
    return (int)e;
}

// Test helper: set counters so the next kernel invocation sees call next_k,
// and write ready flags consistent with next_k completed handshakes (including
// across a uint32 wrap). Production init leaves the header zeroed (next_k=0).
void p2p_seed_call_count(uintptr_t my_base, unsigned int next_k) {
    if (!my_base) return;
    unsigned int hdr[AR_HEAD_BYTES / sizeof(unsigned int)];
    memset(hdr, 0, sizeof(hdr));
    for (int b = 0; b < AR_MAX_BLOCKS; ++b) {
        hdr[b] = next_k;
    }
    unsigned int* ready0 = hdr + 64;
    unsigned int* ready1 = hdr + 64 + AR_MAX_BLOCKS;
    if (next_k == 1u) {
        for (int b = 0; b < AR_MAX_BLOCKS; ++b) ready0[b] = 1u;
    } else if (next_k > 1u) {
        if (next_k & 1u) {
            for (int b = 0; b < AR_MAX_BLOCKS; ++b) {
                ready0[b] = next_k;
                ready1[b] = next_k - 1u;
            }
        } else {
            for (int b = 0; b < AR_MAX_BLOCKS; ++b) {
                ready0[b] = next_k - 1u;
                ready1[b] = next_k;
            }
        }
    }
    cudaMemcpy((void*)my_base, hdr, AR_HEAD_BYTES, cudaMemcpyHostToDevice);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>
uintptr_t p2p_alloc(size_t stage_bytes);
void p2p_free(uintptr_t base);
std::string p2p_get_handle(uintptr_t base);
uintptr_t p2p_open_handle(const std::string& bytes);
void p2p_close_handle(uintptr_t peer_base);
void p2p_ar_launch(uintptr_t in, uintptr_t out, uintptr_t peer_base, uintptr_t my_base,
                   long n, int nblocks, uintptr_t stream, long stage_bytes);
void p2p_ag_launch(uintptr_t in, uintptr_t out, uintptr_t peer_base, uintptr_t my_base,
                   long n, int nblocks, uintptr_t stream, long stage_bytes, long rank);
int p2p_enable_peer(int peer);
void p2p_seed_call_count(uintptr_t my_base, unsigned int next_k);
py::bytes get_handle_bytes(uintptr_t base) { return py::bytes(p2p_get_handle(base)); }
uintptr_t open_handle_bytes(const py::bytes& b) { return p2p_open_handle(std::string(b)); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("p2p_alloc", &p2p_alloc);
    m.def("p2p_free", &p2p_free);
    m.def("p2p_get_handle", &get_handle_bytes);
    m.def("p2p_open_handle", &open_handle_bytes);
    m.def("p2p_close_handle", &p2p_close_handle);
    m.def("p2p_ar_launch", &p2p_ar_launch);
    m.def("p2p_ag_launch", &p2p_ag_launch);
    m.def("p2p_enable_peer", &p2p_enable_peer);
    m.def("p2p_seed_call_count", &p2p_seed_call_count);
}
"""


class _P2PAR:
    def __init__(self):
        self.ext = None
        self.ready = False
        self.base = 0
        self.peer_base = 0
        self.stage_bytes = _MAX_BYTES
        self.ag_base = 0
        self.peer_ag_base = 0
        self.rank = 0

    def init(self) -> bool:
        global _init_failed
        import torch.distributed as dist

        if self.ready:
            return True
        if not (_ENABLED and dist.is_available() and dist.is_initialized()):
            return False
        # Init runs TP collectives, so it is attempted once; failure is sticky.
        _init_failed = True
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != 2:
            logging.info("RTP_LLM_P2P_AR: world_size != 2, disabled")
            return False
        self.rank = rank
        dev = torch.cuda.current_device()
        # Local steps can fail on one rank only: catch that so every rank still
        # joins both exchanges below with fixed-size tensors, then decide
        # readiness from all ranks' status.
        h = b""
        try:
            from torch.utils.cpp_extension import load_inline

            build_dir = os.environ.get("RTP_LLM_P2P_AR_BUILD") or os.path.join(
                os.environ.get("TMPDIR", "/tmp"), f"rtp_p2p_ar_build_{os.getuid()}"
            )
            os.makedirs(build_dir, mode=0o700, exist_ok=True)
            self.ext = load_inline(
                name="rtp_p2p_ar",
                cpp_sources=[_CPP_SRC],
                cuda_sources=[_CUDA_SRC],
                functions=None,
                verbose=False,
                extra_cuda_cflags=["-O2", "-std=c++17"],
                build_directory=build_dir,
            )
            ng = torch.cuda.device_count()
            if ng < 2:
                raise RuntimeError(f"device_count={ng}")
            peer_dev = 1 - dev
            rc = self.ext.p2p_enable_peer(peer_dev)
            if rc != 0:
                raise RuntimeError(f"enable_peer rc={rc}")
            self.base = self.ext.p2p_alloc(self.stage_bytes)
            self.ag_base = self.ext.p2p_alloc(self.stage_bytes)
            if self.base and self.ag_base:
                h = self.ext.p2p_get_handle(self.base) + self.ext.p2p_get_handle(
                    self.ag_base
                )
        except Exception:
            logging.exception("RTP_LLM_P2P_AR: local init failed")
        ok = len(h) == 128
        status = None
        try:
            # exchange both IPC handles (AR base + AG base) in one all_gather
            ht = torch.frombuffer(
                bytearray(h if ok else bytes(128)), dtype=torch.uint8
            ).cuda()
            out = [
                torch.empty(128, dtype=torch.uint8, device=f"cuda:{dev}")
                for _ in range(2)
            ]
            dist.all_gather(out, ht)
            if ok:
                peer_handles = bytes(out[1 - rank].cpu().numpy().tobytes())
                self.peer_base = self.ext.p2p_open_handle(peer_handles[:64])
                self.peer_ag_base = self.ext.p2p_open_handle(peer_handles[64:])
                ok = bool(self.peer_base and self.peer_ag_base)
            mine = [int(ok), int(_AG_ENABLED), self.stage_bytes]
            st = torch.tensor(mine, dtype=torch.int64, device=f"cuda:{dev}")
            sts = [torch.empty_like(st) for _ in range(2)]
            dist.all_gather(sts, st)
            status = [s.tolist() for s in sts]
            self.ready = all(s[0] == 1 and s[1:] == mine[1:] for s in status)
        except Exception:
            logging.exception("RTP_LLM_P2P_AR: init exchange failed")
        if not self.ready:
            logging.warning(
                "RTP_LLM_P2P_AR: not ready on all ranks (per-rank [ok, ag, stage_bytes]=%s), "
                "falling back to NCCL",
                status,
            )
            self._release(sync=status is not None)
            return False
        _init_failed = False
        logging.info(
            "RTP_LLM_P2P_AR: initialized (stage=%d bytes x2, ag=%d bytes x2)",
            self.stage_bytes,
            self.stage_bytes,
        )
        return True

    def _release(self, sync: bool) -> None:
        import torch.distributed as dist

        if self.ext is not None:
            self.ext.p2p_close_handle(self.peer_base)
            self.ext.p2p_close_handle(self.peer_ag_base)
        if sync:
            # the peer must close its mappings of our buffers before we free them
            dist.barrier()
        if self.ext is not None:
            self.ext.p2p_free(self.base)
            self.ext.p2p_free(self.ag_base)
        self.base = self.ag_base = self.peer_base = self.peer_ag_base = 0

    def can_handle(self, t: torch.Tensor) -> bool:
        return (
            self.ready
            and t.dtype == torch.bfloat16
            and t.is_cuda
            and t.is_contiguous()
            and t.numel() % 8 == 0
            and t.numel() * 2 <= self.stage_bytes
            # 16B-aligned for uint4 access; the same op has the same alignment
            # on both ranks, so the decision stays rank-consistent.
            and t.data_ptr() % 16 == 0
        )

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        n = t.numel()
        nblocks = min(16, max(1, (n // 8 + 4095) // 4096))
        self.ext.p2p_ar_launch(
            t.data_ptr(),
            t.data_ptr(),
            self.peer_base,
            self.base,
            n,
            nblocks,
            torch.cuda.current_stream().cuda_stream,
            self.stage_bytes,
        )
        return t

    def seed_next_call(self, next_k: int) -> None:
        """Test helper: next kernel invocation on each buffer sees call next_k.

        Ready flags are written to match next_k completed handshakes, including
        across a uint32 wrap. Both the all-reduce and all-gather headers are
        seeded. Not used on the production path.
        """
        if self.ext is None or not self.base:
            return
        torch.cuda.synchronize()
        k = next_k & 0xFFFFFFFF
        self.ext.p2p_seed_call_count(self.base, k)
        if self.ag_base:
            self.ext.p2p_seed_call_count(self.ag_base, k)

    def all_gather(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        """TP2 one-shot all-gather; returns [2*t.shape[0]] + t.shape[1:] or None."""
        if not (self.can_handle(t) and _AG_ENABLED):
            return None
        n = t.numel()
        out = torch.empty(
            [2 * t.shape[0]] + list(t.shape[1:]), dtype=t.dtype, device=t.device
        )
        nblocks = min(16, max(1, (n // 8 + 4095) // 4096))
        self.ext.p2p_ag_launch(
            t.data_ptr(),
            out.data_ptr(),
            self.peer_ag_base,
            self.ag_base,
            n,
            nblocks,
            torch.cuda.current_stream().cuda_stream,
            self.stage_bytes,
            self.rank,
        )
        return out


_comm: Optional[_P2PAR] = None
_init_failed = False


def get_p2p_ar() -> Optional[_P2PAR]:
    global _comm
    if not _ENABLED or _init_failed:
        return None
    if _comm is None:
        # Never initialize during CUDA graph capture (allocs + a collective are
        # not capturable); fall back to NCCL for that call.
        if torch.cuda.is_current_stream_capturing():
            return None
        _comm = _P2PAR()
        if not _comm.init():
            _comm = None
    return _comm
