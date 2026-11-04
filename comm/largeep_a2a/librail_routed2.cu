// librail_routed2.cu - routed expert-parallel dispatch/combine for EP = 2 nodes x local_world GPUs over eRDMA,
// CUDA-graph capturable, TP-replicated decode tokens (TP group = tp consecutive ranks on one node).
//
// Per call (one MoE layer), every rank r (node N, local l, TP rank tr = r % tp) does:
//   dispatch  owner tr of its TP group owns rows [tr*cap, min(n_pad,(tr+1)*cap)), cap = ceil(n_pad/tp). Each owned
//             row goes once to every destination rank hosting >=1 of its topk experts (dest = id / experts_per_rank,
//             routing read from the int32 ids inside the payload row). Receiver layout: gathered[world src][cap]
//             payload rows, compacted per source in ascending owner-row order; rows [k_sd, cap) get ids = -1.
//             same-node destinations: direct CUDA-IPC P2P writes into the receiver's gathered slab.
//             other-node destinations: the owner sends the rows with >=1 remote destination ONCE over its rail
//             QP (RDMA WRITE, partner = same local id on the other node); the partner forwards them by P2P to every
//             destination on its node (positions recomputed from the ids in the payload, so both paths agree).
//   combine   receiver d pushes partial row (s, j) back: same-node owner -> owner's comb_dir[d][j];
//             other-node owner -> owner's rail partner (on d's node) comb_fwd[d][j]. The partner pre-reduces the
//             same-node contributions of every forwarded row (fp32, ascending rank order) -> bf16 -> one RDMA WRITE
//             back to the owner. The owner sums its same-node contributions (fp32, ascending rank order) + the
//             pre-reduced remote row, rounds to the y transport dtype and writes its y slice into its own y slab
//             and its tp-1 TP mates' y slabs (P2P all-gather); every rank then copies the full y[n_pad] to out.
// Arrival signalling:
//   * P2P legs: writer fences (__threadfence_system), the last block of the grid writes a monotone flag word
//     FV = t<<20 | n_pad<<8 | rows into the receiver's slab (GEQ spin on t; n_pad mismatch -> err[3]).
//   * RDMA legs, flag_mode 0: payload RDMA WRITE (signaled); the proxy posts an 8B inline flag WRITE
//     (FV) only after the payload's CQE (ACK-gated, proven on eRDMA).
//   * RDMA legs, flag_mode 2: one RDMA WRITE WITH IMM (imm = dir<<31 | (t&0x7ff)<<20 | n_pad<<8 | rows); the
//     RECEIVER's proxy takes the recv CQE as the arrival event and stores FV into a host-mapped pinned flag that
//     one GPU thread polls and mirrors into VRAM for the other blocks. One wire crossing per leg.
// Call sequence t = *ep + 1 is read by every kernel at execution time and the final combine block stores
// *ep = t, so one capture serves every replay. Every leg signals every call (rows may be 0), which makes all
// staging buffers single-buffered (each buffer's reuse at t+1 is causally after its last reader at t; see
// README "buffer reuse").
// Graph-capture contract: rr2_dispatch/rr2_combine only launch kernels into the caller's stream. The proxy
// thread touches only ibverbs and host memory (no CUDA calls). Spin waits time out after spin_ns into device
// error counters instead of hanging.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define RR2_LW_MAX 8
#define RR2_W_MAX 16
#define RR2_TP_MAX 8
#define RR2_CAP_MAX 64
#define RR2_TOPK_MAX 16
#define RR2_DESC_RING 64
#define RR2_RECV_RING 128
#define RR2_NTS 14
#define RR2_NTHR 256
#define RR2_NWARP (RR2_NTHR / 32)
#define RR2_HFLAG_OFF 256  // u64 index of the host flags inside the host page (own cache lines)
#define RR2_VERSION 4
#define RR2_ABI 6  // v6: rr2_combine2 (y_local: owner rows only, no TP y all-gather)

extern "C" {

typedef struct {
    int      gpu_index;
    int      rank;              // global rank = node*local_world + local
    int      world;             // 2*local_world
    int      local_world;       // GPUs per node (4 dev, 8 prod)
    int      tp;                // TP group size (consecutive ranks on one node); 4
    int      gid_index;         // 1
    int      n_pad_max;         // max rows per TP group per call (cap = ceil(n_pad/tp) <= 64)
    int      payload_bytes;     // dispatch row bytes (4176), multiple of 16
    int      dim;               // hidden (4096); partial row = dim bf16
    int      topk;              // 6
    int      experts_per_rank;  // 16
    int      ids_offset;        // byte offset of the int32 topk ids in a payload row (<0: payload_bytes - 4*topk)
    int      y_fp32;            // y transport dtype for the intra-group all-gather: 0 = bf16, 1 = fp32
    int      flag_mode;         // 0 = ACK-gated VRAM flag, 2 = WRITE_WITH_IMM + receiver host flag
    int      blocks;            // grid size of every kernel
    int      proxy_cpu;         // -1 = no pinning
    int      ts_calls;          // timestamp ring depth (calls)
    int      ib_port;           // 1
    uint64_t spin_ns;           // per-wait timeout (0 = unbounded)
    char     nic_name[64];
} rr2_config;

typedef struct {
    uint64_t slab_base;
    uint8_t  ipc[64];
    uint8_t  gid[16];
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    int32_t  rank;
    uint32_t proto;
    uint32_t _pad;
} rr2_wire;

typedef struct {
    uint64_t disp_posted, comb_posted, flags_posted, send_cqe, recv_imm, host_errors;
    uint64_t dev_err[4];  // 0 spin timeout, 1 bad flag epoch/rows, 2 remote row-count mismatch, 3 n_pad mismatch
    uint64_t last_d, last_c, ep;
} rr2_stats;

int         rr2_version(void);
void*       rr2_init(const rr2_config* cfg);
const char* rr2_last_error(void* h);
int         rr2_get_wire(void* h, rr2_wire* out);
int         rr2_connect(void* h, const rr2_wire* wires, int n);
void*       rr2_gathered_ptr(void* h);
void*       rr2_y_ptr(void* h);
int         rr2_dispatch(void* h, const void* payload, int n_pad, void* stream);
int         rr2_combine(void* h, const void* partial, void* out, int out_fp32, int n_pad, void* stream);
// y_local=1: out is [cap, dim] and receives only this rank's owned rows (local row i = global row own_base+i); the
// intra-TP-group y all-gather is skipped. Every rank of a TP group must pass the same y_local for a call.
int rr2_combine2(void* h, const void* partial, void* out, int out_fp32, int n_pad, int y_local, void* stream);
int rr2_abi(void);
int rr2_get_stats(void* h, rr2_stats* out);
int rr2_fetch_ts(void* h, uint64_t* out, uint64_t n);
int rr2_sleep_ns(uint64_t ns, void* stream);
// out[4] = RDMA payload rows sent {dispatch, combine}, RDMA payload bytes sent {dispatch, combine} (host counters)
int  rr2_get_traffic(void* h, uint64_t* out4);
void rr2_destroy(void* h);
}

// slab flag words (u64). F_DD/F_DF/F_C indexed by the writer's local id, F_Y by the writer's TP rank.
enum {
    F_DD   = 0,
    F_DF   = 8,
    F_C    = 16,
    F_Y    = 24,
    F_XD   = 32,
    F_XC   = 33,
    F_XDM  = 34,
    F_XCM  = 35,
    NFLAGS = 64
};
// slab meta words (u32)
enum {
    M_CNT     = 0,
    M_KR      = 16,
    M_OWNMASK = 32,
    M_FWDMASK = 32 + RR2_CAP_MAX,
    NMETA     = 32 + 2 * RR2_CAP_MAX
};

struct Args {
    uint64_t*           ep;
    unsigned*           cnt;
    unsigned long long* err;
    uint64_t*           ts;
    uint64_t*           flags;
    uint32_t*           meta;
    uint64_t*           db;  // host page (device pointer): [0] dispatch doorbell, [1] combine doorbell
    uint64_t*           desc_d;
    uint64_t*           desc_c;
    uint64_t*           hflag;  // [0] dispatch arrival, [1] combine arrival (flag_mode 2)
    char*               base;
    char*               peer[RR2_LW_MAX];  // slab base of every local rank (peer[local] == base)
    const char*         payload;
    const char*         partial;
    char*               out;
    uint64_t            o_flags, o_gath, o_dsend, o_dstage, o_cdir, o_cfwd, o_csend, o_cstage, o_y;
    uint64_t            spin_ns;
    uint32_t            ts_calls;
    int                 flag_mode;
    int                 me, node, local, lw, world, tp, tp_rank, grp_base, partner;
    int                 cap, n_pad, own_base, t_own;
    int                 P, ids_off, topk, epr;
    int                 dim, C, y_fp32, out_fp32, y_local;
};

// ---------------- device helpers ----------------

__device__ __forceinline__ uint64_t gtime() {
    uint64_t g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    return g;
}
__device__ __forceinline__ uint64_t vload(const uint64_t* p) {
    return *(const volatile uint64_t*)p;
}
__device__ __forceinline__ void vstore(uint64_t* p, uint64_t v) {
    *(volatile uint64_t*)p = v;
}

__device__ __noinline__ uint64_t spin_geq(const uint64_t* f, uint64_t want, uint64_t spin_ns, unsigned long long* err) {
    uint64_t v = vload(f);
    if (v < want) {
        const uint64_t t0 = gtime();
        for (;;) {
            v = vload(f);
            if (v >= want)
                break;
            if (spin_ns && gtime() - t0 > spin_ns) {
                atomicAdd(err, 1ull);
                break;
            }
        }
    }
    __threadfence();
    return v;
}

// flag word layout: FV(t, n_pad, rows) = t<<20 | n_pad<<8 | rows (rows <= 64, n_pad <= 512). Every leg carries the
// sender's n_pad so a call with mismatched n_pad across ranks is counted in err[3] instead of passing silently.
#define FV(t, np, rows) (((uint64_t)(t) << 20) | ((uint64_t)(np) << 8) | (uint64_t)(rows))

__device__ __forceinline__ int check_rows(const Args& A, uint64_t v, uint64_t t) {
    if ((v >> 20) != t) {
        atomicAdd(A.err + 1, 1ull);
        return 0;
    }
    if ((int)((v >> 8) & 0xfff) != A.n_pad)
        atomicAdd(A.err + 3, 1ull);
    int c = (int)(v & 0xff);
    if (c > A.cap) {
        atomicAdd(A.err + 1, 1ull);
        c = A.cap;
    }
    return c;
}

// RDMA arrival for direction dir. mode 2: the leader polls the host flag and mirrors it into VRAM, everybody
// else polls the mirror (keeps PCIe reads of host memory to one poller per GPU).
__device__ int wait_x(const Args& A, int dir, uint64_t t, bool leader) {
    const uint64_t want = t << 20;
    uint64_t       v;
    if (A.flag_mode == 2) {
        if (leader) {
            v = spin_geq(A.hflag + dir, want, A.spin_ns, A.err);
            vstore(A.flags + F_XDM + dir, v);
        } else {
            v = spin_geq(A.flags + F_XDM + dir, want, A.spin_ns, A.err);
        }
    } else {
        v = spin_geq(A.flags + F_XD + dir, want, A.spin_ns, A.err);
    }
    return check_rows(A, v, t);
}

__device__ __forceinline__ void ts_put(const Args& A, uint64_t t, int k) {
    if (A.ts)
        A.ts[((t - 1) % A.ts_calls) * RR2_NTS + k] = gtime();
}

__device__ __forceinline__ void warp_copy(char* dst, const char* src, int nbytes, int lane) {
    const int4* s = (const int4*)src;
    int4*       d = (int4*)dst;
    const int   n = nbytes >> 4;
    int         i = lane;
    for (; i + 96 < n; i += 128) {
        const int4 a = __ldcg(s + i), b = __ldcg(s + i + 32), c = __ldcg(s + i + 64), e = __ldcg(s + i + 96);
        d[i]      = a;
        d[i + 32] = b;
        d[i + 64] = c;
        d[i + 96] = e;
    }
    for (; i < n; i += 32)
        d[i] = __ldcg(s + i);
}

// every thread fences its own stores; the last block through the counter gets true (and resets the counter)
__device__ __forceinline__ bool epi_last(unsigned* cnt) {
    __shared__ int last;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(cnt, 1u) == gridDim.x - 1;
        if (last)
            atomicExch(cnt, 0u);
    }
    __syncthreads();
    if (last)
        __threadfence();
    return last;
}

struct Lay {
    uint32_t mask[RR2_CAP_MAX];
    uint8_t  pos[RR2_CAP_MAX][RR2_W_MAX + 1];  // [i][d] = slot of row i at dest d; [i][world] = remote-list slot
    int      cnt[RR2_W_MAX + 1];
};

__device__ __forceinline__ uint32_t ids_mask(const char* row_ids, int topk, int E, int epr) {
    const int* ids = (const int*)row_ids;
    uint32_t   m   = 0;
    for (int k = 0; k < topk; ++k) {
        const int e = __ldcg(ids + k);
        if (e >= 0 && e < E)
            m |= 1u << (e / epr);
    }
    return m;
}

__device__ void lay_scan(Lay& L, int n, int W, uint32_t rbits) {
    const int tid = threadIdx.x;
    if (tid <= W) {
        int c = 0;
        if (tid < W) {
            for (int i = 0; i < n; ++i) {
                L.pos[i][tid] = (uint8_t)c;
                c += (L.mask[i] >> tid) & 1u;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                L.pos[i][W] = (uint8_t)c;
                c += (L.mask[i] & rbits) ? 1 : 0;
            }
        }
        L.cnt[tid] = c;
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t node_bits(const Args& A) {
    return ((1u << A.lw) - 1u) << (A.node * A.lw);
}
__device__ __forceinline__ uint32_t remote_bits(const Args& A) {
    return ((1u << A.world) - 1u) & ~node_bits(A);
}

__device__ __forceinline__ void acc_add(float* acc, const int4 v) {
    const uint32_t u[4] = {(uint32_t)v.x, (uint32_t)v.y, (uint32_t)v.z, (uint32_t)v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        acc[2 * q] += __uint_as_float(u[q] << 16);
        acc[2 * q + 1] += __uint_as_float(u[q] & 0xffff0000u);
    }
}
__device__ __forceinline__ uint32_t bf2(float a, float b) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *(uint32_t*)&r;
}
__device__ __forceinline__ int4 pack_bf16(const float* a) {
    return make_int4((int)bf2(a[0], a[1]), (int)bf2(a[2], a[3]), (int)bf2(a[4], a[5]), (int)bf2(a[6], a[7]));
}
__device__ __forceinline__ void unpack_bf16(const int4 v, float* a) {
    const uint32_t u[4] = {(uint32_t)v.x, (uint32_t)v.y, (uint32_t)v.z, (uint32_t)v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        a[2 * q]     = __uint_as_float(u[q] << 16);
        a[2 * q + 1] = __uint_as_float(u[q] & 0xffff0000u);
    }
}

// store 8 consecutive y values (element offset e8*8 of row) as fp32 or bf16
__device__ __forceinline__ void store8(char* row, int e8, const float* a, bool fp32) {
    if (fp32) {
        float4* p = (float4*)row + 2 * e8;
        p[0]      = make_float4(a[0], a[1], a[2], a[3]);
        p[1]      = make_float4(a[4], a[5], a[6], a[7]);
    } else {
        ((int4*)row)[e8] = pack_bf16(a);
    }
}
__device__ __forceinline__ void load8(const char* row, int e8, float* a, bool fp32) {
    if (fp32) {
        const float4* p = (const float4*)row + 2 * e8;
        const float4  x = __ldcg(p), y = __ldcg(p + 1);
        a[0] = x.x;
        a[1] = x.y;
        a[2] = x.z;
        a[3] = x.w;
        a[4] = y.x;
        a[5] = y.y;
        a[6] = y.z;
        a[7] = y.w;
    } else {
        unpack_bf16(__ldcg((const int4*)row + e8), a);
    }
}

// ---------------- dispatch ----------------

__global__ void __launch_bounds__(RR2_NTHR) k_d1(Args A) {
    __shared__ Lay L;
    const int      tid = threadIdx.x, lane = tid & 31;
    const int      gw = blockIdx.x * RR2_NWARP + (tid >> 5), nw = gridDim.x * RR2_NWARP;
    const uint64_t t = vload(A.ep) + 1;
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 0);
    const int      W     = A.world;
    const uint32_t rbits = remote_bits(A);
    for (int i = tid; i < A.t_own; i += blockDim.x)
        L.mask[i] = ids_mask(A.payload + (size_t)(A.own_base + i) * A.P + A.ids_off, A.topk, W * A.epr, A.epr);
    __syncthreads();
    lay_scan(L, A.t_own, W, rbits);
    if (blockIdx.x == 0)
        for (int i = tid; i < A.t_own; i += blockDim.x)
            A.meta[M_OWNMASK + i] = L.mask[i];
    // phase 1: rows with >=1 other-node destination -> RDMA send staging (contiguous), ring the doorbell
    char* dsend = A.base + A.o_dsend;
    for (int i = gw; i < A.t_own; i += nw)
        if (L.mask[i] & rbits)
            warp_copy(dsend + (size_t)L.pos[i][W] * A.P, A.payload + (size_t)(A.own_base + i) * A.P, A.P, lane);
    if (epi_last(A.cnt + 0) && tid == 0) {
        vstore(A.desc_d + (t % RR2_DESC_RING), FV(0, A.n_pad, L.cnt[W]));
        __threadfence_system();
        vstore(A.db + 0, t);
        ts_put(A, t, 1);
    }
    // phase 2: same-node destinations (self included), compacted per (me -> d) in ascending row order
    const int nj = A.t_own * A.lw;
    for (int j = gw; j < nj; j += nw) {
        const int i = j / A.lw, l = j - i * A.lw, d = A.node * A.lw + l;
        if ((L.mask[i] >> d) & 1u)
            warp_copy(A.peer[l] + A.o_gath + ((size_t)A.me * A.cap + L.pos[i][d]) * A.P,
                      A.payload + (size_t)(A.own_base + i) * A.P,
                      A.P,
                      lane);
    }
    if (epi_last(A.cnt + 1)) {
        if (tid < A.lw)
            vstore((uint64_t*)(A.peer[tid] + A.o_flags) + F_DD + A.local, FV(t, A.n_pad, L.cnt[A.node * A.lw + tid]));
        if (tid == 0)
            ts_put(A, t, 2);
    }
}

__global__ void __launch_bounds__(RR2_NTHR) k_d2(Args A) {
    __shared__ Lay L;
    __shared__ int s_kr;
    __shared__ int s_in[RR2_W_MAX];
    const int      tid = threadIdx.x, lane = tid & 31;
    const int      gw = blockIdx.x * RR2_NWARP + (tid >> 5), nw = gridDim.x * RR2_NWARP;
    const uint64_t t = vload(A.ep) + 1;
    const int      W = A.world;
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 3);
    if (tid == 0)
        s_kr = wait_x(A, 0, t, blockIdx.x == 0);
    __syncthreads();
    const int kr = s_kr;
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 4);
    // forward the rail partner's rows to every destination on my node (self included)
    const char*    stage = A.base + A.o_dstage;
    const uint32_t nb    = node_bits(A);
    for (int i = tid; i < kr; i += blockDim.x)
        L.mask[i] = ids_mask(stage + (size_t)i * A.P + A.ids_off, A.topk, W * A.epr, A.epr) & nb;
    __syncthreads();
    lay_scan(L, kr, W, 0);
    if (blockIdx.x == 0) {
        for (int i = tid; i < kr; i += blockDim.x)
            A.meta[M_FWDMASK + i] = L.mask[i];
        if (tid == 0)
            A.meta[M_KR] = (uint32_t)kr;
    }
    const int nj = kr * A.lw;
    for (int j = gw; j < nj; j += nw) {
        const int r = j / A.lw, l = j - r * A.lw, d = A.node * A.lw + l;
        if ((L.mask[r] >> d) & 1u)
            warp_copy(A.peer[l] + A.o_gath + ((size_t)A.partner * A.cap + L.pos[r][d]) * A.P,
                      stage + (size_t)r * A.P,
                      A.P,
                      lane);
    }
    if (epi_last(A.cnt + 2)) {
        if (tid < A.lw)
            vstore((uint64_t*)(A.peer[tid] + A.o_flags) + F_DF + A.local, FV(t, A.n_pad, L.cnt[A.node * A.lw + tid]));
    }
    if (blockIdx.x != 0)
        return;
    // block 0: wait for every inbound leg (direct from each same-node owner, forwarded by each same-node rail
    // partner of an other-node owner), record per-source row counts, invalidate the unfilled rows
    if (tid < 2 * A.lw) {
        const bool     fwd  = tid >= A.lw;
        const int      l    = fwd ? tid - A.lw : tid;
        const uint64_t v    = spin_geq(A.flags + (fwd ? F_DF : F_DD) + l, t << 20, A.spin_ns, A.err);
        const int      c    = check_rows(A, v, t);
        const int      src  = (fwd ? (1 - A.node) : A.node) * A.lw + l;
        s_in[src]           = c;
        A.meta[M_CNT + src] = (uint32_t)c;
    }
    __syncthreads();
    const int ntot = W * A.cap * A.topk;
    for (int x = tid; x < ntot; x += blockDim.x) {
        const int row = x / A.topk, k = x - row * A.topk;
        const int s = row / A.cap, j = row - s * A.cap;
        if (j >= s_in[s])
            *(int*)(A.base + A.o_gath + (size_t)row * A.P + A.ids_off + 4 * k) = -1;
    }
    if (tid == 0)
        ts_put(A, t, 5);
}

// ---------------- combine ----------------

__global__ void __launch_bounds__(RR2_NTHR) k_c1(Args A) {
    __shared__ int s_off[RR2_W_MAX + 1];
    const int      tid = threadIdx.x, lane = tid & 31;
    const int      gw = blockIdx.x * RR2_NWARP + (tid >> 5), nw = gridDim.x * RR2_NWARP;
    const uint64_t t = vload(A.ep) + 1;
    const int      W = A.world;
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 6);
    if (tid == 0) {
        int o = 0;
        for (int s = 0; s < W; ++s) {
            s_off[s] = o;
            if (s != A.me && s != A.partner)
                o += (int)A.meta[M_CNT + s];
        }
        s_off[W] = o;
    }
    __syncthreads();
    // partial rows for other owners: same-node owner -> comb_dir[me]; other-node owner -> its rail partner's
    // comb_fwd[me] (that partner is the same-node rank with the owner's local id). Rows of my own tokens and of
    // my rail partner's tokens are read in place by k_c2.
    const int nrow = s_off[W];
    const int half = A.C >> 1;
    for (int j = gw; j < 2 * nrow; j += nw) {
        const int r = j >> 1, h = j & 1;
        int       s = 0;
        while (s_off[s + 1] <= r)
            ++s;
        const int  jj   = r - s_off[s];
        const int  ls   = s % A.lw;
        const bool same = (s / A.lw) == A.node;
        char* dst = A.peer[ls] + (same ? A.o_cdir : A.o_cfwd) + ((size_t)A.local * A.cap + jj) * A.C + (size_t)h * half;
        warp_copy(dst, A.partial + ((size_t)s * A.cap + jj) * A.C + (size_t)h * half, half, lane);
    }
    if (epi_last(A.cnt + 3)) {
        if (tid < A.lw && tid != A.local)
            vstore((uint64_t*)(A.peer[tid] + A.o_flags) + F_C + A.local, FV(t, A.n_pad, 0));
        if (tid == 0)
            ts_put(A, t, 7);
    }
}

__global__ void __launch_bounds__(RR2_NTHR) k_c2(Args A) {
    __shared__ Lay L;
    const int      tid = threadIdx.x, lane = tid & 31;
    const int      gw = blockIdx.x * RR2_NWARP + (tid >> 5), nw = gridDim.x * RR2_NWARP;
    const uint64_t t   = vload(A.ep) + 1;
    const int      W   = A.world;
    const int      NCH = A.C >> 11;  // 2KB (1024 bf16) chunks per row
    if (tid < A.lw && tid != A.local)
        check_rows(A, spin_geq(A.flags + F_C + tid, t << 20, A.spin_ns, A.err), t);
    __syncthreads();
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 8);
    // phase A (rail partner role): pre-reduce my node's contributions to the forwarded rows -> RDMA staging
    const int kr = (int)A.meta[M_KR];
    for (int i = tid; i < kr; i += blockDim.x)
        L.mask[i] = A.meta[M_FWDMASK + i];
    __syncthreads();
    lay_scan(L, kr, W, 0);
    char* csend = A.base + A.o_csend;
    for (int j = gw; j < kr * NCH; j += nw) {
        const int r = j / NCH, ch = j - r * NCH;
        float     acc[4][8];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int e = 0; e < 8; ++e)
                acc[q][e] = 0.f;
        const uint32_t m = L.mask[r];
        for (int l = 0; l < A.lw; ++l) {
            const int d = A.node * A.lw + l;
            if (!((m >> d) & 1u))
                continue;
            const char* src = (l == A.local) ? A.partial + ((size_t)A.partner * A.cap + L.pos[r][d]) * A.C :
                                               A.base + A.o_cfwd + ((size_t)l * A.cap + L.pos[r][d]) * A.C;
            const int4* s4  = (const int4*)src + ch * 128;
            int4        v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[q] = __ldcg(s4 + lane + 32 * q);
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc_add(acc[q], v[q]);
        }
        int4* o4 = (int4*)(csend + (size_t)r * A.C) + ch * 128;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            o4[lane + 32 * q] = pack_bf16(acc[q]);
    }
    if (epi_last(A.cnt + 4) && tid == 0) {
        vstore(A.desc_c + (t % RR2_DESC_RING), FV(0, A.n_pad, kr));
        __threadfence_system();
        vstore(A.db + 1, t);
        ts_put(A, t, 9);
    }
    // phase B (owner role): same-node contributions (overlapping the RDMA leg) + pre-reduced remote row
    const uint32_t rbits = remote_bits(A);
    for (int i = tid; i < A.t_own; i += blockDim.x)
        L.mask[i] = A.meta[M_OWNMASK + i];
    __syncthreads();
    lay_scan(L, A.t_own, W, rbits);
    // the combine arrival is consumed lazily by the first warp that needs a remote row; warp 0 of block 0 is
    // the leader (polls the host flag in flag_mode 2, mirrors it for the others) and always consumes it first,
    // even with no remote rows, so the partner's next send is always ordered after this call (buffer reuse).
    bool have_x = false;
    auto take_x = [&]() {
        if (lane == 0) {
            const int kx = wait_x(A, 1, t, gw == 0);
            if (gw == 0) {
                ts_put(A, t, 10);
                if (kx != L.cnt[W])
                    atomicAdd(A.err + 2, 1ull);
            }
        }
        __syncwarp();
        have_x = true;
    };
    if (gw == 0)
        take_x();
    const bool   yf = A.y_fp32 != 0, of = A.out_fp32 != 0;
    const size_t yrow = (size_t)A.dim * (yf ? 4 : 2), orow = (size_t)A.dim * (of ? 4 : 2);
    const char*  cstage = A.base + A.o_cstage;
    for (int j = gw; j < A.t_own * NCH; j += nw) {
        const int i = j / NCH, ch = j - i * NCH;
        float     acc[4][8];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int e = 0; e < 8; ++e)
                acc[q][e] = 0.f;
        const uint32_t m = L.mask[i];
        for (int l = 0; l < A.lw; ++l) {
            const int d = A.node * A.lw + l;
            if (!((m >> d) & 1u))
                continue;
            const char* src = (l == A.local) ? A.partial + ((size_t)A.me * A.cap + L.pos[i][d]) * A.C :
                                               A.base + A.o_cdir + ((size_t)l * A.cap + L.pos[i][d]) * A.C;
            const int4* s4  = (const int4*)src + ch * 128;
            int4        v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[q] = __ldcg(s4 + lane + 32 * q);
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc_add(acc[q], v[q]);
        }
        if (m & rbits) {
            if (!have_x)
                take_x();
            const int4* s4 = (const int4*)(cstage + (size_t)L.pos[i][W] * A.C) + ch * 128;
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc_add(acc[q], __ldcg(s4 + lane + 32 * q));
        }
        const int row = A.own_base + i;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int e8 = ch * 128 + lane + 32 * q;
            float*    a  = acc[q];
            if (!yf) {  // round once so every TP mate holds bit-identical y
                float r8[8];
                unpack_bf16(pack_bf16(a), r8);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    a[e] = r8[e];
            }
            if (A.y_local) {
                store8(A.out + (size_t)i * orow, e8, a, of);
            } else {
                for (int mt = 0; mt < A.tp; ++mt)
                    store8(A.peer[A.grp_base + mt] + A.o_y + (size_t)row * yrow, e8, a, yf);
                if (A.out)
                    store8(A.out + (size_t)row * orow, e8, a, of);
            }
        }
    }
    if (epi_last(A.cnt + 5)) {
        if (!A.y_local && tid < A.tp && tid != A.tp_rank)
            vstore((uint64_t*)(A.peer[A.grp_base + tid] + A.o_flags) + F_Y + A.tp_rank, FV(t, A.n_pad, 0));
        if (tid == 0)
            ts_put(A, t, 12);  // owner reduce + y P2P all-gather writes done (last block)
    }
    // phase C: wait for the TP mates' y slices; copy them out
    if (!A.y_local && tid < A.tp && tid != A.tp_rank)
        check_rows(A, spin_geq(A.flags + F_Y + tid, t << 20, A.spin_ns, A.err), t);
    __syncthreads();
    if (blockIdx.x == 0 && tid == 0)
        ts_put(A, t, 13);  // all TP mates' y slices arrived (block 0)
    if (A.out && !A.y_local) {
        const int   nrest = A.n_pad - A.t_own;
        const char* ybuf  = A.base + A.o_y;
        for (int j = gw; j < nrest * NCH; j += nw) {
            const int idx = j / NCH, ch = j - idx * NCH;
            const int row = idx < A.own_base ? idx : idx + A.t_own;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int e8 = ch * 128 + lane + 32 * q;
                float     a[8];
                load8(ybuf + (size_t)row * yrow, e8, a, yf);
                store8(A.out + (size_t)row * orow, e8, a, of);
            }
        }
    }
    if (epi_last(A.cnt + 6) && tid == 0) {
        ts_put(A, t, 11);
        vstore(A.ep, t);
    }
}

__global__ void k_sleep(uint64_t ns) {
    const uint64_t t0 = gtime();
    while (gtime() - t0 < ns) {}
}

// ---------------- host side ----------------

namespace {

struct Rr2 {
    rr2_config cfg{};
    char       errmsg[256]{};
    int        me = 0, node = 0, local = 0, lw = 0, W = 0, tp = 0, tp_rank = 0, grp_base = 0, partner = 0;
    int        cap_max = 0, P = 0, C = 0, ids_off = 0, ysz = 2;
    char*      base    = nullptr;
    size_t     bytes   = 0;
    size_t     o_flags = 0, o_ep = 0, o_cnt = 0, o_err = 0, o_meta = 0, o_ts = 0, o_gath = 0, o_dsend = 0, o_dstage = 0,
           o_cdir = 0, o_cfwd = 0, o_csend = 0, o_cstage = 0, o_y = 0;
    uint64_t*             hp          = nullptr;
    uint64_t*             hp_d        = nullptr;
    cudaStream_t          stat_stream = nullptr;  // private non-blocking stream for host-side counter reads
    uint64_t*             stat_h      = nullptr;  // pinned [8] staging for those reads
    char*                 peer[RR2_LW_MAX]{};
    bool                  peer_open[RR2_LW_MAX]{};
    uint8_t               ipc[64]{};
    ibv_context*          ctx = nullptr;
    ibv_pd*               pd  = nullptr;
    ibv_mr*               mr  = nullptr;
    ibv_cq*               cq  = nullptr;
    ibv_qp*               qp  = nullptr;
    ibv_gid               gid{};
    uint32_t              psn      = 0;
    uint64_t              r_dstage = 0, r_cstage = 0, r_fxd = 0, r_fxc = 0;
    uint32_t              rkey = 0;
    ibv_sge               recv_sge{};
    ibv_recv_wr           recv_wr{};
    pthread_t             th{};
    bool                  th_started = false;
    std::atomic<bool>     connected{false};
    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> disp_posted{0}, comb_posted{0}, flags_posted{0}, send_cqe{0}, recv_imm{0}, host_errors{0},
        last_d{0}, last_c{0};
    std::atomic<uint64_t> rows_sent[2]{}, bytes_sent[2]{};
    Args                  tmpl{};
};

void seterr(Rr2* r, const char* what) {
    snprintf(r->errmsg, sizeof r->errmsg, "%s (errno %d %s)", what, errno, strerror(errno));
}
void seterr_cuda(Rr2* r, const char* what, cudaError_t e) {
    snprintf(r->errmsg, sizeof r->errmsg, "%s: %s", what, cudaGetErrorString(e));
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

uint32_t proto_of(const rr2_config& c) {
    uint32_t  h   = 2166136261u;
    const int v[] = {RR2_VERSION,
                     c.world,
                     c.local_world,
                     c.tp,
                     c.n_pad_max,
                     c.payload_bytes,
                     c.dim,
                     c.topk,
                     c.experts_per_rank,
                     c.ids_offset,
                     c.y_fp32,
                     c.flag_mode};
    for (int x : v) {
        h ^= (uint32_t)x;
        h *= 16777619u;
    }
    return h;
}

// wr_id: bit63 flag-WR marker; payload WRs carry dir (bit62), rows (bits 0..15), seq (bits 16..61)
constexpr uint64_t WR_FLAG = 1ull << 63;
constexpr uint64_t WR_DIR  = 1ull << 62;

void* proxy_main(void* arg) {
    Rr2* r = (Rr2*)arg;
    if (r->cfg.proxy_cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(r->cfg.proxy_cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    while (!r->connected.load(std::memory_order_acquire) && !r->stop.load(std::memory_order_relaxed))
        usleep(100);
    volatile uint64_t* db           = r->hp;
    volatile uint64_t* desc[2]      = {r->hp + 8, r->hp + 8 + RR2_DESC_RING};
    uint64_t*          hflag        = r->hp + RR2_HFLAG_OFF;
    const bool         imm          = r->cfg.flag_mode == 2;
    const uint64_t     row_bytes[2] = {(uint64_t)r->P, (uint64_t)r->C};
    char*              src[2]       = {r->base + r->o_dsend, r->base + r->o_csend};
    const uint64_t     raddr[2]     = {r->r_dstage, r->r_cstage};
    const uint64_t     rflag[2]     = {r->r_fxd, r->r_fxc};
    uint64_t           next[2] = {1, 1}, rseq[2] = {0, 0};
    int                outstanding = 0;
    const int          sq_cap      = 200;
    uint64_t           flagval_store[256];  // inline data is copied at post time; kept for clarity
    unsigned           fv   = 0;
    auto               fail = [&](const char* m) {
        r->host_errors.fetch_add(1);
        snprintf(r->errmsg, sizeof r->errmsg, "proxy: %s (errno %d)", m, errno);
    };
    auto post_flag = [&](int dir, uint64_t v) {
        uint64_t* val = &flagval_store[(fv++) & 255];
        *val          = v;
        ibv_sge     s{(uint64_t)val, 8, 0};
        ibv_send_wr w{}, *bad = nullptr;
        w.wr_id               = WR_FLAG;
        w.sg_list             = &s;
        w.num_sge             = 1;
        w.opcode              = IBV_WR_RDMA_WRITE;
        w.send_flags          = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        w.wr.rdma.remote_addr = rflag[dir];
        w.wr.rdma.rkey        = r->rkey;
        if (ibv_post_send(r->qp, &w, &bad)) {
            fail("post flag");
            return;
        }
        ++outstanding;
        r->flags_posted.fetch_add(1, std::memory_order_relaxed);
    };
    auto pump = [&]() -> int {
        ibv_wc    wc[32];
        const int k = ibv_poll_cq(r->cq, 32, wc);
        if (k < 0) {
            fail("poll cq");
            return 0;
        }
        for (int z = 0; z < k; ++z) {
            const ibv_wc& c = wc[z];
            if (c.status != IBV_WC_SUCCESS) {
                r->host_errors.fetch_add(1);
                snprintf(r->errmsg,
                         sizeof r->errmsg,
                         "wc error %s opcode %d wr_id %llx",
                         ibv_wc_status_str(c.status),
                         (int)c.opcode,
                         (unsigned long long)c.wr_id);
                if (!(c.opcode & IBV_WC_RECV))
                    --outstanding;
                continue;
            }
            if (c.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                const uint32_t im  = ntohl(c.imm_data);
                const int      dir = (int)(im >> 31);
                const uint64_t lo = (im >> 20) & 0x7ff, low20 = im & 0xfffff;
                const uint64_t seq = ++rseq[dir];
                if ((seq & 0x7ff) != lo)
                    fail("imm sequence mismatch");
                __atomic_store_n(&hflag[dir], (seq << 20) | low20, __ATOMIC_RELEASE);
                r->recv_imm.fetch_add(1, std::memory_order_relaxed);
                ibv_recv_wr* bad = nullptr;
                if (ibv_post_recv(r->qp, &r->recv_wr, &bad))
                    fail("post recv");
                continue;
            }
            --outstanding;
            r->send_cqe.fetch_add(1, std::memory_order_relaxed);
            if (c.wr_id & WR_FLAG)
                continue;
            if (!imm) {
                const int      dir = (c.wr_id & WR_DIR) ? 1 : 0;
                const uint64_t seq = (c.wr_id >> 20) & ((1ull << 42) - 1), low20 = c.wr_id & 0xfffff;
                post_flag(dir, (seq << 20) | low20);
            }
        }
        return k;
    };
    auto post_payload = [&](int dir, uint64_t seq, uint64_t low20) {  // low20 = n_pad<<8 | rows
        const uint64_t rows = low20 & 0xff;
        while (outstanding > sq_cap && !r->stop.load(std::memory_order_relaxed))
            pump();
        if (!imm && rows == 0) {
            post_flag(dir, (seq << 20) | low20);
            return;
        }
        // a 0-row leg in imm mode still writes 16 bytes (nobody reads the staging rows of a 0-row call)
        ibv_sge s{(uint64_t)src[dir], (uint32_t)(rows ? rows * row_bytes[dir] : 16), r->mr->lkey};
        r->rows_sent[dir].fetch_add(rows, std::memory_order_relaxed);
        r->bytes_sent[dir].fetch_add(rows * row_bytes[dir], std::memory_order_relaxed);
        ibv_send_wr w{}, *bad = nullptr;
        w.wr_id   = (dir ? WR_DIR : 0) | (seq << 20) | low20;
        w.sg_list = &s;
        w.num_sge = 1;
        w.opcode  = imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
        if (imm)
            w.imm_data = htonl((uint32_t)(((uint64_t)dir << 31) | ((seq & 0x7ff) << 20) | low20));
        w.send_flags          = IBV_SEND_SIGNALED;
        w.wr.rdma.remote_addr = raddr[dir];
        w.wr.rdma.rkey        = r->rkey;
        if (ibv_post_send(r->qp, &w, &bad)) {
            fail("post payload");
            return;
        }
        ++outstanding;
    };
    timespec last_work{};
    clock_gettime(CLOCK_MONOTONIC, &last_work);
    const char*  idle_env = getenv("RR2_PROXY_IDLE_MS");
    const double idle_s   = idle_env ? atof(idle_env) / 1e3 : 0.0;  // 0 = always spin (lowest latency)
    for (uint64_t spins = 0;; ++spins) {
        bool work = false;
        for (int dir = 0; dir < 2; ++dir) {
            while (db[dir] >= next[dir]) {
                const uint64_t seq   = next[dir]++;
                uint64_t       low20 = desc[dir][seq % RR2_DESC_RING] & 0xfffff;
                if ((low20 & 0xff) > (uint64_t)r->cap_max) {
                    fail("bad descriptor");
                    low20 &= ~0xffull;
                }
                post_payload(dir, seq, low20);
                (dir ? r->comb_posted : r->disp_posted).fetch_add(1, std::memory_order_relaxed);
                (dir ? r->last_c : r->last_d).store(seq, std::memory_order_relaxed);
                work = true;
            }
        }
        if (pump() > 0)
            work = true;
        if (r->stop.load(std::memory_order_relaxed))
            break;
        if (work) {
            clock_gettime(CLOCK_MONOTONIC, &last_work);
            continue;
        }
        if (idle_s > 0 && (spins & 1023) == 0) {
            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);
            const double idle = (now.tv_sec - last_work.tv_sec) + 1e-9 * (now.tv_nsec - last_work.tv_nsec);
            if (idle > idle_s)
                usleep(20);  // engine idle between requests: stop burning the core
        }
        cpu_relax();
    }
    for (int i = 0; i < 1000 && outstanding > 0; ++i) {
        pump();
        usleep(100);
    }
    return nullptr;
}

}  // namespace

extern "C" {

int rr2_version(void) {
    return RR2_VERSION;
}

void* rr2_init(const rr2_config* cfg) {
    if (!cfg)
        return nullptr;
    Rr2* r              = new Rr2();
    r->cfg              = *cfg;
    const rr2_config& c = r->cfg;
    if (c.local_world < 1 || c.local_world > RR2_LW_MAX || c.world != 2 * c.local_world || c.tp < 1 || c.tp > RR2_TP_MAX
        || c.local_world % c.tp || c.rank < 0 || c.rank >= c.world || c.n_pad_max < 1
        || (c.n_pad_max + c.tp - 1) / c.tp > RR2_CAP_MAX || c.payload_bytes <= 0 || (c.payload_bytes & 15) || c.dim <= 0
        || (c.dim % 1024) || c.topk < 1 || c.topk > RR2_TOPK_MAX || c.experts_per_rank < 1 || c.blocks < 1
        || c.blocks > 1024 || (c.flag_mode != 0 && c.flag_mode != 2) || c.ts_calls < 1) {
        snprintf(r->errmsg, sizeof r->errmsg, "invalid config");
        return r;
    }
    r->me       = c.rank;
    r->lw       = c.local_world;
    r->W        = c.world;
    r->node     = c.rank / c.local_world;
    r->local    = c.rank % c.local_world;
    r->tp       = c.tp;
    r->tp_rank  = c.rank % c.tp;
    r->grp_base = r->local - r->tp_rank;
    r->partner  = (1 - r->node) * r->lw + r->local;
    r->cap_max  = (c.n_pad_max + c.tp - 1) / c.tp;
    r->P        = c.payload_bytes;
    r->C        = c.dim * 2;
    r->ids_off  = c.ids_offset >= 0 ? c.ids_offset : c.payload_bytes - 4 * c.topk;
    if (r->ids_off < 0 || (r->ids_off & 3) || r->ids_off + 4 * c.topk > c.payload_bytes) {
        snprintf(r->errmsg, sizeof r->errmsg, "invalid ids_offset");
        return r;
    }
    r->ysz         = c.y_fp32 ? 4 : 2;
    cudaError_t ce = cudaSetDevice(c.gpu_index);
    if (ce != cudaSuccess) {
        seterr_cuda(r, "cudaSetDevice", ce);
        return r;
    }
    size_t off  = 0;
    auto   take = [&](size_t n) {
        size_t o = off;
        off += (n + 255) & ~size_t(255);
        return o;
    };
    const size_t capm = (size_t)r->cap_max;
    r->o_flags        = take(NFLAGS * 8);
    r->o_ep           = take(8);
    r->o_cnt          = take(16 * 4);
    r->o_err          = take(8 * 8);
    r->o_meta         = take(NMETA * 4);
    r->o_ts           = take((size_t)c.ts_calls * RR2_NTS * 8);
    r->o_gath         = take((size_t)r->W * capm * r->P);
    r->o_dsend        = take(capm * r->P);
    r->o_dstage       = take(capm * r->P);
    r->o_cdir         = take((size_t)r->lw * capm * r->C);
    r->o_cfwd         = take((size_t)r->lw * capm * r->C);
    r->o_csend        = take(capm * r->C);
    r->o_cstage       = take(capm * r->C);
    r->o_y            = take((size_t)c.tp * capm * c.dim * r->ysz);
    r->bytes          = (off + 65535) & ~size_t(65535);
    ce                = cudaMalloc(&r->base, r->bytes);
    if (ce != cudaSuccess) {
        seterr_cuda(r, "cudaMalloc slab", ce);
        return r;
    }
    cudaMemset(r->base, 0, r->bytes);
    {  // gathered rows start invalid
        int neg[RR2_TOPK_MAX];
        for (int k = 0; k < RR2_TOPK_MAX; ++k)
            neg[k] = -1;
        for (size_t row = 0; row < (size_t)r->W * capm; ++row)
            cudaMemcpy(r->base + r->o_gath + row * r->P + r->ids_off, neg, 4 * c.topk, cudaMemcpyHostToDevice);
    }
    unsigned one = 1;
    if (cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)r->base) != CUDA_SUCCESS) {
        snprintf(r->errmsg, sizeof r->errmsg, "cuPointerSetAttribute SYNC_MEMOPS failed");
        return r;
    }
    ce = cudaHostAlloc((void**)&r->hp, 4096, cudaHostAllocMapped);
    if (ce != cudaSuccess) {
        seterr_cuda(r, "cudaHostAlloc", ce);
        return r;
    }
    memset(r->hp, 0, 4096);
    cudaHostGetDevicePointer((void**)&r->hp_d, r->hp, 0);
    if (cudaHostAlloc((void**)&r->stat_h, 64, cudaHostAllocDefault) != cudaSuccess
        || cudaStreamCreateWithFlags(&r->stat_stream, cudaStreamNonBlocking) != cudaSuccess) {
        snprintf(r->errmsg, sizeof r->errmsg, "stats stream/staging alloc failed");
        return r;
    }
    ce = cudaIpcGetMemHandle((cudaIpcMemHandle_t*)r->ipc, r->base);
    if (ce != cudaSuccess) {
        seterr_cuda(r, "cudaIpcGetMemHandle", ce);
        return r;
    }
    int          ndev = 0;
    ibv_device** dl   = ibv_get_device_list(&ndev);
    for (int i = 0; i < ndev; ++i)
        if (!strncmp(c.nic_name, ibv_get_device_name(dl[i]), 64))
            r->ctx = ibv_open_device(dl[i]);
    if (dl)
        ibv_free_device_list(dl);
    if (!r->ctx) {
        seterr(r, "open rdma device");
        return r;
    }
    const int port = c.ib_port > 0 ? c.ib_port : 1;
    r->pd          = ibv_alloc_pd(r->ctx);
    if (!r->pd || ibv_query_gid(r->ctx, port, c.gid_index, &r->gid)) {
        seterr(r, "alloc pd / query gid");
        return r;
    }
    r->mr = ibv_reg_mr(r->pd, r->base, r->bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!r->mr) {
        seterr(r, "ibv_reg_mr on VRAM");
        return r;
    }
    r->cq = ibv_create_cq(r->ctx, 1024, nullptr, nullptr, 0);
    if (!r->cq) {
        seterr(r, "ibv_create_cq");
        return r;
    }
    ibv_qp_init_attr ia{};
    ia.send_cq             = r->cq;
    ia.recv_cq             = r->cq;
    ia.qp_type             = IBV_QPT_RC;
    ia.cap.max_send_wr     = 256;
    ia.cap.max_recv_wr     = RR2_RECV_RING + 8;
    ia.cap.max_send_sge    = 1;
    ia.cap.max_recv_sge    = 1;
    ia.cap.max_inline_data = 32;
    r->qp                  = ibv_create_qp(r->pd, &ia);
    if (!r->qp) {
        seterr(r, "ibv_create_qp");
        return r;
    }
    ibv_qp_attr a{};
    a.qp_state        = IBV_QPS_INIT;
    a.port_num        = port;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        seterr(r, "qp init");
        return r;
    }
    if (c.flag_mode == 2) {
        r->recv_sge        = ibv_sge{(uint64_t)r->base, 8, r->mr->lkey};
        r->recv_wr.wr_id   = 1;
        r->recv_wr.sg_list = &r->recv_sge;
        r->recv_wr.num_sge = 1;
        for (int i = 0; i < RR2_RECV_RING; ++i) {
            ibv_recv_wr* bad = nullptr;
            if (ibv_post_recv(r->qp, &r->recv_wr, &bad)) {
                seterr(r, "ibv_post_recv ring");
                return r;
            }
        }
    }
    r->psn = ((uint32_t)getpid() * 7919u + (uint32_t)c.rank * 104729u) & 0xffffff;
    // kernel argument template
    Args& A = r->tmpl;
    memset(&A, 0, sizeof A);
    A.ep     = (uint64_t*)(r->base + r->o_ep);
    A.cnt    = (unsigned*)(r->base + r->o_cnt);
    A.err    = (unsigned long long*)(r->base + r->o_err);
    A.ts     = (uint64_t*)(r->base + r->o_ts);
    A.flags  = (uint64_t*)(r->base + r->o_flags);
    A.meta   = (uint32_t*)(r->base + r->o_meta);
    A.db     = r->hp_d;
    A.desc_d = r->hp_d + 8;
    A.desc_c = r->hp_d + 8 + RR2_DESC_RING;
    A.hflag  = r->hp_d + RR2_HFLAG_OFF;
    A.base   = r->base;
    for (int l = 0; l < RR2_LW_MAX; ++l)
        A.peer[l] = r->base;
    A.o_flags   = r->o_flags;
    A.o_gath    = r->o_gath;
    A.o_dsend   = r->o_dsend;
    A.o_dstage  = r->o_dstage;
    A.o_cdir    = r->o_cdir;
    A.o_cfwd    = r->o_cfwd;
    A.o_csend   = r->o_csend;
    A.o_cstage  = r->o_cstage;
    A.o_y       = r->o_y;
    A.spin_ns   = c.spin_ns;
    A.ts_calls  = (uint32_t)c.ts_calls;
    A.flag_mode = c.flag_mode;
    A.me        = r->me;
    A.node      = r->node;
    A.local     = r->local;
    A.lw        = r->lw;
    A.world     = r->W;
    A.tp        = r->tp;
    A.tp_rank   = r->tp_rank;
    A.grp_base  = r->grp_base;
    A.partner   = r->partner;
    A.P         = r->P;
    A.ids_off   = r->ids_off;
    A.topk      = c.topk;
    A.epr       = c.experts_per_rank;
    A.dim       = c.dim;
    A.C         = r->C;
    A.y_fp32    = c.y_fp32;
    // force module load now (never inside a capture)
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, k_d1);
    cudaFuncGetAttributes(&fa, k_d2);
    cudaFuncGetAttributes(&fa, k_c1);
    cudaFuncGetAttributes(&fa, k_c2);
    cudaDeviceSynchronize();
    ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        seterr_cuda(r, "init", ce);
        return r;
    }
    if (pthread_create(&r->th, nullptr, proxy_main, r)) {
        seterr(r, "pthread_create");
        return r;
    }
    r->th_started = true;
    return r;
}

const char* rr2_last_error(void* h) {
    return h ? ((Rr2*)h)->errmsg : "null handle";
}

int rr2_get_wire(void* h, rr2_wire* out) {
    Rr2* r = (Rr2*)h;
    if (!r || !out || !r->qp)
        return -1;
    memset(out, 0, sizeof *out);
    out->slab_base = (uint64_t)r->base;
    memcpy(out->ipc, r->ipc, 64);
    memcpy(out->gid, r->gid.raw, 16);
    out->qpn   = r->qp->qp_num;
    out->psn   = r->psn;
    out->rkey  = r->mr->rkey;
    out->rank  = r->me;
    out->proto = proto_of(r->cfg);
    return 0;
}

int rr2_connect(void* h, const rr2_wire* wires, int n) {
    Rr2* r = (Rr2*)h;
    if (!r || !wires || n != r->W || !r->qp)
        return -1;
    const uint32_t pr = proto_of(r->cfg);
    for (int i = 0; i < n; ++i)
        if (wires[i].rank != i || wires[i].proto != pr) {
            snprintf(r->errmsg,
                     sizeof r->errmsg,
                     "wire %d mismatch (rank %d proto %08x vs mine %08x)",
                     i,
                     wires[i].rank,
                     wires[i].proto,
                     pr);
            return -2;
        }
    cudaSetDevice(r->cfg.gpu_index);
    for (int l = 0; l < r->lw; ++l) {
        if (l == r->local) {
            r->peer[l] = r->base;
            continue;
        }
        if (r->peer_open[l])
            continue;
        const rr2_wire&    w = wires[r->node * r->lw + l];
        cudaIpcMemHandle_t hd;
        memcpy(&hd, w.ipc, 64);
        cudaError_t e = cudaIpcOpenMemHandle((void**)&r->peer[l], hd, cudaIpcMemLazyEnablePeerAccess);
        if (e != cudaSuccess) {
            seterr_cuda(r, "cudaIpcOpenMemHandle", e);
            return -3;
        }
        r->peer_open[l] = true;
    }
    for (int l = 0; l < r->lw; ++l)
        r->tmpl.peer[l] = r->peer[l];
    const rr2_wire& p = wires[r->partner];
    r->r_dstage       = p.slab_base + r->o_dstage;
    r->r_cstage       = p.slab_base + r->o_cstage;
    r->r_fxd          = p.slab_base + r->o_flags + F_XD * 8;
    r->r_fxc          = p.slab_base + r->o_flags + F_XC * 8;
    r->rkey           = p.rkey;
    const int   port  = r->cfg.ib_port > 0 ? r->cfg.ib_port : 1;
    ibv_qp_attr a{};
    a.qp_state           = IBV_QPS_RTR;
    a.path_mtu           = IBV_MTU_1024;
    a.dest_qp_num        = p.qpn;
    a.rq_psn             = p.psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer      = 12;
    a.ah_attr.is_global  = 1;
    a.ah_attr.port_num   = port;
    memcpy(a.ah_attr.grh.dgid.raw, p.gid, 16);
    a.ah_attr.grh.sgid_index = r->cfg.gid_index;
    a.ah_attr.grh.hop_limit  = 64;
    if (ibv_modify_qp(r->qp,
                      &a,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                          | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        seterr(r, "qp rtr");
        return -4;
    }
    a               = {};
    a.qp_state      = IBV_QPS_RTS;
    a.timeout       = 14;
    a.retry_cnt     = 7;
    a.rnr_retry     = 7;
    a.sq_psn        = r->psn;
    a.max_rd_atomic = 1;
    if (ibv_modify_qp(r->qp,
                      &a,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN
                          | IBV_QP_MAX_QP_RD_ATOMIC)) {
        seterr(r, "qp rts");
        return -4;
    }
    r->connected.store(true, std::memory_order_release);
    return 0;
}

void* rr2_gathered_ptr(void* h) {
    return h ? ((Rr2*)h)->base + ((Rr2*)h)->o_gath : nullptr;
}
void* rr2_y_ptr(void* h) {
    return h ? ((Rr2*)h)->base + ((Rr2*)h)->o_y : nullptr;
}

static bool geom(Rr2* r, Args& A, int n_pad) {
    if (n_pad < 1 || n_pad > r->cfg.n_pad_max || !r->connected.load(std::memory_order_acquire))
        return false;
    A.n_pad       = n_pad;
    A.cap         = (n_pad + r->tp - 1) / r->tp;
    A.own_base    = r->tp_rank * A.cap;
    const int end = n_pad < A.own_base + A.cap ? n_pad : A.own_base + A.cap;
    A.t_own       = end > A.own_base ? end - A.own_base : 0;
    if (A.t_own == 0)
        A.own_base = n_pad;
    return true;
}

int rr2_dispatch(void* h, const void* payload, int n_pad, void* stream) {
    Rr2* r = (Rr2*)h;
    if (!r || !payload)
        return -1;
    Args A = r->tmpl;
    if (!geom(r, A, n_pad)) {
        snprintf(r->errmsg, sizeof r->errmsg, "dispatch: bad n_pad %d or not connected", n_pad);
        return -2;
    }
    A.payload      = (const char*)payload;
    cudaStream_t s = (cudaStream_t)stream;
    k_d1<<<r->cfg.blocks, RR2_NTHR, 0, s>>>(A);
    k_d2<<<r->cfg.blocks, RR2_NTHR, 0, s>>>(A);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        seterr_cuda(r, "dispatch launch", e);
        return (int)e;
    }
    return 0;
}

int rr2_abi(void) {
    return RR2_ABI;
}

int rr2_combine(void* h, const void* partial, void* out, int out_fp32, int n_pad, void* stream) {
    return rr2_combine2(h, partial, out, out_fp32, n_pad, 0, stream);
}

int rr2_combine2(void* h, const void* partial, void* out, int out_fp32, int n_pad, int y_local, void* stream) {
    Rr2* r = (Rr2*)h;
    if (!r || !partial)
        return -1;
    if (y_local && !out) {
        snprintf(r->errmsg, sizeof r->errmsg, "combine: y_local requires an output buffer");
        return -3;
    }
    Args A = r->tmpl;
    if (!geom(r, A, n_pad)) {
        snprintf(r->errmsg, sizeof r->errmsg, "combine: bad n_pad %d or not connected", n_pad);
        return -2;
    }
    A.partial      = (const char*)partial;
    A.out          = (char*)out;
    A.out_fp32     = out_fp32 ? 1 : 0;
    A.y_local      = y_local ? 1 : 0;
    cudaStream_t s = (cudaStream_t)stream;
    k_c1<<<r->cfg.blocks, RR2_NTHR, 0, s>>>(A);
    k_c2<<<r->cfg.blocks, RR2_NTHR, 0, s>>>(A);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        seterr_cuda(r, "combine launch", e);
        return (int)e;
    }
    return 0;
}

int rr2_get_stats(void* h, rr2_stats* out) {
    Rr2* r = (Rr2*)h;
    if (!r || !out)
        return -1;
    memset(out, 0, sizeof *out);
    out->disp_posted  = r->disp_posted.load();
    out->comb_posted  = r->comb_posted.load();
    out->flags_posted = r->flags_posted.load();
    out->send_cqe     = r->send_cqe.load();
    out->recv_imm     = r->recv_imm.load();
    out->host_errors  = r->host_errors.load();
    out->last_d       = r->last_d.load();
    out->last_c       = r->last_c.load();
    // private non-blocking stream + pinned staging: never touches the legacy stream, so it is safe from a
    // monitoring thread while another thread captures or replays graphs
    if (r->base && r->stat_stream && r->stat_h) {
        cudaSetDevice(r->cfg.gpu_index);
        if (cudaMemcpyAsync(r->stat_h, r->base + r->o_err, 32, cudaMemcpyDeviceToHost, r->stat_stream) == cudaSuccess
            && cudaMemcpyAsync(r->stat_h + 4, r->base + r->o_ep, 8, cudaMemcpyDeviceToHost, r->stat_stream)
                   == cudaSuccess
            && cudaStreamSynchronize(r->stat_stream) == cudaSuccess) {
            for (int i = 0; i < 4; ++i)
                out->dev_err[i] = r->stat_h[i];
            out->ep = r->stat_h[4];
        }
    }
    return 0;
}

int rr2_fetch_ts(void* h, uint64_t* out, uint64_t n) {
    Rr2* r = (Rr2*)h;
    if (!r || !out)
        return -1;
    const uint64_t lim = (uint64_t)r->cfg.ts_calls * RR2_NTS;
    const uint64_t k   = n < lim ? n : lim;
    cudaSetDevice(r->cfg.gpu_index);
    if (cudaMemcpy(out, r->base + r->o_ts, k * 8, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    return (int)k;
}

int rr2_get_traffic(void* h, uint64_t* out4) {
    Rr2* r = (Rr2*)h;
    if (!r || !out4)
        return -1;
    for (int d = 0; d < 2; ++d) {
        out4[d]     = r->rows_sent[d].load();
        out4[2 + d] = r->bytes_sent[d].load();
    }
    return 0;
}

int rr2_sleep_ns(uint64_t ns, void* stream) {
    k_sleep<<<1, 1, 0, (cudaStream_t)stream>>>(ns);
    return (int)cudaGetLastError();
}

void rr2_destroy(void* h) {
    Rr2* r = (Rr2*)h;
    if (!r)
        return;
    r->stop.store(true);
    if (r->th_started)
        pthread_join(r->th, nullptr);
    if (r->qp)
        ibv_destroy_qp(r->qp);
    if (r->cq)
        ibv_destroy_cq(r->cq);
    if (r->mr)
        ibv_dereg_mr(r->mr);
    if (r->pd)
        ibv_dealloc_pd(r->pd);
    if (r->ctx)
        ibv_close_device(r->ctx);
    if (r->base) {
        cudaSetDevice(r->cfg.gpu_index);
        for (int l = 0; l < r->lw; ++l)
            if (r->peer_open[l])
                cudaIpcCloseMemHandle(r->peer[l]);
        cudaFree(r->base);
    }
    if (r->hp)
        cudaFreeHost(r->hp);
    if (r->stat_stream)
        cudaStreamDestroy(r->stat_stream);
    if (r->stat_h)
        cudaFreeHost(r->stat_h);
    delete r;
}

}  // extern "C"
