// MegaTopk: exact top-k across a DCP group (K3.1 DSA decode).
//
// Each rank holds the logits of its Page-RR share of every row. A row is owned either by one CTA
// or by a cluster of 2, 4, 8 or 16 CTAs that split its real length evenly and merge the coarse
// histogram through distributed shared memory; a span that fits one tile stays in registers,
// a longer one is streamed tile by tile from global memory on every pass. The row leader (the
// single CTA, or CTA 0 of the cluster) then selects the row's global top-k with two in-kernel
// exchange rounds:
//   X1  every rank broadcasts its 1024-bin coarse histogram from its local top-k start bin up,
//       so all ranks agree on the global threshold bin b* and on each rank's count inside it;
//   X2  every rank broadcasts the fp32 values that fall in b*; all ranks then find the exact
//       k-th key theta and split ties by (rank, local position).
// Each rank writes its slots above b* while gathering its b* values, then its b* elements above theta
// and theta-valued ones up to its quota. The deterministic instance writes each group in local
// position order and gives the quota to the earliest positions (block scans); the fast instance lets
// warps claim output ranges with atomics, so the order and which equal-valued elements fill the
// quota vary between runs while the selected values stay exact.
// Heavy ties (too many values in b*) fall back to distributed 8-bit radix rounds on the fp32 key.
// Exchanges use LL words {data0, epoch, data1, epoch} written and read as two 64-bit scalars: the
// PTX memory model makes each strong 64-bit access single-copy atomic (vector accesses only per
// element), so a reader validates data by its epoch and needs no flag or fence. Words are
// broadcast with multimem.st through the NVLS multicast address (unicast stores to every peer
// when it is unavailable).
#include "rtp_llm/models_py/bindings/cuda/kernels/mega_topk.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace cg = cooperative_groups;

namespace rtp_llm {
namespace {

constexpr int kMaxRanks      = 32;
constexpr int kMaxCtasPerRow = 16;
// 16-byte LL words per exchange slot. X1: header + 1024 u32 counts; X2: 512 values;
// radix fallback: 256 u32 counts.
constexpr int kSlotWords                = 513;
constexpr int kMaxBoundaryValuesPerRank = 512;                  // X2 values one rank may send
constexpr int kMaxBoundaryValues        = 4096;                 // X2 values of all ranks, held in shared memory
constexpr int kMinKvPageTokens          = 16;                   // smallest kernel KV page supported
constexpr int kFloat4PerThread          = 4;                    // a register tile: 16 elements per thread
constexpr int kElemsPerThread           = kFloat4PerThread * 4;
// X1 header flags: the rank sent its whole histogram (it holds fewer than k elements); its counts
// are u32, two per word (else u16, four per word).
constexpr uint32_t kX1WholeHistogram = 1, kX1U32Counts = 2;
// A round's first wait (a peer's X1 header) this long means the peer will never write: trap rather
// than hang. Each poll is a system-scope load of at least ~100 ns, so 2^28 polls take tens of seconds.
constexpr uint32_t kMaxSpins = 1u << 28;
#ifndef MEGA_TOPK_STALL_DIAG
#define MEGA_TOPK_STALL_DIAG 0
#endif

// CTAs of 1024 threads, one per SM, or of 512 threads, two per SM, for more rows than SMs: rows
// then progress side by side on an SM instead of one after another.
template<int kThreads>
struct CtaShape {
    static constexpr int kWarpsPerCta    = kThreads / 32;
    static constexpr int kCtasPerSm      = 1024 / kThreads;  // also __launch_bounds__: 64 registers per thread
    // fp16 ordered key, top 10 bits (topk_v3 extract_coarse_bin<10>): fewer bins put so many values
    // of long rows in b* that X2 overflows into the radix fallback. Thread t holds bins from the top.
    static constexpr int kCoarseBins     = 1024;
    static constexpr int kBinsPerThread  = kCoarseBins / kThreads;
    static constexpr int kTileElems      = kFloat4PerThread * 4 * kThreads;
    static constexpr int kMaxSpanPages   = kTileElems / kMinKvPageTokens + 1;  // spans need not be page aligned
};

struct Params {
    const float*   logits;
    int64_t        logits_row_stride;
    const int*     local_lens;
    const int*     positions;  // the row's global query position: its length over the group is + 1
    const int*     block_table;
    int64_t        block_table_row_stride;
    int            rows_per_request;
    int            kv_page_shift;  // log2 of the tokens per kernel KV page of block_table
    int*           out_slots;
    int64_t        out_row_stride;
    int*           out_counts;
    const int64_t* peer_slot_bases;      // [ranks] each rank's exchange workspace
    int64_t        multicast_slot_base;  // NVLS multicast address of the workspace, or 0
    int*           round_epochs;         // [num_slots] last exchange round of each slot
    int            num_slots;            // exchange slots per (parity, rank); one per row owner
    int            num_rows, k, rank, num_ranks;
    int            ctas_per_row;  // 1: single CTAs that stream rows longer than a tile
    int            global_page_shift;  // 0: write KV slots; else log2 of the Page-RR page tokens: write
                                       // global positions (local page j of rank r is global page j * ranks + r)
};

// ---------------------------------------------------------------- memory primitives
// An LL half {data, epoch} as one 64-bit scalar: data in the low word, as in the uint4 view.
__device__ __forceinline__ uint64_t pack_ll_half(uint32_t data, uint32_t epoch) {
    return static_cast<uint64_t>(data) | (static_cast<uint64_t>(epoch) << 32);
}
__device__ __forceinline__ void st_ll_word(uint4* addr, uint32_t data0, uint32_t data1, uint32_t epoch) {
    asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1,%2};" ::"l"(addr),
                 "l"(pack_ll_half(data0, epoch)),
                 "l"(pack_ll_half(data1, epoch))
                 : "memory");
}
__device__ __forceinline__ void multicast_st_ll_word(uint4* addr, uint32_t data0, uint32_t data1, uint32_t epoch) {
    // multimem.st has no vector form for 64-bit types.
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(pack_ll_half(data0, epoch))
                 : "memory");
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(reinterpret_cast<uint64_t*>(addr) + 1),
                 "l"(pack_ll_half(data1, epoch))
                 : "memory");
}
__device__ __forceinline__ uint4 ld_ll_word(const uint4* addr) {
    uint64_t lo, hi;
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0,%1}, [%2];" : "=l"(lo), "=l"(hi) : "l"(addr) : "memory");
    return make_uint4(static_cast<uint32_t>(lo),
                      static_cast<uint32_t>(lo >> 32),
                      static_cast<uint32_t>(hi),
                      static_cast<uint32_t>(hi >> 32));
}

// Counts in an LL word: four u16 (data0 low/high, data1 low/high) or two u32 (data0, data1).
__device__ __forceinline__ uint32_t count_at(uint4 word, int i, bool u32_counts) {
    if (u32_counts)
        return i == 0 ? word.x : word.z;
    const uint32_t data = (i < 2) ? word.x : word.z;
    return (data >> ((i & 1) * 16)) & 0xffffu;
}


// One row owner's exchange slot. Workspace layout per rank: [2 parity][num_slots][src rank][words];
// a round's parity is that of its epoch, so consecutive rounds never share words.
struct LLExchange {
    const Params&     p;
    int               slot;
    __device__ uint4* word_ptr(int64_t rank_base, uint32_t epoch, int src_rank) const {
        return reinterpret_cast<uint4*>(rank_base)
               + ((static_cast<int64_t>(epoch & 1) * p.num_slots + slot) * p.num_ranks + src_rank) * kSlotWords;
    }
    // Word `word` of this rank's part of the slot, on every rank (this one included).
    __device__ void broadcast(uint32_t epoch, int word, uint32_t data0, uint32_t data1) const {
        if (p.multicast_slot_base) {
            multicast_st_ll_word(word_ptr(p.multicast_slot_base, epoch, p.rank) + word, data0, data1, epoch);
        } else {
            for (int dst = 0; dst < p.num_ranks; ++dst)
                st_ll_word(word_ptr(p.peer_slot_bases[dst], epoch, p.rank) + word, data0, data1, epoch);
        }
    }
    // counts[0, 2) or counts[0, 4) as one word, in count_at's layout.
    __device__ void broadcast_counts(uint32_t epoch, int word, const uint32_t* counts, bool u32_counts) const {
        if (u32_counts)
            broadcast(epoch, word, counts[0], counts[1]);
        else
            broadcast(epoch, word, counts[0] | (counts[1] << 16), counts[2] | (counts[3] << 16));
    }
    // A word left unwritten for a whole 32-bit epoch cycle could still carry the epoch of a later
    // round that reads it, and pass as that round's data. So each round also rewrites one word
    // outside the words [0, written) it sends, cycling through the slot every kSlotWords rounds of
    // one parity: every word is rewritten long before the epoch wraps. Readers only read the words
    // a round sends, so the filler is never read as data.
    __device__ void refresh(uint32_t epoch, uint32_t written) const {
        const uint32_t word = (epoch >> 1) % kSlotWords;
        if (word >= written)
            broadcast(epoch, static_cast<int>(word), 0u, 0u);
    }
    // Word `word` of `src_rank`'s part, read locally once it carries `epoch`. A peer that never
    // starts a round (a missing call, a diverged epoch, an owner that cannot become resident) would
    // leave its waiters spinning forever, so the round's first wait (kBounded, the X1 header) traps
    // after kMaxSpins polls; one trap ends the whole kernel. Only that wait is bounded: a counter in
    // the waits of register-heavy loops spills in the 64-register 512-thread instance (~1.6 us per
    // call). Built with MEGA_TOPK_STALL_DIAG=1 the trap first prints what it waited for; that printf
    // call gives every instance a stack frame and costs the normal path 2-5 us, so it is off by
    // default.
    template<bool kBounded = false>
    __device__ uint4 wait_word(uint32_t epoch, int src_rank, int word) const {
        const uint4* addr = word_ptr(p.peer_slot_bases[p.rank], epoch, src_rank) + word;
        for (uint32_t spins = 0;; ++spins) {
            const uint4 value = ld_ll_word(addr);
            if (value.y == epoch && value.w == epoch)
                return value;
            if (kBounded && spins == kMaxSpins) {
#if MEGA_TOPK_STALL_DIAG
                printf("[mega_topk] stalled: rank %d block %d thread %d slot %d waits for rank %d word %d "
                       "epoch %u, sees epochs %u/%u (rows %d, owners %d, ctas/row %d)\n",
                       p.rank, blockIdx.x, threadIdx.x, slot, src_rank, word, epoch, value.y, value.w,
                       p.num_rows, gridDim.x / p.ctas_per_row, p.ctas_per_row);
#endif
                asm volatile("trap;");
            }
        }
    }
};

// ---------------------------------------------------------------- keys
template<int kCoarseBins>
__device__ __forceinline__ uint32_t coarse_bin(float x) {
    const uint16_t bits = __half_as_ushort(__float2half_rn(x));
    if ((bits & 0x7fffu) > 0x7c00u)
        return kCoarseBins - 1;
    const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u);
    return key >> (16 - __builtin_ctz(kCoarseBins));
}
// Total order of fp32 bit patterns; coarse_bin is non-decreasing in it.
__device__ __forceinline__ uint32_t ordered_key(uint32_t bits) {
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ int kv_slot(int page, int pos, int page_shift) {
    return (page << page_shift) | (pos & ((1 << page_shift) - 1));
}
// Global position of this rank's local position `pos` under Page-RR sharding.
__device__ __forceinline__ int global_position(int pos, const Params& p) {
    const int shift = p.global_page_shift;
    return (((pos >> shift) * p.num_ranks + p.rank) << shift) | (pos & ((1 << shift) - 1));
}

// Fast instance: consecutive indices for `count` items of this thread, claimed with one atomic per
// warp on `cursor` (shared or distributed shared memory). Returns this thread's first index.
// Whole warp only.
__device__ __forceinline__ uint32_t warp_claim(uint32_t count, uint32_t* cursor) {
    const int lane = threadIdx.x & 31;
    uint32_t  inc  = count;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o)
            inc += n;
    }
    const uint32_t total = __shfl_sync(0xffffffffu, inc, 31);
    uint32_t       base  = 0;
    if (lane == 31 && total)
        base = atomicAdd(cursor, total);
    return __shfl_sync(0xffffffffu, base, 31) + inc - count;
}

// ---------------------------------------------------------------- block scans
template<int kWarpsPerCta>
struct ScanSmem {
    uint32_t warp_u32[kWarpsPerCta + 1];
    uint4    warp_u32x4[kWarpsPerCta + 1];
};

// Warp 0 scans the warp totals; its lanes past the warp count carry zero.
template<int kWarpsPerCta>
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, ScanSmem<kWarpsPerCta>& s, uint32_t& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t  inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o)
            inc += n;
    }
    if (lane == 31)
        s.warp_u32[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const uint32_t w  = lane < kWarpsPerCta ? s.warp_u32[lane] : 0u;
        uint32_t       wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t n = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= o)
                wi += n;
        }
        if (lane < kWarpsPerCta)
            s.warp_u32[lane] = wi - w;
        if (lane == 31)
            s.warp_u32[kWarpsPerCta] = wi;
    }
    __syncthreads();
    const uint32_t r = s.warp_u32[warp] + inc - v;
    total            = s.warp_u32[kWarpsPerCta];
    __syncthreads();
    return r;
}

__device__ __forceinline__ uint4 add_u32x4(uint4 a, uint4 b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ uint4 shfl_up_u32x4(uint4 v, int o) {
    return make_uint4(__shfl_up_sync(0xffffffffu, v.x, o),
                      __shfl_up_sync(0xffffffffu, v.y, o),
                      __shfl_up_sync(0xffffffffu, v.z, o),
                      __shfl_up_sync(0xffffffffu, v.w, o));
}
// Scan of eight packed 16-bit counters at once (block totals stay < 65536).
template<int kWarpsPerCta>
__device__ __forceinline__ uint4 block_exclusive_scan_u16x8(uint4 v, ScanSmem<kWarpsPerCta>& s, uint4& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint4     inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint4 n = shfl_up_u32x4(inc, o);
        if (lane >= o)
            inc = add_u32x4(inc, n);
    }
    if (lane == 31)
        s.warp_u32x4[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const uint4 w  = lane < kWarpsPerCta ? s.warp_u32x4[lane] : make_uint4(0u, 0u, 0u, 0u);
        uint4       wi = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint4 n = shfl_up_u32x4(wi, o);
            if (lane >= o)
                wi = add_u32x4(wi, n);
        }
        if (lane < kWarpsPerCta)
            s.warp_u32x4[lane] = make_uint4(wi.x - w.x, wi.y - w.y, wi.z - w.z, wi.w - w.w);
        if (lane == 31)
            s.warp_u32x4[kWarpsPerCta] = wi;
    }
    __syncthreads();
    const uint4 base = s.warp_u32x4[warp];
    const uint4 r = make_uint4(base.x + inc.x - v.x, base.y + inc.y - v.y, base.z + inc.z - v.z, base.w + inc.w - v.w);
    total         = s.warp_u32x4[kWarpsPerCta];
    __syncthreads();
    return r;
}
__device__ __forceinline__ void u16x8_add(uint4& v, int j, uint32_t n) {
    const uint32_t a = n << ((j & 1) * 16);
    if ((j >> 1) == 0)
        v.x += a;
    else if ((j >> 1) == 1)
        v.y += a;
    else if ((j >> 1) == 2)
        v.z += a;
    else
        v.w += a;
}
__device__ __forceinline__ uint32_t u16x8_get(uint4 v, int j) {
    const uint32_t w = (j >> 1) == 0 ? v.x : (j >> 1) == 1 ? v.y : (j >> 1) == 2 ? v.z : v.w;
    return (w >> ((j & 1) * 16)) & 0xffffu;
}
// Set bits per float4 of this thread (4 mask bits each), packed for block_exclusive_scan_u16x8.
__device__ __forceinline__ uint4 set_bits_per_float4(uint32_t mask, int float4_per_thread) {
    uint4 c = make_uint4(0, 0, 0, 0);
    for (int j = 0; j < float4_per_thread; ++j)
        u16x8_add(c, j, __popc((mask >> (4 * j)) & 0xfu));
    return c;
}

template<int kThreads>
struct Smem {
    using Shape = CtaShape<kThreads>;
    uint32_t bin_counts[Shape::kCoarseBins];  // coarse histogram; 256-digit counts during radix rounds
    // X2: every rank's values in b*, grouped by rank; radix fallback: the global digit counts.
    uint32_t boundary_values[kMaxBoundaryValues];
    uint32_t boundary_pos[kMaxBoundaryValuesPerRank];  // local positions of this rank's values in b*
    uint32_t span_pages[Shape::kMaxSpanPages];  // KV page ids covering this CTA's span of a register-resident row
    ScanSmem<Shape::kWarpsPerCta> scan;
    // Row leader state, read by the other CTAs of a cluster through DSMEM.
    uint32_t x1_start_bin[kMaxRanks], x1_num_bins[kMaxRanks], x1_flags[kMaxRanks];
    uint32_t bstar_count[kMaxRanks];           // each rank's elements in b*
    uint32_t theta_count[kMaxRanks];           // each rank's elements equal to theta
    uint32_t boundary_offset[kMaxRanks + 1];   // each rank's first index in boundary_values
    // Each CTA's span: elements above b* and in b*; under heavy ties, b* elements above theta and at theta.
    uint32_t cta_above[kMaxCtasPerRow], cta_bstar[kMaxCtasPerRow], cta_gt[kMaxCtasPerRow], cta_eq[kMaxCtasPerRow];
    uint32_t own_start_bin, own_top_bin;       // this rank's X1 segment
    uint32_t picked_digit, count_above_digit;  // radix digit search result
    uint32_t still_needed_in_bstar, use_radix_fallback, bstar;
    // Radix fallback: the key range of the row's b* elements (min == max: every one is theta), and
    // whether a digit filled the remaining places exactly (the selection is then key >= prefix).
    uint32_t fallback_min_key, fallback_max_key, fallback_filled;
    // This CTA's own: its count in b*, its first X2 value index, and per-CTA prefixes and totals.
    uint32_t own_bstar_count, own_above_count, my_boundary_offset, prefix_a, prefix_b, total_a, total_b;
    uint32_t out_cursor, bstar_cursor, eq_cursor;  // fast instance: claimed output, b* list and tie slots
};

// Warp 0 only: the digit holding the `rank_from_top`-th largest key in counts[256], and the
// number of keys with larger digits. Lane l owns digits 255 - 8l - i.
template<typename SmemT>
__device__ __forceinline__ void find_digit_of_rank(const uint32_t* counts, uint32_t rank_from_top, SmemT& sm) {
    const int lane = threadIdx.x & 31;
    uint32_t  c[8], s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        c[i] = counts[255 - 8 * lane - i];
        s += c[i];
    }
    uint32_t inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t n = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o)
            inc += n;
    }
    uint32_t above = inc - s;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (above < rank_from_top && rank_from_top <= above + c[i]) {
            sm.picked_digit      = 255 - 8 * lane - i;
            sm.count_above_digit = above;
        }
        above += c[i];
    }
}

// Source rank of flattened word i (word_end holds cumulative word counts per rank) and that
// rank's first flattened word. Unrolled so word_end stays in registers instead of local memory.
__device__ __forceinline__ int src_rank_of_word(uint32_t i, const uint32_t (&word_end)[kMaxRanks], uint32_t& first) {
    int src = 0;
    first   = 0;
#pragma unroll
    for (int r = 0; r < kMaxRanks; ++r) {
        if (i >= word_end[r]) {
            src   = r + 1;
            first = word_end[r];
        }
    }
    return src;
}

// The next exchange round. Rounds never use epoch 0, which a zeroed workspace word carries, and
// keep alternating parity across the 32-bit wrap (0xffffffff, odd, is followed by 2).
__device__ __forceinline__ uint32_t next_round(uint32_t& round_epoch) {
    round_epoch = round_epoch == 0xffffffffu ? 2u : round_epoch + 1;
    return round_epoch;
}

// ---------------------------------------------------------------- mask scans and claims
// Block-wide position ranks of a mask's set bits: per float4 column j, the 16-bit counters
// first + j hold the bits before this thread and the block total.
struct MaskScan {
    uint4    exclusive, per_float4_total;
    uint32_t total;
    int      first;
};
__device__ __forceinline__ void sum_mask_scan_total(MaskScan& s) {
    s.total = 0;
    for (int j = 0; j < kFloat4PerThread; ++j)
        s.total += u16x8_get(s.per_float4_total, s.first + j);
}
template<int kWarpsPerCta>
__device__ __forceinline__ MaskScan scan_mask(uint32_t mask, ScanSmem<kWarpsPerCta>& scan) {
    MaskScan s;
    s.first     = 0;
    s.exclusive = block_exclusive_scan_u16x8(set_bits_per_float4(mask, kFloat4PerThread), scan, s.per_float4_total);
    sum_mask_scan_total(s);
    return s;
}
// Two masks in one scan: counters 0..3 and 4..7.
template<int kWarpsPerCta>
__device__ __forceinline__ void
scan_two_masks(uint32_t mask_a, uint32_t mask_b, ScanSmem<kWarpsPerCta>& scan, MaskScan& sa, MaskScan& sb) {
    uint4 c = set_bits_per_float4(mask_a, kFloat4PerThread);
    for (int j = 0; j < kFloat4PerThread; ++j)
        u16x8_add(c, kFloat4PerThread + j, __popc((mask_b >> (4 * j)) & 0xfu));
    sa.exclusive = sb.exclusive = block_exclusive_scan_u16x8(c, scan, sa.per_float4_total);
    sb.per_float4_total         = sa.per_float4_total;
    sa.first                    = 0;
    sb.first                    = kFloat4PerThread;
    sum_mask_scan_total(sa);
    sum_mask_scan_total(sb);
}
// Deterministic instance: calls emit(element i, index) for the set bits, indexed in position
// order from `before`.
template<typename Emit>
__device__ __forceinline__ void for_each_in_order(uint32_t mask, const MaskScan& s, uint32_t before, Emit&& emit) {
#pragma unroll
    for (int j = 0; j < kFloat4PerThread; ++j) {
        uint32_t r = before + u16x8_get(s.exclusive, s.first + j);
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (mask >> (4 * j + q) & 1)
                emit(4 * j + q, r++);
        before += u16x8_get(s.per_float4_total, s.first + j);
    }
}
// Fast instance: calls emit(element i, index) for the set bits, with indices claimed on `cursor`.
// Whole warp only. Unrolled over i: a runtime element index would push the tile to local memory.
template<typename Emit>
__device__ __forceinline__ void for_each_claimed(uint32_t mask, uint32_t* cursor, Emit&& emit) {
    uint32_t r = warp_claim(__popc(mask), cursor);
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
        if (mask >> i & 1)
            emit(i, r++);
}

// ---------------------------------------------------------------- the kernel
// One CTA of the persistent grid. Row owners, a single CTA or (kClusterRow) a cluster of
// p.ctas_per_row CTAs, take rows with a stride; in a cluster a row that fits one tile runs on CTA 0
// alone. A row goes through
//   histogram -> X1: b* -> write above b*, gather b* -> X2: theta -> write this rank's share of b*
// or, under heavy ties, radix rounds over the spans in place of X2. The single-CTA instance keeps
// every leader access in plain shared memory and every barrier a block barrier.
//
// run() owns this CTA's span of each row and the passes over it; the steps that only touch shared
// memory and the exchange (X1, X2, the radix digit exchange, per-CTA prefixes) are member functions.
// The span stays as locals and lambdas of run(): wrapping it in a struct measured 0.3-0.9 us slower
// per call (ptxas then keeps the row geometry out of uniform registers).
template<int kThreads, bool kClusterRow, bool kDeterministic>
struct MegaTopkCta {
    using SmemT                          = Smem<kThreads>;
    static constexpr int kWarpsPerCta    = CtaShape<kThreads>::kWarpsPerCta;
    static constexpr int kCoarseBins     = CtaShape<kThreads>::kCoarseBins;
    static constexpr int kBinsPerThread  = CtaShape<kThreads>::kBinsPerThread;
    static constexpr int kTileElems      = CtaShape<kThreads>::kTileElems;
    static_assert(!kClusterRow || kBinsPerThread == 1, "the cluster histogram merge maps one bin per thread");

    // Exclusive prefix over the row's CTAs of two per-CTA counts, and their totals.
    struct CtaPrefix {
        uint32_t before_a, before_b, total_a, total_b;
    };

    const Params&     p;
    SmemT&            sm;
    cg::cluster_group cluster;
    const int         tid;
    const int         cta_in_cluster;
    const bool        is_row_leader;
    const int         owner;  // also this owner's exchange slot
    const LLExchange  exchange;
    const uint32_t    k;
    uint32_t          round_epoch;  // the leader's last exchange round

    __device__ __forceinline__ MegaTopkCta(const Params& p_, SmemT& sm_):
        p(p_),
        sm(sm_),
        cluster(cg::this_cluster()),
        tid(threadIdx.x),
        cta_in_cluster(kClusterRow ? static_cast<int>(cluster.block_rank()) : 0),
        is_row_leader(cta_in_cluster == 0),
        owner(blockIdx.x / (kClusterRow ? p_.ctas_per_row : 1)),
        exchange{p_, owner},
        k(p_.k),
        round_epoch(is_row_leader ? p_.round_epochs[owner] : 0u) {}

    // Barrier over a row's CTAs; also orders DSMEM accesses.
    __device__ __forceinline__ void sync(bool clustered) {
        if constexpr (kClusterRow) {
            if (clustered) {
                cluster.sync();
                return;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ void run() {
        constexpr int kElemsPerThread = kFloat4PerThread * 4;
        const int     num_owners      = gridDim.x / (kClusterRow ? p.ctas_per_row : 1);
        SmemT* const  cluster_leader_smem = leader_smem_of_cluster();
        for (int row = owner; row < p.num_rows; row += num_owners) {
            // A padding row (position -1) selects nothing, whatever its local length.
            const int local_len = p.positions[row] < 0 ? 0 : p.local_lens[row];
            // A row that fits one tile skips the cluster: CTA 0 takes it alone and the others move on.
            // Every DSMEM access of a clustered row follows that row's first cluster barrier, which
            // CTA 0 reaches only after finishing such a row.
            const bool clustered = kClusterRow && local_len > kTileElems;
            if (kClusterRow && !clustered && !is_row_leader)
                continue;
            const int    ctas_per_row = clustered ? p.ctas_per_row : 1;
            const int    cta_in_row   = clustered ? cta_in_cluster : 0;
            SmemT* const leader_smem  = clustered ? cluster_leader_smem : &sm;
            auto         row_sync     = [&]() { sync(clustered); };

            // ---- this CTA's span of the row
            // [span_begin, span_begin + span_len): the real length split evenly over the row's CTAs in
            // multiples of 4 for float4 loads. A one-tile span stays in registers across every pass; a
            // longer one is reloaded tile by tile on each pass.
            const int span_stride =
                clustered ? (local_len + ctas_per_row * 4 - 1) / (ctas_per_row * 4) * 4 : local_len;
            const int    span_begin        = min(cta_in_row * span_stride, local_len);
            const int    span_len          = min(local_len - span_begin, span_stride);
            const int    num_tiles         = max(1, (span_len + kTileElems - 1) / kTileElems);
            const bool   span_in_registers = num_tiles == 1;
            const float* span_logits = p.logits + static_cast<int64_t>(row) * p.logits_row_stride + span_begin;
            const int    page_shift  = p.kv_page_shift;
            const int*   row_block_table =
                p.block_table + static_cast<int64_t>(row / p.rows_per_request) * p.block_table_row_stride;
            const int  first_page       = span_begin >> page_shift;
            const int* span_block_table = row_block_table + first_page;
            float4     tile_values[kFloat4PerThread];
            uint32_t   valid_mask = 0;  // one bit per element of this thread in the current tile
            // Element i = 4j + q of this thread sits at tile position (j * kThreads + tid) * 4 + q,
            // so (CTA in row, tile, j, tid, q) order is position order.
            auto value_at = [&](int i) -> float {
                const float4& f = tile_values[i >> 2];
                return (i & 3) == 0 ? f.x : (i & 3) == 1 ? f.y : (i & 3) == 2 ? f.z : f.w;
            };
            auto pos_in_tile = [&](int i) -> int { return ((i >> 2) * kThreads + tid) * 4 + (i & 3); };
            auto load_tile   = [&](int tile) {
                const int     tile_len = min(span_len - tile * kTileElems, kTileElems);
                const float4* in       = reinterpret_cast<const float4*>(span_logits + tile * kTileElems);
#pragma unroll
                for (int j = 0; j < kFloat4PerThread; ++j) {
                    const int i4   = j * kThreads + tid;
                    tile_values[j] = i4 * 4 < tile_len ? in[i4] : make_float4(0.f, 0.f, 0.f, 0.f);
                }
                valid_mask = 0;
#pragma unroll
                for (int i = 0; i < kElemsPerThread; ++i)
                    valid_mask |= static_cast<uint32_t>(pos_in_tile(i) < tile_len) << i;
            };
            // body(tile) with the tile loaded. Prefetching the next tile into shared memory (cp.async),
            // and staging spans of up to three tiles there, both measured no faster: a tile's cost is
            // its per-element work, not its loads.
            auto for_each_tile = [&](auto&& body) {
                if (span_in_registers) {
                    body(0);
                    return;
                }
                for (int tile = 0; tile < num_tiles; ++tile) {
                    load_tile(tile);
                    body(tile);
                }
            };
            if (span_in_registers) {
                load_tile(0);
                // Read by the writes after several barriers; overlaps the exchanges.
                const int span_pages = span_len ? ((span_begin + span_len - 1) >> page_shift) - first_page + 1 : 0;
                for (int i = tid; i < span_pages; i += kThreads)
                    sm.span_pages[i] = span_block_table[i];
            }
            auto kv_slot_of = [&](int pos_in_span) -> int {
                const int pos  = span_begin + pos_in_span;  // local position in the row
                if (p.global_page_shift)
                    return global_position(pos, p);
                const int i    = (pos >> page_shift) - first_page;
                const int page = span_in_registers ? static_cast<int>(sm.span_pages[i]) : span_block_table[i];
                return kv_slot(page, pos, page_shift);
            };
            int* out_row = p.out_slots + static_cast<int64_t>(row) * p.out_row_stride;
            // Writes the KV slots of `mask`'s elements of `tile` in position order from output index
            // `before` (deterministic instance), or at indices claimed on the leader's cursor (fast).
            auto write_in_order = [&](uint32_t mask, const MaskScan& s, int tile, uint32_t before) {
                for_each_in_order(mask, s, before, [&](int i, uint32_t o) {
                    out_row[o] = kv_slot_of(tile * kTileElems + pos_in_tile(i));
                });
            };
            auto write_claimed = [&](uint32_t mask, int tile) {
                for_each_claimed(mask, &leader_smem->out_cursor, [&](int i, uint32_t o) {
                    out_row[o] = kv_slot_of(tile * kTileElems + pos_in_tile(i));
                });
            };

            // The leader's count of this rank's slots of the row; it pads the rest of the row with -1.
            uint32_t row_selected = 0;
            if (p.positions[row] < static_cast<int>(k)) {
                // Rows of global length <= k keep every valid element: span_begin precede this span.
                uint32_t written = 0;
                for_each_tile([&](int tile) {
                    const MaskScan s = scan_mask(valid_mask, sm.scan);
                    write_in_order(valid_mask, s, tile, span_begin + written);
                    written += s.total;
                });
                row_selected = local_len;
            } else {
                uint32_t bstar = 0;
                uint32_t span_above_bstar = 0, span_in_bstar = 0;  // register-resident span only
                // This tile's valid elements in b*, and in `above` those in higher bins.
                auto bstar_masks = [&](uint32_t& above) -> uint32_t {
                    if (span_in_registers) {
                        above = span_above_bstar;
                        return span_in_bstar;
                    }
                    uint32_t in_bstar = 0;
                    above             = 0;
#pragma unroll
                    for (int i = 0; i < kElemsPerThread; ++i) {
                        const uint32_t bin = coarse_bin<kCoarseBins>(value_at(i));
                        above |= static_cast<uint32_t>(bin > bstar) << i;
                        in_bstar |= static_cast<uint32_t>(bin == bstar) << i;
                    }
                    above &= valid_mask;
                    return in_bstar & valid_mask;
                };

                // ---- row histogram; a register-resident span keeps its bins (two per word) until b*
                // turns them into masks
                for (int b = tid; b < kCoarseBins; b += kThreads)
                    sm.bin_counts[b] = 0;
                if (tid == 0)
                    sm.own_start_bin = sm.own_top_bin = 0;
                __syncthreads();
                uint32_t span_bins[kElemsPerThread / 2];
                for_each_tile([&](int) {
#pragma unroll
                    for (int i = 0; i < kElemsPerThread; i += 2) {
                        const uint32_t bin0 = coarse_bin<kCoarseBins>(value_at(i)),
                                       bin1 = coarse_bin<kCoarseBins>(value_at(i + 1));
                        span_bins[i / 2]    = bin0 | (bin1 << 16);
                        if (valid_mask >> i & 1)
                            atomicAdd(&sm.bin_counts[bin0], 1u);
                        if (valid_mask >> (i + 1) & 1)
                            atomicAdd(&sm.bin_counts[bin1], 1u);
                    }
                });
                __syncthreads();
                uint32_t own_bin_count[kBinsPerThread];  // this CTA's, before the cluster merge
#pragma unroll
                for (int j = 0; j < kBinsPerThread; ++j)
                    own_bin_count[j] = sm.bin_counts[bin_of(j)];
                if (clustered)
                    merge_cluster_histogram(cta_in_row, ctas_per_row);

                // ---- X1 (leader): b*
                uint32_t rank_count_above[kBinsPerThread] = {};  // this rank's elements above each bin
                if (is_row_leader)
                    find_bstar(rank_count_above);
                row_sync();
                bstar                         = leader_smem->bstar;
                const bool use_radix_fallback = leader_smem->use_radix_fallback;
                uint32_t   still_needed       = leader_smem->still_needed_in_bstar;
                if (span_in_registers) {
#pragma unroll
                    for (int i = 0; i < kElemsPerThread; ++i) {
                        const uint32_t bin = (span_bins[i / 2] >> ((i & 1) * 16)) & 0xffffu;
                        span_above_bstar |= static_cast<uint32_t>(bin > bstar) << i;
                        span_in_bstar |= static_cast<uint32_t>(bin == bstar) << i;
                    }
                    span_above_bstar &= valid_mask;
                    span_in_bstar &= valid_mask;
                }

                // ---- where this CTA's writes go
                // Deterministic: this CTA's elements above b* and in b*, from its own histogram, place
                // its position-ordered writes among the row's CTAs; unclustered rows reuse the X1 scan.
                uint32_t own_count_above_bstar = 0;  // this thread's own elements in bins above b*
#pragma unroll
                for (int j = 0; j < kBinsPerThread; ++j) {
                    if (bin_of(j) == bstar) {
                        sm.own_bstar_count = own_bin_count[j];
                        sm.own_above_count = rank_count_above[j];
                    }
                    own_count_above_bstar += bin_of(j) > bstar ? own_bin_count[j] : 0u;
                }
                uint32_t above_before = 0, above_total = 0, bstar_before = 0;
                if constexpr (!kDeterministic) {
                    if (tid == 0)
                        sm.my_boundary_offset = leader_smem->boundary_offset[p.rank];
                    __syncthreads();
                } else if (clustered) {
                    uint32_t cta_above;
                    block_exclusive_scan(own_count_above_bstar, sm.scan, cta_above);
                    if (tid == 0)
                        sm.my_boundary_offset = leader_smem->boundary_offset[p.rank];
                    const CtaPrefix c = cta_prefix(clustered,
                                                   cta_in_row,
                                                   ctas_per_row,
                                                   leader_smem->cta_above,
                                                   leader_smem->cta_bstar,
                                                   cta_above,
                                                   sm.own_bstar_count);
                    above_before      = c.before_a;
                    above_total       = c.total_a;
                    bstar_before      = c.before_b;
                } else {
                    if (tid == 0)
                        sm.my_boundary_offset = sm.boundary_offset[p.rank];
                    __syncthreads();
                    above_total = sm.own_above_count;
                }
                const uint32_t my_offset = sm.my_boundary_offset;

                // ---- one pass: write the slots above b* and gather this rank's b* values and
                // positions into the leader's shared memory (heavy ties leave b* to the radix rounds)
                uint32_t above_written = 0, bstar_written = 0;
                for_each_tile([&](int tile) {
                    uint32_t       above;
                    const uint32_t in_bstar     = bstar_masks(above) & (use_radix_fallback ? 0u : ~0u);
                    auto           gather_value = [&](int i, uint32_t r) {
                        leader_smem->boundary_values[my_offset + r] = __float_as_uint(value_at(i));
                        leader_smem->boundary_pos[r] = span_begin + tile * kTileElems + pos_in_tile(i);
                    };
                    if constexpr (!kDeterministic) {
                        write_claimed(above, tile);
                        if (!use_radix_fallback)
                            for_each_claimed(in_bstar, &leader_smem->bstar_cursor, gather_value);
                    } else {
                        MaskScan sa, sb;
                        scan_two_masks(above, in_bstar, sm.scan, sa, sb);
                        write_in_order(above, sa, tile, above_before + above_written);
                        above_written += sa.total;
                        if (!use_radix_fallback) {
                            for_each_in_order(in_bstar, sb, bstar_before + bstar_written, gather_value);
                            bstar_written += sb.total;
                        }
                    }
                });

                if (!use_radix_fallback) {
                    // ---- X2 (leader): theta from the gathered b* values, then this rank's share of b*
                    row_sync();
                    if (is_row_leader) {
                        const uint32_t theta_key = find_theta_in_gathered(my_offset, still_needed);
                        row_selected             = select_gathered(out_row,
                                                       row_block_table,
                                                       above_total,
                                                       my_offset,
                                                       theta_key,
                                                       tie_quota(sm.theta_count, still_needed));
                    }
                } else {
                    // ---- heavy ties: radix rounds over the fp32 keys of the row's b* elements, counted in
                    // place over the spans. The key bits shared by the row's smallest and largest b* key
                    // are known; the rest go in digits of up to 8 bits, top down, until theta is exact or
                    // a digit fills the remaining places.
                    uint32_t bstar_min_key = 0xffffffffu, bstar_max_key = 0;
                    for_each_tile([&](int) {
                        uint32_t       above;
                        const uint32_t in_bstar = bstar_masks(above);
#pragma unroll
                        for (int i = 0; i < kElemsPerThread; ++i) {
                            if (in_bstar >> i & 1) {
                                const uint32_t key = ordered_key(__float_as_uint(value_at(i)));
                                bstar_min_key      = min(bstar_min_key, key);
                                bstar_max_key      = max(bstar_max_key, key);
                            }
                        }
                    });
                    __syncthreads();  // the write pass's block scans are done with the scan scratch
                    block_key_range(bstar_min_key, bstar_max_key);
                    if (tid == 0) {
                        atomicMin(&leader_smem->fallback_min_key, bstar_min_key);
                        atomicMax(&leader_smem->fallback_max_key, bstar_max_key);
                    }
                    row_sync();
                    if (is_row_leader)
                        exchange_key_range();
                    row_sync();
                    const uint32_t min_key    = leader_smem->fallback_min_key;
                    int unresolved_bits = min_key == leader_smem->fallback_max_key ?
                                              0 :
                                              32 - __clz(min_key ^ leader_smem->fallback_max_key);
                    uint32_t key_prefix = unresolved_bits == 32 ? 0u : min_key >> unresolved_bits;
                    bool     filled     = false;
                    while (unresolved_bits > 0) {
                        const int digit_bits = min(8, unresolved_bits);
                        const int low_bits   = unresolved_bits - digit_bits;
                        // Each CTA counts its span in its own shared memory; the others then add their
                        // nonzero digits to the leader's counts, one DSMEM atomic per digit.
                        if (tid < 256) {
                            sm.bin_counts[tid] = 0;  // this rank's digit counts, in the leader
                            if (is_row_leader)
                                sm.boundary_values[tid] = 0;  // every rank's
                        }
                        row_sync();
                        for_each_tile([&](int) {
                            uint32_t       above;
                            const uint32_t in_bstar = bstar_masks(above);
#pragma unroll
                            for (int i = 0; i < kElemsPerThread; ++i) {
                                if (!(in_bstar >> i & 1))
                                    continue;
                                const uint32_t key = ordered_key(__float_as_uint(value_at(i)));
                                if (unresolved_bits == 32 || (key >> unresolved_bits) == key_prefix)
                                    atomicAdd(&sm.bin_counts[(key >> low_bits) & ((1u << digit_bits) - 1u)], 1u);
                            }
                        });
                        if (clustered && !is_row_leader) {
                            __syncthreads();
                            if (tid < 256 && sm.bin_counts[tid])
                                atomicAdd(&leader_smem->bin_counts[tid], sm.bin_counts[tid]);
                        }
                        row_sync();
                        if (is_row_leader)
                            exchange_digit_counts(still_needed, low_bits == 0);
                        row_sync();
                        key_prefix = (key_prefix << digit_bits) | leader_smem->picked_digit;
                        still_needed -= leader_smem->count_above_digit;
                        unresolved_bits = low_bits;
                        if (leader_smem->fallback_filled) {
                            filled = true;
                            break;
                        }
                    }
                    // Exact: theta and `quota` of its elements. Filled: every key from theta up, no ties.
                    const uint32_t theta_key = key_prefix << unresolved_bits;
                    const uint32_t quota     = filled ? 0u : tie_quota(leader_smem->theta_count, still_needed);
                    // This tile's b* elements above theta, and in `eq` those equal to it.
                    auto theta_masks = [&](uint32_t& eq) -> uint32_t {
                        uint32_t       above, gt = 0;
                        const uint32_t in_bstar = bstar_masks(above);
                        eq                      = 0;
#pragma unroll
                        for (int i = 0; i < kElemsPerThread; ++i) {
                            if (in_bstar >> i & 1) {
                                const uint32_t key = ordered_key(__float_as_uint(value_at(i)));
                                gt |= static_cast<uint32_t>(filled ? key >= theta_key : key > theta_key) << i;
                                eq |= static_cast<uint32_t>(!filled && key == theta_key) << i;
                            }
                        }
                        return gt;
                    };
                    // b* elements above theta and `quota` of those at theta, after the slots above b*.
                    if constexpr (!kDeterministic) {
                        for_each_tile([&](int tile) {
                            uint32_t eq;
                            uint32_t selected = theta_masks(eq);
                            for_each_claimed(eq, &leader_smem->eq_cursor, [&](int i, uint32_t r) {
                                selected |= static_cast<uint32_t>(r < quota) << i;
                            });
                            write_claimed(selected, tile);
                        });
                        row_sync();  // every CTA's claims are in before the leader reads the count
                        row_selected = leader_smem->out_cursor;
                    } else {
                        // In position order: in a cluster each CTA first learns how many the CTAs before
                        // it hold.
                        CtaPrefix c{};
                        if (clustered) {
                            uint32_t gt_count = 0, eq_count = 0;
                            for_each_tile([&](int) {
                                uint32_t       eq;
                                const uint32_t gt = theta_masks(eq);
                                gt_count += __popc(gt);
                                eq_count += __popc(eq);
                            });
                            uint32_t cta_gt, cta_eq;
                            block_exclusive_scan(gt_count, sm.scan, cta_gt);
                            block_exclusive_scan(eq_count, sm.scan, cta_eq);
                            c = cta_prefix(clustered,
                                           cta_in_row,
                                           ctas_per_row,
                                           leader_smem->cta_gt,
                                           leader_smem->cta_eq,
                                           cta_gt,
                                           cta_eq);
                        }
                        uint32_t written = 0, eq_seen = 0;
                        for_each_tile([&](int tile) {
                            uint32_t       eq;
                            uint32_t       selected = theta_masks(eq);
                            const MaskScan se       = scan_mask(eq, sm.scan);
                            for_each_in_order(eq, se, c.before_b + eq_seen, [&](int i, uint32_t r) {
                                selected |= static_cast<uint32_t>(r < quota) << i;
                            });
                            eq_seen += se.total;
                            const MaskScan ss = scan_mask(selected, sm.scan);
                            write_in_order(selected, ss, tile, above_total + c.before_a + min(quota, c.before_b) + written);
                            written += ss.total;
                        });
                        row_selected = clustered ? above_total + c.total_a + min(quota, c.total_b) : above_total + written;
                    }
                }
            }
            // trtllm-gen reads index tiles past seq_lens and skips -1 entries only: pad the tail.
            if (is_row_leader) {
                for (uint32_t i = row_selected + tid; i < k; i += kThreads)
                    out_row[i] = -1;
                if (tid == 0)
                    p.out_counts[row] = row_selected;
            }
            row_sync();
        }
        if (is_row_leader && tid == 0)
            p.round_epochs[owner] = round_epoch;
    }

    __device__ __forceinline__ SmemT* leader_smem_of_cluster() {
        if constexpr (kClusterRow)
            return cluster.map_shared_rank(&sm, 0);
        else
            return &sm;
    }

    // ------------------------------------------------------------ helpers
    // Thread t holds bins kCoarseBins - 1 - (t * kBinsPerThread + j), from the top.
    __device__ __forceinline__ uint32_t bin_of(int j) const {
        return kCoarseBins - 1 - (tid * kBinsPerThread + j);
    }
    // Exclusive counts above each of this thread's bins, of the counts in `counts`, and their total.
    __device__ __forceinline__ void scan_bins(const uint32_t (&counts)[kBinsPerThread],
                                              uint32_t (&above)[kBinsPerThread],
                                              uint32_t& total) {
        uint32_t thread_sum = 0;
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j)
            thread_sum += counts[j];
        uint32_t running = block_exclusive_scan(thread_sum, sm.scan, total);
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j) {
            above[j] = running;
            running += counts[j];
        }
    }
    // theta-valued elements go to lower ranks first: this rank's share of the still_needed ones.
    __device__ __forceinline__ uint32_t tie_quota(const uint32_t* theta_count, uint32_t still_needed) const {
        uint32_t before = 0;
        for (int r = 0; r < p.rank; ++r)
            before += theta_count[r];
        return still_needed > before ? min(still_needed - before, theta_count[p.rank]) : 0u;
    }
    // Exclusive prefix over the row's CTAs of two per-CTA counts and their totals: each CTA leaves
    // its counts in the leader's `counts_a` and `counts_b`, then warp 0 scans them.
    __device__ __forceinline__ CtaPrefix cta_prefix(bool      clustered,
                                                    int       cta_in_row,
                                                    int       ctas_per_row,
                                                    uint32_t* counts_a,
                                                    uint32_t* counts_b,
                                                    uint32_t  cta_a,
                                                    uint32_t  cta_b) {
        if (tid == 0) {
            counts_a[cta_in_row] = cta_a;
            counts_b[cta_in_row] = cta_b;
        }
        sync(clustered);
        if (tid < 32) {
            const uint32_t a = tid < ctas_per_row ? counts_a[tid] : 0u, b = tid < ctas_per_row ? counts_b[tid] : 0u;
            uint32_t       inc_a = a, inc_b = b;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t na = __shfl_up_sync(0xffffffffu, inc_a, o), nb = __shfl_up_sync(0xffffffffu, inc_b, o);
                if (tid >= o) {
                    inc_a += na;
                    inc_b += nb;
                }
            }
            if (tid == cta_in_row) {
                sm.prefix_a = inc_a - a;
                sm.prefix_b = inc_b - b;
            }
            if (tid == 31) {
                sm.total_a = inc_a;
                sm.total_b = inc_b;
            }
        }
        __syncthreads();
        return {sm.prefix_a, sm.prefix_b, sm.total_a, sm.total_b};
    }
    // CTA c sums bins [c * kCoarseBins / C, (c + 1) * kCoarseBins / C) over the cluster and writes
    // the sums back to every CTA: thread t owns bin start + t / C of CTA t % C.
    __device__ __forceinline__ void merge_cluster_histogram(int cta_in_row, int ctas_per_row) {
        if constexpr (kClusterRow) {
            cluster.sync();
            const int bin       = cta_in_row * (kCoarseBins / ctas_per_row) + tid / ctas_per_row;
            uint32_t* bin_count = cluster.map_shared_rank(&sm.bin_counts[bin], tid % ctas_per_row);
            uint32_t  sum       = *bin_count;
            for (int o = 1; o < ctas_per_row; o <<= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            *bin_count = sum;
            cluster.sync();
        }
    }

    // ------------------------------------------------------------ X1 (leader): b*
    // Every rank broadcasts its histogram from its local top-k start bin up, so all ranks agree on
    // b*, the bin of the global k-th element, and on each rank's count in it. rank_count_above:
    // this rank's elements above each of this thread's bins.
    __device__ __forceinline__ void find_bstar(uint32_t (&rank_count_above)[kBinsPerThread]) {
        const uint32_t epoch = next_round(round_epoch);
        send_x1_segment(epoch, rank_count_above);
        pick_bstar(receive_x1_segments(epoch));
        read_bstar_counts(epoch);
    }
    __device__ __forceinline__ bool x1_u32_counts(int src) const {
        return sm.x1_flags[src] & kX1U32Counts;
    }
    // log2 of the counts per word of rank src: two u32 or four u16. Kept a shift: with a 2-or-4 divisor
    // the compiler emitted general integer division on the exchange's critical path.
    __device__ __forceinline__ int x1_word_shift(int src) const {
        return x1_u32_counts(src) ? 1 : 2;
    }
    // Word 0 is the header {start bin | bins << 16; flags}, then the counts of bins [start, top].
    // The start is the local top-k start bin, or the lowest non-empty bin when the rank holds fewer
    // than k elements (then its whole histogram goes out).
    __device__ __forceinline__ void send_x1_segment(uint32_t epoch, uint32_t (&count_above)[kBinsPerThread]) {
        uint32_t count[kBinsPerThread], total;
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j)
            count[j] = sm.bin_counts[bin_of(j)];
        scan_bins(count, count_above, total);
        const bool whole_histogram = total < k;
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j) {
            if (whole_histogram ? (count[j] > 0 && count_above[j] + count[j] == total) :
                                  (count_above[j] < k && k <= count_above[j] + count[j]))
                sm.own_start_bin = bin_of(j);
            if (count[j] > 0 && count_above[j] == 0)
                sm.own_top_bin = bin_of(j);
        }
        __syncthreads();
        const uint32_t start      = sm.own_start_bin;
        const uint32_t bins       = total ? sm.own_top_bin - start + 1 : 0;
        const bool     u32_counts = total > 0xffffu;  // once a bin may pass u16
        const int      shift      = u32_counts ? 1 : 2;
        const uint32_t per_word   = 1u << shift;
        const uint32_t words      = (bins + per_word - 1) >> shift;
        for (uint32_t word = tid; word <= words; word += kThreads) {
            if (word == 0) {
                exchange.broadcast(epoch,
                                   0,
                                   start | (bins << 16),
                                   (whole_histogram ? kX1WholeHistogram : 0u) | (u32_counts ? kX1U32Counts : 0u));
                continue;
            }
            uint32_t counts[4];
#pragma unroll
            for (uint32_t t = 0; t < 4; ++t) {
                const uint32_t b = start + per_word * (word - 1) + t;
                counts[t]        = (t < per_word && b < start + bins) ? sm.bin_counts[b] : 0u;
            }
            exchange.broadcast_counts(epoch, word, counts, u32_counts);
        }
        if (tid == kThreads - 1)
            exchange.refresh(epoch, words + 1);
        __syncthreads();  // bin_counts turns into the global histogram
    }
    // Sums every rank's segment into sm.bin_counts. Returns the bin from which up that sum is the
    // exact global histogram: the highest start of a rank that sent only part of its histogram.
    __device__ __forceinline__ uint32_t receive_x1_segments(uint32_t epoch) {
        for (int b = tid; b < kCoarseBins; b += kThreads)
            sm.bin_counts[b] = 0;
        if (tid < p.num_ranks) {
            const uint4 header   = exchange.template wait_word<true>(epoch, tid, 0);
            sm.x1_start_bin[tid] = header.x & 0xffffu;
            sm.x1_num_bins[tid]  = header.x >> 16;
            sm.x1_flags[tid]     = header.z;
        }
        __syncthreads();
        uint32_t word_end[kMaxRanks], total_words = 0;
#pragma unroll
        for (int r = 0; r < kMaxRanks; ++r) {
            const int shift = x1_word_shift(r);
            total_words += r < p.num_ranks ? (sm.x1_num_bins[r] + (1u << shift) - 1) >> shift : 0u;
            word_end[r] = total_words;
        }
        for (uint32_t i = tid; i < total_words; i += kThreads) {
            uint32_t       first;
            const int      src         = src_rank_of_word(i, word_end, first);
            const uint32_t word        = i - first;
            const uint32_t per_word    = 1u << x1_word_shift(src);
            const uint4    counts_word = exchange.wait_word(epoch, src, 1 + word);
            for (uint32_t t = 0; t < per_word; ++t) {
                const uint32_t bin_offset = per_word * word + t;
                const uint32_t count      = count_at(counts_word, t, x1_u32_counts(src));
                if (bin_offset < sm.x1_num_bins[src] && count)
                    atomicAdd(&sm.bin_counts[sm.x1_start_bin[src] + bin_offset], count);
            }
        }
        uint32_t floor_bin = 0;
        for (int r = 0; r < p.num_ranks; ++r)
            if (!(sm.x1_flags[r] & kX1WholeHistogram))
                floor_bin = max(floor_bin, sm.x1_start_bin[r]);
        __syncthreads();
        return floor_bin;
    }
    // b* and how many of its elements the top-k still takes.
    __device__ __forceinline__ void pick_bstar(uint32_t floor_bin) {
        uint32_t count[kBinsPerThread], above[kBinsPerThread], total;
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j)
            count[j] = bin_of(j) >= floor_bin ? sm.bin_counts[bin_of(j)] : 0u;
        scan_bins(count, above, total);
#pragma unroll
        for (int j = 0; j < kBinsPerThread; ++j) {
            if (above[j] < k && k <= above[j] + count[j]) {
                sm.bstar                 = bin_of(j);
                sm.still_needed_in_bstar = k - above[j];
            }
        }
        __syncthreads();
    }
    // Each rank's count in b*, from its segment, and where its b* values go in boundary_values;
    // the radix fallback when they do not fit. Also resets the fast instance's claim cursors.
    __device__ __forceinline__ void read_bstar_counts(uint32_t epoch) {
        if (tid < p.num_ranks) {
            const uint32_t start = sm.x1_start_bin[tid], b = sm.bstar;
            const int      shift    = x1_word_shift(tid);
            uint32_t       count    = 0;
            if (b >= start && b < start + sm.x1_num_bins[tid]) {
                const uint32_t bin_offset = b - start;
                count                     = count_at(exchange.wait_word(epoch, tid, 1 + (bin_offset >> shift)),
                                 bin_offset & ((1u << shift) - 1),
                                 x1_u32_counts(tid));
            }
            sm.bstar_count[tid] = count;
            sm.theta_count[tid] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t offset = 0, max_count = 0;
            for (int r = 0; r < p.num_ranks; ++r) {
                sm.boundary_offset[r] = offset;
                offset += sm.bstar_count[r];
                max_count = max(max_count, sm.bstar_count[r]);
            }
            sm.boundary_offset[p.num_ranks] = offset;
            sm.out_cursor = sm.bstar_cursor = sm.eq_cursor = 0;
            sm.use_radix_fallback = offset > kMaxBoundaryValues || max_count > kMaxBoundaryValuesPerRank;
            sm.fallback_min_key   = 0xffffffffu;
            sm.fallback_max_key   = 0;
            sm.fallback_filled    = 0;
        }
    }

    // ------------------------------------------------------------ X2 (leader): theta
    // Gathers every rank's b* values, then finds theta, the exact key of rank still_needed among
    // them, and each rank's count at it. Leaves in still_needed the theta-valued elements to take.
    __device__ __forceinline__ uint32_t find_theta_in_gathered(uint32_t my_offset, uint32_t& still_needed) {
        exchange_boundary_values(my_offset);
        const uint32_t num_values = sm.boundary_offset[p.num_ranks];
        // Values sharing one coarse bin: skip the key bits common to all of them, then 8-bit digits
        // (10-bit digits measured slower: 1024-bin clears and picks).
        uint32_t min_key, max_key;
        boundary_key_range(num_values, min_key, max_key);
        int      unresolved_bits = min_key == max_key ? 0 : 32 - __clz(min_key ^ max_key);
        uint32_t key_prefix      = unresolved_bits == 32 ? 0u : min_key >> unresolved_bits;
        while (unresolved_bits > 0) {
            const int digit_bits = min(8, unresolved_bits);
            unresolved_bits -= digit_bits;
            if (tid < 256)
                sm.bin_counts[tid] = 0;
            __syncthreads();
            for (uint32_t i = tid; i < num_values; i += kThreads) {
                const uint32_t key = ordered_key(sm.boundary_values[i]);
                if (unresolved_bits + digit_bits >= 32 || (key >> (unresolved_bits + digit_bits)) == key_prefix)
                    atomicAdd(&sm.bin_counts[(key >> unresolved_bits) & ((1u << digit_bits) - 1u)], 1u);
            }
            __syncthreads();
            if (tid < 32)
                find_digit_of_rank(sm.bin_counts, still_needed, sm);
            __syncthreads();
            key_prefix = (key_prefix << digit_bits) | sm.picked_digit;
            still_needed -= sm.count_above_digit;
        }
        for (uint32_t i = tid; i < num_values; i += kThreads) {
            if (ordered_key(sm.boundary_values[i]) == key_prefix) {
                int src = 0;
                while (i >= sm.boundary_offset[src + 1])
                    ++src;
                atomicAdd(&sm.theta_count[src], 1u);
            }
        }
        __syncthreads();
        return key_prefix;
    }
    // Broadcasts this rank's b* values, two per word, and receives the other ranks' into place.
    __device__ __forceinline__ void exchange_boundary_values(uint32_t my_offset) {
        const uint32_t epoch    = next_round(round_epoch);
        const uint32_t my_count = sm.bstar_count[p.rank];
        for (uint32_t word = tid; word < (my_count + 1) / 2; word += kThreads)
            exchange.broadcast(epoch,
                               word,
                               sm.boundary_values[my_offset + 2 * word],
                               2 * word + 1 < my_count ? sm.boundary_values[my_offset + 2 * word + 1] : 0u);
        if (tid == kThreads - 1)
            exchange.refresh(epoch, (my_count + 1) / 2);
        uint32_t word_end[kMaxRanks], total_words = 0;
#pragma unroll
        for (int r = 0; r < kMaxRanks; ++r) {
            total_words += (r < p.num_ranks && r != p.rank) ? (sm.bstar_count[r] + 1) / 2 : 0u;
            word_end[r] = total_words;
        }
        for (uint32_t i = tid; i < total_words; i += kThreads) {
            uint32_t       first;
            const int      src         = src_rank_of_word(i, word_end, first);
            const uint32_t word        = i - first;
            const uint4    values_word = exchange.wait_word(epoch, src, word);
            sm.boundary_values[sm.boundary_offset[src] + 2 * word] = values_word.x;
            if (2 * word + 1 < sm.bstar_count[src])
                sm.boundary_values[sm.boundary_offset[src] + 2 * word + 1] = values_word.z;
        }
        __syncthreads();
    }
    // Block-wide min and max of the gathered values' keys.
    __device__ __forceinline__ void boundary_key_range(uint32_t num_values, uint32_t& min_key, uint32_t& max_key) {
        min_key = 0xffffffffu;
        max_key = 0;
        for (uint32_t i = tid; i < num_values; i += kThreads) {
            const uint32_t key = ordered_key(sm.boundary_values[i]);
            min_key            = min(min_key, key);
            max_key            = max(max_key, key);
        }
        block_key_range(min_key, max_key);
    }
    // Block-wide min and max of every thread's min_key and max_key, returned to every thread.
    __device__ __forceinline__ void block_key_range(uint32_t& min_key, uint32_t& max_key) {
        const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            min_key = min(min_key, __shfl_xor_sync(0xffffffffu, min_key, o));
            max_key = max(max_key, __shfl_xor_sync(0xffffffffu, max_key, o));
        }
        if (lane == 0) {
            sm.scan.warp_u32x4[warp].x = min_key;
            sm.scan.warp_u32x4[warp].y = max_key;
        }
        __syncthreads();
        min_key = lane < kWarpsPerCta ? sm.scan.warp_u32x4[lane].x : 0xffffffffu;
        max_key = lane < kWarpsPerCta ? sm.scan.warp_u32x4[lane].y : 0u;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            min_key = min(min_key, __shfl_xor_sync(0xffffffffu, min_key, o));
            max_key = max(max_key, __shfl_xor_sync(0xffffffffu, max_key, o));
        }
    }
    // This rank's gathered b* elements above theta and `quota` of those at theta, after its
    // `above_total` slots above b*. Returns this rank's count of the row's slots.
    __device__ __forceinline__ uint32_t select_gathered(int*       out_row,
                                                        const int* row_block_table,
                                                        uint32_t   above_total,
                                                        uint32_t   my_offset,
                                                        uint32_t   theta_key,
                                                        uint32_t   quota) {
        const uint32_t n   = sm.bstar_count[p.rank];  // <= kMaxBoundaryValuesPerRank <= threads
        const uint32_t key = tid < n ? ordered_key(sm.boundary_values[my_offset + tid]) : 0u;
        const bool     gt = tid < n && key > theta_key, eq = tid < n && key == theta_key;
        auto           kv_slot_at = [&](int pos) {
            if (p.global_page_shift)
                return global_position(pos, p);
            return kv_slot(row_block_table[pos >> p.kv_page_shift], pos, p.kv_page_shift);
        };
        if constexpr (!kDeterministic) {
            // warp_claim needs every lane: claim the tie rank unconditionally.
            const uint32_t eq_rank = warp_claim(eq ? 1u : 0u, &sm.eq_cursor);
            const bool     take    = gt || (eq && eq_rank < quota);
            const uint32_t out     = warp_claim(take ? 1u : 0u, &sm.out_cursor);
            if (take)
                out_row[out] = kv_slot_at(static_cast<int>(sm.boundary_pos[tid]));
            __syncthreads();
            return sm.out_cursor;
        } else {
            // In position order, the first `quota` at theta.
            uint4          totals;
            const uint4    before    = block_exclusive_scan_u16x8(
                make_uint4(static_cast<uint32_t>(gt) | (static_cast<uint32_t>(eq) << 16), 0u, 0u, 0u), sm.scan, totals);
            const uint32_t gt_before = before.x & 0xffffu, eq_before = before.x >> 16;
            if (gt || (eq && eq_before < quota))
                out_row[above_total + gt_before + min(eq_before, quota)] =
                    kv_slot_at(static_cast<int>(sm.boundary_pos[tid]));
            return above_total + (totals.x & 0xffffu) + min(totals.x >> 16, quota);
        }
    }

    // ------------------------------------------------------------ heavy ties (leader)
    // Merges every rank's b* key range, one word per rank. When it is one key, every b* element is
    // theta and each rank's count at theta is its count in b*.
    __device__ __forceinline__ void exchange_key_range() {
        const uint32_t epoch = next_round(round_epoch);
        if (tid == 0)
            exchange.broadcast(epoch, 0, sm.fallback_min_key, sm.fallback_max_key);
        if (tid == kThreads - 1)
            exchange.refresh(epoch, 1);
        __syncthreads();  // thread 0 has read this rank's range
        if (tid < p.num_ranks && tid != p.rank) {
            const uint4 range = exchange.wait_word(epoch, tid, 0);
            atomicMin(&sm.fallback_min_key, range.x);
            atomicMax(&sm.fallback_max_key, range.z);
        }
        __syncthreads();
        if (tid < p.num_ranks && sm.fallback_min_key == sm.fallback_max_key)
            sm.theta_count[tid] = sm.bstar_count[tid];
    }
    // Sums every rank's digit counts (in the width of its X1 counts), picks the digit of the
    // still_needed-th key and, in the last round, reads each rank's count at it.
    __device__ __forceinline__ void exchange_digit_counts(uint32_t still_needed, bool last_round) {
        const uint32_t epoch    = next_round(round_epoch);
        const int shift = x1_word_shift(p.rank);
        if (tid < (256 >> shift))
            exchange.broadcast_counts(epoch, tid, &sm.bin_counts[tid << shift], x1_u32_counts(p.rank));
        if (tid == kThreads - 1)
            exchange.refresh(epoch, 256 >> shift);
        uint32_t* global_digit_counts = sm.boundary_values;
        for (int i = tid; i < 128 * p.num_ranks; i += kThreads) {
            const int      src = i / 128, word = i % 128;
            const int      src_shift    = x1_word_shift(src);
            const uint32_t src_per_word = 1u << src_shift;
            if (word >= (256 >> src_shift))
                continue;
            const uint4 counts_word = exchange.wait_word(epoch, src, word);
            for (uint32_t t = 0; t < src_per_word; ++t)
                atomicAdd(&global_digit_counts[src_per_word * word + t], count_at(counts_word, t, x1_u32_counts(src)));
        }
        __syncthreads();
        if (tid < 32)
            find_digit_of_rank(global_digit_counts, still_needed, sm);
        __syncthreads();
        // The picked digit holds exactly the remaining places: every rank stops here.
        if (tid == 0)
            sm.fallback_filled = !last_round && global_digit_counts[sm.picked_digit] == still_needed - sm.count_above_digit;
        if (last_round && tid < p.num_ranks) {
            const int src_shift = x1_word_shift(tid);
            sm.theta_count[tid] = count_at(exchange.wait_word(epoch, tid, sm.picked_digit >> src_shift),
                                           sm.picked_digit & ((1u << src_shift) - 1),
                                           x1_u32_counts(tid));
        }
    }
};

template<int kThreads, bool kClusterRow, bool kDeterministic>
__global__ void __launch_bounds__(kThreads, CtaShape<kThreads>::kCtasPerSm) mega_topk_kernel(const Params p) {
    __shared__ Smem<kThreads> sm;
    MegaTopkCta<kThreads, kClusterRow, kDeterministic>(p, sm).run();
}

// Per-instance launch attribute, set once: cluster instances opt in to clusters of 16 (beyond the
// portable 8). A device without them reports none resident, so they are never launched.
template<int kThreads, bool kClusterRow, bool kDeterministic = true>
void allow_cluster_16() {
    static const bool done = [] {
        if (kClusterRow
            && cudaFuncSetAttribute(mega_topk_kernel<kThreads, kClusterRow, kDeterministic>,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed,
                                    1)
                   != cudaSuccess)
            (void)cudaGetLastError();
        return true;
    }();
    (void)done;
}

template<int kThreads, bool kClusterRow, bool kDeterministic>
cudaError_t launch(const Params& p, int num_owners, cudaStream_t stream) {
    allow_cluster_16<kThreads, kClusterRow, kDeterministic>();
    cudaLaunchConfig_t  config = {};
    cudaLaunchAttribute attr[1];
    config.gridDim           = dim3(num_owners * p.ctas_per_row, 1, 1);
    config.blockDim          = dim3(kThreads, 1, 1);
    config.stream            = stream;
    attr[0].id               = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = p.ctas_per_row;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    config.attrs             = attr;
    config.numAttrs          = p.ctas_per_row > 1 ? 1 : 0;  // single-CTA rows launch without clusters
    return cudaLaunchKernelEx(&config, mega_topk_kernel<kThreads, kClusterRow, kDeterministic>, p);
}

// Clusters (of 1024-thread CTAs) that can be co-resident: every cluster waits on its peers on
// other ranks, so the persistent grid must never exceed what runs at once.
int max_resident_clusters(int ctas_per_row, int sms, cudaStream_t stream) {
    cudaLaunchConfig_t  config = {};
    cudaLaunchAttribute attr[1];
    config.gridDim           = dim3(sms / ctas_per_row * ctas_per_row, 1, 1);
    config.blockDim          = dim3(1024, 1, 1);
    config.stream            = stream;
    attr[0].id               = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = ctas_per_row;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    config.attrs             = attr;
    config.numAttrs          = 1;
    const auto kernel        = mega_topk_kernel<1024, true, true>;
    allow_cluster_16<1024, true>();
    // Clusters of 16 exceed the portable limit of 8; a device without them reports none.
    if (ctas_per_row > 8
        && cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return clusters;
}

// Co-resident 512-thread single CTAs (two per SM).
int max_resident_two_ctas_per_sm(int sms) {
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mega_topk_kernel<512, false, true>, 512, 0)
        != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return sms * per_sm;
}

// Resident table: 1024-thread single CTAs (SMs), clusters of 2, 4, 8 and 16 of them, then
// 512-thread single CTAs. Exchange slots: the largest entry.
constexpr int kResidentTableSize = 6;
constexpr int kTwoCtasPerSmIndex = 5;
int           resident_table_index(int ctas_per_row) {
    return __builtin_ctz(static_cast<unsigned>(ctas_per_row));
}

}  // namespace

std::vector<int64_t> mega_topk_resident_clusters() {
    const c10::cuda::CUDAGuard guard(c10::cuda::current_device());
    int                        sms;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c10::cuda::current_device()));
    auto                 stream = at::cuda::getCurrentCUDAStream();
    std::vector<int64_t> table(kResidentTableSize);
    // Single CTAs: one per SM (1024 threads at 64 registers fill an SM's register file).
    table[0] = sms;
    for (int ctas : {2, 4, 8, 16})
        table[resident_table_index(ctas)] = max_resident_clusters(ctas, sms, stream);
    table[kTwoCtasPerSmIndex] = max_resident_two_ctas_per_sm(sms);
    return table;
}

int64_t mega_topk_workspace_bytes(int64_t world, int64_t slots) {
    return 2 * slots * world * kSlotWords * static_cast<int64_t>(sizeof(uint4));
}

void mega_topk(const torch::Tensor&        logits,
                   const torch::Tensor&        local_lens,
                   const torch::Tensor&        positions,
                   const torch::Tensor&        block_table,
                   int64_t                     rows_per_request,
                   int64_t                     kv_page_tokens,
                   torch::Tensor&              output,
                   torch::Tensor&              count,
                   const torch::Tensor&        peers,
                   int64_t                     multicast_ptr,
                   torch::Tensor&              epochs,
                   int64_t                     rank,
                   int64_t                     world,
                   int64_t                     k,
                   int64_t                     max_local_kvlen,
                   const std::vector<int64_t>& resident,
                   bool                        deterministic,
                   int64_t                     global_page_tokens) {
    const int rows = logits.size(0);
    TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kFloat32 && logits.dim() == 2 && logits.stride(1) == 1
                    && logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "logits must be float32 [rows, width] with 16-byte aligned rows");
    for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{&local_lens, &positions, &count})
        TORCH_CHECK(t->device() == logits.device() && t->dtype() == torch::kInt32 && t->is_contiguous()
                        && t->numel() == rows,
                    "local_lens, positions and count must be int32 [rows] on the logits device");
    for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{&block_table, &output, &peers, &epochs})
        TORCH_CHECK(t->device() == logits.device(), "all tensors must be on the logits device");
    TORCH_CHECK(block_table.dtype() == torch::kInt32 && block_table.dim() == 2 && block_table.stride(1) == 1,
                "block_table must be int32 [requests, pages]");
    TORCH_CHECK(rows_per_request > 0 && block_table.size(0) * rows_per_request >= rows, "block_table has too few rows");
    TORCH_CHECK(kv_page_tokens >= kMinKvPageTokens && (kv_page_tokens & (kv_page_tokens - 1)) == 0,
                "kv_page_tokens must be a power of two of at least ",
                kMinKvPageTokens);
    TORCH_CHECK(global_page_tokens == 0
                    || (global_page_tokens >= kv_page_tokens && (global_page_tokens & (global_page_tokens - 1)) == 0),
                "global_page_tokens must be 0 or a power of two of at least kv_page_tokens");
    TORCH_CHECK(output.dtype() == torch::kInt32 && output.dim() == 2 && output.stride(1) == 1 && output.size(0) == rows
                    && output.size(1) >= k,
                "output must be int32 [rows, >= k]");
    TORCH_CHECK(world > 0 && world <= kMaxRanks && rank >= 0 && rank < world, "world must be 1..", kMaxRanks);
    TORCH_CHECK(peers.dtype() == torch::kInt64 && peers.is_contiguous() && peers.numel() == world,
                "peers must be int64 [world]");
    TORCH_CHECK(epochs.dtype() == torch::kInt32 && epochs.is_contiguous() && epochs.numel() > 0,
                "epochs must be int32 [slots]");
    TORCH_CHECK(k > 0 && k <= 32768, "k must be in 1..32768");
    TORCH_CHECK(max_local_kvlen > 0, "max_local_kvlen must be positive");
    TORCH_CHECK(resident.size() == kResidentTableSize, "resident must come from mega_topk_resident_clusters");
    if (rows == 0)
        return;
    const c10::cuda::CUDAGuard guard(logits.device());
    // Launch shape, from the row count and the bound on any rank's local row length only, so every
    // rank runs the same grid. Under CUDA graphs the logits are as wide as the global capacity, so
    // max_local_kvlen is the tighter bound; in eager mode the logits width may be.
    max_local_kvlen = std::min<int64_t>(logits.size(1), max_local_kvlen);
    // Rows longer than a tile split over the most CTAs that still run every row in one wave of
    // co-resident clusters, but no more than one tile each of the longest row needs; spans longer
    // than a tile stream. With more rows than fit, single CTAs take one row each.
    const int64_t tiles = (max_local_kvlen + CtaShape<1024>::kTileElems - 1) / CtaShape<1024>::kTileElems;
    int           ctas_per_row = 1;
    for (int ctas = kMaxCtasPerRow; ctas >= 2; ctas /= 2) {
        if (ctas < 2 * tiles && rows <= resident[resident_table_index(ctas)]) {
            ctas_per_row = ctas;
            break;
        }
    }
    Params p{logits.data_ptr<float>(),
             logits.stride(0),
             local_lens.data_ptr<int>(),
             positions.data_ptr<int>(),
             block_table.data_ptr<int>(),
             block_table.stride(0),
             static_cast<int>(rows_per_request),
             __builtin_ctzll(static_cast<uint64_t>(kv_page_tokens)),
             output.data_ptr<int>(),
             output.stride(0),
             count.data_ptr<int>(),
             peers.data_ptr<int64_t>(),
             multicast_ptr,
             epochs.data_ptr<int>(),
             static_cast<int>(epochs.numel()),
             rows,
             static_cast<int>(k),
             static_cast<int>(rank),
             static_cast<int>(world),
             ctas_per_row,
             global_page_tokens ? __builtin_ctzll(static_cast<uint64_t>(global_page_tokens)) : 0};
    // More rows than SMs: 512-thread CTAs, two per SM, so rows progress side by side.
    const bool two_ctas_per_sm = ctas_per_row == 1 && rows > resident[0] && resident[kTwoCtasPerSmIndex] > resident[0];
    // Persistent grid: at most one owner per row, no more than can be co-resident (every owner
    // waits on its peers on other ranks), one exchange slot each.
    const int max_resident = static_cast<int>(resident[two_ctas_per_sm ? kTwoCtasPerSmIndex : resident_table_index(ctas_per_row)]);
    const int num_owners   = std::min({rows, max_resident, static_cast<int>(epochs.numel())});
    TORCH_CHECK(num_owners > 0, "no co-resident cluster of ", ctas_per_row, " CTAs");
    auto        stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    // Both instances share the resource footprint the resident table was measured for.
    auto launch_shape = [&](auto deterministic_tag) {
        constexpr bool kDet = decltype(deterministic_tag)::value;
        if (two_ctas_per_sm)
            return launch<512, false, kDet>(p, num_owners, stream);
        if (ctas_per_row == 1)
            return launch<1024, false, kDet>(p, num_owners, stream);
        return launch<1024, true, kDet>(p, num_owners, stream);
    };
    const cudaError_t err =
        deterministic ? launch_shape(std::true_type{}) : launch_shape(std::false_type{});
    C10_CUDA_CHECK(err);
}

}  // namespace rtp_llm
