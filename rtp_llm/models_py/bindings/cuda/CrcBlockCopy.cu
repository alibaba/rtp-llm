#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace crc_block_copy_internal {
// Per-message verdict written by the CRC kernel's fold-completing thread when
// a load footer comparison fails; read back only for failed items.
struct Result {
    uint32_t       expected;
    uint32_t       actual;
    nvcompStatus_t status;
};
static_assert(sizeof(Result) == 12, "CRC result transfer layout must remain unchanged");
}  // namespace crc_block_copy_internal
namespace {
using crc_block_copy_internal::Result;

struct CudaFailure: std::runtime_error {
    explicit CudaFailure(cudaError_t error): std::runtime_error(cudaGetErrorString(error)) {}
};

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw CudaFailure(error);
}

size_t checkedProduct(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("CRC workspace size overflow");
    return a * b;
}

constexpr uint32_t kCrc32cPoly             = 0x82F63B78u;
constexpr int      kCrcThreads             = 256;
constexpr int      kMaxCrcBlocksPerMessage = 32;
constexpr size_t   kCrcBytesPerBlockHint   = 16 * 1024;
// H2D wave pipelining: a batch of at least kH2dPipelineMinBytes splits into at
// most kMaxH2dWaveCount record waves so the copy engine streams wave w+1 in
// while the SMs CRC wave w, converging H2D from DMA+CRC to DMA+CRC/waves.
// Smaller batches keep the serial path: wave events and launches cost more
// than the CRC time they could hide.
constexpr int    kMaxH2dWaveCount     = 4;
constexpr size_t kH2dPipelineMinBytes = size_t{4} << 20;
// Row stride for every shift-matrix level. A stride of 32 words would place
// rows[bit] of every level on shared bank `bit`, so lanes that diverge into
// different shift levels serialize 32-way on every load; 33 spreads them.
constexpr int kPowerStride = 33;
constexpr size_t kCrcTableWords = 8 * 256;

// Bound the ordinary host cache even when callers submit arbitrary lengths.
constexpr size_t kFoldTableCacheEntries = 16;

struct WorkspaceLayout {
    size_t stride, staging_bytes, result_bytes, partials_bytes, fold_mats_bytes, final_const_bytes;
    size_t table_bytes, power_bytes, arrived_offset, tiles_offset, pack_bytes, alias_bytes;
    size_t device_bytes{0}, host_bytes{0};

    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::overflow_error("CRC workspace size overflow");
        return a + b;
    }

    WorkspaceLayout(size_t items, size_t payload, size_t tiles) {
        if (!items || !payload || !tiles || items > unsigned(std::numeric_limits<int>::max())
            || tiles > unsigned(std::numeric_limits<int>::max()))
            throw std::invalid_argument("invalid CRC capacity");
        stride            = CrcCopyWorkspace::encodedBytes(payload);
        staging_bytes     = checkedProduct(items, stride);
        result_bytes      = checkedProduct(items, sizeof(Result));
        partials_bytes    = checkedProduct(checkedProduct(items, kMaxCrcBlocksPerMessage), 2 * sizeof(uint32_t));
        fold_mats_bytes   = checkedProduct(checkedProduct(items, 33 * kPowerStride), sizeof(uint32_t));
        final_const_bytes = checkedProduct(items, sizeof(uint32_t));
        if (payload > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("CRC payload too large");
        size_t levels = 0;
        for (size_t value = payload; value != 0; value >>= 1)
            ++levels;
        table_bytes    = 8 * 256 * sizeof(uint32_t);
        power_bytes    = checkedProduct(checkedProduct(levels, kPowerStride), sizeof(uint32_t));
        arrived_offset = add(checkedProduct(items, sizeof(size_t)), 7) / 8 * 8;
        tiles_offset   = add(add(arrived_offset, checkedProduct(items, sizeof(unsigned int))), 7) / 8 * 8;
        pack_bytes     = add(tiles_offset, checkedProduct(tiles, sizeof(CopyTile)));
        alias_bytes    = checkedProduct(items, sizeof(void*));
        for (const auto bytes : {staging_bytes,
                                 result_bytes,
                                 partials_bytes,
                                 fold_mats_bytes,
                                 final_const_bytes,
                                 table_bytes,
                                 power_bytes,
                                 pack_bytes,
                                 alias_bytes})
            device_bytes = add(device_bytes, bytes);
        for (const auto bytes : {pack_bytes,
                                 fold_mats_bytes,
                                 final_const_bytes,
                                 alias_bytes,
                                 result_bytes,
                                 checkedProduct(items, sizeof(bool))})
            host_bytes = add(host_bytes, bytes);
    }
};

std::vector<uint32_t> buildCrc32cTables() {
    std::vector<uint32_t> tables(kCrcTableWords);
    for (int index = 0; index < 256; ++index) {
        uint32_t entry = static_cast<uint32_t>(index);
        for (int bit = 0; bit < 8; ++bit)
            entry = (entry >> 1) ^ (entry & 1u ? kCrc32cPoly : 0u);
        tables[index] = entry;
    }
    for (int level = 1; level < 8; ++level)
        for (int index = 0; index < 256; ++index) {
            const uint32_t previous     = tables[(level - 1) * 256 + index];
            tables[level * 256 + index] = (previous >> 8) ^ tables[previous & 0xFFu];
        }
    return tables;
}

using Gf2Matrix = std::array<uint32_t, 32>;

Gf2Matrix identityMatrix() {
    Gf2Matrix m{};
    for (int bit = 0; bit < 32; ++bit)
        m[bit] = 1u << bit;
    return m;
}

uint32_t hostMatvec(const Gf2Matrix& m, uint32_t state) {
    uint32_t out = 0;
    for (int bit = 0; bit < 32; ++bit)
        out ^= ((state >> bit) & 1u) * m[bit];
    return out;
}

// N(a+b) = N(a) . N(b): rows of the composed map are N(b)'s rows advanced by a.
Gf2Matrix composeShift(const Gf2Matrix& a, const Gf2Matrix& b) {
    Gf2Matrix out{};
    for (int bit = 0; bit < 32; ++bit)
        out[bit] = hostMatvec(a, b[bit]);
    return out;
}

int shiftPowerLevels(size_t max_payload) {
    int levels = 0;
    for (size_t value = max_payload; value != 0; value >>= 1)
        ++levels;
    return levels;
}

// Row j of each level is the image of basis vector e_j under the linear map
// that advances one raw CRC32C state by 2^level zero bytes; powers of the same
// map commute, so any byte count is the product of the rows for its set bits.
std::vector<uint32_t> buildShiftPowers(size_t max_payload) {
    const auto tables = buildCrc32cTables();
    const int             levels = shiftPowerLevels(max_payload);
    std::vector<uint32_t> powers(static_cast<size_t>(levels) * kPowerStride, 0);
    // Level 0 rows: M1(e_j) = (e_j >> 8) ^ T0[e_j & 0xFF].
    for (int row = 0; row < 32; ++row) {
        const uint32_t basis = 1u << row;
        powers[row]          = (basis >> 8) ^ tables[basis & 0xFFu];
    }
    for (int level = 1; level < levels; ++level)
        for (int row = 0; row < 32; ++row) {
            const uint32_t source = powers[(level - 1) * kPowerStride + row];
            uint32_t       out    = 0;
            for (int bit = 0; bit < 32; ++bit)
                if ((source >> bit) & 1u)
                    out ^= powers[(level - 1) * kPowerStride + bit];
            powers[level * kPowerStride + row] = out;
        }
    return powers;
}

uint32_t hostApplyShift(const std::vector<uint32_t>& powers, uint32_t state, size_t bytes) {
    uint32_t remaining = static_cast<uint32_t>(bytes);
    while (remaining != 0) {
        const int level = __builtin_ctz(remaining);
        Gf2Matrix step{};
        for (int bit = 0; bit < 32; ++bit)
            step[bit] = powers[static_cast<size_t>(level) * kPowerStride + bit];
        state = hostMatvec(step, state);
        remaining &= remaining - 1;
    }
    return state;
}

// Mirrors the kernel partition: slice is the pow2 round-up of
// ceil(len/blocks), blocks 0..full-1 carry `slice` bytes, block `full` carries
// the ragged remainder, the rest are empty. Returns 33 slots of kPowerStride
// words: slot k (0..31) is N(k*slice), slot 32 is N(ragged) (identity when the
// batch divides evenly). The fold tree shifts by its right group's exact byte
// count - k*slice, plus the ragged remainder when that group straddles the
// ragged block - so one matrix lookup replaces the per-bit matrix chain.
std::vector<uint32_t> buildFoldMatrices(size_t len, int blocks, const std::vector<uint32_t>& powers) {
    size_t slice = 1;
    while (slice < (len + blocks - 1) / blocks)
        slice <<= 1;
    const size_t full   = len / slice;
    const size_t ragged = len - full * slice;
    const auto   base   = [&] {
        // A single ragged block can round slice beyond the power table's
        // capacity. N(slice/2) is in range; squaring it gives N(slice).
        const bool split_slice = slice > len;
        Gf2Matrix  m           = identityMatrix();
        uint32_t   remaining   = static_cast<uint32_t>(split_slice ? slice / 2 : slice);
        while (remaining != 0) {
            const int level = __builtin_ctz(remaining);
            Gf2Matrix step{};
            for (int bit = 0; bit < 32; ++bit)
                step[bit] = powers[static_cast<size_t>(level) * kPowerStride + bit];
            m = composeShift(m, step);
            remaining &= remaining - 1;
        }
        return split_slice ? composeShift(m, m) : m;
    }();
    Gf2Matrix             m = identityMatrix();
    std::vector<uint32_t> mats(33 * kPowerStride, 0);
    for (int slot = 0; slot <= 31; ++slot) {
        for (int bit = 0; bit < 32; ++bit)
            mats[static_cast<size_t>(slot) * kPowerStride + bit] = m[bit];
        m = composeShift(m, base);
    }
    const Gf2Matrix tail = ragged ?
                               [&] {
                                   Gf2Matrix t         = identityMatrix();
                                   uint32_t  remaining = static_cast<uint32_t>(ragged);
                                   while (remaining != 0) {
                                       const int level = __builtin_ctz(remaining);
                                       Gf2Matrix step{};
                                       for (int bit = 0; bit < 32; ++bit)
                                           step[bit] = powers[static_cast<size_t>(level) * kPowerStride + bit];
                                       t = composeShift(t, step);
                                       remaining &= remaining - 1;
                                   }
                                   return t;
                               }() :
                               identityMatrix();
    for (int bit = 0; bit < 32; ++bit)
        mats[static_cast<size_t>(32) * kPowerStride + bit] = tail[bit];
    return mats;
}

struct Allocation {
    Allocation()                             = default;
    Allocation(const Allocation&)            = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&&)                 = delete;
    Allocation& operator=(Allocation&&)      = delete;

    void* pointer{nullptr};
    bool  pinned{false};
    void  allocate(size_t bytes, bool host = false, unsigned flags = cudaHostAllocDefault) {
        pinned = host;
        checkCuda(host ? cudaHostAlloc(&pointer, bytes, flags) : cudaMalloc(&pointer, bytes));
    }
    ~Allocation() {
        if (pointer) {
            if (pinned)
                cudaFreeHost(pointer);
            else
                cudaFree(pointer);
        }
    }
    template<typename T>
    T* as() const {
        return static_cast<T*>(pointer);
    }
};

__device__ __forceinline__ uint32_t crcMatvec(const uint32_t* __restrict__ rows, uint32_t state) {
    // Bit-masked terms combined through an explicit XOR tree: the 32 shared
    // loads are independent and the reduce is log-depth, so the matvec is not
    // a serial LDS->XOR chain.
    uint32_t t[32];
#pragma unroll
    for (int bit = 0; bit < 32; ++bit)
        t[bit] = rows[bit] & static_cast<uint32_t>(static_cast<int>(state << (31 - bit)) >> 31);
#pragma unroll
    for (int stride = 16; stride >= 1; stride >>= 1)
#pragma unroll
        for (int i = 0; i < stride; ++i)
            t[i] ^= t[i + stride];
    return t[0];
}

// Shift state forward by `bytes` positions: one matvec per set bit of bytes.
__device__ __forceinline__ uint32_t applyShift(const uint32_t* __restrict__ powers, uint32_t state, uint32_t bytes) {
    while (bytes != 0) {
        const int level = __ffs(bytes) - 1;
        state           = crcMatvec(powers + level * kPowerStride, state);
        bytes &= bytes - 1;
    }
    return state;
}

// Slicing-by-8 CRC32C over count records at batch indices
// [msg_offset, msg_offset + count), each at staging + batch_index*stride.
// Messages are msg_offset + blockIdx.x, power-of-two slices within a message
// are blockIdx.y. Every block participates in the arrival counter; the last one
// folds all slice partials through per-message fold matrices, then seals the
// record (store) or renders the footer verdict (load) in the same thread, so
// no second kernel launch is needed. arrived must be zero before launch.
__global__ void crc32cBatchedKernel(unsigned char* __restrict__ staging,
                                    size_t stride,
                                    const size_t* __restrict__ lengths,
                                    size_t       msg_offset,
                                    Result*      results,
                                    bool*        failed,
                                    bool         store,
                                    void* const* host_records,
                                    const uint32_t* __restrict__ crc_tables,
                                    const uint32_t* __restrict__ shift_powers,
                                    int power_levels,
                                    const uint32_t* __restrict__ fold_mats,
                                    const uint32_t* __restrict__ final_consts,
                                    uint32_t* partials,
                                    unsigned int* __restrict__ arrived,
                                    int blocks) {
    __shared__ uint32_t     tables[8][256];
    __shared__ uint32_t     warp_states[kCrcThreads / 32];
    __shared__ uint32_t     warp_bytes[kCrcThreads / 32];
    __shared__ unsigned int arrival_ticket;
    __shared__ uint32_t     s_powers[32 * kPowerStride];
    __shared__ uint32_t     s_fold_mats[33 * kPowerStride];

    for (int index = threadIdx.x; index < 8 * 256; index += blockDim.x)
        reinterpret_cast<uint32_t*>(tables)[index] = crc_tables[index];
    // Byte counts are 32-bit, so bit_length(max shift) <= 32: s_powers always
    // covers every level. A runtime ternary on the powers pointer would force
    // the matvec loads through global memory instead of these LDS reads.
    for (int index = threadIdx.x; index < power_levels * kPowerStride; index += blockDim.x)
        s_powers[index] = shift_powers[index];
    __syncthreads();

    const size_t msg     = msg_offset + blockIdx.x;
    const size_t len     = lengths[msg];
    const int    slice_i = blockIdx.y;
    // Round the slice up to a power of two so reduction lengths stay single-bit.
    size_t       slice   = 1;
    const size_t minimum = (len + blocks - 1) / blocks;
    while (slice < minimum)
        slice <<= 1;
    const size_t begin       = static_cast<size_t>(slice_i) * slice;
    const size_t end         = begin < len ? (begin + slice < len ? begin + slice : len) : begin;
    const size_t block_bytes = begin < end ? end - begin : 0;

    size_t run = 1;
    while (run * blockDim.x < block_bytes)
        run <<= 1;
    const size_t tbegin = begin + static_cast<size_t>(threadIdx.x) * run;
    const size_t tend   = tbegin < end ? (tbegin + run < end ? tbegin + run : end) : tbegin;

    uint32_t             crc = 0;
    const unsigned char* p   = staging + msg * stride + tbegin;
    size_t               n   = tend > tbegin ? tend - tbegin : 0;
    // Peel to 16-byte alignment, then uint4 loads in an unrolled counted loop
    // let the compiler hoist the loads ahead of the serial CRC chain. The
    // per-byte fallback cannot prove alignment, so it compiles to 1-byte loads.
    while (n != 0 && (reinterpret_cast<uintptr_t>(p) & 15) != 0) {
        crc = (crc >> 8) ^ tables[0][(crc ^ *p++) & 0xFFu];
        --n;
    }
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0u && n >= 16) {
        const uint4* quads = reinterpret_cast<const uint4*>(p);
        const int    steps = static_cast<int>(n >> 4);
#pragma unroll 4
        for (int i = 0; i < steps; ++i) {
            const uint4 v = quads[i];
            crc ^= v.x;
            crc = tables[7][crc & 0xFFu] ^ tables[6][(crc >> 8) & 0xFFu] ^ tables[5][(crc >> 16) & 0xFFu]
                  ^ tables[4][crc >> 24] ^ tables[3][v.y & 0xFFu] ^ tables[2][(v.y >> 8) & 0xFFu]
                  ^ tables[1][(v.y >> 16) & 0xFFu] ^ tables[0][v.y >> 24];
            crc ^= v.z;
            crc = tables[7][crc & 0xFFu] ^ tables[6][(crc >> 8) & 0xFFu] ^ tables[5][(crc >> 16) & 0xFFu]
                  ^ tables[4][crc >> 24] ^ tables[3][v.w & 0xFFu] ^ tables[2][(v.w >> 8) & 0xFFu]
                  ^ tables[1][(v.w >> 16) & 0xFFu] ^ tables[0][v.w >> 24];
        }
        p = reinterpret_cast<const unsigned char*>(quads + steps);
        n -= static_cast<size_t>(steps) << 4;
    }
    if (n >= 8) {
        uint32_t lo, hi;
        memcpy(&lo, p, 4);
        memcpy(&hi, p + 4, 4);
        crc ^= lo;
        crc = tables[7][crc & 0xFFu] ^ tables[6][(crc >> 8) & 0xFFu] ^ tables[5][(crc >> 16) & 0xFFu]
              ^ tables[4][crc >> 24] ^ tables[3][hi & 0xFFu] ^ tables[2][(hi >> 8) & 0xFFu]
              ^ tables[1][(hi >> 16) & 0xFFu] ^ tables[0][hi >> 24];
        p += 8;
        n -= 8;
    }
    while (n != 0) {
        crc = (crc >> 8) ^ tables[0][(crc ^ *p++) & 0xFFu];
        --n;
    }

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    uint32_t  bytes = static_cast<uint32_t>(tend > tbegin ? tend - tbegin : 0);
    // CRC combine is associative, so a shfl_down tree works: each lane absorbs
    // the segment group to its right, shifted by that group's exact byte count
    // (not just a single-bit run length, which ragged tails would violate).
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t right       = __shfl_down_sync(0xFFFFFFFFu, crc, offset);
        const uint32_t right_bytes = __shfl_down_sync(0xFFFFFFFFu, bytes, offset);
        crc                        = applyShift(s_powers, crc, right_bytes) ^ right;
        bytes += right_bytes;
    }
    if (lane == 0) {
        warp_states[warp] = crc;
        warp_bytes[warp]  = bytes;
    }
    __syncthreads();

    const int warps = static_cast<int>((blockDim.x + 31) >> 5);
    if (warp == 0) {
        uint32_t state = lane < warps ? warp_states[lane] : 0;
        uint32_t count = lane < warps ? warp_bytes[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const uint32_t right       = __shfl_down_sync(0xFFFFFFFFu, state, offset);
            const uint32_t right_count = __shfl_down_sync(0xFFFFFFFFu, count, offset);
            state                      = applyShift(s_powers, state, right_count) ^ right;
            count += right_count;
        }
        if (lane == 0) {
            partials[2 * (msg * blocks + slice_i)]     = state;
            partials[2 * (msg * blocks + slice_i) + 1] = count;
            __threadfence();
            arrival_ticket = atomicAdd(&arrived[msg], 1u);
        }
        __syncwarp();
        if (arrival_ticket == static_cast<unsigned int>(blocks - 1)) {
            __threadfence();
            // Stage this message's fold matrices into shared: the matvecs then
            // hit LDS instead of global, and the kPowerStride layout spreads
            // the per-lane slot divergence across banks.
            for (int index = lane; index < 33 * kPowerStride; index += 32)
                s_fold_mats[index] = fold_mats[msg * (33 * kPowerStride) + index];
            __syncwarp();
            // Partition facts are uniform: blocks 0..full-1 carry exactly
            // `slice` bytes, block `full` carries the ragged remainder R, the
            // rest are empty. The tree shifts each left group by its right
            // group's exact byte count: k*slice, plus R when that group
            // straddles the ragged block (slot 32 = N(R); powers of the base
            // map commute, so the extra matvec's order does not matter).
            const int  full       = static_cast<int>(len / slice);
            uint32_t   fold_state = lane < blocks ? partials[2 * (msg * blocks + lane)] : 0;
            uint32_t   kcnt       = lane < full ? 1u : 0u;
            const bool has_ragged = full < blocks && len - static_cast<size_t>(full) * slice != 0;
            uint32_t   rflag      = has_ragged && lane == full ? 1u : 0u;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const uint32_t right = __shfl_down_sync(0xFFFFFFFFu, fold_state, offset);
                const uint32_t rk    = __shfl_down_sync(0xFFFFFFFFu, kcnt, offset);
                const uint32_t rr    = __shfl_down_sync(0xFFFFFFFFu, rflag, offset);
                if (rr != 0u)
                    fold_state = crcMatvec(s_fold_mats + 32 * kPowerStride, fold_state);
                fold_state = crcMatvec(s_fold_mats + rk * kPowerStride, fold_state) ^ right;
                kcnt += rk;
                rflag += rr;
            }
            if (lane == 0) {
                // crc32c(D) = ~(shift_len(~0) ^ raw(0, D)): fold_state is raw(0, D)
                // and final_consts[msg] is the precomputed shift_len(~0).
                const uint32_t checksum = ~(final_consts[msg] ^ fold_state);
                // The message's bytes are final exactly when the fold completes,
                // so the seal (store) or footer verdict (load) happens here
                // instead of in a second kernel launch.
                const size_t   footer_offset = ((len + 19) & ~size_t(15)) - 4;
                unsigned char* base          = staging + msg * stride;
                if (store) {
                    // Padding is outside the CRC domain. Clear it while sealing
                    // instead of submitting one tiny memset per backing.
                    for (size_t offset = len; offset < footer_offset; ++offset)
                        base[offset] = 0;
                    reinterpret_cast<uint32_t*>(base + footer_offset)[0] = checksum;
                    if (host_records != nullptr) {
                        // Mirror the sealed tail into the host record through its
                        // device alias so the D2H DMA can stay payload-only and
                        // overlap this kernel. Byte stores keep unaligned host
                        // records exact.
                        unsigned char* tail = static_cast<unsigned char*>(host_records[msg]) + len;
                        for (size_t offset = len; offset < footer_offset; ++offset)
                            tail[offset - len] = 0;
                        for (size_t byte = 0; byte < 4; ++byte)
                            tail[footer_offset - len + byte] = reinterpret_cast<const unsigned char*>(&checksum)[byte];
                    }
                    failed[msg] = false;
                } else {
                    const uint32_t footer  = reinterpret_cast<const uint32_t*>(base + footer_offset)[0];
                    const bool     failure = footer != checksum;
                    failed[msg]            = failure;
                    if (failure)
                        results[msg] = {footer, checksum, nvcompSuccess};
                }
            }
        }
    }
}

}  // namespace

namespace crc_block_copy_internal {
cudaError_t launchCrc32cBatched(unsigned char*  staging,
                                size_t          stride,
                                const size_t*   lengths,
                                size_t          count,
                                size_t          msg_offset,
                                Result*         results,
                                bool*           failed,
                                bool            store,
                                void* const*    host_records,
                                const uint32_t* crc_tables,
                                const uint32_t* shift_powers,
                                int             power_levels,
                                const uint32_t* fold_mats,
                                const uint32_t* final_consts,
                                uint32_t*       partials,
                                unsigned int*   arrived,
                                int             blocks,
                                cudaStream_t    stream) {
    const dim3 grid(static_cast<unsigned int>(count), static_cast<unsigned int>(blocks));
    crc32cBatchedKernel<<<grid, kCrcThreads, 0, stream>>>(staging,
                                                          stride,
                                                          lengths,
                                                          msg_offset,
                                                          results,
                                                          failed,
                                                          store,
                                                          host_records,
                                                          crc_tables,
                                                          shift_powers,
                                                          power_levels,
                                                          fold_mats,
                                                          final_consts,
                                                          partials,
                                                          arrived,
                                                          blocks);
    return cudaGetLastError();
}
}  // namespace crc_block_copy_internal

struct CrcCopyWorkspace::Impl {
    int          device;
    size_t       max_items, max_payload, max_tiles, stride;
    size_t       device_bytes{0};
    size_t       host_bytes{0};
    int          power_levels{1};
    cudaStream_t stream{nullptr}, copy_stream{nullptr};
    cudaEvent_t  gather_done{nullptr}, copy_done{nullptr};
    cudaEvent_t  wave_done[kMaxH2dWaveCount]{};
    Allocation   staging, results;
    Allocation   pack_host, pack_device, crc_tables, shift_powers, partials, aliases, host_aliases;
    Allocation   fold_mats, final_consts, fold_mats_host, final_consts_host;
    Allocation   host_results, host_failed;
    bool*        device_failed{nullptr};
    // The signature tracks the valid prefix of the per-message GPU tables.
    // Host matrices depend only on (length, blocks), independent of batch size,
    // tile addresses and message contents. Keep a bounded cache per workspace;
    // its pure CPU values remain valid even after a failed CUDA submission.
    struct FoldTable {
        size_t                length;
        int                   blocks;
        std::vector<uint32_t> matrices;
        uint32_t              final_constant;
    };
    std::vector<uint32_t>  powers_host;
    std::vector<size_t>    fold_lengths;
    int                    fold_blocks{-1};
    std::vector<FoldTable> fold_table_cache;
    size_t                 fold_table_evict{0};
    // Fixed offsets inside the pack buffers; arrived stays zero on the host
    // copy, so every descriptor upload also resets it on the device.
    size_t                                        lengths_offset, arrived_offset, tiles_offset;
    std::mutex                                    mutex;
    std::condition_variable                       idle;
    std::shared_ptr<CrcDumpWriter>                dump_writer;
    CrcCopyState                                  state{CrcCopyState::IDLE};
    CrcCopyDirection                              direction{CrcCopyDirection::STORE};
    std::vector<CrcCopyItem>                      batch;
    std::vector<CrcCopyFailure>                   failures;
    bool                                          diagnostic_cuda_error{false};
    bool                                          diagnostic_pending{false};
    std::function<int(const char*, void*, void*)> test_hook;

    void checkpoint(const char* point) {
        if (test_hook)
            checkCuda(static_cast<cudaError_t>(test_hook(point, stream, copy_stream)));
    }

    Impl(int device_index, size_t items, size_t payload, size_t tile_count, std::shared_ptr<CrcDumpWriter> writer):
        device(device_index),
        max_items(items),
        max_payload(payload),
        max_tiles(tile_count),
        stride(CrcCopyWorkspace::encodedBytes(payload)),
        dump_writer(std::move(writer)) {
        if (device < 0)
            throw std::invalid_argument("invalid CRC capacity");
        const WorkspaceLayout layout(items, payload, tile_count);
        const auto            tables = buildCrc32cTables();
        const auto            powers = buildShiftPowers(payload);
        power_levels                 = static_cast<int>(powers.size() / kPowerStride);
        powers_host                  = powers;
        fold_lengths.reserve(items);
        lengths_offset = 0;
        arrived_offset = layout.arrived_offset;
        tiles_offset   = layout.tiles_offset;
        checkCuda(cudaSetDevice(device));
        staging.allocate(layout.staging_bytes);
        results.allocate(layout.result_bytes);
        partials.allocate(layout.partials_bytes);
        fold_mats.allocate(layout.fold_mats_bytes);
        final_consts.allocate(layout.final_const_bytes);
        fold_mats_host.allocate(layout.fold_mats_bytes, true);
        final_consts_host.allocate(layout.final_const_bytes, true);
        crc_tables.allocate(layout.table_bytes);
        shift_powers.allocate(layout.power_bytes);
        pack_device.allocate(layout.pack_bytes);
        aliases.allocate(layout.alias_bytes);
        pack_host.allocate(layout.pack_bytes, true);
        host_aliases.allocate(layout.alias_bytes, true);
        host_results.allocate(layout.result_bytes, true);
        std::memset(pack_host.pointer, 0, layout.pack_bytes);
        checkCuda(
            cudaMemcpy(crc_tables.pointer, tables.data(), tables.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(shift_powers.pointer, powers_host.data(), layout.power_bytes, cudaMemcpyHostToDevice));
        host_failed.allocate(items * sizeof(bool), true, cudaHostAllocMapped);
        checkCuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_failed), host_failed.pointer, 0));
        // Create streams and events last: no constructor-failure path can strand work.
        try {
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            checkCuda(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
            checkCuda(cudaEventCreateWithFlags(&gather_done, cudaEventDisableTiming));
            checkCuda(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
            for (auto& event : wave_done)
                checkCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        } catch (...) {
            destroyHandles();
            throw;
        }
        device_bytes = layout.device_bytes;
        host_bytes   = layout.host_bytes;
    }

    void drainOrAbort() noexcept {
        // Returning while submitted work still owns caller pointers is unsafe.
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess && test_hook)
            error = static_cast<cudaError_t>(test_hook("drain", stream, copy_stream));
        if (error == cudaSuccess)
            error = cudaStreamSynchronize(stream);
        if (error == cudaSuccess)
            error = cudaStreamSynchronize(copy_stream);
        if (error != cudaSuccess) {
            std::fprintf(stderr,
                         "CRC cannot establish CUDA completion: %s; aborting before pointer reuse\n",
                         cudaGetErrorString(error));
            std::abort();
        }
    }

    // Teardown is reached only after the transfer engine has drained its pools;
    // a CUDA error here means the device/context is already unhealthy. Report
    // it instead of turning a shutdown into SIGABRT.
    void drainForTeardown() noexcept {
        const auto  select = cudaSetDevice(device);
        cudaError_t error  = select;
        if (error == cudaSuccess)
            error = cudaStreamSynchronize(stream);
        if (error == cudaSuccess)
            error = cudaStreamSynchronize(copy_stream);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "CRC teardown could not establish CUDA completion: %s\n", cudaGetErrorString(error));
        }
    }

    void destroyHandles() noexcept {
        if (gather_done)
            cudaEventDestroy(gather_done);
        if (copy_done)
            cudaEventDestroy(copy_done);
        for (auto& event : wave_done)
            if (event)
                cudaEventDestroy(event);
        if (copy_stream)
            cudaStreamDestroy(copy_stream);
        if (stream)
            cudaStreamDestroy(stream);
    }

    ~Impl() {
        if (state != CrcCopyState::IDLE)
            drainOrAbort();
        else if (stream)
            drainForTeardown();
        // Allocation destructors run on this same, explicitly selected device.
        destroyHandles();
    }

    CrcCopyFailure captureSnapshot(const CrcCopyItem& item, const CrcCopyFailure& failure) {
        auto snapshot = failure;
        try {
            checkCuda(cudaSetDevice(device));
            const size_t encoded = CrcCopyWorkspace::encodedBytes(item.payload_bytes);
            const auto*  host    = static_cast<const uint8_t*>(item.host);
            snapshot.cpu_snapshot.assign(host, host + encoded);
            snapshot.staging_snapshot.resize(encoded);
            checkpoint("snapshot");
            checkCuda(cudaMemcpy(snapshot.staging_snapshot.data(),
                                 staging.as<unsigned char>() + failure.item_index * stride,
                                 encoded,
                                 cudaMemcpyDeviceToHost));
        } catch (const CudaFailure& error) {
            diagnostic_cuda_error = true;
            snapshot.staging_snapshot.clear();
            snapshot.capture_error = error.what();
        } catch (const std::exception& error) {
            snapshot.staging_snapshot.clear();
            snapshot.capture_error = error.what();
        } catch (...) {
            snapshot.staging_snapshot.clear();
            snapshot.capture_error = "unknown CRC snapshot error";
        }
        return snapshot;
    }

    template<typename T>
    T* packDeviceAt(size_t offset) const {
        return reinterpret_cast<T*>(static_cast<char*>(pack_device.pointer) + offset);
    }

    template<typename T>
    T* packHostAt(size_t offset) const {
        return reinterpret_cast<T*>(static_cast<char*>(pack_host.pointer) + offset);
    }

    struct PreparedBatch {
        size_t count{0};
        size_t tile_count{0};
        size_t largest_payload{0};
        int    crc_blocks{1};
        bool   upload_fold{false};
    };

    const FoldTable& foldTable(size_t length, int blocks) {
        const auto found = std::find_if(fold_table_cache.begin(), fold_table_cache.end(), [&](const auto& table) {
            return table.length == length && table.blocks == blocks;
        });
        if (found != fold_table_cache.end())
            return *found;
        checkpoint("fold_build");
        FoldTable table{length,
                        blocks,
                        buildFoldMatrices(length, blocks, powers_host),
                        hostApplyShift(powers_host, 0xFFFFFFFFu, length)};
        if (fold_table_cache.size() < kFoldTableCacheEntries) {
            fold_table_cache.push_back(std::move(table));
            return fold_table_cache.back();
        }
        // FIFO replacement avoids bookkeeping and allocation on cache hits.
        const size_t index      = fold_table_evict;
        fold_table_cache[index] = std::move(table);
        fold_table_evict        = (index + 1) % kFoldTableCacheEntries;
        return fold_table_cache[index];
    }

    // A smaller matching batch leaves the longer valid GPU prefix intact.
    // The preceding operation must retire before this slot can be used again.
    void prepareFoldTables(PreparedBatch& prepared) {
        const size_t* lengths = packHostAt<size_t>(lengths_offset);
        if (fold_blocks == prepared.crc_blocks && fold_lengths.size() >= prepared.count
            && std::equal(lengths, lengths + prepared.count, fold_lengths.begin()))
            return;
        fold_blocks = -1;
        fold_lengths.assign(lengths, lengths + prepared.count);
        uint32_t* mats   = fold_mats_host.as<uint32_t>();
        uint32_t* finals = final_consts_host.as<uint32_t>();
        for (size_t index = 0; index < prepared.count; ++index) {
            const auto& table = foldTable(lengths[index], prepared.crc_blocks);
            std::copy(table.matrices.begin(), table.matrices.end(), mats + index * (33 * kPowerStride));
            finals[index] = table.final_constant;
        }
        prepared.upload_fold = true;
    }

    void uploadFoldTables(const PreparedBatch& prepared) {
        if (!prepared.upload_fold)
            return;
        checkpoint("fold_upload");
        checkCuda(cudaMemcpyAsync(fold_mats.pointer,
                                  fold_mats_host.pointer,
                                  prepared.count * (33 * kPowerStride) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
        checkCuda(cudaMemcpyAsync(final_consts.pointer,
                                  final_consts_host.pointer,
                                  prepared.count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
        fold_blocks = prepared.crc_blocks;
    }

    // One walk validates and builds the descriptors; the single packed upload
    // is separate from this pure CPU plan. Partial writes stay in private pinned
    // scratch; no rejected request has submitted asynchronous CUDA accesses.
    std::optional<PreparedBatch> prepareBatch(const std::vector<CrcCopyItem>& batch) {
        if (batch.empty() || batch.size() > max_items)
            return std::nullopt;
        PreparedBatch prepared;
        prepared.count        = batch.size();
        size_t*   lengths     = packHostAt<size_t>(lengths_offset);
        CopyTile* tiles       = packHostAt<CopyTile>(tiles_offset);
        size_t    total_tiles = 0;
        for (size_t index = 0; index < prepared.count; ++index) {
            const auto& item = batch[index];
            if (!item.host || !item.payload_bytes || item.payload_bytes > max_payload)
                return std::nullopt;
            const size_t    encoded = CrcCopyWorkspace::encodedBytes(item.payload_bytes);
            const uintptr_t address = reinterpret_cast<uintptr_t>(item.host);
            if (item.capacity_bytes < encoded || address > std::numeric_limits<uintptr_t>::max() - encoded)
                return std::nullopt;
            // Overlapping destinations can invalidate an earlier item's footer.
            for (size_t previous = 0; previous < index; ++previous) {
                const uintptr_t other       = reinterpret_cast<uintptr_t>(batch[previous].host);
                const size_t    other_bytes = CrcCopyWorkspace::encodedBytes(batch[previous].payload_bytes);
                if (address < other + other_bytes && other < address + encoded)
                    return std::nullopt;
            }
            if (item.tiles.empty() || item.tiles.size() > max_tiles - total_tiles)
                return std::nullopt;
            lengths[index]           = item.payload_bytes;
            prepared.largest_payload = std::max(prepared.largest_payload, item.payload_bytes);
            size_t end               = 0;
            size_t write             = total_tiles;
            for (const auto& tile : item.tiles) {
                if (!tile.device || !tile.bytes || tile.offset != end || tile.bytes > item.payload_bytes - end
                    || reinterpret_cast<uintptr_t>(tile.device) > std::numeric_limits<uintptr_t>::max() - tile.bytes)
                    return std::nullopt;
                tiles[write++] = {tile.device, index * stride + tile.offset, tile.bytes};
                end += tile.bytes;
            }
            if (end != item.payload_bytes)
                return std::nullopt;
            total_tiles += item.tiles.size();
        }
        prepared.tile_count = total_tiles;
        // largest_payload is nonzero, so ceil is at least 1; the min clamps at 32.
        prepared.crc_blocks = static_cast<int>(std::min<size_t>(
            kMaxCrcBlocksPerMessage, (prepared.largest_payload + kCrcBytesPerBlockHint - 1) / kCrcBytesPerBlockHint));
        return prepared;
    }

    // Binds the workspace-owned buffers in one place; only the message range,
    // direction, and optional host alias array vary between the store, serial
    // load, and wave load sites.
    void
    launchCrc(const PreparedBatch& prepared, size_t count, size_t msg_offset, bool store, void* const* host_records) {
        checkCuda(crc_block_copy_internal::launchCrc32cBatched(staging.as<unsigned char>(),
                                                               stride,
                                                               packDeviceAt<size_t>(lengths_offset),
                                                               count,
                                                               msg_offset,
                                                               results.as<Result>(),
                                                               device_failed,
                                                               store,
                                                               host_records,
                                                               crc_tables.as<uint32_t>(),
                                                               shift_powers.as<uint32_t>(),
                                                               power_levels,
                                                               fold_mats.as<uint32_t>(),
                                                               final_consts.as<uint32_t>(),
                                                               partials.as<uint32_t>(),
                                                               packDeviceAt<unsigned int>(arrived_offset),
                                                               prepared.crc_blocks,
                                                               stream));
    }

    // Called only after the complete pipeline has converged. Results and
    // failure storage were reserved before the first metadata upload.
    void collectResults() {
        checkCuda(cudaSetDevice(device));
        for (size_t index = 0; index < batch.size(); ++index) {
            if (!host_failed.as<bool>()[index])
                continue;
            diagnostic_pending = true;
            checkCuda(cudaMemcpyAsync(host_results.as<Result>() + index,
                                      results.as<Result>() + index,
                                      sizeof(Result),
                                      cudaMemcpyDeviceToHost,
                                      stream));
            checkpoint("result_copy");
        }
        checkCuda(cudaStreamSynchronize(stream));
        diagnostic_pending = false;
        for (size_t index = 0; index < batch.size(); ++index) {
            if (!host_failed.as<bool>()[index])
                continue;
            const auto&    observed = host_results.as<Result>()[index];
            CrcCopyFailure failure{
                index, observed.expected, observed.actual, CrcCopyState::CRC_MISMATCH, int(observed.status)};
            failure.checked_footer_available = direction == CrcCopyDirection::LOAD;
            failures.push_back(std::move(failure));
            if (dump_writer) {
                const auto& verdict = failures.back();
                dump_writer->write(batch[index], verdict, [&] { return captureSnapshot(batch[index], verdict); });
            }
            if (diagnostic_cuda_error)
                break;
        }
    }

    bool prepareHostAliases(const std::vector<CrcCopyItem>& items, bool require_pinned) {
        for (size_t index = 0; index < items.size(); ++index) {
            cudaPointerAttributes attributes{};
            // CUDA 13 reports ordinary host allocations as Unregistered. Any
            // actual probe error is a device error, never a pageable fallback.
            checkCuda(cudaPointerGetAttributes(&attributes, items[index].host));
            const bool pinned = attributes.type == cudaMemoryTypeHost;
            if (require_pinned && !pinned)
                return false;
            host_aliases.as<void*>()[index] = pinned ? attributes.devicePointer : nullptr;
        }
        return true;
    }

    void enqueueStore(const PreparedBatch& prepared) {
        checkCuda(
            launchCopyTiles(packDeviceAt<CopyTile>(tiles_offset), prepared.tile_count, staging.pointer, false, stream));
        checkCuda(cudaEventRecord(gather_done, stream));
        uploadFoldTables(prepared);
        const bool   mapped       = std::all_of(host_aliases.as<void*>(),
                                        host_aliases.as<void*>() + prepared.count,
                                        [](void* alias) { return alias != nullptr; });
        void* const* host_records = nullptr;
        if (mapped) {
            checkCuda(cudaMemcpyAsync(
                aliases.pointer, host_aliases.pointer, prepared.count * sizeof(void*), cudaMemcpyHostToDevice, stream));
            host_records = aliases.as<void*>();
        }
        checkpoint("store_crc");
        launchCrc(prepared, prepared.count, 0, true, host_records);
        if (mapped) {
            checkCuda(cudaStreamWaitEvent(copy_stream, gather_done));
            checkpoint("store_dma");
            for (size_t index = 0; index < prepared.count; ++index)
                checkCuda(cudaMemcpyAsync(batch[index].host,
                                          staging.as<unsigned char>() + index * stride,
                                          batch[index].payload_bytes,
                                          cudaMemcpyDeviceToHost,
                                          copy_stream));
            checkCuda(cudaEventRecord(copy_done, copy_stream));
            checkpoint("join");
            checkCuda(cudaStreamWaitEvent(stream, copy_done));
        } else {
            for (size_t index = 0; index < prepared.count; ++index)
                checkCuda(cudaMemcpyAsync(batch[index].host,
                                          staging.as<unsigned char>() + index * stride,
                                          CrcCopyWorkspace::encodedBytes(batch[index].payload_bytes),
                                          cudaMemcpyDeviceToHost,
                                          stream));
        }
    }

    void enqueueLoad(const PreparedBatch& prepared) {
        size_t total_bytes = 0;
        for (const auto& item : batch)
            total_bytes += CrcCopyWorkspace::encodedBytes(item.payload_bytes);
        const size_t waves =
            total_bytes >= kH2dPipelineMinBytes ? std::min<size_t>(kMaxH2dWaveCount, prepared.count) : 1;
        if (waves == 1) {
            for (size_t index = 0; index < prepared.count; ++index)
                checkCuda(cudaMemcpyAsync(staging.as<unsigned char>() + index * stride,
                                          batch[index].host,
                                          CrcCopyWorkspace::encodedBytes(batch[index].payload_bytes),
                                          cudaMemcpyHostToDevice,
                                          stream));
            uploadFoldTables(prepared);
            launchCrc(prepared, prepared.count, 0, false, nullptr);
        } else {
            uploadFoldTables(prepared);
            const size_t base  = prepared.count / waves;
            const size_t extra = prepared.count % waves;
            size_t       begin = 0;
            for (size_t wave = 0; wave < waves; ++wave) {
                const size_t count = base + (wave < extra ? 1 : 0);
                for (size_t index = begin; index < begin + count; ++index)
                    checkCuda(cudaMemcpyAsync(staging.as<unsigned char>() + index * stride,
                                              batch[index].host,
                                              CrcCopyWorkspace::encodedBytes(batch[index].payload_bytes),
                                              cudaMemcpyHostToDevice,
                                              copy_stream));
                checkCuda(cudaEventRecord(wave_done[wave], copy_stream));
                checkCuda(cudaStreamWaitEvent(stream, wave_done[wave]));
                launchCrc(prepared, count, begin, false, nullptr);
                begin += count;
            }
        }
        // A failed load still scatters. Only settlement may adopt its targets.
        checkpoint("scatter");
        checkCuda(
            launchCopyTiles(packDeviceAt<CopyTile>(tiles_offset), prepared.tile_count, staging.pointer, true, stream));
    }

    CrcCopyState submit(const std::vector<CrcCopyItem>& items, CrcCopyDirection operation, bool require_pinned) {
        if (state != CrcCopyState::IDLE)
            return CrcCopyState::BUSY;
        std::optional<PreparedBatch> prepared;
        try {
            prepared = prepareBatch(items);
            if (!prepared || (operation != CrcCopyDirection::STORE && operation != CrcCopyDirection::LOAD))
                return CrcCopyState::INVALID_ARGS;
            checkCuda(cudaSetDevice(device));
            if (!prepareHostAliases(items, require_pinned))
                return CrcCopyState::INVALID_ARGS;
            prepareFoldTables(*prepared);
            // All potentially allocating operation metadata is stable before
            // the first enqueue, including lazy dump captures and failures.
            batch = items;
            failures.clear();
            failures.reserve(items.size());
        } catch (...) {
            batch.clear();
            fold_blocks = -1;
            return CrcCopyState::DEVICE_ERROR;
        }
        direction             = operation;
        diagnostic_cuda_error = false;
        diagnostic_pending    = false;
        // Even a failing first CUDA API may have started work. Any exception
        // from here drains both streams before releasing workspace ownership.
        state = CrcCopyState::PENDING;
        try {
            checkCuda(cudaMemcpyAsync(pack_device.pointer,
                                      pack_host.pointer,
                                      tiles_offset + prepared->tile_count * sizeof(CopyTile),
                                      cudaMemcpyHostToDevice,
                                      stream));
            checkpoint("metadata");
            if (direction == CrcCopyDirection::STORE)
                enqueueStore(*prepared);
            else
                enqueueLoad(*prepared);
        } catch (...) {
            failDevice();
            return finish();
        }
        return state;
    }

    CrcCopyState failDevice() noexcept {
        fold_blocks = -1;
        drainOrAbort();
        state = CrcCopyState::DEVICE_ERROR;
        return state;
    }

    CrcCopyState poll() {
        if (state == CrcCopyState::IDLE)
            throw std::invalid_argument("CRC workspace has no active operation");
        if (state == CrcCopyState::PENDING) {
            try {
                checkCuda(cudaSetDevice(device));
                checkpoint("query");
                const auto ready = cudaStreamQuery(stream);
                if (ready == cudaErrorNotReady)
                    return CrcCopyState::PENDING;
                checkCuda(ready);
                state = CrcCopyState::OK;
                for (size_t index = 0; index < batch.size(); ++index)
                    if (host_failed.as<bool>()[index]) {
                        return diagnose();
                    }
            } catch (...) {
                return failDevice();
            }
        }
        return state;
    }

    CrcCopyState diagnose() {
        state = CrcCopyState::CRC_MISMATCH;
        try {
            collectResults();
        } catch (const CudaFailure&) {
            diagnostic_cuda_error = true;
        } catch (...) {
            diagnostic_cuda_error = diagnostic_cuda_error || diagnostic_pending;
            // Allocation/CPU diagnostic failures do not change physical safety
            // or erase a confirmed mismatch. No CUDA work is pending here unless
            // a CUDA call failed, in which case both streams must be drained.
        }
        if (diagnostic_cuda_error)
            return failDevice();
        return state;
    }

    void requireTerminal() const {
        if (state != CrcCopyState::OK && state != CrcCopyState::CRC_MISMATCH && state != CrcCopyState::DEVICE_ERROR)
            throw std::invalid_argument("CRC operation is not safe to retire");
    }

    CrcCopyState finish() {
        requireTerminal();
        const auto result = state;
        batch.clear();
        failures.clear();
        state = CrcCopyState::IDLE;
        idle.notify_all();
        return result;
    }

    CrcCopyState synchronous(const std::vector<CrcCopyItem>& items,
                             CrcCopyDirection                operation,
                             std::vector<CrcCopyFailure>&    output) {
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [&] { return state == CrcCopyState::IDLE; });
        output.clear();
        const auto submitted = submit(items, operation, false);
        if (state == CrcCopyState::IDLE)
            return submitted;
        if (state == CrcCopyState::PENDING) {
            try {
                checkCuda(cudaSetDevice(device));
                checkCuda(cudaStreamSynchronize(stream));
            } catch (...) {
                failDevice();
            }
        }
        poll();
        output = std::move(failures);
        return finish();
    }
};

CrcCopyMemoryUsage CrcCopyWorkspace::memoryUsage(size_t items, size_t payload, size_t tiles) {
    const WorkspaceLayout layout(items, payload, tiles);
    return {layout.device_bytes, layout.host_bytes};
}

bool CrcCopyWorkspace::available() {
    return true;
}

size_t CrcCopyWorkspace::deviceMemoryBytes() const {
    return impl_->device_bytes;
}

size_t CrcCopyWorkspace::hostMemoryBytes() const {
    return impl_->host_bytes;
}

CrcCopyWorkspace::CrcCopyWorkspace(
    int device, size_t items, size_t payload, size_t tiles, std::shared_ptr<CrcDumpWriter> dump_writer):
    impl_(std::make_unique<Impl>(device, items, payload, tiles, std::move(dump_writer))) {}

CrcCopyWorkspace::~CrcCopyWorkspace() = default;

CrcCopyState CrcCopyWorkspace::store(const std::vector<CrcCopyItem>& items, std::vector<CrcCopyFailure>& failures) {
    return impl_->synchronous(items, CrcCopyDirection::STORE, failures);
}

CrcCopyState CrcCopyWorkspace::load(const std::vector<CrcCopyItem>& items, std::vector<CrcCopyFailure>& failures) {
    return impl_->synchronous(items, CrcCopyDirection::LOAD, failures);
}

CrcCopyState CrcCopyWorkspace::trySubmit(const std::vector<CrcCopyItem>& items, CrcCopyDirection direction) {
    std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
    if (!lock.owns_lock())
        return CrcCopyState::BUSY;
    return impl_->submit(items, direction, true);
}

CrcCopyState CrcCopyWorkspace::poll() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->poll();
}

const std::vector<CrcCopyFailure>& CrcCopyWorkspace::failures() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->requireTerminal();
    return impl_->failures;
}

CrcCopyState CrcCopyWorkspace::finish() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->finish();
}

void CrcCopyWorkspace::setTestHook(std::function<int(const char*, void*, void*)> hook) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state != CrcCopyState::IDLE)
        throw std::logic_error("cannot change hooks on an active CRC workspace");
    impl_->test_hook = std::move(hook);
}

}  // namespace rtp_llm
