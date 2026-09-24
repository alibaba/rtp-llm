#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"
#include "rtp_llm/models_py/bindings/cuda/CrcBlockCopyInternal.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace {
using crc_block_copy_internal::BackingInfo;
using crc_block_copy_internal::Result;
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
void checkNvcomp(nvcompStatus_t error) {
    if (error != nvcompSuccess)
        throw std::runtime_error("nvCOMP CRC launch failed: " + std::to_string(int(error)));
}
size_t checkedProduct(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("CRC workspace size overflow");
    return a * b;
}
struct Allocation {
    void* pointer{nullptr};
    bool  pinned{false};
    void  allocate(size_t bytes, bool host = false) {
        pinned = host;
        checkCuda(host ? cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) : cudaMalloc(&pointer, bytes));
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
__global__ void finishCrc(unsigned char*        staging,
                          size_t                stride,
                          const size_t*         lengths,
                          const uint32_t*       crc,
                          const nvcompStatus_t* status,
                          Result*               results,
                          size_t                count,
                          bool                  store) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const size_t footer_offset = ((lengths[index] + 19) & ~size_t(15)) - 4;
    auto*        footer        = reinterpret_cast<uint32_t*>(staging + index * stride + footer_offset);
    if (store) {
        // Padding is outside the CRC domain. Clear it while sealing instead of
        // submitting one tiny cudaMemsetAsync per backing on the host thread.
        for (size_t offset = lengths[index]; offset < footer_offset; ++offset)
            staging[index * stride + offset] = 0;
        *footer = crc[index];
    }
    results[index] = {*footer, crc[index], status[index]};
}

constexpr unsigned kCombineThreads = 128;

// For CRC32C, init == xorout == 0xffffffff. Finalized segment checksums
// therefore combine as XOR_i Z_suffix_bytes[i](crc[i]), without an affine term.
__global__ void combineAndFinish(unsigned char*        staging,
                                 size_t                stride,
                                 const BackingInfo*    backings,
                                 const uint32_t*       matrices,
                                 const uint32_t*       checksums,
                                 const nvcompStatus_t* statuses,
                                 Result*               results,
                                 bool                  store) {
    __shared__ uint32_t partial_crc[kCombineThreads];
    __shared__ size_t   first_error[kCombineThreads];
    const size_t        backing = blockIdx.x;
    const auto          info    = backings[backing];
    const auto*         matrix  = matrices + info.matrix_offset;
    uint32_t            crc     = 0;
    size_t              failed  = info.segment_count;
    for (size_t segment = threadIdx.x; segment < info.segment_count; segment += blockDim.x) {
        const size_t index = info.first_segment + segment;
        if (statuses[index] != nvcompSuccess) {
            failed = min(failed, segment);
            continue;  // A failed message's checksum need not be initialized.
        }
        const uint32_t value   = checksums[index];
        uint32_t       shifted = 0;
#pragma unroll
        for (size_t bit = 0; bit < 32; ++bit)
            shifted ^= matrix[bit * info.segment_count + segment] & (uint32_t(0) - ((value >> bit) & 1U));
        crc ^= shifted;
    }
    partial_crc[threadIdx.x] = crc;
    first_error[threadIdx.x] = failed;
    __syncthreads();
    for (unsigned step = kCombineThreads / 2; step; step >>= 1) {
        if (threadIdx.x < step) {
            partial_crc[threadIdx.x] ^= partial_crc[threadIdx.x + step];
            first_error[threadIdx.x] = min(first_error[threadIdx.x], first_error[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t footer_offset = ((info.payload_bytes + 19) & ~size_t(15)) - 4;
        auto*        base          = staging + backing * stride;
        auto*        footer        = reinterpret_cast<uint32_t*>(base + footer_offset);
        if (store) {
            for (size_t offset = info.payload_bytes; offset < footer_offset; ++offset)
                base[offset] = 0;
            *footer = partial_crc[0];
        }
        const auto status =
            first_error[0] == info.segment_count ? nvcompSuccess : statuses[info.first_segment + first_error[0]];
        results[backing] = {*footer, partial_crc[0], status};
    }
}

using Matrix = std::array<uint32_t, 32>;

uint32_t applyMatrix(const Matrix& matrix, uint32_t value) {
    uint32_t result = 0;
    for (size_t bit = 0; bit < 32; ++bit)
        if ((value >> bit) & 1U)
            result ^= matrix[bit];
    return result;
}

const std::array<Matrix, sizeof(size_t) * 8>& zeroBytePowers() {
    static const auto powers = [] {
        std::array<Matrix, sizeof(size_t) * 8> result{};
        for (size_t bit = 0; bit < 32; ++bit) {
            uint32_t value = uint32_t(1) << bit;
            for (int zero_bit = 0; zero_bit < 8; ++zero_bit)
                value = (value >> 1) ^ ((value & 1U) ? 0x82f63b78U : 0);
            result[0][bit] = value;
        }
        for (size_t power = 1; power < result.size(); ++power)
            for (size_t bit = 0; bit < 32; ++bit)
                result[power][bit] = applyMatrix(result[power - 1], result[power - 1][bit]);
        return result;
    }();
    return powers;
}
}  // namespace

namespace crc_block_copy_internal {
void makeSuffixMatrices(size_t payload, uint32_t* out) {
    const size_t segments = segmentCount(payload);
    if (!segments || !out)
        throw std::invalid_argument("invalid CRC matrix shape");
    checkedProduct(checkedProduct(segments, 32), sizeof(uint32_t));
    const auto& powers = zeroBytePowers();
    size_t      suffix = payload;
    for (size_t segment = 0; segment < segments; ++segment) {
        suffix -= std::min(suffix, kSegmentBytes);
        for (size_t bit = 0; bit < 32; ++bit) {
            uint32_t value     = uint32_t(1) << bit;
            size_t   remaining = suffix;
            for (size_t power = 0; remaining; ++power, remaining >>= 1)
                if (remaining & 1)
                    value = applyMatrix(powers[power], value);
            out[bit * segments + segment] = value;
        }
    }
}

cudaError_t launchCombineAndFinish(unsigned char*        staging,
                                   size_t                stride,
                                   const BackingInfo*    backings,
                                   const uint32_t*       matrices,
                                   const uint32_t*       checksums,
                                   const nvcompStatus_t* statuses,
                                   Result*               results,
                                   size_t                count,
                                   bool                  store,
                                   cudaStream_t          stream) {
    combineAndFinish<<<count, kCombineThreads, 0, stream>>>(
        staging, stride, backings, matrices, checksums, statuses, results, store);
    return cudaGetLastError();
}
}  // namespace crc_block_copy_internal

struct CrcBlockCopyBatch::Impl {
    struct Configuration {
        size_t                   messages{0};
        size_t                   largest{0};
        nvcompBatchedCRC32Opts_t opts{};
    };
    int          device;
    size_t       max_items, max_payload, max_tiles, stride;
    size_t       max_segments{1}, max_messages{0}, matrix_slot_words{0};
    cudaStream_t stream{nullptr};
    Allocation   staging, tiles, inputs, lengths, checksums, statuses, results;
    Allocation   host_tiles, host_inputs, host_lengths, host_results;
    Allocation   backings, host_backings, matrices, host_matrices;
    // Both caches are bounded by max_items. A whole batch can have at most that
    // many distinct payload shapes, even when every backing has a different P.
    std::vector<Configuration> options;
    size_t                     next_option{0}, next_matrix_slot{0};
    std::map<size_t, size_t>   matrix_shapes;  // payload -> slot
    std::vector<size_t>        matrix_payloads, item_matrix_slots;
    std::vector<unsigned char> protected_matrix_slots;
    std::array<bool, 3>        reported_segmented_success{};
    std::mutex                 mutex;

    Impl(int device_index, size_t items, size_t payload, size_t tile_count):
        device(device_index),
        max_items(items),
        max_payload(payload),
        max_tiles(tile_count),
        stride(CrcBlockCopyBatch::encodedBytes(payload)) {
        if (device < 0 || !items || !payload || !tile_count || items > unsigned(std::numeric_limits<int>::max())
            || tile_count > unsigned(std::numeric_limits<int>::max()))
            throw std::invalid_argument("invalid CRC capacity");
        // Check every capacity before allocating anything on the GPU. Small
        // workspaces which can never select segmented CRC retain N messages.
        const bool can_segment = crc_block_copy_internal::shouldSegment(items, payload);
        max_segments           = can_segment ? crc_block_copy_internal::segmentCount(payload) : 1;
        max_messages           = checkedProduct(items, max_segments);
        if (max_messages > unsigned(std::numeric_limits<int>::max()))
            throw std::invalid_argument("too many CRC segment messages");
        const size_t staging_bytes  = checkedProduct(items, stride);
        const size_t tile_bytes     = checkedProduct(tile_count, sizeof(CopyTile));
        const size_t input_bytes    = checkedProduct(max_messages, sizeof(void*));
        const size_t length_bytes   = checkedProduct(max_messages, sizeof(size_t));
        const size_t checksum_bytes = checkedProduct(max_messages, sizeof(uint32_t));
        const size_t status_bytes   = checkedProduct(max_messages, sizeof(nvcompStatus_t));
        const size_t result_bytes   = checkedProduct(items, sizeof(Result));
        const size_t backing_bytes  = can_segment ? checkedProduct(items, sizeof(BackingInfo)) : 0;
        matrix_slot_words           = can_segment ? checkedProduct(max_segments, 32) : 0;
        const size_t matrix_bytes   = checkedProduct(checkedProduct(items, matrix_slot_words), sizeof(uint32_t));
        options.resize(items);
        if (can_segment) {
            matrix_payloads.resize(items, 0);
            item_matrix_slots.resize(items, 0);
            protected_matrix_slots.resize(items, 0);
        }
        checkCuda(cudaSetDevice(device));
        staging.allocate(staging_bytes);
        tiles.allocate(tile_bytes);
        inputs.allocate(input_bytes);
        lengths.allocate(length_bytes);
        checksums.allocate(checksum_bytes);
        statuses.allocate(status_bytes);
        results.allocate(result_bytes);
        host_tiles.allocate(tile_bytes, true);
        host_inputs.allocate(input_bytes, true);
        host_lengths.allocate(length_bytes, true);
        host_results.allocate(result_bytes, true);
        if (can_segment) {
            backings.allocate(backing_bytes);
            host_backings.allocate(backing_bytes, true);
            matrices.allocate(matrix_bytes);
            host_matrices.allocate(matrix_bytes, true);
        }
        // Create the stream last: no constructor-failure path can strand work.
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    void drainOrAbort() noexcept {
        // Returning while submitted work still owns caller pointers is unsafe.
        const auto select = cudaSetDevice(device);
        const auto error  = select == cudaSuccess ? cudaStreamSynchronize(stream) : select;
        if (error != cudaSuccess) {
            std::fprintf(stderr,
                         "CRC cannot establish CUDA completion: %s; aborting before pointer reuse\n",
                         cudaGetErrorString(error));
            std::abort();
        }
    }
    ~Impl() {
        if (stream) {
            drainOrAbort();
            cudaStreamDestroy(stream);
        }
    }
    nvcompBatchedCRC32Opts_t getOptions(size_t messages, size_t largest) {
        for (const auto& entry : options)
            if (entry.messages == messages && entry.largest == largest)
                return entry.opts;
        nvcompBatchedCRC32Opts_t opts{};
        opts.spec = nvcompCRC32_C;
        checkNvcomp(nvcompBatchedCRC32GetHeuristicConf(nullptr, messages, &opts.kernel_conf, largest, stream));
        options[next_option] = {messages, largest, opts};
        next_option          = (next_option + 1) % max_items;
        return opts;
    }
    void invalidateMatrices() noexcept {
        matrix_shapes.clear();
        std::fill(matrix_payloads.begin(), matrix_payloads.end(), size_t(0));
        next_matrix_slot = 0;
    }
    void prepareMatrices(const std::vector<CrcCopyItem>& batch) {
        std::fill(protected_matrix_slots.begin(), protected_matrix_slots.end(), 0);
        // Protect every existing hit before allocating misses: the first new
        // shape must not evict a slot required by a later item in this batch.
        for (size_t index = 0; index < batch.size(); ++index) {
            const auto entry         = matrix_shapes.find(batch[index].payload_bytes);
            item_matrix_slots[index] = entry == matrix_shapes.end() ? max_items : entry->second;
            if (entry != matrix_shapes.end())
                protected_matrix_slots[entry->second] = 1;
        }
        for (size_t index = 0; index < batch.size(); ++index) {
            if (item_matrix_slots[index] != max_items)
                continue;
            const size_t payload = batch[index].payload_bytes;
            const auto   entry   = matrix_shapes.find(payload);
            if (entry != matrix_shapes.end()) {
                // Another item in this batch installed the same shape.
                item_matrix_slots[index] = entry->second;
                continue;
            }
            size_t slot = max_items;
            for (size_t attempt = 0; attempt < max_items; ++attempt) {
                const size_t candidate = next_matrix_slot;
                next_matrix_slot       = (next_matrix_slot + 1) % max_items;
                if (!protected_matrix_slots[candidate]) {
                    slot = candidate;
                    break;
                }
            }
            if (slot == max_items)
                throw std::logic_error("CRC matrix cache cannot hold current batch");
            if (matrix_payloads[slot])
                matrix_shapes.erase(matrix_payloads[slot]);
            const size_t offset = slot * matrix_slot_words;
            auto*        source = host_matrices.as<uint32_t>() + offset;
            crc_block_copy_internal::makeSuffixMatrices(payload, source);
            const size_t bytes = crc_block_copy_internal::segmentCount(payload) * 32 * sizeof(uint32_t);
            checkCuda(cudaMemcpyAsync(matrices.as<uint32_t>() + offset, source, bytes, cudaMemcpyHostToDevice, stream));
            matrix_shapes.emplace(payload, slot);
            matrix_payloads[slot]        = payload;
            protected_matrix_slots[slot] = 1;
            item_matrix_slots[index]     = slot;
        }
        // Slots remain protected until this call's final stream completion.
        // Any exceptional exit drains first and invalidates all cache entries,
        // including uploads which were enqueued but never confirmed complete.
    }
    void reportSegmentedSuccess(size_t count, size_t messages, bool store, bool scatter) {
        const size_t operation = store ? 0 : (scatter ? 1 : 2);
        if (!reported_segmented_success[operation]) {
            std::fprintf(stderr,
                         "CRC segmented op=%s backings=%zu segments=%zu segment_bytes=%zu\n",
                         store ? "store" : (scatter ? "load" : "validate"),
                         count,
                         messages,
                         crc_block_copy_internal::kSegmentBytes);
            reported_segmented_success[operation] = true;
        }
    }
    bool valid(const std::vector<CrcCopyItem>& batch, bool need_tiles) const {
        if (batch.empty() || batch.size() > max_items)
            return false;
        size_t total_tiles = 0;
        for (size_t index = 0; index < batch.size(); ++index) {
            const auto& item = batch[index];
            if (!item.host || !item.payload_bytes || item.payload_bytes > max_payload)
                return false;
            const size_t    encoded = CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
            const uintptr_t address = reinterpret_cast<uintptr_t>(item.host);
            if (item.capacity_bytes < encoded || address > std::numeric_limits<uintptr_t>::max() - encoded)
                return false;
            // Overlapping destinations can invalidate an earlier item's footer.
            for (size_t previous = 0; previous < index; ++previous) {
                const uintptr_t other       = reinterpret_cast<uintptr_t>(batch[previous].host);
                const size_t    other_bytes = CrcBlockCopyBatch::encodedBytes(batch[previous].payload_bytes);
                if (address < other + other_bytes && other < address + encoded)
                    return false;
            }
            if (!need_tiles)
                continue;
            if (item.tiles.empty() || item.tiles.size() > max_tiles - total_tiles)
                return false;
            total_tiles += item.tiles.size();
            size_t end = 0;
            for (const auto& tile : item.tiles) {
                if (!tile.device || !tile.bytes || tile.offset != end || tile.bytes > item.payload_bytes - end
                    || reinterpret_cast<uintptr_t>(tile.device) > std::numeric_limits<uintptr_t>::max() - tile.bytes)
                    return false;
                end += tile.bytes;
            }
            if (end != item.payload_bytes)
                return false;
        }
        return true;
    }
    CrcCopyStatus run(const std::vector<CrcCopyItem>& batch, bool store, bool scatter) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!valid(batch, store || scatter))
            return CrcCopyStatus::INVALID_ARGS;
        try {
            checkCuda(cudaSetDevice(device));
            const size_t count   = batch.size();
            size_t       largest = 0;
            for (const auto& item : batch)
                largest = std::max(largest, item.payload_bytes);
            const bool segmented = crc_block_copy_internal::shouldSegment(count, largest);
            if (segmented)
                prepareMatrices(batch);
            size_t tile_count = 0, messages = 0, largest_message = 0;
            for (size_t index = 0; index < count; ++index) {
                const auto&  item     = batch[index];
                const size_t segments = segmented ? crc_block_copy_internal::segmentCount(item.payload_bytes) : 1;
                if (segments > max_messages - messages)
                    throw std::logic_error("CRC message capacity exceeded");
                if (segmented)
                    host_backings.as<BackingInfo>()[index] = {
                        item.payload_bytes, messages, segments, item_matrix_slots[index] * matrix_slot_words};
                size_t offset = 0;
                for (size_t segment = 0; segment < segments; ++segment) {
                    const size_t bytes =
                        segmented ? std::min(item.payload_bytes - offset, crc_block_copy_internal::kSegmentBytes) :
                                    item.payload_bytes;
                    host_inputs.as<const void*>()[messages] = staging.as<unsigned char>() + index * stride + offset;
                    host_lengths.as<size_t>()[messages]     = bytes;
                    largest_message                         = std::max(largest_message, bytes);
                    offset += bytes;
                    ++messages;
                }
                if (store || scatter)
                    for (const auto& tile : item.tiles)
                        host_tiles.as<CopyTile>()[tile_count++] = {
                            tile.device, index * stride + tile.offset, tile.bytes};
            }
            const auto opts = getOptions(messages, largest_message);
            checkCuda(cudaMemcpyAsync(
                inputs.pointer, host_inputs.pointer, messages * sizeof(void*), cudaMemcpyHostToDevice, stream));
            checkCuda(cudaMemcpyAsync(
                lengths.pointer, host_lengths.pointer, messages * sizeof(size_t), cudaMemcpyHostToDevice, stream));
            if (tile_count)
                checkCuda(cudaMemcpyAsync(
                    tiles.pointer, host_tiles.pointer, tile_count * sizeof(CopyTile), cudaMemcpyHostToDevice, stream));
            if (segmented)
                checkCuda(cudaMemcpyAsync(backings.pointer,
                                          host_backings.pointer,
                                          count * sizeof(BackingInfo),
                                          cudaMemcpyHostToDevice,
                                          stream));
            for (size_t index = 0; index < count; ++index) {
                const auto&  item    = batch[index];
                const size_t encoded = CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
                auto*        packed  = staging.as<unsigned char>() + index * stride;
                if (!store)
                    checkCuda(cudaMemcpyAsync(packed, item.host, encoded, cudaMemcpyHostToDevice, stream));
            }
            if (store) {
                checkCuda(launchCopyTiles(tiles.as<CopyTile>(), tile_count, staging.pointer, false, stream));
            }
            checkNvcomp(nvcompBatchedCRC32Async(inputs.as<const void*>(),
                                                lengths.as<size_t>(),
                                                messages,
                                                checksums.as<uint32_t>(),
                                                opts,
                                                nvcompCRC32OnlySegment,
                                                statuses.as<nvcompStatus_t>(),
                                                stream));
            if (segmented) {
                checkCuda(crc_block_copy_internal::launchCombineAndFinish(staging.as<unsigned char>(),
                                                                          stride,
                                                                          backings.as<BackingInfo>(),
                                                                          matrices.as<uint32_t>(),
                                                                          checksums.as<uint32_t>(),
                                                                          statuses.as<nvcompStatus_t>(),
                                                                          results.as<Result>(),
                                                                          count,
                                                                          store,
                                                                          stream));
            } else {
                finishCrc<<<(count + 255) / 256, 256, 0, stream>>>(staging.as<unsigned char>(),
                                                                   stride,
                                                                   lengths.as<size_t>(),
                                                                   checksums.as<uint32_t>(),
                                                                   statuses.as<nvcompStatus_t>(),
                                                                   results.as<Result>(),
                                                                   count,
                                                                   store);
                checkCuda(cudaGetLastError());
            }
            if (store)
                for (size_t index = 0; index < count; ++index)
                    checkCuda(cudaMemcpyAsync(batch[index].host,
                                              staging.as<unsigned char>() + index * stride,
                                              CrcBlockCopyBatch::encodedBytes(batch[index].payload_bytes),
                                              cudaMemcpyDeviceToHost,
                                              stream));
            checkCuda(cudaMemcpyAsync(
                host_results.pointer, results.pointer, count * sizeof(Result), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaStreamSynchronize(stream));
            CrcCopyStatus result = CrcCopyStatus::OK;
            for (size_t index = 0; index < count; ++index) {
                const auto& observed = host_results.as<Result>()[index];
                if (observed.status != nvcompSuccess || (!store && observed.expected != observed.actual)) {
                    std::fprintf(stderr,
                                 "CRC item=%zu payload=%zu expected=%08x actual=%08x nvcomp_status=%d\n",
                                 index,
                                 batch[index].payload_bytes,
                                 observed.expected,
                                 observed.actual,
                                 int(observed.status));
                    if (observed.status != nvcompSuccess)
                        result = CrcCopyStatus::CRC_COMPUTE_ERROR;
                    else if (result == CrcCopyStatus::OK)
                        result = CrcCopyStatus::CRC_MISMATCH;
                }
            }
            if (result != CrcCopyStatus::OK)
                return result;
            if (!scatter) {
                if (segmented)
                    reportSegmentedSuccess(count, messages, store, scatter);
                return result;
            }
            // All verdicts reached the CPU before any destination can be written.
            checkCuda(launchCopyTiles(tiles.as<CopyTile>(), tile_count, staging.pointer, true, stream));
            checkCuda(cudaStreamSynchronize(stream));
            if (segmented)
                reportSegmentedSuccess(count, messages, store, scatter);
            return CrcCopyStatus::OK;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CRC device execution failed: %s\n", error.what());
            drainOrAbort();
            invalidateMatrices();
            return CrcCopyStatus::DEVICE_ERROR;
        } catch (...) {
            drainOrAbort();
            invalidateMatrices();
            return CrcCopyStatus::DEVICE_ERROR;
        }
    }
};

bool CrcBlockCopyBatch::available() {
    return true;
}
CrcBlockCopyBatch::CrcBlockCopyBatch(int device, size_t items, size_t payload, size_t tiles):
    impl_(std::make_unique<Impl>(device, items, payload, tiles)) {}
CrcBlockCopyBatch::~CrcBlockCopyBatch() = default;
CrcCopyStatus CrcBlockCopyBatch::store(const std::vector<CrcCopyItem>& items) {
    return impl_->run(items, true, false);
}
CrcCopyStatus CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>& items) {
    return impl_->run(items, false, true);
}
CrcCopyStatus CrcBlockCopyBatch::validate(const std::vector<CrcCopyItem>& items) {
    return impl_->run(items, false, false);
}
}  // namespace rtp_llm
