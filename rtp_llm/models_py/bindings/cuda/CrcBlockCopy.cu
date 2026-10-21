#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"
#include "rtp_llm/models_py/bindings/cuda/CrcBlockCopyInternal.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace {
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
                          bool*                 failed,
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
    const bool failure = status[index] != nvcompSuccess || (!store && *footer != crc[index]);
    if (failure)
        results[index] = {*footer, crc[index], status[index]};
    failed[index] = failure;
}

}  // namespace

namespace crc_block_copy_internal {
cudaError_t launchFinishCrc(unsigned char*        staging,
                            size_t                stride,
                            const size_t*         lengths,
                            const uint32_t*       checksums,
                            const nvcompStatus_t* statuses,
                            Result*               results,
                            bool*                 failed,
                            size_t                count,
                            bool                  store,
                            cudaStream_t          stream) {
    finishCrc<<<(count + 255) / 256, 256, 0, stream>>>(
        staging, stride, lengths, checksums, statuses, results, failed, count, store);
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
    size_t       device_bytes{0};
    cudaStream_t stream{nullptr};
    Allocation   staging, tiles, inputs, lengths, checksums, statuses, results;
    Allocation   host_tiles, host_inputs, host_lengths, host_results, host_failed;
    bool*        device_failed{nullptr};
    std::vector<Configuration>     options;
    size_t                         next_option{0};
    std::mutex                     mutex;
    std::shared_ptr<CrcDumpWriter> dump_writer;

    Impl(int device_index, size_t items, size_t payload, size_t tile_count, std::shared_ptr<CrcDumpWriter> writer):
        device(device_index),
        max_items(items),
        max_payload(payload),
        max_tiles(tile_count),
        stride(CrcBlockCopyBatch::encodedBytes(payload)),
        dump_writer(std::move(writer)) {
        if (device < 0 || !items || !payload || !tile_count || items > unsigned(std::numeric_limits<int>::max())
            || tile_count > unsigned(std::numeric_limits<int>::max()))
            throw std::invalid_argument("invalid CRC capacity");
        // Check every capacity before allocating anything on the GPU.
        const size_t staging_bytes  = checkedProduct(items, stride);
        const size_t tile_bytes     = checkedProduct(tile_count, sizeof(CopyTile));
        const size_t input_bytes    = checkedProduct(items, sizeof(void*));
        const size_t length_bytes   = checkedProduct(items, sizeof(size_t));
        const size_t checksum_bytes = checkedProduct(items, sizeof(uint32_t));
        const size_t status_bytes   = checkedProduct(items, sizeof(nvcompStatus_t));
        const size_t result_bytes   = checkedProduct(items, sizeof(Result));
        options.resize(items);
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
        host_failed.pinned = true;
        checkCuda(cudaHostAlloc(&host_failed.pointer, items * sizeof(bool), cudaHostAllocMapped));
        checkCuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_failed), host_failed.pointer, 0));
        // Create the stream last: no constructor-failure path can strand work.
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        device_bytes =
            staging_bytes + tile_bytes + input_bytes + length_bytes + checksum_bytes + status_bytes + result_bytes;
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

    bool valid(const std::vector<CrcCopyItem>& batch) const {
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

    CrcCopyFailure captureSnapshot(const CrcCopyItem& item, const CrcCopyFailure& failure) {
        auto snapshot = failure;
        try {
            const size_t encoded = CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
            const auto*  host    = static_cast<const uint8_t*>(item.host);
            snapshot.cpu_snapshot.assign(host, host + encoded);
            snapshot.staging_snapshot.resize(encoded);
            checkCuda(cudaMemcpy(snapshot.staging_snapshot.data(),
                                 staging.as<unsigned char>() + failure.item_index * stride,
                                 encoded,
                                 cudaMemcpyDeviceToHost));
        } catch (const std::exception& error) {
            snapshot.staging_snapshot.clear();
            snapshot.capture_error = error.what();
        } catch (...) {
            snapshot.staging_snapshot.clear();
            snapshot.capture_error = "unknown CRC snapshot error";
        }
        return snapshot;
    }

    struct PreparedBatch {
        size_t count{0};
        size_t tile_count{0};
        size_t largest_payload{0};
    };

    PreparedBatch prepareBatch(const std::vector<CrcCopyItem>& batch) {
        checkCuda(cudaSetDevice(device));
        PreparedBatch prepared;
        prepared.count = batch.size();
        for (size_t index = 0; index < prepared.count; ++index) {
            const auto& item                     = batch[index];
            host_inputs.as<const void*>()[index] = staging.as<unsigned char>() + index * stride;
            host_lengths.as<size_t>()[index]     = item.payload_bytes;
            prepared.largest_payload             = std::max(prepared.largest_payload, item.payload_bytes);
            for (const auto& tile : item.tiles)
                host_tiles.as<CopyTile>()[prepared.tile_count++] = {
                    tile.device, index * stride + tile.offset, tile.bytes};
        }
        checkCuda(cudaMemcpyAsync(
            inputs.pointer, host_inputs.pointer, prepared.count * sizeof(void*), cudaMemcpyHostToDevice, stream));
        checkCuda(cudaMemcpyAsync(
            lengths.pointer, host_lengths.pointer, prepared.count * sizeof(size_t), cudaMemcpyHostToDevice, stream));
        checkCuda(cudaMemcpyAsync(
            tiles.pointer, host_tiles.pointer, prepared.tile_count * sizeof(CopyTile), cudaMemcpyHostToDevice, stream));
        return prepared;
    }

    void computeCrc(const PreparedBatch& batch, bool write_footer) {
        const auto opts = getOptions(batch.count, batch.largest_payload);
        checkNvcomp(nvcompBatchedCRC32Async(inputs.as<const void*>(),
                                            lengths.as<size_t>(),
                                            batch.count,
                                            checksums.as<uint32_t>(),
                                            opts,
                                            nvcompCRC32OnlySegment,
                                            statuses.as<nvcompStatus_t>(),
                                            stream));
        checkCuda(crc_block_copy_internal::launchFinishCrc(staging.as<unsigned char>(),
                                                           stride,
                                                           lengths.as<size_t>(),
                                                           checksums.as<uint32_t>(),
                                                           statuses.as<nvcompStatus_t>(),
                                                           results.as<Result>(),
                                                           device_failed,
                                                           batch.count,
                                                           write_footer,
                                                           stream));
    }

    CrcCopyStatus collectResults(const std::vector<CrcCopyItem>& batch,
                                 bool                            checked_footer_available,
                                 std::vector<CrcCopyFailure>*    failures) {
        // The slot owns the mapped flags. Only read them after the GPU has finished
        // this stream; every active item overwrites its flag on each invocation.
        checkCuda(cudaStreamSynchronize(stream));
        CrcCopyStatus result = CrcCopyStatus::OK;
        for (size_t index = 0; index < batch.size(); ++index) {
            if (!host_failed.as<bool>()[index])
                continue;
            // Full diagnostics stay on the GPU until a failure needs them.
            checkCuda(cudaMemcpyAsync(host_results.as<Result>() + index,
                                      results.as<Result>() + index,
                                      sizeof(Result),
                                      cudaMemcpyDeviceToHost,
                                      stream));
            checkCuda(cudaStreamSynchronize(stream));
            const auto& observed = host_results.as<Result>()[index];
            const auto failure_status =
                observed.status != nvcompSuccess ? CrcCopyStatus::CRC_COMPUTE_ERROR : CrcCopyStatus::CRC_MISMATCH;
            if (result == CrcCopyStatus::OK || failure_status == CrcCopyStatus::CRC_COMPUTE_ERROR)
                result = failure_status;
            CrcCopyFailure failure{index, observed.expected, observed.actual, failure_status, int(observed.status)};
            failure.checked_footer_available = checked_footer_available;
            if (failures)
                failures->push_back(failure);
            if (dump_writer)
                dump_writer->write(batch[index], failure, [&] { return captureSnapshot(batch[index], failure); });
        }
        return result;
    }

    CrcCopyStatus store(const std::vector<CrcCopyItem>& batch, std::vector<CrcCopyFailure>* failures) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failures)
            failures->clear();
        if (!valid(batch))
            return CrcCopyStatus::INVALID_ARGS;
        try {
            if (failures)
                failures->reserve(batch.size());
            const auto prepared = prepareBatch(batch);
            checkCuda(launchCopyTiles(tiles.as<CopyTile>(), prepared.tile_count, staging.pointer, false, stream));
            computeCrc(prepared, /*write_footer=*/true);
            for (size_t index = 0; index < batch.size(); ++index)
                checkCuda(cudaMemcpyAsync(batch[index].host,
                                          staging.as<unsigned char>() + index * stride,
                                          CrcBlockCopyBatch::encodedBytes(batch[index].payload_bytes),
                                          cudaMemcpyDeviceToHost,
                                          stream));
            return collectResults(batch, /*checked_footer_available=*/false, failures);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CRC device execution failed: %s\n", error.what());
            drainOrAbort();
            return CrcCopyStatus::DEVICE_ERROR;
        } catch (...) {
            drainOrAbort();
            return CrcCopyStatus::DEVICE_ERROR;
        }
    }

    CrcCopyStatus load(const std::vector<CrcCopyItem>& batch, std::vector<CrcCopyFailure>* failures) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failures)
            failures->clear();
        if (!valid(batch))
            return CrcCopyStatus::INVALID_ARGS;
        try {
            if (failures)
                failures->reserve(batch.size());
            const auto prepared = prepareBatch(batch);
            for (size_t index = 0; index < batch.size(); ++index)
                checkCuda(cudaMemcpyAsync(staging.as<unsigned char>() + index * stride,
                                          batch[index].host,
                                          CrcBlockCopyBatch::encodedBytes(batch[index].payload_bytes),
                                          cudaMemcpyHostToDevice,
                                          stream));
            computeCrc(prepared, /*write_footer=*/false);
            const auto result = collectResults(batch, /*checked_footer_available=*/true, failures);
            if (result != CrcCopyStatus::OK)
                return result;
            // All verdicts reached the CPU before any destination can be written.
            checkCuda(launchCopyTiles(tiles.as<CopyTile>(), prepared.tile_count, staging.pointer, true, stream));
            checkCuda(cudaStreamSynchronize(stream));
            return CrcCopyStatus::OK;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CRC device execution failed: %s\n", error.what());
            drainOrAbort();
            return CrcCopyStatus::DEVICE_ERROR;
        } catch (...) {
            drainOrAbort();
            return CrcCopyStatus::DEVICE_ERROR;
        }
    }
};

bool CrcBlockCopyBatch::available() {
    return true;
}

size_t CrcBlockCopyBatch::deviceMemoryBytes() const {
    return impl_->device_bytes;
}

CrcBlockCopyBatch::CrcBlockCopyBatch(
    int device, size_t items, size_t payload, size_t tiles, std::shared_ptr<CrcDumpWriter> dump_writer):
    impl_(std::make_unique<Impl>(device, items, payload, tiles, std::move(dump_writer))) {}

CrcBlockCopyBatch::~CrcBlockCopyBatch() = default;

CrcCopyStatus CrcBlockCopyBatch::store(const std::vector<CrcCopyItem>& items,
                                       std::vector<CrcCopyFailure>*    failures) {
    return impl_->store(items, failures);
}

CrcCopyStatus CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>& items,
                                      std::vector<CrcCopyFailure>*    failures) {
    return impl_->load(items, failures);
}

}  // namespace rtp_llm
