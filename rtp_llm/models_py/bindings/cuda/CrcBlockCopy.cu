#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace {

// A CUDA failure can leave device work in flight. Do not unwind and return
// its source, destination, or workspace to an allocator in that state.
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "memory cache CRC CUDA failure: %s\n", cudaGetErrorString(error));
        std::abort();
    }
}
void checkNvcomp(nvcompStatus_t status) {
    if (status != nvcompSuccess) {
        std::fprintf(stderr, "memory cache CRC launch failure: %d\n", static_cast<int>(status));
        std::abort();
    }
}

struct DeviceResult {
    uint32_t       crc;
    nvcompStatus_t status;
    CrcBlockFooter footer;
    uint32_t       valid;
};

__device__ void copyBytes(unsigned char* dst, const unsigned char* src, size_t bytes) {
    if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
        for (size_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) {
            reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
        }
        for (size_t i = bytes / 16 * 16 + threadIdx.x; i < bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

__global__ void copyTiles(const CrcBlockCopyTile* tiles, int count, unsigned char* staging, bool to_device) {
    if (blockIdx.x >= count)
        return;
    const auto tile = tiles[blockIdx.x];
    if (!tile.is_cuda)
        return;
    auto* pool   = static_cast<unsigned char*>(tile.address);
    auto* packed = staging + tile.offset;
    copyBytes(to_device ? pool : packed, to_device ? packed : pool, tile.bytes);
}

__global__ void validateBlock(const CrcBlockFooter* footer, DeviceResult* result) {
    result->footer = *footer;
    result->valid  = result->status == nvcompSuccess && result->crc == footer->crc32c;
}

}  // namespace

struct CrcBlockCopy::Impl {
    struct HostCopy {
        void*  host;
        size_t offset, width, rows, host_pitch, staging_pitch;
    };
    int                                        device{-1};
    cudaStream_t                               stream{nullptr};
    size_t                                     capacity{0};
    size_t                                     tile_capacity{0};
    unsigned char*                             staging{nullptr};
    unsigned char*                             host_staging{nullptr};
    CrcBlockCopyTile*                          host_tiles{nullptr};
    CrcBlockCopyTile*                          device_tiles{nullptr};
    CrcBlockFooter*                            host_footer{nullptr};
    const void**                               input{nullptr};
    size_t*                                    length{nullptr};
    size_t*                                    host_length{nullptr};
    DeviceResult*                              device_result{nullptr};
    DeviceResult*                              host_result{nullptr};
    std::map<size_t, nvcompBatchedCRC32Opts_t> options;
    int                                        max_pitch{0};
    std::vector<HostCopy>                      host_copies;
    std::vector<size_t>                        host_candidates;
    std::vector<uint8_t>                       host_used;

    size_t planHostCopies(const std::vector<CrcBlockCopyTile>& tiles) {
        std::fill(host_used.begin(), host_used.begin() + tiles.size(), 0);
        size_t run_count = 0;
        for (size_t first = 0; first < tiles.size(); ++first) {
            if (tiles[first].is_cuda || host_used[first])
                continue;
            const auto&         tile             = tiles[first];
            CUdeviceptr         allocation_base  = 0;
            size_t              allocation_bytes = 0;
            CUmemorytype        memory_type{};
            CUpointer_attribute attributes[] = {CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                                CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                                                CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
            void*               values[]     = {&allocation_base, &allocation_bytes, &memory_type};
            const auto          query =
                cuPointerGetAttributes(3, attributes, values, reinterpret_cast<CUdeviceptr>(tile.address));
            if (query != CUDA_SUCCESS) {
                std::fprintf(stderr, "memory cache CRC host allocation query failed: %d\n", static_cast<int>(query));
                std::abort();
            }
            const auto inside_allocation = [&](const CrcBlockCopyTile& candidate) {
                const auto address = reinterpret_cast<uintptr_t>(candidate.address);
                return memory_type == CU_MEMORYTYPE_HOST && address >= allocation_base
                       && candidate.bytes <= allocation_bytes
                       && address - allocation_base <= allocation_bytes - candidate.bytes;
            };
            if (!inside_allocation(tile)) {
                host_copies[run_count++] = {tile.address, tile.offset, tile.bytes, 1, 0, 0};
                host_used[first]         = 1;
                continue;
            }
            size_t candidates = 0;
            for (size_t index = first; index < tiles.size(); ++index) {
                if (!tiles[index].is_cuda && !host_used[index] && tiles[index].bytes == tile.bytes
                    && inside_allocation(tiles[index]))
                    host_candidates[candidates++] = index;
            }
            HostCopy run{tile.address, tile.offset, tile.bytes, 1, 0, 0};
            size_t   best_step = 1;
            // Adjacent rows handle state pools; alternate rows handle SWA slots
            // interleaved with CSA/HCA. Validate every actual address and offset.
            for (size_t step : {size_t(1), size_t(2)}) {
                if (candidates <= step)
                    continue;
                const auto& second = tiles[host_candidates[step]];
                const auto  base   = reinterpret_cast<uintptr_t>(tile.address);
                const auto  next   = reinterpret_cast<uintptr_t>(second.address);
                if (next <= base)
                    continue;
                const size_t host_pitch    = next - base;
                const size_t staging_pitch = second.offset - tile.offset;
                if (host_pitch < tile.bytes || staging_pitch < tile.bytes || host_pitch > static_cast<size_t>(max_pitch)
                    || staging_pitch > static_cast<size_t>(max_pitch))
                    continue;
                size_t rows = 1;
                for (size_t pos = step; pos < candidates; pos += step) {
                    const auto& candidate = tiles[host_candidates[pos]];
                    if (reinterpret_cast<uintptr_t>(candidate.address) != base + rows * host_pitch
                        || candidate.offset != tile.offset + rows * staging_pitch)
                        break;
                    ++rows;
                }
                if (rows > run.rows) {
                    run.rows          = rows;
                    run.host_pitch    = host_pitch;
                    run.staging_pitch = staging_pitch;
                    best_step         = step;
                }
            }
            for (size_t row = 0; row < run.rows; ++row)
                host_used[host_candidates[row * best_step]] = 1;
            host_copies[run_count++] = run;
        }
        return run_count;
    }

    void copyHostTiles(size_t count, bool to_device) {
        for (size_t index = 0; index < count; ++index) {
            const auto& run       = host_copies[index];
            auto*       packed    = staging + run.offset;
            const auto  direction = to_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if (run.rows == 1) {
                checkCuda(cudaMemcpyAsync(
                    to_device ? run.host : packed, to_device ? packed : run.host, run.width, direction, stream));
            } else {
                checkCuda(cudaMemcpy2DAsync(to_device ? run.host : packed,
                                            to_device ? run.host_pitch : run.staging_pitch,
                                            to_device ? packed : run.host,
                                            to_device ? run.staging_pitch : run.host_pitch,
                                            run.width,
                                            run.rows,
                                            direction,
                                            stream));
            }
        }
    }

    void checkPayload(size_t bytes) const {
        if (bytes == 0 || options.find(bytes) == options.end())
            throw std::invalid_argument("CRC workspace payload size mismatch");
    }

    size_t prepareTiles(size_t bytes, const std::vector<CrcBlockCopyTile>& tiles, bool& complete, bool& has_device) {
        checkPayload(bytes);
        if (tiles.empty() || tiles.size() > tile_capacity)
            throw std::invalid_argument("CRC workspace tile count mismatch");
        size_t covered  = 0;
        bool   has_host = false;
        complete        = true;
        has_device      = false;
        for (const auto& tile : tiles) {
            if (!tile.address || tile.bytes == 0 || tile.offset < covered || tile.offset > bytes
                || tile.bytes > bytes - tile.offset)
                throw std::invalid_argument("invalid CRC copy tile");
            complete   = complete && tile.offset == covered;
            covered    = tile.offset + tile.bytes;
            has_host   = has_host || !tile.is_cuda;
            has_device = has_device || tile.is_cuda;
        }
        complete = complete && covered == bytes;
        checkCuda(cudaSetDevice(device));
        const size_t host_count = has_host ? planHostCopies(tiles) : 0;
        if (has_device) {
            std::copy(tiles.begin(), tiles.end(), host_tiles);
            checkCuda(cudaMemcpyAsync(
                device_tiles, host_tiles, tiles.size() * sizeof(CrcBlockCopyTile), cudaMemcpyHostToDevice, stream));
        }
        return host_count;
    }

    void calculateCrc(size_t bytes, uint32_t* output) {
        *host_length = bytes;
        checkCuda(cudaMemcpyAsync(length, host_length, sizeof(size_t), cudaMemcpyHostToDevice, stream));
        checkNvcomp(nvcompBatchedCRC32Async(
            input, length, 1, output, options.at(bytes), nvcompCRC32OnlySegment, &device_result->status, stream));
    }

    void captureFailure(CrcBlockCopyResult& result, size_t bytes, const std::function<bool()>& capture) {
        try {
            if (result.success || !capture || !capture())
                return;
            result.staging_snapshot.resize(bytes);
            checkCuda(cudaMemcpyAsync(result.staging_snapshot.data(), staging, bytes, cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaMemcpyAsync(host_footer,
                                      staging + CrcBlockCopy::footerOffset(bytes),
                                      sizeof(CrcBlockFooter),
                                      cudaMemcpyDeviceToHost,
                                      stream));
            checkCuda(cudaStreamSynchronize(stream));
            result.staging_footer = *host_footer;
        } catch (const std::bad_alloc&) {
            result.staging_snapshot.clear();
        }
    }

    ~Impl() {
        if (device < 0) {
            return;
        }
        checkCuda(cudaSetDevice(device));
        if (stream) {
            checkCuda(cudaStreamSynchronize(stream));
        }
        cudaFree(staging);
        cudaFreeHost(host_staging);
        cudaFree(device_tiles);
        cudaFree(input);
        cudaFree(length);
        cudaFree(device_result);
        cudaFreeHost(host_tiles);
        cudaFreeHost(host_footer);
        cudaFreeHost(host_length);
        cudaFreeHost(host_result);
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

bool CrcBlockCopy::supported() {
    return true;
}

CrcBlockCopy::CrcBlockCopy(const std::vector<size_t>& sizes, size_t max_tiles): impl_(std::make_unique<Impl>()) {
    if (sizes.empty() || max_tiles == 0) {
        throw std::invalid_argument("CRC workspace requires payload sizes and tiles");
    }
    auto& work = *impl_;
    checkCuda(cudaGetDevice(&work.device));
    checkCuda(cudaDeviceGetAttribute(&work.max_pitch, cudaDevAttrMaxPitch, work.device));
    work.capacity      = storageBytes(*std::max_element(sizes.begin(), sizes.end()));
    work.tile_capacity = max_tiles;
    work.host_copies.resize(max_tiles);
    work.host_candidates.resize(max_tiles);
    work.host_used.resize(max_tiles);
    checkCuda(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking));
    checkCuda(cudaMalloc(&work.staging, work.capacity));
    checkCuda(cudaMallocHost(&work.host_staging, work.capacity));
    checkCuda(cudaMallocHost(&work.host_tiles, max_tiles * sizeof(CrcBlockCopyTile)));
    checkCuda(cudaMalloc(&work.device_tiles, max_tiles * sizeof(CrcBlockCopyTile)));
    checkCuda(cudaMallocHost(&work.host_footer, sizeof(CrcBlockFooter)));
    checkCuda(cudaMalloc(&work.input, sizeof(void*)));
    checkCuda(cudaMalloc(&work.length, sizeof(size_t)));
    checkCuda(cudaMallocHost(&work.host_length, sizeof(size_t)));
    checkCuda(cudaMalloc(&work.device_result, sizeof(DeviceResult)));
    checkCuda(cudaMallocHost(&work.host_result, sizeof(DeviceResult)));
    checkCuda(cudaMemcpyAsync(work.input, &work.staging, sizeof(void*), cudaMemcpyHostToDevice, work.stream));
    for (const auto bytes : sizes) {
        if (bytes == 0 || work.options.count(bytes)) {
            continue;
        }
        nvcompBatchedCRC32Opts_t opts{};
        opts.spec = nvcompCRC32_C;
        checkNvcomp(nvcompBatchedCRC32GetHeuristicConf(nullptr, 1, &opts.kernel_conf, bytes, work.stream));
        work.options.emplace(bytes, opts);
        // Warm up the exact single-message configuration without touching a cache entry.
        checkCuda(cudaMemsetAsync(work.staging, 0, storageBytes(bytes), work.stream));
        *work.host_length = bytes;
        checkCuda(cudaMemcpyAsync(work.length, work.host_length, sizeof(size_t), cudaMemcpyHostToDevice, work.stream));
        checkNvcomp(nvcompBatchedCRC32Async(work.input,
                                            work.length,
                                            1,
                                            &work.device_result->crc,
                                            opts,
                                            nvcompCRC32OnlySegment,
                                            &work.device_result->status,
                                            work.stream));
        checkCuda(cudaStreamSynchronize(work.stream));
    }
}

CrcBlockCopy::~CrcBlockCopy() = default;

void CrcBlockCopy::gather(size_t bytes, const std::vector<CrcBlockCopyTile>& tiles, bool preserve_payload) {
    auto&        work     = *impl_;
    bool         complete = false, has_device = false;
    const size_t host_count = work.prepareTiles(bytes, tiles, complete, has_device);
    if (!preserve_payload) {
        const size_t clear_begin = complete ? bytes : 0;
        if (clear_begin < footerOffset(bytes))
            checkCuda(cudaMemsetAsync(work.staging + clear_begin, 0, footerOffset(bytes) - clear_begin, work.stream));
    }
    work.copyHostTiles(host_count, false);
    if (has_device) {
        copyTiles<<<tiles.size(), 256, 0, work.stream>>>(work.device_tiles, tiles.size(), work.staging, false);
        checkCuda(cudaGetLastError());
    }
}

CrcBlockCopyResult CrcBlockCopy::store(
    void* host, size_t bytes, bool host_is_pinned, const std::function<bool()>& capture_failure) {
    auto& work = *impl_;
    work.checkPayload(bytes);
    if (!host)
        throw std::invalid_argument("null CRC output block");
    checkCuda(cudaSetDevice(work.device));
    auto* footer = reinterpret_cast<CrcBlockFooter*>(work.staging + footerOffset(bytes));
    work.calculateCrc(bytes, &footer->crc32c);
    checkCuda(cudaMemcpyAsync(host_is_pinned ? host : work.host_staging,
                              work.staging,
                              storageBytes(bytes),
                              cudaMemcpyDeviceToHost,
                              work.stream));
    checkCuda(cudaMemcpyAsync(&work.host_result->status,
                              &work.device_result->status,
                              sizeof(nvcompStatus_t),
                              cudaMemcpyDeviceToHost,
                              work.stream));
    checkCuda(cudaStreamSynchronize(work.stream));
    if (!host_is_pinned)
        std::memcpy(host, work.host_staging, storageBytes(bytes));
    CrcBlockCopyResult result;
    result.success        = work.host_result->status == nvcompSuccess;
    result.output_written = true;
    result.gpu_crc_status = static_cast<uint32_t>(work.host_result->status);
    result.failure_stage  = CrcBlockCopyResult::FailureStage::CRC_COMPUTE;
    work.captureFailure(result, bytes, capture_failure);
    return result;
}

CrcBlockCopyResult CrcBlockCopy::loadAndValidate(const void*                  host,
                                                 size_t                       bytes,
                                                 bool                         host_is_pinned,
                                                 const std::function<bool()>& capture_failure) {
    auto& work = *impl_;
    work.checkPayload(bytes);
    if (!host)
        throw std::invalid_argument("null CRC input block");
    checkCuda(cudaSetDevice(work.device));
    const void* input = host;
    if (!host_is_pinned) {
        std::memcpy(work.host_staging, host, storageBytes(bytes));
        input = work.host_staging;
    }
    checkCuda(cudaMemcpyAsync(work.staging, input, storageBytes(bytes), cudaMemcpyHostToDevice, work.stream));
    work.calculateCrc(bytes, &work.device_result->crc);
    const auto* footer = reinterpret_cast<const CrcBlockFooter*>(work.staging + footerOffset(bytes));
    validateBlock<<<1, 1, 0, work.stream>>>(footer, work.device_result);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpyAsync(
        work.host_result, work.device_result, sizeof(DeviceResult), cudaMemcpyDeviceToHost, work.stream));
    checkCuda(cudaStreamSynchronize(work.stream));
    CrcBlockCopyResult result;
    result.success          = work.host_result->valid != 0;
    result.expected_crc     = work.host_result->footer.crc32c;
    result.actual_crc       = work.host_result->crc;
    result.checked_footer   = work.host_result->footer;
    result.gpu_crc_observed = true;
    result.gpu_crc_status   = static_cast<uint32_t>(work.host_result->status);
    result.failure_stage = work.host_result->status != nvcompSuccess ? CrcBlockCopyResult::FailureStage::CRC_COMPUTE :
                                                                       CrcBlockCopyResult::FailureStage::SOURCE_CRC;
    work.captureFailure(result, bytes, capture_failure);
    return result;
}

void CrcBlockCopy::scatter(size_t bytes, const std::vector<CrcBlockCopyTile>& tiles) {
    auto&        work     = *impl_;
    bool         complete = false, has_device = false;
    const size_t host_count = work.prepareTiles(bytes, tiles, complete, has_device);
    if (has_device) {
        copyTiles<<<tiles.size(), 256, 0, work.stream>>>(work.device_tiles, tiles.size(), work.staging, true);
        checkCuda(cudaGetLastError());
    }
    work.copyHostTiles(host_count, true);
    checkCuda(cudaStreamSynchronize(work.stream));
}

}  // namespace rtp_llm
