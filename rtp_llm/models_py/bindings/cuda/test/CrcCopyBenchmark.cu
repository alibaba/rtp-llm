// Manual end-to-end cache-copy microbenchmark. See crc_copy_benchmark_test.
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"
#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if CUDART_VERSION < 13000
#error "CrcCopyBenchmark requires --config=cuda13 (cudaMemcpy3DBatchAsync)."
#endif

namespace rtp_llm::crc_copy_benchmark {
namespace {
constexpr int           kMaxBackings     = 32;
constexpr int           kEvictMultiplier = 8;
constexpr unsigned char deviceGuard      = 0xd3;
const char*             syncNames[]{"integrated_crc", "copy1d_batch", "copy3d_batch", "main_staged", "gather_control"};

void requireSync(bool ok, std::string_view reason) {
    if (!ok)
        throw std::runtime_error(std::string(reason));
}
void checkedCuda(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line) + ": "
                                 + cudaGetErrorString(error));
    }
}
#define cu(expression) checkedCuda((expression), #expression, __LINE__)

struct Buffer {
    void* p{nullptr};
    bool  pinned;
    explicit Buffer(size_t bytes, bool host = false): pinned(host) {
        if (host)
            cu(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
        else
            cu(cudaMalloc(&p, bytes));
    }
    ~Buffer() {
        if (pinned)
            cudaFreeHost(p);
        else
            cudaFree(p);
    }
    Buffer(const Buffer&)                   = delete;
    Buffer&        operator=(const Buffer&) = delete;
    unsigned char* bytes() const {
        return static_cast<unsigned char*>(p);
    }
};
struct Stream {
    cudaStream_t s{};
    Stream() {
        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    ~Stream() {
        cudaStreamDestroy(s);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
};
size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

__host__ __device__ unsigned char pattern(uint64_t index) {
    index ^= 0x123456789abcdef0ULL;
    index ^= index >> 30;
    index *= 0xbf58476d1ce4e5b9ULL;
    index ^= index >> 27;
    index *= 0x94d049bb133111ebULL;
    index ^= index >> 31;
    return static_cast<unsigned char>(index);
}
__global__ void initRandom(unsigned char* bytes, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        bytes[i] = pattern(i);
}
__global__ void evictCache(unsigned char* bytes, size_t count, unsigned int salt) {
    auto* data = reinterpret_cast<uint4*>(bytes);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count / 16; i += size_t(gridDim.x) * blockDim.x) {
        uint4 value = data[i];
        value.x += salt;
        value.y ^= salt;
        value.z += 1;
        value.w ^= value.x;
        data[i] = value;
    }
}
uint32_t cpuCrc(const unsigned char* bytes, size_t count) {
    static const auto table = [] {
        std::vector<uint32_t> result(256);
        for (unsigned int i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit)
                value = (value >> 1) ^ ((value & 1) ? 0x82f63b78U : 0);
            result[i] = value;
        }
        return result;
    }();
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < count; ++i)
        crc = table[(crc ^ bytes[i]) & 255] ^ (crc >> 8);
    return ~crc;
}
void knownVector() {
    const auto* bytes = reinterpret_cast<const unsigned char*>("123456789");
    requireSync(cpuCrc(bytes, 9) == 0xe3069283U, "CPU CRC32C oracle does not match the standard vector");
}

struct Layout {
    std::string         name;
    std::vector<size_t> sizes;
    size_t              payload() const {
        return std::accumulate(sizes.begin(), sizes.end(), size_t(0));
    }
};
Layout orderedLayout(bool full) {
    Layout     result{full ? "full" : "prefill_cp8_no_spec_swa", {}};
    const auto groups = full ? std::vector<std::pair<int, size_t>>{{31, 1152}, {30, 19008}, {30, 4224}} :
                               std::vector<std::pair<int, size_t>>{{61, 9360}, {30, 2048}, {30, 8192}};
    for (const auto& group : groups)
        result.sizes.insert(result.sizes.end(), group.first, group.second);
    return result;
}
Layout full() {
    return orderedLayout(true);
}
Layout swa() {
    return orderedLayout(false);
}

// A test-only gathered-record control: identical record stride and transfer
// bytes, but no CRC, footer seal, Result copy or intermediate host verdict.
// Retaining pointer/length metadata uploads makes its boundary explicit; this
// is not the production staged strategy or an exact ablation of future CRC code.
class GatherControl {
public:
    explicit GatherControl(size_t max_payload):
        maxPayload_(max_payload),
        stride_(CrcBlockCopyBatch::encodedBytes(max_payload)),
        staging_(kMaxBackings * stride_),
        deviceTiles_(kMaxBackings * 121 * sizeof(CopyTile)),
        deviceInputs_(kMaxBackings * sizeof(void*)),
        deviceLengths_(kMaxBackings * sizeof(size_t)),
        hostTiles_(kMaxBackings * 121 * sizeof(CopyTile), true),
        hostInputs_(kMaxBackings * sizeof(void*), true),
        hostLengths_(kMaxBackings * sizeof(size_t), true) {
        cu(cudaMemsetAsync(staging_.p, 0, kMaxBackings * stride_, stream_.s));
        cu(cudaStreamSynchronize(stream_.s));
    }
    ~GatherControl() {
        // Submitted operations must not outlive any of these allocations.
        if (cudaStreamSynchronize(stream_.s) != cudaSuccess)
            std::abort();
    }
    CrcCopyStatus store(const std::vector<CrcCopyItem>& items) {
        return run(items, true);
    }
    CrcCopyStatus load(const std::vector<CrcCopyItem>& items) {
        return run(items, false);
    }

private:
    bool valid(const std::vector<CrcCopyItem>& items) const {
        if (items.empty() || items.size() > kMaxBackings)
            return false;
        size_t total_tiles = 0;
        for (size_t index = 0; index < items.size(); ++index) {
            const auto& item = items[index];
            if (!item.host || !item.payload_bytes || item.payload_bytes > maxPayload_)
                return false;
            const size_t    encoded = CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
            const uintptr_t address = reinterpret_cast<uintptr_t>(item.host);
            if (item.capacity_bytes < encoded || address > std::numeric_limits<uintptr_t>::max() - encoded)
                return false;
            for (size_t previous = 0; previous < index; ++previous) {
                const uintptr_t other       = reinterpret_cast<uintptr_t>(items[previous].host);
                const size_t    other_bytes = CrcBlockCopyBatch::encodedBytes(items[previous].payload_bytes);
                if (address < other + other_bytes && other < address + encoded)
                    return false;
            }
            if (item.tiles.empty() || item.tiles.size() > kMaxBackings * 121 - total_tiles)
                return false;
            total_tiles += item.tiles.size();
            size_t offset = 0;
            for (const auto& tile : item.tiles) {
                if (!tile.device || !tile.bytes || tile.offset != offset || tile.bytes > item.payload_bytes - offset)
                    return false;
                offset += tile.bytes;
            }
            if (offset != item.payload_bytes)
                return false;
        }
        return true;
    }
    CrcCopyStatus run(const std::vector<CrcCopyItem>& items, bool store) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(items))
            return CrcCopyStatus::INVALID_ARGS;
        cu(cudaSetDevice(0));
        size_t count = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            static_cast<const void**>(hostInputs_.p)[i] = staging_.bytes() + i * stride_;
            static_cast<size_t*>(hostLengths_.p)[i]     = items[i].payload_bytes;
            for (const auto& tile : items[i].tiles)
                static_cast<CopyTile*>(hostTiles_.p)[count++] = {
                    static_cast<unsigned char*>(tile.device), i * stride_ + tile.offset, tile.bytes};
        }
        cu(cudaMemcpyAsync(
            deviceInputs_.p, hostInputs_.p, items.size() * sizeof(void*), cudaMemcpyHostToDevice, stream_.s));
        cu(cudaMemcpyAsync(
            deviceLengths_.p, hostLengths_.p, items.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream_.s));
        cu(cudaMemcpyAsync(deviceTiles_.p, hostTiles_.p, count * sizeof(CopyTile), cudaMemcpyHostToDevice, stream_.s));
        if (store) {
            cu(launchCopyTiles(static_cast<const CopyTile*>(deviceTiles_.p), count, staging_.p, false, stream_.s));
            for (size_t i = 0; i < items.size(); ++i)
                cu(cudaMemcpyAsync(items[i].host,
                                   staging_.bytes() + i * stride_,
                                   CrcBlockCopyBatch::encodedBytes(items[i].payload_bytes),
                                   cudaMemcpyDeviceToHost,
                                   stream_.s));
        } else {
            for (size_t i = 0; i < items.size(); ++i)
                cu(cudaMemcpyAsync(staging_.bytes() + i * stride_,
                                   items[i].host,
                                   CrcBlockCopyBatch::encodedBytes(items[i].payload_bytes),
                                   cudaMemcpyHostToDevice,
                                   stream_.s));
            cu(launchCopyTiles(static_cast<const CopyTile*>(deviceTiles_.p), count, staging_.p, true, stream_.s));
        }
        cu(cudaStreamSynchronize(stream_.s));
        return CrcCopyStatus::OK;
    }
    size_t     maxPayload_, stride_;
    Stream     stream_;
    Buffer     staging_, deviceTiles_, deviceInputs_, deviceLengths_, hostTiles_, hostInputs_, hostLengths_;
    std::mutex mutex_;
};

struct Plan {
    int                                n;
    std::vector<int>                   blocks;
    std::vector<CrcCopyItem>           items;
    std::unique_ptr<FrameworkCopyPlan> nativePlans;
};
struct Workspace {
    Layout                             layout;
    size_t                             p, e, hostStride, sourceBytes, evictBytes, stride;
    int                                poolBlocks;
    Stream                             stream;
    std::unique_ptr<Buffer>            source, eviction, host;
    std::vector<std::unique_ptr<Plan>> plans;
    std::vector<size_t>                planStarts;
    std::vector<int>                   rotationCounts;

    Workspace(Layout shape, size_t l2): layout(std::move(shape)) {
        p                    = layout.payload();
        e                    = CrcBlockCopyBatch::encodedBytes(p);
        stride               = CrcBlockCopyBatch::encodedBytes(swa().payload());
        hostStride           = alignUp(e, 4096);
        const size_t minimum = std::max(kEvictMultiplier * l2, size_t(4 * kMaxBackings) * p);
        poolBlocks           = int((minimum + p - 1) / p);
        sourceBytes          = size_t(poolBlocks) * p;
        evictBytes           = alignUp(kEvictMultiplier * l2, 16);
        source               = std::make_unique<Buffer>(sourceBytes);
        eviction             = std::make_unique<Buffer>(evictBytes);
        host                 = std::make_unique<Buffer>(kMaxBackings * hostStride, true);
        initRandom<<<4096, 256, 0, stream.s>>>(eviction->bytes(), evictBytes);
        cu(cudaGetLastError());
        cu(cudaStreamSynchronize(stream.s));
        std::mt19937 rng(20260922);
        for (int n = 1; n <= kMaxBackings; ++n) {
            planStarts.push_back(plans.size());
            const int rotations = (poolBlocks + n - 1) / n;
            rotationCounts.push_back(rotations);
            std::vector<int> candidates(poolBlocks);
            std::iota(candidates.begin(), candidates.end(), 0);
            std::shuffle(candidates.begin(), candidates.end(), rng);
            for (int rotation = 0; rotation < rotations; ++rotation) {
                auto plan = std::make_unique<Plan>();
                plan->n   = n;

                for (int b = 0; b < n; ++b) {
                    const int block = candidates[(rotation * n + b) % poolBlocks];
                    plan->blocks.push_back(block);
                    CrcCopyItem item{host->bytes() + size_t(b) * hostStride, p, hostStride, {}};
                    size_t      prefix = 0;
                    for (size_t width : layout.sizes) {
                        auto* device = source->bytes() + prefix * size_t(poolBlocks) + size_t(block) * width;
                        item.tiles.push_back({device, prefix, width});
                        prefix += width;
                    }
                    plan->items.push_back(std::move(item));
                }
                plan->nativePlans = std::make_unique<FrameworkCopyPlan>(plan->items);
                plans.push_back(std::move(plan));
            }
        }
    }
    int rotationCount(int n) const {
        return rotationCounts.at(size_t(n - 1));
    }
    size_t planIndex(int n, int rotation) const {
        return planStarts.at(size_t(n - 1)) + rotation % rotationCount(n);
    }
    void flush(unsigned int salt) {
        evictCache<<<4096, 256, 0, stream.s>>>(eviction->bytes(), evictBytes, salt + 1);
        cu(cudaGetLastError());
        cu(cudaStreamSynchronize(stream.s));
    }
};

struct CheckLayer {
    size_t offset, width;
};
__global__ void checkSyncPool(const unsigned char* actual,
                              const unsigned char* oracle,
                              const CheckLayer*    layers,
                              size_t               poolBlocks,
                              const unsigned char* selected,
                              bool                 acceptSelected,
                              unsigned int*        errors) {
    const auto   layer = layers[blockIdx.x];
    const size_t span  = layer.width * poolBlocks;
    for (size_t i = size_t(blockIdx.y) * blockDim.x + threadIdx.x; i < span; i += size_t(gridDim.y) * blockDim.x) {
        const size_t        absolute = layer.offset + i;
        const unsigned char expected = acceptSelected && selected[i / layer.width] ? oracle[absolute] : deviceGuard;
        if (actual[absolute] != expected)
            atomicExch(errors, 1U);
    }
}

struct Benchmark {
    Workspace                  w;
    FrameworkCopies            framework;
    CrcBlockCopyBatch          production;
    GatherControl              control;
    std::vector<unsigned char> cpuOracle;
    std::vector<uint32_t>      recordCrc;
    std::vector<CheckLayer>    layers;
    std::vector<size_t>        groupRows;
    Buffer                     deviceOracle, deviceLayers, selectedBlocks, checkErrors;
    volatile uintptr_t         metadataSink = 0;
    std::mutex                 copyMutex;

    Benchmark(Layout layout, size_t l2):
        w(layout, l2),
        framework(0),
        production(0, kMaxBackings, swa().payload(), kMaxBackings * 121),
        control(swa().payload()),
        cpuOracle(w.sourceBytes),
        recordCrc(w.poolBlocks),
        deviceOracle(w.sourceBytes),
        deviceLayers(layout.sizes.size() * sizeof(CheckLayer)),
        selectedBlocks(w.poolBlocks),
        checkErrors(sizeof(unsigned int)) {
        groupRows = layout.name == "full" ? std::vector<size_t>{31, 30, 30} : std::vector<size_t>{61, 30, 30};
        requireSync(std::accumulate(groupRows.begin(), groupRows.end(), size_t(0)) == layout.sizes.size(),
                    "pool groups do not cover layout");
        requireSync(w.sourceBytes >= 8 * l2, "device pool smaller than 8 L2");
        cudaPointerAttributes attr{};
        cu(cudaPointerGetAttributes(&attr, w.host->p));
        requireSync(attr.type == cudaMemoryTypeHost, "final host allocation is not pinned");
        cu(cudaPointerGetAttributes(&attr, w.source->p));
        requireSync(attr.type == cudaMemoryTypeDevice, "copy pool is not device memory");
        for (size_t i = 0; i < cpuOracle.size(); ++i)
            cpuOracle[i] = pattern(i);
        size_t prefix = 0;
        for (size_t width : layout.sizes) {
            layers.push_back({prefix * size_t(w.poolBlocks), width});
            prefix += width;
        }
        cu(cudaMemcpyAsync(deviceOracle.p, cpuOracle.data(), cpuOracle.size(), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaMemcpyAsync(
            deviceLayers.p, layers.data(), layers.size() * sizeof(CheckLayer), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
        // CPU-only whole-record CRCs, computed once for every physical block.
        // Host input construction and checksum calculation are never timed.
        std::vector<unsigned char> packed(w.p);
        for (int block = 0; block < w.poolBlocks; ++block) {
            packCpuBlock(block, packed.data());
            recordCrc[block] = cpuCrc(packed.data(), packed.size());
        }
    }

    void packCpuBlock(int block, unsigned char* output) const {
        size_t offset = 0;
        for (const auto& layer : layers) {
            std::memcpy(output + offset, cpuOracle.data() + layer.offset + size_t(block) * layer.width, layer.width);
            offset += layer.width;
        }
    }

    void restoreDeviceOracle() {
        cu(cudaMemcpyAsync(w.source->p, deviceOracle.p, w.sourceBytes, cudaMemcpyDeviceToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void prepareHost(size_t ix) {
        const auto& plan = *w.plans[ix];
        std::memset(w.host->p, 0xa5, 32 * w.hostStride);
        for (int b = 0; b < plan.n; ++b) {
            auto* host = w.host->bytes() + size_t(b) * w.hostStride;
            packCpuBlock(plan.blocks[b], host);
            // Valid legacy records can have arbitrary padding outside payload.
            std::memset(host + w.p, 0x7b, w.e - 4 - w.p);
            std::memcpy(host + w.e - 4, &recordCrc[plan.blocks[b]], 4);
        }
    }

    void primeCpuMetadata(size_t ix) {
        uintptr_t sum = 0;
        for (const auto& item : w.plans[ix]->items) {
            sum += reinterpret_cast<uintptr_t>(item.host) ^ item.payload_bytes ^ item.capacity_bytes;
            for (const auto& tile : item.tiles)
                sum += reinterpret_cast<uintptr_t>(tile.device) ^ tile.offset ^ tile.bytes;
        }
        // All variants receive the same CPU-only priming; never dereference
        // a host payload or a GPU pointer while warming descriptor metadata.
        sum += w.plans[ix]->nativePlans->touchMetadata();
        for (size_t rows : groupRows)
            sum += rows;
        metadataSink = sum;
    }

    std::vector<cudaMemcpy3DBatchOp> build3d(size_t ix, bool store) const {
        std::vector<cudaMemcpy3DBatchOp> ops;
        ops.reserve(w.plans[ix]->items.size() * groupRows.size());
        for (const auto& item : w.plans[ix]->items) {
            size_t first = 0;
            for (size_t rows : groupRows) {
                requireSync(rows >= 2 && first + rows <= item.tiles.size(), "invalid 3D pool group");
                const auto& head  = item.tiles[first];
                const auto  begin = reinterpret_cast<uintptr_t>(head.device);
                const auto  next  = reinterpret_cast<uintptr_t>(item.tiles[first + 1].device);
                requireSync(next > begin && next - begin >= head.bytes, "invalid 3D device pitch");
                const size_t pitch = next - begin;
                for (size_t r = 0; r < rows; ++r) {
                    const auto& tile = item.tiles[first + r];
                    requireSync(tile.bytes == head.bytes && tile.offset == head.offset + r * head.bytes
                                    && reinterpret_cast<uintptr_t>(tile.device) == begin + r * pitch,
                                "non-affine tile inside declared pool");
                }
                cudaMemcpy3DBatchOp op{};
                op.src.type = op.dst.type = cudaMemcpyOperandTypePointer;
                auto* host                = static_cast<unsigned char*>(item.host) + head.offset;
                op.src.op.ptr.ptr         = store ? head.device : host;
                op.dst.op.ptr.ptr         = store ? host : head.device;
                op.src.op.ptr.rowLength   = store ? pitch : head.bytes;
                op.dst.op.ptr.rowLength   = store ? head.bytes : pitch;
                op.src.op.ptr.layerHeight = op.dst.op.ptr.layerHeight = rows;
                op.extent                                             = make_cudaExtent(head.bytes, rows, 1);
                op.srcAccessOrder                                     = cudaMemcpySrcAccessOrderStream;
                op.flags                                              = 0;
                ops.push_back(op);
                first += rows;
            }
            requireSync(first == item.tiles.size(), "uncovered 3D tiles");
        }
        return ops;
    }

    rtp_llm::CrcCopyStatus invoke(size_t ix, int variant, bool store) {
        const auto& items = w.plans[ix]->items;
        if (variant == 0)
            return store ? production.store(items) : production.load(items);
        if (variant == 4)
            return store ? control.store(items) : control.load(items);
        if (variant == 1) {
            framework.copyBatch(*w.plans[ix]->nativePlans, store);
        } else if (variant == 3) {
            framework.copyStaged(*w.plans[ix]->nativePlans, store);
        } else if (variant == 2) {
            cu(cudaSetDevice(0));
            auto        ops    = build3d(ix, store);
            const auto  stream = framework.copy3dStream();
            cudaError_t submission;
            {
                std::lock_guard<std::mutex> lock(copyMutex);
                submission = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
            }
            const auto completion = cudaStreamSynchronize(stream);
            cu(submission);
            cu(completion);
        } else {
            throw std::runtime_error("unknown copy variant");
        }
        return rtp_llm::CrcCopyStatus::OK;
    }

    void poisonDevice(size_t ix) {
        std::vector<unsigned char> selected(w.poolBlocks, 0);
        for (int block : w.plans[ix]->blocks)
            selected[block] = 1;
        cu(cudaMemsetAsync(w.source->p, deviceGuard, w.sourceBytes, w.stream.s));
        cu(cudaMemcpyAsync(selectedBlocks.p, selected.data(), selected.size(), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void verifyDevice(bool validPayload) {
        cu(cudaMemsetAsync(checkErrors.p, 0, sizeof(unsigned int), w.stream.s));
        checkSyncPool<<<dim3(unsigned(layers.size()), 32), 256, 0, w.stream.s>>>(
            w.source->bytes(),
            deviceOracle.bytes(),
            static_cast<const CheckLayer*>(deviceLayers.p),
            w.poolBlocks,
            selectedBlocks.bytes(),
            validPayload,
            static_cast<unsigned int*>(checkErrors.p));
        cu(cudaGetLastError());
        unsigned int errors = 0;
        cu(cudaMemcpyAsync(&errors, checkErrors.p, sizeof(errors), cudaMemcpyDeviceToHost, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
        requireSync(errors == 0,
                    validPayload ? "H2D payload or unselected guard mismatch" : "CRC failure wrote a GPU destination");
    }

    void verifyHost(size_t ix, bool hasCrc, bool inputRecord = false, bool copiesEncoded = false) {
        const auto&                plan = *w.plans[ix];
        std::vector<unsigned char> packed(w.p);
        for (int b = 0; b < 32; ++b) {
            const auto* host = w.host->bytes() + size_t(b) * w.hostStride;
            if (b < plan.n) {
                packCpuBlock(plan.blocks[b], packed.data());
                requireSync(std::memcmp(host, packed.data(), w.p) == 0, "CPU payload oracle mismatch");
                if (hasCrc) {
                    uint32_t footer = 0;
                    std::memcpy(&footer, host + w.e - 4, 4);
                    requireSync(footer == recordCrc[plan.blocks[b]], "whole-record CPU CRC mismatch");
                    for (size_t i = w.p; i < w.e - 4; ++i)
                        requireSync(host[i] == (inputRecord ? 0x7b : 0), "CRC padding mismatch");
                }
            }
            // The gather control transfers E bytes without sealing a footer or
            // padding. Its payload is defined; only bytes beyond E are guards.
            const size_t copied = b < plan.n ? (hasCrc || copiesEncoded ? w.e : w.p) : 0;
            for (size_t i = copied; i < w.hostStride; ++i)
                requireSync(host[i] == 0xa5, "host non-payload guard overwritten");
        }
    }

    void verify(size_t ix, int variant, bool store) {
        try {
            if (store) {
                std::memset(w.host->p, 0xa5, 32 * w.hostStride);
                requireSync(invoke(ix, variant, true) == rtp_llm::CrcCopyStatus::OK, "D2H call failed");
                verifyHost(ix, variant == 0, false, variant == 4);
            } else {
                prepareHost(ix);
                poisonDevice(ix);
                requireSync(invoke(ix, variant, false) == rtp_llm::CrcCopyStatus::OK, "H2D call failed");
                verifyDevice(true);
                verifyHost(ix, true, true);
            }

        } catch (const std::exception& error) {
            std::fprintf(stderr,
                         "correctness failed layout=%s bs=%d variant=%s direction=%s rotation=%zu: %s\n",
                         w.layout.name.c_str(),
                         w.plans[ix]->n,
                         syncNames[variant],
                         store ? "d2h" : "h2d",
                         ix - w.planStarts[w.plans[ix]->n - 1],
                         error.what());
            throw;
        }
    }

    void verifyCorruption(size_t ix, int variant) {
        const int n = w.plans[ix]->n;
        prepareHost(ix);
        poisonDevice(ix);
        requireSync(invoke(ix, variant, false) == rtp_llm::CrcCopyStatus::OK, "initial good CRC load failed");
        verifyDevice(true);
        const int    indices[]{0, n / 2, n - 1};
        const size_t offsets[]{0, w.p / 2, w.p - 1};
        for (int location = 0; location < 3; ++location) {
            for (bool footer : {false, true}) {
                auto* damaged =
                    w.host->bytes() + size_t(indices[location]) * w.hostStride + (footer ? w.e - 4 : offsets[location]);
                *damaged ^= 1;
                poisonDevice(ix);
                requireSync(invoke(ix, variant, false) == rtp_llm::CrcCopyStatus::CRC_MISMATCH,
                            "corrupted whole record was accepted");
                verifyDevice(false);
                *damaged ^= 1;
                requireSync(invoke(ix, variant, false) == rtp_llm::CrcCopyStatus::OK,
                            "good record rejected after failed load");
                verifyDevice(true);
            }
        }
    }
};

// A separate small allocation tests ragged unaligned tiles and changing payload
// shapes. It checks GPU bytes against CPU records, including guards after every
// item; a store/load round trip alone would miss a shared serialization bug.
template<class Backend>
static void mixedCrcSelfTest() {
    const size_t               maxPayload = swa().payload(), slot = alignUp(maxPayload + 64, 16);
    const size_t               hostSlot = alignUp(maxPayload + 64, 16);
    Buffer                     device(32 * slot), host(32 * hostSlot, true);
    Stream                     setup;
    Backend                    backend(0, 32, maxPayload, 32 * 3);
    const std::vector<size_t>  sizes{16383, 16384, 16385, 32768, 32769, 35, full().payload(), swa().payload()};
    std::vector<unsigned char> expected(32 * slot), observed(32 * slot);
    int                        round = 0;
    for (int n : {32, 1, 7, 8, 32}) {
        std::fill(expected.begin(), expected.end(), deviceGuard);
        std::memset(host.p, 0xa5, 32 * hostSlot);
        std::vector<rtp_llm::CrcCopyItem> items;
        for (int b = 0; b < n; ++b) {
            const size_t bytes   = sizes[(b + round) % sizes.size()];
            auto*        payload = expected.data() + size_t(b) * slot + 17;
            auto*        record  = host.bytes() + size_t(b) * hostSlot + 3;
            for (size_t i = 0; i < bytes; ++i)
                payload[i] = pattern(i + size_t(b + 1) * 1000003 + round * 733);
            std::memcpy(record, payload, bytes);
            const size_t encoded = rtp_llm::CrcBlockCopyBatch::encodedBytes(bytes);
            std::memset(record + bytes, 0x7b, encoded - 4 - bytes);
            const uint32_t crc = cpuCrc(payload, bytes);
            std::memcpy(record + encoded - 4, &crc, 4);
            auto* target = device.bytes() + size_t(b) * slot + 17;
            items.push_back(
                {record, bytes, hostSlot - 3, {{target, 0, 3}, {target + 3, 3, 17}, {target + 20, 20, bytes - 20}}});
        }
        auto validateItems = items;
        for (auto& item : validateItems)
            item.tiles.clear();
        requireSync(backend.validate(validateItems) == rtp_llm::CrcCopyStatus::OK, "mixed CPU record validation");
        auto check = [&](bool good) {
            cu(cudaMemcpy(observed.data(), device.p, observed.size(), cudaMemcpyDeviceToHost));
            if (good)
                requireSync(observed == expected, "mixed load payload/guard mismatch");
            else
                requireSync(
                    std::all_of(observed.begin(), observed.end(), [](unsigned char c) { return c == deviceGuard; }),
                    "mixed rejected batch wrote a destination");
        };
        auto poison = [&] {
            cu(cudaMemsetAsync(device.p, deviceGuard, 32 * slot, setup.s));
            cu(cudaStreamSynchronize(setup.s));
        };
        poison();
        requireSync(backend.load(items) == rtp_llm::CrcCopyStatus::OK, "mixed old-record load failed");
        check(true);
        auto* corrupt = static_cast<unsigned char*>(items.back().host) + items.back().payload_bytes - 1;
        *corrupt ^= 1;
        poison();
        requireSync(backend.load(items) == rtp_llm::CrcCopyStatus::CRC_MISMATCH, "mixed corruption accepted");
        check(false);
        *corrupt ^= 1;
        requireSync(backend.load(items) == rtp_llm::CrcCopyStatus::OK, "mixed stale verdict after failure");
        check(true);
        // Poison the entire output so footer/padding must be produced by store.
        std::memset(host.p, 0xa5, 32 * hostSlot);
        requireSync(backend.store(items) == rtp_llm::CrcCopyStatus::OK, "mixed store failed");
        for (int b = 0; b < n; ++b) {
            const auto&  item    = items[b];
            const auto*  record  = static_cast<const unsigned char*>(item.host);
            const auto*  payload = expected.data() + size_t(b) * slot + 17;
            const size_t encoded = rtp_llm::CrcBlockCopyBatch::encodedBytes(item.payload_bytes);
            requireSync(std::memcmp(record, payload, item.payload_bytes) == 0, "mixed store CPU payload mismatch");
            uint32_t footer = 0;
            std::memcpy(&footer, record + encoded - 4, 4);
            requireSync(footer == cpuCrc(payload, item.payload_bytes), "mixed store CPU CRC mismatch");
            for (size_t i = item.payload_bytes; i < encoded - 4; ++i)
                requireSync(record[i] == 0, "mixed store padding not cleared");
            const auto* base = host.bytes() + size_t(b) * hostSlot;
            for (size_t i = 0; i < 3; ++i)
                requireSync(base[i] == 0xa5, "mixed host prefix overwritten");
            for (size_t i = encoded + 3; i < hostSlot; ++i)
                requireSync(base[i] == 0xa5, "mixed host suffix overwritten");
        }
        requireSync(backend.validate(validateItems) == rtp_llm::CrcCopyStatus::OK, "mixed new-record validation");
        poison();
        requireSync(backend.load(items) == rtp_llm::CrcCopyStatus::OK, "mixed new-record load failed");
        check(true);
        ++round;
    }
}

struct Options {
    int         iterations        = 100;
    int         warmup            = 30;
    int         repeat            = 80;
    uint32_t    seed              = 20261004;
    bool        correctnessOnly   = false;
    bool        tileThreadProfile = false;
    bool        profileCapture    = false;
    bool        exclude1dH2d      = false;
    bool        seedProvided      = false;
    std::string output;
};
uint32_t parseUnsigned(const std::string& text, const char* name) {
    requireSync(!text.empty() && text.front() != '-', std::string("invalid ") + name);
    size_t     end   = 0;
    const auto value = std::stoull(text, &end);
    requireSync(end == text.size() && value <= std::numeric_limits<uint32_t>::max(), std::string("invalid ") + name);
    return static_cast<uint32_t>(value);
}
Options parseOptions(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--profile-capture") {
            result.profileCapture = true;
            continue;
        }
        if (key == "--tile-thread-profile") {
            result.tileThreadProfile = true;
            continue;
        }
        if (key == "--correctness-only") {
            result.correctnessOnly = true;
            continue;
        }
        if (key == "--exclude-1d-h2d") {
            result.exclude1dH2d = true;
            continue;
        }
        if (key == "--help") {
            std::cout
                << "CrcCopyBenchmark --output FILE [--iterations 100] [--warmup 30] [--repeat 80] [--seed UINT]\n"
                   "                 [--correctness-only] [--exclude-1d-h2d] [--tile-thread-profile] [--profile-capture]\n"
                   "Five paths, both directions, FULL/SWA, local backing BS=1..32. CUDA13 required.\n"
                   "--exclude-1d-h2d explicitly omits that path; it never substitutes another copy implementation.\n";
            std::exit(0);
        }
        std::string value;
        const auto  equal = key.find('=');
        if (equal != std::string::npos) {
            value = key.substr(equal + 1);
            key.resize(equal);
        } else {
            requireSync(i + 1 < argc, "missing option value for " + key);
            value = argv[++i];
        }
        if (key == "--output") {
            result.output = value;
            continue;
        }
        const auto number = parseUnsigned(value, key.c_str());
        if (key == "--seed") {
            result.seed         = number;
            result.seedProvided = true;
            continue;
        }
        requireSync(number <= static_cast<uint32_t>(std::numeric_limits<int>::max()), "option value exceeds INT_MAX");
        if (key == "--iterations")
            result.iterations = static_cast<int>(number);
        else if (key == "--warmup")
            result.warmup = static_cast<int>(number);
        else if (key == "--repeat")
            result.repeat = static_cast<int>(number);
        else
            throw std::runtime_error("unknown option: " + key);
    }
    requireSync(result.iterations > 0 && result.warmup <= std::numeric_limits<int>::max() - result.iterations,
                "iterations must be positive and iterations+warmup must fit INT_MAX");
    if (!result.seedProvided)
        result.seed = 20260924U + static_cast<uint32_t>(result.repeat);
    requireSync(!result.output.empty(), "--output FILE is required");
    return result;
}

void metadata(std::ostream& os, const Options& options, const cudaDeviceProp& prop, int driver, int runtime) {
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    os << "{\"type\":\"metadata\",\"implementation\":\"crc_copy_benchmark_v1\",\"seed\":" << options.seed
       << ",\"variants\":[\"integrated_crc\",\"copy1d_batch\",\"copy3d_batch\",\"main_staged\",\"gather_control\"]"
       << ",\"exclude_1d_h2d\":" << (options.exclude1dH2d ? "true" : "false") << ",\"excluded_cases\":"
       << (options.exclude1dH2d ?
               "[{\"direction\":\"h2d\",\"variant\":\"copy1d_batch\",\"reason\":\"explicit --exclude-1d-h2d\"}]" :
               "[]")
       << ",\"repeat\":" << options.repeat << ",\"evict_multiplier\":" << kEvictMultiplier
       << ",\"cpu_metadata\":\"warm\",\"host_gap\":0,\"descriptor_bytes\":24,\"layout_order\":\"production\""
       << ",\"l2_bytes\":" << prop.l2CacheSize << ",\"timing\":\"wall\",\"regime\":\"cold\""
       << ",\"gpu\":\"" << prop.name << "\",\"sm\":" << prop.major * 10 + prop.minor << ",\"gpu_uuid\":\"" << uuid.str()
       << "\",\"driver\":" << driver << ",\"runtime\":" << runtime << ",\"iterations\":" << options.iterations
       << ",\"warmup\":" << options.warmup << ",\"local_backing_count\":true,\"gen_num_per_cycle\":0,\"profiled\":false"
       << ",\"correctness_only\":" << (options.correctnessOnly ? "true" : "false")
       << ",\"boundary\":\"prebuilt per-API inputs; complete synchronous calls including internal descriptors, checks, locks, metadata H2D, kernels, CPU pack/unpack for main_staged, data transfers, synchronization and CRC verdict\""
       << ",\"production_source\":\"directly linked current workspace CrcBlockCopyBatch and DeviceHostCopyStrategy\""
       << ",\"gather_control\":\"test-local gathered-record control, fixed max-payload stride, pointer/length/tile metadata, E transfer, no CRC/footer/Result; one final sync; not production main_staged\""
       << ",\"copy3d_stream\":\"independent Torch nondefault pooled stream\""
       << ",\"copy3d_submit_mutex\":\"benchmark-local mutex; production private mutex is not exported\""
       << ",\"host_input\":\"independent CPU whole-record oracle prepared outside timing\""
       << ",\"block_counts\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]"
       << ",\"seq_size_per_block\":128,\"cp_size\":8,\"tp_size\":8,\"cp_mode\":\"CP_RR\""
       << ",\"h2d_host_payload\":\"recently prepared on CPU once per shared round; host cache is not claimed cold\",\"fallback_allowed\":false}\n";
}

// Profile the production tile launcher with its actual two staging layouts.
// Metadata construction/upload, cold-cache eviction, setup and validation are
// excluded from the event interval. CUDA events measure stream latency; an
// independent profiler is needed to separate kernel duration from launch gaps.
constexpr int profileThreads[]{32, 64, 128, 256, 512, 1024};
constexpr int profileBlocks[]{1, 2, 4, 8, 16, 32};

struct Events {
    cudaEvent_t begin{}, end{};
    Events() {
        cu(cudaEventCreate(&begin));
        cu(cudaEventCreate(&end));
    }
    ~Events() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};

__global__ void checkProfilePool(const unsigned char* actual,
                                 const CheckLayer*    layers,
                                 size_t               poolBlocks,
                                 const unsigned char* selected,
                                 bool                 scatter,
                                 unsigned int*        errors) {
    const auto   layer = layers[blockIdx.x];
    const size_t span  = layer.width * poolBlocks;
    for (size_t i = size_t(blockIdx.y) * blockDim.x + threadIdx.x; i < span; i += size_t(gridDim.y) * blockDim.x) {
        const size_t absolute = layer.offset + i;
        const auto   expected = !scatter || selected[i / layer.width] ? pattern(absolute) : deviceGuard;
        if (actual[absolute] != expected)
            atomicExch(errors, 1U);
    }
}

struct TileProfile {
    Workspace                  w;
    Buffer                     staging, descriptors, deviceLayers, selectedBlocks, errors;
    Events                     events;
    std::vector<CopyTile>      tiles;
    std::vector<unsigned char> expected, observed;
    size_t                     stagingBytes;

    TileProfile(Layout layout, size_t l2):
        w(std::move(layout), l2),
        staging(kMaxBackings * w.stride + 64),
        descriptors(kMaxBackings * 121 * sizeof(CopyTile)),
        deviceLayers(w.layout.sizes.size() * sizeof(CheckLayer)),
        selectedBlocks(w.poolBlocks),
        errors(sizeof(unsigned int)),
        expected(kMaxBackings * w.stride + 64),
        observed(expected.size()),
        stagingBytes(expected.size()) {
        std::vector<CheckLayer> layers;
        size_t                  prefix = 0;
        for (size_t width : w.layout.sizes) {
            layers.push_back({prefix * size_t(w.poolBlocks), width});
            prefix += width;
        }
        cu(cudaMemcpyAsync(
            deviceLayers.p, layers.data(), layers.size() * sizeof(CheckLayer), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void prepare(size_t ix, bool crcStride, bool scatter, bool guardCheck) {
        tiles.clear();
        if (guardCheck)
            std::fill(expected.begin(), expected.end(), deviceGuard);
        size_t      compactOffset = 0;
        const auto& plan          = *w.plans[ix];
        for (size_t b = 0; b < plan.items.size(); ++b) {
            for (const auto& tile : plan.items[b].tiles) {
                const size_t offset = crcStride ? b * w.stride + tile.offset : alignUp(compactOffset, 16);
                tiles.push_back({tile.device, offset, tile.bytes});
                compactOffset = offset + tile.bytes;
                if (guardCheck) {
                    const size_t absolute = static_cast<unsigned char*>(tile.device) - w.source->bytes();
                    for (size_t i = 0; i < tile.bytes; ++i)
                        expected[offset + i] = pattern(absolute + i);
                }
            }
        }
        cu(cudaMemcpyAsync(
            descriptors.p, tiles.data(), tiles.size() * sizeof(CopyTile), cudaMemcpyHostToDevice, w.stream.s));
        if (scatter && guardCheck) {
            cu(cudaMemcpyAsync(staging.p, expected.data(), stagingBytes, cudaMemcpyHostToDevice, w.stream.s));
        } else if (scatter) {
            // Prepare correct scatter input outside timing. Every measured
            // candidate subsequently evicts this setup from L2.
            cu(launchCopyTiles(
                static_cast<const CopyTile*>(descriptors.p), tiles.size(), staging.p, false, w.stream.s));
        } else if (guardCheck) {
            cu(cudaMemsetAsync(staging.p, deviceGuard, stagingBytes, w.stream.s));
        }
        if (guardCheck) {
            if (scatter) {
                cu(cudaMemsetAsync(w.source->p, deviceGuard, w.sourceBytes, w.stream.s));
            } else {
                initRandom<<<4096, 256, 0, w.stream.s>>>(w.source->bytes(), w.sourceBytes);
                cu(cudaGetLastError());
            }
            std::vector<unsigned char> selected(w.poolBlocks, 0);
            for (int b : plan.blocks)
                selected[b] = 1;
            cu(cudaMemcpyAsync(selectedBlocks.p, selected.data(), selected.size(), cudaMemcpyHostToDevice, w.stream.s));
            // Host vectors must remain alive until their asynchronous transfers finish.
            cu(cudaStreamSynchronize(w.stream.s));
        }
    }

    void verify(size_t ix, bool crcStride, bool scatter, int threads) {
        prepare(ix, crcStride, scatter, true);
        cu(launchCopyTiles(
            static_cast<const CopyTile*>(descriptors.p), tiles.size(), staging.p, scatter, w.stream.s, threads));
        cu(cudaMemsetAsync(errors.p, 0, sizeof(unsigned int), w.stream.s));
        checkProfilePool<<<dim3(unsigned(w.layout.sizes.size()), 32), 256, 0, w.stream.s>>>(
            w.source->bytes(),
            static_cast<const CheckLayer*>(deviceLayers.p),
            w.poolBlocks,
            selectedBlocks.bytes(),
            scatter,
            static_cast<unsigned int*>(errors.p));
        cu(cudaGetLastError());
        unsigned int error = 0;
        cu(cudaMemcpyAsync(&error, errors.p, sizeof(error), cudaMemcpyDeviceToHost, w.stream.s));
        cu(cudaMemcpyAsync(observed.data(), staging.p, stagingBytes, cudaMemcpyDeviceToHost, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
        requireSync(error == 0, "tile profile device payload or unselected pool guard mismatch");
        requireSync(observed == expected, "tile profile staging payload, backing gaps or allocation guards mismatch");
    }

    double measure(bool scatter, int threads, unsigned int salt) {
        // Queue eviction before both events on this stream. Do not synchronize
        // after eviction: that would add a host submission gap to every sample.
        evictCache<<<4096, 256, 0, w.stream.s>>>(w.eviction->bytes(), w.evictBytes, salt + 1);
        cu(cudaGetLastError());
        cu(cudaEventRecord(events.begin, w.stream.s));
        cu(launchCopyTiles(
            static_cast<const CopyTile*>(descriptors.p), tiles.size(), staging.p, scatter, w.stream.s, threads));
        cu(cudaEventRecord(events.end, w.stream.s));
        cu(cudaEventSynchronize(events.end));
        float elapsed = 0;
        cu(cudaEventElapsedTime(&elapsed, events.begin, events.end));
        return double(elapsed) * 1000;
    }
};

int runTileThreadProfile(
    std::ostream& os, const Options& options, const cudaDeviceProp& prop, int driver, int runtime) {
    os << "{\"type\":\"metadata\",\"implementation\":\"tile_copy_thread_profile_v1\",\"seed\":" << options.seed
       << ",\"repeat\":" << options.repeat << ",\"iterations\":" << options.iterations
       << ",\"warmup\":" << options.warmup << ",\"correctness_only\":" << (options.correctnessOnly ? "true" : "false")
       << ",\"threads\":[32,64,128,256,512,1024],\"blocks\":[1,2,4,8,16,32]"
       << ",\"measured_copy_launches_per_round\":6,\"scatter_setup_copy_launches_per_round\":1"
       << ",\"production_default_threads\":" << kCopyTileThreads << ",\"gpu\":\"" << prop.name
       << "\",\"sm\":" << prop.major * 10 + prop.minor << ",\"driver\":" << driver << ",\"runtime\":" << runtime
       << ",\"l2_bytes\":" << prop.l2CacheSize << ",\"evict_multiplier\":" << kEvictMultiplier
       << ",\"timing\":\"cuda_event_stream_latency\",\"regime\":\"cold\""
       << ",\"seq_size_per_block\":128,\"cp_size\":8,\"tp_size\":8,\"cp_mode\":\"CP_RR\""
       << ",\"profile_capture\":" << (options.profileCapture ? "true" : "false")
       << ",\"boundary\":\"production launchCopyTiles only between CUDA events; descriptors preuploaded; independent 8xL2 eviction before every candidate outside events; no PCIe payload copy or CRC in interval; event latency may include launch gaps\"}\n";
    std::mt19937 sourceRng(options.seed), orderRng(options.seed ^ 0x728193U);
    for (bool isFull : {true, false}) {
        TileProfile profile(orderedLayout(isFull), prop.l2CacheSize);
        auto&       w = profile.w;
        os << "{\"type\":\"layout\",\"layout\":\"" << w.layout.name << "\",\"payload_bytes\":" << w.p
           << ",\"tiles\":" << w.layout.sizes.size() << ",\"crc_staging_stride\":" << w.stride
           << ",\"source_bytes\":" << w.sourceBytes << ",\"pool_blocks\":" << w.poolBlocks
           << ",\"evict_bytes\":" << w.evictBytes << "}\n";
        for (bool crcStride : {true, false}) {
            const char* packing = crcStride ? "crc_records" : "main_staged";
            for (bool scatter : {false, true}) {
                const char* direction = scatter ? "scatter" : "gather";
                for (int n : profileBlocks) {
                    for (int threads : profileThreads) {
                        for (int rotation : {0, 1})
                            profile.verify(w.planIndex(n, rotation), crcStride, scatter, threads);
                        os << "{\"type\":\"correctness\",\"layout\":\"" << w.layout.name << "\",\"packing\":\""
                           << packing << "\",\"direction\":\"" << direction << "\",\"blocks\":" << n
                           << ",\"threads\":" << threads
                           << ",\"rotations_checked\":[0,1],\"full_payload_and_guards\":true,\"success\":true}\n";
                    }
                    if (options.correctnessOnly)
                        continue;
                    initRandom<<<4096, 256, 0, w.stream.s>>>(w.source->bytes(), w.sourceBytes);
                    cu(cudaGetLastError());
                    std::vector<int> sources(w.rotationCount(n));
                    std::iota(sources.begin(), sources.end(), 0);
                    std::shuffle(sources.begin(), sources.end(), sourceRng);
                    std::vector<int> order(std::begin(profileThreads), std::end(profileThreads));
                    for (int round = -options.warmup; round < options.iterations; ++round) {
                        if (round == 0 && options.profileCapture)
                            cu(cudaProfilerStart());
                        const int source = sources[(round + options.warmup) % sources.size()];
                        profile.prepare(w.planIndex(n, source), crcStride, scatter, false);
                        std::shuffle(order.begin(), order.end(), orderRng);
                        for (size_t position = 0; position < order.size(); ++position) {
                            const int    threads = order[position];
                            const double us      = profile.measure(scatter,
                                                              threads,
                                                              unsigned(round + options.warmup) * unsigned(order.size())
                                                                  + unsigned(position));
                            if (round >= 0)
                                os << "{\"type\":\"sample\",\"layout\":\"" << w.layout.name << "\",\"packing\":\""
                                   << packing << "\",\"direction\":\"" << direction << "\",\"blocks\":" << n
                                   << ",\"threads\":" << threads << ",\"repeat\":" << options.repeat
                                   << ",\"round\":" << round << ",\"position\":" << position
                                   << ",\"source_plan\":" << source << ",\"us\":" << us << "}\n";
                        }
                    }
                    if (options.profileCapture)
                        cu(cudaProfilerStop());
                    os.flush();
                }
            }
        }
    }
    os << "{\"type\":\"complete\",\"success\":true}\n";
    os.flush();
    requireSync(bool(os), "failed to write thread profile output");
    return 0;
}

int run(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);
    std::ofstream os(options.output);
    requireSync(bool(os), "cannot open output: " + options.output);
    os << std::setprecision(12);
    cu(cudaSetDevice(0));
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop, 0));
    requireSync(prop.l2CacheSize > 0, "known GPU L2 size is required");
    int driver = 0, runtime = 0;
    cu(cudaDriverGetVersion(&driver));
    cu(cudaRuntimeGetVersion(&runtime));
    requireSync(driver >= 13000 && runtime >= 13000 && CrcBlockCopyBatch::available(),
                "CUDA13 runtime, driver and CRC backend are required");
    if (options.tileThreadProfile)
        return runTileThreadProfile(os, options, prop, driver, runtime);
    metadata(os, options, prop, driver, runtime);
    os.flush();
    knownVector();
    mixedCrcSelfTest<CrcBlockCopyBatch>();
    os << "{\"type\":\"mixed_crc_selftest\",\"variant\":\"integrated_crc\",\"success\":true}\n";
    std::mt19937 source_rng(options.seed), order_rng(options.seed ^ 0x728193U);
    for (bool is_full : {true, false}) {
        Benchmark benchmark(orderedLayout(is_full), prop.l2CacheSize);
        auto&     w = benchmark.w;
        os << "{\"type\":\"layout\",\"layout\":\"" << w.layout.name << "\",\"payload_bytes\":" << w.p
           << ",\"encoded_bytes\":" << w.e << ",\"host_stride\":" << w.hostStride << ",\"staging_stride\":" << w.stride
           << ",\"tiles\":" << w.layout.sizes.size() << ",\"tile_bytes\":[";
        for (size_t t = 0; t < w.layout.sizes.size(); ++t) {
            if (t)
                os << ',';
            os << w.layout.sizes[t];
        }
        os << "],\"source_bytes\":" << w.sourceBytes << ",\"destination_bytes\":" << w.sourceBytes
           << ",\"pool_blocks\":" << w.poolBlocks << ",\"evict_bytes\":" << w.evictBytes
           << ",\"crc_payload_bytes_per_backing\":" << w.e << ",\"baseline_payload_bytes_per_backing\":" << w.p
           << ",\"gather_control_data_bytes_per_backing\":" << w.e << ",\"gather_control_result_bytes_per_backing\":0"
           << ",\"crc_result_bytes_per_backing\":12,\"host_pinned_verified\":true,\"source_device_verified\":true"
           << ",\"copy3d_operations_per_backing\":3,\"copy3d_expanded_rows_per_backing\":" << w.layout.sizes.size()
           << ",\"copy3d_max_device_pitch\":"
           << *std::max_element(w.layout.sizes.begin(), w.layout.sizes.end()) * size_t(w.poolBlocks)
           << ",\"copy3d_max_host_pitch\":" << *std::max_element(w.layout.sizes.begin(), w.layout.sizes.end())
           << ",\"guard_check\":\"entire device pool against CPU oracle plus unselected sentinel\"}\n";
        benchmark.restoreDeviceOracle();
        for (bool store : {true, false}) {
            for (int n = 1; n <= kMaxBackings; ++n) {
                for (int variant = 0; variant < 5; ++variant) {
                    if (!store && variant == 1 && options.exclude1dH2d) {
                        os << "{\"type\":\"excluded_correctness\",\"direction\":\"h2d\",\"layout\":\"" << w.layout.name
                           << "\",\"blocks\":" << n
                           << ",\"variant\":\"copy1d_batch\",\"reason\":\"explicit --exclude-1d-h2d\"}\n";
                        continue;
                    }
                    for (int rotation : {0, 1})
                        benchmark.verify(w.planIndex(n, rotation), variant, store);
                    os << "{\"type\":\"correctness\",\"direction\":\"" << (store ? "d2h" : "h2d") << "\",\"layout\":\""
                       << w.layout.name << "\",\"blocks\":" << n << ",\"variant\":\"" << syncNames[variant]
                       << "\",\"rotations_checked\":[0,1],\"success\":true}\n";
                }
            }
        }
        for (int n : {1, 8, 16, 32}) {
            benchmark.verifyCorruption(w.planIndex(n, 1), 0);
            os << "{\"type\":\"crc_corruption\",\"direction\":\"h2d\",\"layout\":\"" << w.layout.name
               << "\",\"blocks\":" << n << ",\"variant\":\"integrated_crc\",\"cases\":6,\"corruption_cases\":6"
               << ",\"all_targets_unchanged\":true,\"recovery_success\":true,\"good_bad_good\":true"
               << ",\"entire_pool_guards\":true,\"success\":true}\n";
        }
        benchmark.restoreDeviceOracle();
        os.flush();
        if (options.correctnessOnly)
            continue;
        for (bool store : {true, false}) {
            for (int n = 1; n <= kMaxBackings; ++n) {
                std::vector<int> sources(w.rotationCount(n));
                std::iota(sources.begin(), sources.end(), 0);
                std::shuffle(sources.begin(), sources.end(), source_rng);
                std::vector<int> order{0, 1, 2, 3, 4};
                if (!store && options.exclude1dH2d)
                    order.erase(order.begin() + 1);
                for (int round = -options.warmup; round < options.iterations; ++round) {
                    const int    source = sources[(round + options.warmup) % sources.size()];
                    const size_t ix     = w.planIndex(n, source);
                    if (!store)
                        benchmark.prepareHost(ix);
                    std::shuffle(order.begin(), order.end(), order_rng);
                    for (size_t position = 0; position < order.size(); ++position) {
                        const int variant = order[position];
                        w.flush(static_cast<unsigned int>(round + options.warmup) * 5U
                                + static_cast<unsigned int>(position));
                        benchmark.primeCpuMetadata(ix);
                        const auto   begin  = std::chrono::steady_clock::now();
                        const auto   status = benchmark.invoke(ix, variant, store);
                        const double us =
                            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
                        requireSync(status == CrcCopyStatus::OK, "measured copy call failed");
                        if (round >= 0)
                            os << "{\"type\":\"sample\",\"direction\":\"" << (store ? "d2h" : "h2d")
                               << "\",\"layout\":\"" << w.layout.name << "\",\"blocks\":" << n << ",\"variant\":\""
                               << syncNames[variant] << "\",\"regime\":\"cold\",\"timing\":\"wall\""
                               << ",\"repeat\":" << options.repeat << ",\"round\":" << round
                               << ",\"position\":" << position << ",\"source_plan\":" << source << ",\"us\":" << us
                               << "}\n";
                    }
                }
                os.flush();
            }
        }
    }
    os << "{\"type\":\"complete\",\"success\":true}\n";
    os.flush();
    requireSync(bool(os), "failed to write benchmark output");
    return 0;
}
}  // namespace
}  // namespace rtp_llm::crc_copy_benchmark

int main(int argc, char** argv) {
    try {
        return rtp_llm::crc_copy_benchmark::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "CRC copy benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
