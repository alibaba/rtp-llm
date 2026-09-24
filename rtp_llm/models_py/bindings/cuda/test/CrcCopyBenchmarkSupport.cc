#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>

namespace rtp_llm::crc_copy_benchmark {
namespace {
void requireDone(const StrategyResult& result, const char* name) {
    if (result.status != StrategyStatus::DONE) {
        throw std::runtime_error(std::string(name) + " did not complete; fallback is forbidden; strategy_status="
                                 + std::to_string(static_cast<int>(result.status))
                                 + " copy_status=" + std::to_string(static_cast<int>(result.copy_status)));
    }
}
}  // namespace

struct FrameworkCopyPlan::Impl {
    DeviceHostCopyPlan store;
    DeviceHostCopyPlan load;

    explicit Impl(const std::vector<CrcCopyItem>& items) {
        store.device_to_host = true;
        load.device_to_host  = false;
        size_t count         = 0;
        for (const auto& item : items)
            count += item.tiles.size();
        store.copy_tiles.reserve(count);
        load.copy_tiles.reserve(count);
        for (const auto& item : items) {
            for (const auto& tile : item.tiles) {
                DeviceHostCopyTile copy;
                copy.host_addr    = static_cast<unsigned char*>(item.host) + tile.offset;
                copy.device_addr  = tile.device;
                copy.host_offset  = tile.offset;
                copy.bytes        = tile.bytes;
                copy.device_index = 0;
                store.copy_tiles.push_back(copy);
                load.copy_tiles.push_back(copy);
            }
        }
    }
};

FrameworkCopyPlan::FrameworkCopyPlan(const std::vector<CrcCopyItem>& items): impl_(std::make_unique<Impl>(items)) {}
FrameworkCopyPlan::~FrameworkCopyPlan() = default;

uintptr_t FrameworkCopyPlan::touchMetadata() const {
    uintptr_t sum = 0;
    for (const auto* plan : {&impl_->store, &impl_->load}) {
        sum += plan->device_to_host ^ plan->group_set_id ^ reinterpret_cast<uintptr_t>(plan->host.base)
               ^ plan->host.payload_bytes ^ plan->host.capacity_bytes;
        for (const auto& tile : plan->copy_tiles)
            sum += reinterpret_cast<uintptr_t>(tile.host_addr) ^ reinterpret_cast<uintptr_t>(tile.device_addr)
                   ^ tile.host_offset ^ tile.bytes ^ uintptr_t(tile.device_index) ^ tile.member_group_id
                   ^ tile.local_layer_index;
    }
    return sum;
}

struct FrameworkCopies::Impl {
    CudaBatchDeviceHostCopyStrategy batch;
    StagedSmDeviceHostCopyStrategy  staged;
    at::cuda::CUDAStream            stream;

    explicit Impl(int device): stream(at::cuda::getStreamFromPool(false, device)) {}
};

FrameworkCopies::FrameworkCopies(int device) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    impl_ = std::make_unique<Impl>(device);
}

FrameworkCopies::~FrameworkCopies() = default;

void FrameworkCopies::copyBatch(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->batch.tryExecute(store ? plan.impl_->store : plan.impl_->load, DeviceHostCopyOptions{}),
                "CUDA 1D batch");
}

void FrameworkCopies::copyStaged(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->staged.tryExecute(store ? plan.impl_->store : plan.impl_->load, DeviceHostCopyOptions{}),
                "main staged");
}

cudaStream_t FrameworkCopies::copy3dStream() const {
    return impl_->stream.stream();
}

}  // namespace rtp_llm::crc_copy_benchmark
