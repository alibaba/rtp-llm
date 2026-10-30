#include "rtp_llm/cpp/model_rpc/RegisteredCacheStaging.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <c10/core/DeviceGuard.h>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace rtp_llm {

RegisteredCacheStaging::RegisteredCacheStaging(size_t bytes, std::shared_ptr<MemoryUtil> memory):
    source_bytes(bytes), memory_(std::move(memory)) {
    if (!memory_) {
        throw std::invalid_argument("typed TP staging needs a memory registration owner");
    }
    const auto registered_bytes = registeredCacheBytes(bytes);
    tensor_                     = torch::empty({static_cast<int64_t>(registered_bytes)},
                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    if (memory_->isRdmaMode()) {
        if (!memory_->regUserMr(tensor_.data_ptr(), registered_bytes, false, registered_bytes)) {
            throw std::runtime_error("failed to register typed TP staging memory");
        }
        registered_ = true;
    }
}

RegisteredCacheStaging::RegisteredCacheStaging(size_t                                  bytes,
                                               std::shared_ptr<RegisteredCacheStaging> owner,
                                               size_t                                  offset):
    source_bytes(bytes), owner_(std::move(owner)), tensor_(owner_->tensor().narrow(0, offset, bytes)) {}

std::shared_ptr<RegisteredCacheStaging> RegisteredCacheStagingPool::allocate(size_t bytes) {
    if (bytes == 0 || bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()) - 63) {
        throw std::invalid_argument("invalid typed TP staging size");
    }
    const size_t aligned = (bytes + 63) & ~size_t(63);
    if (!current_ || aligned > current_->source_bytes - used_) {
        const auto minimum  = registeredCacheBytes(aligned);
        size_t     capacity = std::max(slab_bytes_, minimum);
        while (true) {
            try {
                // One slab must be one MR: a transport block cannot span MRs.
                current_    = std::make_shared<RegisteredCacheStaging>(capacity, memory_);
                slab_bytes_ = capacity;
                break;
            } catch (const std::runtime_error&) {
                if (capacity == minimum) {
                    throw;
                }
                capacity = std::max(minimum, registeredCacheBytes(capacity / 2));
            }
        }
        used_ = 0;
    }
    auto slice = std::shared_ptr<RegisteredCacheStaging>(new RegisteredCacheStaging(bytes, current_, used_));
    used_ += aligned;
    return slice;
}

RegisteredCacheStaging::~RegisteredCacheStaging() {
    if (registered_ && !memory_->deregUserMr(tensor_.data_ptr(), false)) {
        RTP_LLM_LOG_WARNING("failed to deregister typed TP staging memory");
    }
}

void reassembleCacheTransfers(const std::vector<PendingCacheCopy>& copies, int device_index) {
    if (copies.empty()) {
        return;
    }
    c10::DeviceGuard                               guard(torch::Device(torch::kCUDA, device_index));
    std::unordered_map<void*, std::vector<size_t>> groups;
    size_t                                         scratch_bytes = 0;
    for (size_t index = 0; index < copies.size(); ++index) {
        const auto& copy = copies[index];
        if (!copy.destination.is_cuda || copy.destination.addr == nullptr
            || copy.destination.device_index != device_index
            || copy.destination.size_bytes != copy.plan.destination_bytes || !copy.source
            || copy.source->source_bytes != copy.plan.source_bytes) {
            throw std::invalid_argument("typed TP cache destination or source has changed");
        }
        copy.plan.validateSlices();
        groups[copy.destination.addr].push_back(index);
        scratch_bytes = std::max(scratch_bytes, copy.destination.size_bytes);
    }
    // Validate all groups before the first H2D write.
    for (const auto& [address, indices] : groups) {
        std::vector<TpCacheTransferPlan> plans;
        plans.reserve(indices.size());
        for (size_t index : indices) {
            plans.push_back(copies[index].plan);
        }
        validateCompleteTransfer(plans);
    }
    auto scratch = torch::empty({static_cast<int64_t>(scratch_bytes)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    for (const auto& [address, indices] : groups) {
        const auto& destination_info = copies[indices.front()].destination;
        for (size_t index : indices) {
            const auto& copy   = copies[index];
            const auto* source = static_cast<const uint8_t*>(copy.source->tensor().data_ptr());
            auto*       output = static_cast<uint8_t*>(scratch.data_ptr());
            for (const auto& slice : copy.plan.slices) {
                std::memcpy(output + slice.destination_offset, source + slice.source_offset, slice.bytes);
            }
        }
        auto destination = torch::from_blob(address,
                                            {static_cast<int64_t>(destination_info.size_bytes)},
                                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        destination.copy_(scratch.narrow(0, 0, destination_info.size_bytes), false);
    }
}

}  // namespace rtp_llm
