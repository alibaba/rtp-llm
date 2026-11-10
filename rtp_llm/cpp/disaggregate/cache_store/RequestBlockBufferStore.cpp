#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <algorithm>
#include <limits>
#include <torch/torch.h>

namespace rtp_llm {

RequestBlockBufferStore::RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util):
    memory_util_(memory_util) {}

void RequestBlockBufferStore::stop() {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    auto                                tmp_buffers = std::move(request_cache_map_);
    lock.unlock();

    // avoid deadlock
    tmp_buffers.clear();
}

bool RequestBlockBufferStore::setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    auto store_request_block_buffer = getOrInsertRequestBlockBuffer(request_block_buffer->getRequestId());
    if (store_request_block_buffer == nullptr) {
        RTP_LLM_LOG_WARNING("set request block buffer failed to get block buffer, request id %s",
                            request_block_buffer->getRequestId().c_str());
        return false;
    }

    auto                                      blocks = request_block_buffer->getBlocks();
    std::vector<std::shared_ptr<BlockBuffer>> valid_blocks;
    std::vector<std::shared_ptr<BlockBuffer>> staging_blocks;
    valid_blocks.reserve(blocks.size());
    for (auto iter : blocks) {
        auto& block = iter.second;
        if (isValidBlock(block)) {
            valid_blocks.push_back(block);
        } else {
            staging_blocks.push_back(block);
        }
    }

    if (!staging_blocks.empty()) {
        auto staged = makeValidBlocks(staging_blocks);
        if (staged.size() != staging_blocks.size()) {
            RTP_LLM_LOG_WARNING("set request block buffer failed to make valid blocks, request id %s, block count %zu",
                                request_block_buffer->getRequestId().c_str(),
                                staging_blocks.size());
            return false;
        }
        valid_blocks.insert(valid_blocks.end(), staged.begin(), staged.end());
    }

    store_request_block_buffer->addBlocks(valid_blocks);
    return true;
}

bool RequestBlockBufferStore::setRequestBlockBufferWatchFunc(const std::string&              requestid,
                                                             RequestBlockBuffer::WatchFunc&& watch_func) {
    auto request_block_buffer = getOrInsertRequestBlockBuffer(requestid);
    if (request_block_buffer == nullptr) {
        RTP_LLM_LOG_WARNING("set request block buffer to request block buffer store failed, request id %s",
                            requestid.c_str());
        return false;
    }
    return request_block_buffer->setWatchFunc(std::move(watch_func));
}

void RequestBlockBufferStore::debugInfo() {
    std::string                         debug = "";
    std::shared_lock<std::shared_mutex> lock(request_cache_map_mutex_);
    std::ostringstream                  oss;
    for (auto block : request_cache_map_) {
        oss << "request id is " << block.first;
        if (block.second == nullptr) {
            oss << " is null";
            continue;
        }
        oss << " block ids: ";
        for (auto s : block.second->getBlocks()) {
            oss << s.first << " ";
        }
        oss << std::endl;
    }
    RTP_LLM_LOG_INFO("reqeut block buffer debug info: %s", oss.str().c_str());
}

std::string RequestBlockBufferStore::debugInfoOnRequest(const std::string& requestid) const {
    std::ostringstream stream;
    auto               request_block_buffer = getRequestBlockBuffer(requestid);
    if (request_block_buffer == nullptr) {
        stream << "request id: " << requestid << " not found or expired";
        return stream.str();
    }
    return request_block_buffer->debugInfo();
}

std::shared_ptr<BlockBuffer> RequestBlockBufferStore::getBlockBuffer(const std::string& requestid,
                                                                     const std::string& blockid) const {
    auto request_block_buffer = getRequestBlockBuffer(requestid);
    if (request_block_buffer == nullptr) {
        return nullptr;
    }
    return request_block_buffer->getBlock(blockid);
}

std::shared_ptr<RequestBlockBuffer> RequestBlockBufferStore::getRequestBlockBuffer(const std::string& requestid) const {
    std::shared_lock<std::shared_mutex> lock(request_cache_map_mutex_);

    auto iter = request_cache_map_.find(requestid);
    if (iter != request_cache_map_.end()) {
        return iter->second;
    }
    return nullptr;
}

std::shared_ptr<RequestBlockBuffer>
RequestBlockBufferStore::getOrInsertRequestBlockBuffer(const std::string& requestid) {
    std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);

    auto iter = request_cache_map_.find(requestid);
    if (iter != request_cache_map_.end()) {
        if (iter->second == nullptr) {
            RTP_LLM_LOG_WARNING("request block buffer store try get expired request block buffer, request id %s",
                                requestid.c_str());
        }
        return iter->second;
    }

    auto ret = request_cache_map_.insert(std::make_pair(requestid, std::make_shared<RequestBlockBuffer>(requestid)));
    if (!ret.second) {
        RTP_LLM_LOG_WARNING("request block buffer store new request block buffer to request map failed, request id %s",
                            requestid.c_str());
        return nullptr;
    }

    return ret.first->second;
}

bool RequestBlockBufferStore::isValidBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (memory_util_->isRdmaMode()) {
        return memory_util_->isMemoryMr(block->addr.get(), block->len, block->gpu_mem, block->adopted);
    }
    return block->gpu_mem == false;
}

std::vector<std::shared_ptr<BlockBuffer>>
RequestBlockBufferStore::makeValidBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (!isRuntimeInitialized()) {
        RTP_LLM_LOG_WARNING("make valid blocks failed, device is null, block count %zu", blocks.size());
        return {};
    }

    // One pinned allocation for the whole submission: page-locking host memory is a
    // device-synchronizing driver call, so a per-block allocation turns a multi-GB
    // publication into hundreds of thousands of them.
    constexpr size_t alignment             = 256;
    constexpr size_t max_tensor_size       = std::numeric_limits<int64_t>::max();
    const bool       rdma                  = memory_util_->isRdmaMode();
    const auto       min_registration_size = rdma ? memory_util_->minRegistrationSize() : 0;
    if (min_registration_size > max_tensor_size) {
        RTP_LLM_LOG_WARNING("minimum staging registration size is out of range: %lu", min_registration_size);
        return {};
    }

    struct StagingBlock {
        const BlockBuffer* block;
        size_t             stride;
    };
    std::vector<StagingBlock> layout;
    layout.reserve(blocks.size());
    size_t total_len = 0;
    for (const auto& block : blocks) {
        const size_t bytes  = std::max<size_t>(block->len, min_registration_size);
        const size_t stride = (bytes + alignment - 1) / alignment * alignment;
        if (stride > max_tensor_size - total_len) {
            RTP_LLM_LOG_WARNING("staging allocation is too large, block count %zu", blocks.size());
            return {};
        }
        layout.push_back({block.get(), stride});
        total_len += stride;
    }
    if (rdma) {
        std::sort(layout.begin(), layout.end(), [](const StagingBlock& a, const StagingBlock& b) {
            return a.stride < b.stride;
        });
    }

    auto tensor =
        torch::empty({static_cast<int64_t>(total_len)}, torch::TensorOptions().dtype(torch::kUInt8)).pin_memory();
    if (!tensor.defined()) {
        RTP_LLM_LOG_WARNING(
            "make valid blocks failed, alloc %zu bytes failed, block count %zu", total_len, blocks.size());
        return {};
    }
    std::vector<std::shared_ptr<BlockBuffer>> staged_blocks;
    staged_blocks.reserve(blocks.size());
    MultiCopyParams copy_params;
    copy_params.multi_dst.reserve(blocks.size());
    copy_params.multi_src.reserve(blocks.size());
    const auto options = torch::TensorOptions().dtype(torch::kUInt8);
    auto*      cursor  = static_cast<char*>(tensor.data_ptr());
    for (auto slot = layout.begin(); slot != layout.end();) {
        const size_t stride = slot->stride;
        // RDMA can split a registration at multiples of its alignment. Equal
        // strides ensure each split falls between blocks.
        auto end = layout.end();
        if (rdma) {
            end = std::find_if(slot, end, [stride](const StagingBlock& entry) { return entry.stride != stride; });
        }
        const size_t bytes              = rdma ? (end - slot) * stride : total_len;
        const bool   needs_registration = rdma || !memory_util_->isMemoryMr(cursor, bytes, false, false);
        if (needs_registration && !memory_util_->regUserMr(cursor, bytes, false, rdma ? stride : 0)) {
            RTP_LLM_LOG_WARNING("make valid blocks failed to reg mr, size %zu, block count %td", bytes, end - slot);
            return {};
        }

        // Readers can outlive the request. Release the registration before
        // dropping the pinned allocation, including on an early return.
        auto region = std::shared_ptr<void>(
            cursor, [tensor, registration = needs_registration ? memory_util_ : nullptr](void* ptr) {
                if (registration) {
                    registration->deregUserMr(ptr, false);
                }
            });
        for (; slot != end; ++slot) {
            const auto* block = slot->block;
            auto        addr  = std::shared_ptr<void>(region, cursor);
            staged_blocks.push_back(std::make_shared<BlockBuffer>(block->key, addr, block->len, false, true));
            copy_params.multi_dst.push_back(
                torch::from_blob(cursor, {static_cast<int64_t>(block->len)}, options.device(torch::kCPU)));
            copy_params.multi_src.push_back(
                torch::from_blob(block->addr.get(),
                                 {static_cast<int64_t>(block->len)},
                                 options.device(block->gpu_mem ? torch::kCUDA : torch::kCPU)));
            cursor += slot->stride;
        }
    }

    const auto copy_begin_us = currentTimeUs();
    execNoBlockCopy(copy_params);
    RTP_LLM_INTERVAL_LOG(120,
                         INFO,
                         "stage block cache once, block count %zu, size %zu bytes, cost %ldus",
                         blocks.size(),
                         total_len,
                         currentTimeUs() - copy_begin_us);
    return staged_blocks;
}

void RequestBlockBufferStore::delRequestBlockBuffer(const std::string& requestid) {
    std::shared_ptr<RequestBlockBuffer> request_block_buffer;
    {
        std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
        auto                                iter = request_cache_map_.find(requestid);
        if (iter != request_cache_map_.end()) {
            request_block_buffer          = iter->second;
            request_cache_map_[requestid] = nullptr;
        }
    }
    if (request_block_buffer) {
        request_block_buffer->notifyRequestDone();
    }

    {
        std::unique_lock<std::shared_mutex> lock(request_cache_map_mutex_);
        // Append-ordered, so the oldest entries are at the front.
        size_t expired_count = 0;
        while (expired_count < expired_request_caches_.size()
               && currentTimeUs() - expired_request_caches_[expired_count].second > expired_request_cache_ttl_us_) {
            request_cache_map_.erase(expired_request_caches_[expired_count].first);
            ++expired_count;
        }
        expired_request_caches_.erase(expired_request_caches_.begin(), expired_request_caches_.begin() + expired_count);
        expired_request_caches_.push_back({requestid, currentTimeUs()});
    }
}

bool RequestBlockBufferStore::regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) {
    std::unique_lock<std::shared_mutex> lock(buffer_map_mutex_);
    for (auto& buffer : buffers) {
        buffer_map_[buffer->key] = buffer;
    }
    RTP_LLM_LOG_INFO("reg user buffer count %d", buffers.size());
    return true;
}

std::shared_ptr<BlockBuffer> RequestBlockBufferStore::findUserBuffer(const std::string& key) {
    std::shared_lock<std::shared_mutex> lock(buffer_map_mutex_);
    auto                                it = buffer_map_.find(key);
    if (it == buffer_map_.end()) {
        RTP_LLM_LOG_INFO("find user buffer failed, key %s, current count %d", key.c_str(), buffer_map_.size());
        return nullptr;
    }
    return it->second;
}

}  // namespace rtp_llm
