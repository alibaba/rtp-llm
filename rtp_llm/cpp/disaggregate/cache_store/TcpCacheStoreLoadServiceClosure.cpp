#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <torch/torch.h>
#include <stdexcept>
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpCacheStoreLoadServiceClosure::~TcpCacheStoreLoadServiceClosure() {
    if (controller_) {
        delete controller_;
    }
    if (request_) {
        delete request_;
    }
    if (response_) {
        delete response_;
    }
}

void TcpCacheStoreLoadServiceClosure::Run() {
    pinThreadToDeviceOnce(device_id_);
    collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("cache load request failed, controller err is %d", controller_->GetErrorCode());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("cache load request failed, response err is %d", response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    try {
        // Validate the complete packet before DMA. Count equality alone cannot
        // detect a duplicate key replacing an omitted block or a truncated payload.
        auto                                      requested_blocks = request_block_buffer_->getBlocks();
        std::vector<std::shared_ptr<BlockBuffer>> destinations;
        destinations.reserve(response_->blocks_size());
        for (int i = 0; i < response_->blocks_size(); i++) {
            const auto& block = response_->blocks(i);
            auto        entry = requested_blocks.find(block.key());
            if (entry == requested_blocks.end() || entry->second == nullptr || block.len() != entry->second->len
                || block.content().size() != block.len() || (block.len() > 0 && entry->second->addr == nullptr)) {
                // Keep callbacks outside try: a throwing callback must never
                // be caught as a copy error and invoked a second time.
                throw std::invalid_argument("invalid or duplicate cache response block: " + block.key());
            }
            destinations.push_back(entry->second);
            requested_blocks.erase(entry);
        }
        for (size_t i = 0; i < destinations.size(); ++i) {
            const auto& block              = response_->blocks(static_cast<int>(i));
            const auto& destination        = destinations[i];
            auto        destination_tensor = torch::from_blob(
                destination->addr.get(),
                {(int64_t)destination->len},
                torch::TensorOptions().dtype(torch::kUInt8).device(destination->gpu_mem ? torch::kCUDA : torch::kCPU));
            auto source_tensor = torch::from_blob(const_cast<char*>(block.content().data()),
                                                  {(int64_t)block.len()},
                                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
            execNoBlockCopy({destination_tensor, source_tensor});
        }
        // Preserve the original per-block copy completion before callback.
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("cache load response copy failed, request %s: %s",
                            request_block_buffer_->getRequestId().c_str(),
                            error.what());
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm
