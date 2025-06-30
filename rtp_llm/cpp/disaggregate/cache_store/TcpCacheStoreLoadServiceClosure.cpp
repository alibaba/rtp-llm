#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"

#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
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

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            RTP_LLM_LOG_WARNING("can not find match block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        auto dst_buffer = unload_block->toDeviceBuffer();
        auto src_buffer = rtp_llm::Buffer(
            rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_UINT8, {block.len()}, block.content().data());
        device_->noBlockCopy({dst_buffer, src_buffer});
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm
