#include "maga_transformer/cpp/disaggregate/cache_store/CacheLoadServiceClosure.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, CacheLoadServiceClosure);

CacheLoadServiceClosure::~CacheLoadServiceClosure() {
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

void CacheLoadServiceClosure::Run() {
    CacheStoreClientLoadMetricsCollector::setResponseReceiveCost(
        collector_, autil::TimeUtility::currentTimeInMicroSeconds() - response_->response_send_start_time_us());
    if (controller_->Failed() || response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        AUTIL_LOG(WARN,
                  "cache load request failed, controller err is %d, response err is %d",
                  controller_->GetErrorCode(),
                  response_->error_code());
        end(false);
        return;
    }

    if (memory_util_->rdmaMode()) {
        // write成功, 直接回调
        end(true);
        return;
    }

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        AUTIL_LOG(WARN, "cache load response block count not equal to request block buffer");
        end(false);
        return;
    }

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            AUTIL_LOG(WARN,
                      "can not find match block %s from response, request is %s",
                      block.key().c_str(),
                      request_block_buffer_->getRequestId().c_str());
            end(false);
            return;
        }

        if (!memory_util_->memcopy(unload_block->addr.get(),
                                   unload_block->gpu_mem,
                                   block.content().data(),
                                   false,
                                   block.len())) {
            AUTIL_LOG(WARN,
                      "copy load response to dst block failed, block %s, request %s",
                      block.key().c_str(),
                      request_block_buffer_->getRequestId().c_str());
            end(false);
            return;
        }
    }
    end(true);
}

void CacheLoadServiceClosure::end(bool success) {
    CacheStoreClientLoadMetricsCollector::markEnd(collector_, success);
    callback_(success);
    delete this;
}

}  // namespace rtp_llm
