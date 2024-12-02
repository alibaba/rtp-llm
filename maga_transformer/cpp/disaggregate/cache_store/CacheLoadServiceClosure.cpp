#include "maga_transformer/cpp/disaggregate/cache_store/CacheLoadServiceClosure.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

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
    if (controller_->Failed()) {
        FT_LOG_WARNING("cache load request failed, controller err is %d", controller_->GetErrorCode());
        end(false, fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        FT_LOG_WARNING("cache load request failed, response err is %d", response_->error_code());
        end(false, fromResponseErrorCode(response_->error_code()));
        return;
    }

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        FT_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            FT_LOG_WARNING(
                      "can not find match block %s from response, request is %s",
                      block.key().c_str(),
                      request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        auto dst_buffer = unload_block->toDeviceBuffer();
        auto src_buffer = fastertransformer::Buffer(
            fastertransformer::MemoryType::MEMORY_CPU,
            fastertransformer::DataType::TYPE_UINT8,
            {block.len()}, block.content().data());
        device_->noBlockCopy({dst_buffer, src_buffer});
    }
    end(true, CacheStoreErrorCode::None);
}

CacheStoreErrorCode CacheLoadServiceClosure::fromArpcErrorCode(arpc::ErrorCode ec) {
    switch (ec) {
        case arpc::ARPC_ERROR_TIMEOUT:
            return CacheStoreErrorCode::CallPrefillTimeout;
        case arpc::ARPC_ERROR_CONNECTION_CLOSED:
        case arpc::ARPC_ERROR_METHOD_NOT_FOUND:
        case arpc::ARPC_ERROR_POST_PACKET:
            return CacheStoreErrorCode::LoadSendRequestFailed;
        case arpc::ARPC_ERROR_PUSH_WORKITEM:
        case arpc::ARPC_ERROR_QUEUE_FULL:
            return CacheStoreErrorCode::PushWorkerItemFailed;
        default:
            return CacheStoreErrorCode::LoadErrorUnknown;
    }
}

CacheStoreErrorCode CacheLoadServiceClosure::fromResponseErrorCode(KvCacheStoreServiceErrorCode ec) {
    switch (ec) {
        case KvCacheStoreServiceErrorCode::EC_SUCCESS:
            return CacheStoreErrorCode::None;
        case KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ:
            return CacheStoreErrorCode::LoadSendRequestFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION:
            return CacheStoreErrorCode::LoadRdmaConnectFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE:
            return CacheStoreErrorCode::LoadRdmaWriteFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER:
            return CacheStoreErrorCode::LoadBufferTimeout;
        default:
            return CacheStoreErrorCode::LoadErrorUnknown;
    }
}

void CacheLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    CacheStoreClientLoadMetricsCollector::markEnd(collector_, success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm
