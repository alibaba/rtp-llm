#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <exception>
#include <memory>
#include <torch/torch.h>
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreDevicePin.h"
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
    try {
        collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());
        if (!tryPinThreadDevice(device_id_, "cache load request")) {
            end(false, CacheStoreErrorCode::LoadErrorUnknown);
            return;
        }

        if (controller_->Failed()) {
            RTP_LLM_LOG_WARNING(
                "cache load request failed, source=client stage=rpc request_id=%s client_ip=%s timeout_ms=%u "
                "arpc_error_code=%d",
                request_block_buffer_->getRequestId().c_str(),
                request_->client_ip().c_str(),
                request_->has_timeout_ms() ? request_->timeout_ms() : 0,
                controller_->GetErrorCode());
            end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
            return;
        }

        if (!response_->has_error_code() || response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
            RTP_LLM_LOG_WARNING(
                "cache load request failed, source=client stage=remote_response request_id=%s "
                "has_error_code=%d response_error_code=%d expected_blocks=%zu response_blocks=%d",
                request_block_buffer_->getRequestId().c_str(),
                response_->has_error_code(),
                response_->error_code(),
                request_block_buffer_->getBlocksCount(),
                response_->blocks_size());
            end(false, response_->has_error_code() ? CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()) :
                                                     CacheStoreErrorCode::LoadErrorUnknown);
            return;
        }

        // TCP Mode 下需要Copy数据
        if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
            RTP_LLM_LOG_WARNING(
                "cache load response validation failed, source=client stage=validate_response request_id=%s "
                "expected_blocks=%zu response_blocks=%d",
                request_block_buffer_->getRequestId().c_str(),
                request_block_buffer_->getBlocksCount(),
                response_->blocks_size());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        for (int i = 0; i < response_->blocks_size(); i++) {
            const auto& block        = response_->blocks(i);
            auto        unload_block = request_block_buffer_->getBlock(block.key());

            if (unload_block == nullptr || block.len() != unload_block->len) {
                RTP_LLM_LOG_WARNING(
                    "cache load response block mismatch, source=client stage=validate_block request_id=%s "
                    "block_index=%d block_key=%s expected_len=%d actual_len=%u block_found=%d",
                    request_block_buffer_->getRequestId().c_str(), i, block.key().c_str(),
                    unload_block == nullptr ? -1 : unload_block->len, block.len(), unload_block != nullptr);
                end(false, CacheStoreErrorCode::LoadBufferTimeout);
                return;
            }

            auto dst_tensor = torch::from_blob(
                unload_block->addr.get(),
                {(int64_t)unload_block->len},
                torch::TensorOptions().dtype(torch::kUInt8).device(unload_block->gpu_mem ? torch::kCUDA : torch::kCPU));
            auto src_tensor = torch::from_blob(const_cast<char*>(block.content().data()),
                                               {(int64_t)block.len()},
                                               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
            execNoBlockCopy({dst_tensor, src_tensor});
        }
        end(true, CacheStoreErrorCode::None);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cache load request exception, source=client stage=closure request_id=%s exception=%s",
                          request_block_buffer_ ? request_block_buffer_->getRequestId().c_str() : "unknown",
                          e.what());
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
    } catch (...) {
        RTP_LLM_LOG_ERROR("cache load request exception, source=client stage=closure request_id=%s exception=unknown",
                          request_block_buffer_ ? request_block_buffer_->getRequestId().c_str() : "unknown");
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
    }
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    std::unique_ptr<TcpCacheStoreLoadServiceClosure> self(this);
    collector_->markEnd(success);
    try {
        callback_(success, ec);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cache load completion callback exception, request_id=%s error_code=%s exception=%s",
                          request_block_buffer_ ? request_block_buffer_->getRequestId().c_str() : "unknown",
                          CacheStoreErrorCodeToString(ec).c_str(),
                          e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("cache load completion callback exception, request_id=%s error_code=%s exception=unknown",
                          request_block_buffer_ ? request_block_buffer_->getRequestId().c_str() : "unknown",
                          CacheStoreErrorCodeToString(ec).c_str());
    }
}

}  // namespace rtp_llm
