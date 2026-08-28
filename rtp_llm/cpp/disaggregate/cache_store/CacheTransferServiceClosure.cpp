#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <exception>
#include <memory>

namespace rtp_llm {

CacheTransferServiceClosure::~CacheTransferServiceClosure() {
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

void CacheTransferServiceClosure::Run() {
    try {
        if (controller_->Failed()) {
            RTP_LLM_LOG_WARNING(
                "cache transfer request failed, source=client stage=rpc transport=rdma request_id=%s peer=%s:%u "
                "timeout_ms=%lu arpc_error_code=%d block_count=%zu",
                transfer_request_->request_id.c_str(), transfer_request_->ip.c_str(), transfer_request_->port,
                transfer_request_->timeout_ms, controller_->GetErrorCode(), transfer_request_->buffer_pairs.size());
            end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
            return;
        }

        if (!response_->has_error_code() || response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
            RTP_LLM_LOG_WARNING(
                "cache transfer request failed, source=client stage=remote_response transport=rdma request_id=%s "
                "peer=%s:%u has_error_code=%d response_error_code=%d error_message=%s block_count=%zu",
                transfer_request_->request_id.c_str(), transfer_request_->ip.c_str(), transfer_request_->port,
                response_->has_error_code(), response_->error_code(), response_->error_msg().c_str(),
                transfer_request_->buffer_pairs.size());
            end(false, response_->has_error_code() ? CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()) :
                                                     CacheStoreErrorCode::LoadErrorUnknown);
            return;
        }

        RTP_LLM_LOG_DEBUG("rdma read service closure success, request %s", transfer_request_->request_id.c_str());
        end(true, CacheStoreErrorCode::None);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cache transfer request exception, source=client stage=closure transport=rdma request_id=%s "
                          "exception=%s",
                          transfer_request_ ? transfer_request_->request_id.c_str() : "unknown",
                          e.what());
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
    } catch (...) {
        RTP_LLM_LOG_ERROR("cache transfer request exception, source=client stage=closure transport=rdma request_id=%s "
                          "exception=unknown",
                          transfer_request_ ? transfer_request_->request_id.c_str() : "unknown");
        end(false, CacheStoreErrorCode::LoadErrorUnknown);
    }
}

void CacheTransferServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    std::unique_ptr<CacheTransferServiceClosure> self(this);
    try {
        transfer_request_->callback(success, ec, transfer_request_->buffer_pairs);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cache transfer completion callback exception, request_id=%s error_code=%s exception=%s",
                          transfer_request_->request_id.c_str(), CacheStoreErrorCodeToString(ec).c_str(), e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("cache transfer completion callback exception, request_id=%s error_code=%s exception=unknown",
                          transfer_request_->request_id.c_str(), CacheStoreErrorCodeToString(ec).c_str());
    }
}

}  // namespace rtp_llm