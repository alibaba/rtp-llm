#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

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
    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("rdma read service closure failed, request %s, controller err is %d",
                            transfer_request_->request_id.c_str(),
                            controller_->GetErrorCode());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("rdma read service closure failed, request %s, response err is %d",
                            transfer_request_->request_id.c_str(),
                            response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    RTP_LLM_LOG_DEBUG("rdma read service closure success, request %s", transfer_request_->request_id.c_str());
    end(true, CacheStoreErrorCode::None);
}

void CacheTransferServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    transfer_request_->callback(success, ec, transfer_request_->buffer_pairs);
    delete this;
}

}  // namespace rtp_llm