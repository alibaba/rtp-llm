#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferServiceImplContext.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"

namespace rtp_llm {

CacheTransferServiceImplContext::CacheTransferServiceImplContext(
    const ::CacheTransferRequest*                        request,
    ::CacheTransferResponse*                             response,
    ::google::protobuf::Closure*                         done,
    const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
    const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks,
    const std::shared_ptr<LockedBlockBufferManager>&     locked_block_buffer_manager,
    const std::shared_ptr<MemoryUtil>&                   memory_util,
    const std::shared_ptr<TransferConnection>&           transfer_connection):
    request_(request),
    response_(response),
    done_(done),
    request_id_(request->request_id()),
    client_ip_(request->client_ip()),
    local_blocks_(local_blocks),
    remote_blocks_(remote_blocks),
    locked_block_buffer_manager_(locked_block_buffer_manager),
    memory_util_(memory_util),
    transfer_connection_(transfer_connection),
    unfinished_count_(local_blocks.size()) {}

void CacheTransferServiceImplContext::run() {

    if (!locked_block_buffer_manager_->lock(local_blocks_)) {
        RTP_LLM_LOG_WARNING("cache store service lock local block failed, request id is %s, request from %s",
                            request_id_.c_str(),
                            client_ip_.c_str());
        runFailed(CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    // transfer blocks
    auto callback = [shared_this = shared_from_this()](bool                                             success,
                                                       CacheStoreErrorCode                              error_code,
                                                       const std::vector<std::shared_ptr<BlockBuffer>>& local_blocks) {
        shared_this->notifyDone(success, error_code, local_blocks);
    };
    transfer_connection_->read(local_blocks_, remote_blocks_, callback, request_->timeout_ms());
}

void CacheTransferServiceImplContext::notifyDone(bool                                             success,
                                                 CacheStoreErrorCode                              error_code,
                                                 const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    // buffer load done, should unlock
    locked_block_buffer_manager_->unlock(blocks);

    std::unique_lock<std::mutex> lock(mutex_);
    if (finished_) {
        RTP_LLM_LOG_WARNING(
            "cache store service transfer notify done, but already finished, request id is %s, request from %s",
            request_id_.c_str(),
            client_ip_.c_str());
        return;
    }

    if (error_code != CacheStoreErrorCode::None) {
        RTP_LLM_LOG_WARNING("cache store service transfer failed, request_id=%s request_from=%s error_code=%s "
                            "callback_blocks=%zu unfinished_blocks=%d total_blocks=%zu",
                            request_id_.c_str(),
                            client_ip_.c_str(),
                            CacheStoreErrorCodeToString(error_code).c_str(),
                            blocks.size(),
                            unfinished_count_,
                            local_blocks_.size());
        finished_ = true;
        runFailed(error_code);
        return;
    }

    if (blocks.size() > static_cast<size_t>(unfinished_count_)) {
        RTP_LLM_LOG_ERROR(
            "cache store service transfer completion invariant violated, request_id=%s request_from=%s "
            "unfinished_blocks=%d callback_blocks=%zu total_blocks=%zu",
            request_id_.c_str(), client_ip_.c_str(), unfinished_count_, blocks.size(), local_blocks_.size());
        finished_ = true;
        runFailed(CacheStoreErrorCode::LoadErrorUnknown);
        return;
    }

    unfinished_count_ -= blocks.size();
    if (unfinished_count_ == 0) {
        finished_ = true;
        runSuccess();
    }
}

void CacheTransferServiceImplContext::runSuccess() {
    response_->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
    done_->Run();
}

void CacheTransferServiceImplContext::runFailed(CacheStoreErrorCode ec) {
    response_->set_error_code(CacheStoreUtil::toKvCacheStoreErrorCode(ec));
    done_->Run();
}

}  // namespace rtp_llm
