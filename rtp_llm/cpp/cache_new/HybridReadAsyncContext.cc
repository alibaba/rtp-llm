#include "rtp_llm/cpp/cache_new/HybridReadAsyncContext.h"
#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache_new/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HybridReadAsyncContext::HybridReadAsyncContext(int64_t                                   request_id,
                                               const std::shared_ptr<KVCacheResourceV1>& resource,
                                               const std::shared_ptr<KVCacheConnector>&  memory_connector,
                                               const std::shared_ptr<KVCacheConnector>&  remote_connector):
    request_id_(request_id),
    resource_(resource),
    memory_connector_(memory_connector),
    remote_connector_(remote_connector) {
    assert(resource != nullptr);
    if (memory_connector_ != nullptr) {
        memory_context_ = memory_connector_->asyncRead(resource, nullptr);
    } else if (remote_connector_ != nullptr) {
        genRemoteContext();
    }
}

void HybridReadAsyncContext::genRemoteContext() const {
    std::string                             unique_id    = "";  // TODO : support lora
    auto                                    trace_id_str = std::to_string(request_id_);
    std::shared_ptr<KVCacheConnector::Meta> remote_connector_meta =
        std::make_shared<RemoteConnectorMeta>(unique_id, trace_id_str);
    remote_context_ = remote_connector_->asyncRead(resource_, remote_connector_meta);
}

void HybridReadAsyncContext::waitDone() {
    if (auto memory_context = memory_context_; memory_context != nullptr) {
        memory_context->waitDone();
        if (remote_connector_ != nullptr) {
            genRemoteContext();
            memory_context_.reset();
        }
    }
    if (remote_context_ != nullptr) {
        remote_context_->waitDone();
        remote_reuse_block_num_ =
            std::static_pointer_cast<RemoteConnectorAsyncContext>(remote_context_)->remote_reuse_block_num();
    }
}

bool HybridReadAsyncContext::done() const {
    if (auto memory_context = memory_context_; memory_context != nullptr) {
        if (!memory_context->done()) {
            return false;
        }
        if (!memory_context->success()) {
            RTP_LLM_LOG_WARNING("memory_connector asyncRead failed");
            return true;
        }
        // TODO : this may be not efficient, have to call done() to enable next read
        if (remote_connector_ != nullptr) {
            genRemoteContext();
            memory_context_.reset();
            return false;
        }
        return true;
    }

    if (remote_context_ != nullptr) {
        if (!remote_context_->done()) {
            return false;
        }
        remote_reuse_block_num_ =
            std::static_pointer_cast<RemoteConnectorAsyncContext>(remote_context_)->remote_reuse_block_num();
    }

    return true;
}

bool HybridReadAsyncContext::success() const {
    // call this function after done
    if (memory_context_ != nullptr) {
        return memory_context_->success();
    }
    if (remote_context_ != nullptr) {
        return remote_context_->success();
    }
    return true;
}

}  // namespace rtp_llm