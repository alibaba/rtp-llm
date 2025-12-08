#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecode.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeWorker.h"

namespace rtp_llm {

P2PConnectorDecode::P2PConnectorDecode(const GptInitParameter&                  gpt_init_parameter,
                                       DeviceBase*                              device_base,
                                       const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator):
    gpt_init_parameter_(gpt_init_parameter), device_base_(device_base), kv_cache_allocator_(kv_cache_allocator) {}

P2PConnectorDecode::~P2PConnectorDecode() {}

bool P2PConnectorDecode::init() {
    if (gpt_init_parameter_.tp_rank_ == 0) {
        scheduler_ = std::make_shared<P2PConnectorDecodeScheduler>(gpt_init_parameter_);
        if (!scheduler_->init()) {
            RTP_LLM_LOG_ERROR("P2PConnectorDecode init failed: scheduler is null");
            return false;
        }
    }

    worker_ = std::make_shared<P2PConnectorDecodeWorker>(gpt_init_parameter_, device_base_, kv_cache_allocator_);
    if (!worker_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecode init failed: worker is null");
        return false;
    }

    RTP_LLM_LOG_INFO("P2PConnectorDecode init success");
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
P2PConnectorDecode::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    if (!scheduler_) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecode::asyncRead: scheduler is null, only rank 0 can call this");
        return nullptr;
    }

    auto decode_meta = std::dynamic_pointer_cast<P2PConnectorDecodeMeta>(meta);
    if (!decode_meta) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecode::asyncRead: meta is not P2PConnectorDecodeMeta");
        return nullptr;
    }

    auto async_context = scheduler_->asyncRead(resource,
                                               decode_meta->requestId(),
                                               decode_meta->uniqueKey(),
                                               decode_meta->prefillIp(),
                                               decode_meta->prefillPort(),
                                               decode_meta->deadlineMs());
    if (!async_context) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecode::asyncRead: scheduler_->asyncRead failed");
        return nullptr;
    }

    return async_context;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
P2PConnectorDecode::asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    RTP_LLM_LOG_ERROR("P2PConnectorDecode::asyncWrite not supported");
    return nullptr;
}

std::shared_ptr<KVCacheConnector::AsyncContext> P2PConnectorDecode::asyncWriteByLayer(
    int layer_id, const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    RTP_LLM_LOG_ERROR("P2PConnectorDecode::asyncWriteByLayer not supported");
    return nullptr;
}

std::shared_ptr<TPBroadcastService::Callback> P2PConnectorDecode::makeCallback() {
    if (!worker_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecode::makeCallback: worker is null");
        return nullptr;
    }
    return std::make_shared<P2PConnectorDecodeWorkerTPCallback>(worker_);
}

}  // namespace rtp_llm
