#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefill.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillWorker.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorStreamStore.h"
#include <thread>
#include <chrono>

namespace rtp_llm {

P2PConnectorPrefill::P2PConnectorPrefill(const GptInitParameter&                  gpt_init_parameter,
                                         DeviceBase*                              device_base,
                                         const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator):
    gpt_init_parameter_(gpt_init_parameter), device_base_(device_base), kv_cache_allocator_(kv_cache_allocator) {}

P2PConnectorPrefill::~P2PConnectorPrefill() = default;

bool P2PConnectorPrefill::init() {
    RTP_LLM_LOG_INFO("P2PConnectorPrefill init start");
    if (gpt_init_parameter_.tp_rank_ == 0) {
        scheduler_ = std::make_shared<P2PConnectorPrefillScheduler>(gpt_init_parameter_);
        if (!scheduler_->init()) {
            RTP_LLM_LOG_ERROR("P2PConnectorPrefill init failed: scheduler is null");
            return false;
        }
        RTP_LLM_LOG_INFO("P2PConnectorPrefill init scheduler success");
    }

    worker_ = std::make_shared<P2PConnectorPrefillWorker>(gpt_init_parameter_, device_base_, kv_cache_allocator_);
    if (!worker_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill init failed: worker is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorPrefill init worker success");

    stream_store_ = std::make_shared<PrefillConnectorStreamStore>();
    if (!stream_store_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill init failed: stream_store init failed");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorPrefill init stream_store success");
    RTP_LLM_LOG_INFO("P2PConnectorPrefill init success");
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
P2PConnectorPrefill::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    RTP_LLM_LOG_ERROR("P2PConnectorPrefill::asyncRead not supported");
    return nullptr;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
P2PConnectorPrefill::asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    RTP_LLM_LOG_ERROR("P2PConnectorPrefill::asyncWrite not supported");
    return nullptr;
}

std::shared_ptr<KVCacheConnector::AsyncContext> P2PConnectorPrefill::asyncWriteByLayer(
    int layer_id, const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    // async write by layer call by each rank
    worker_->writeByLayer(layer_id, resource, meta->requestId(), meta->event());
    RTP_LLM_LOG_INFO("P2PConnectorPrefill::asyncWriteByLayer: writeByLayer success");
    return nullptr;  // writeByLayer 是同步的，返回 nullptr
}

grpc::Status
P2PConnectorPrefill::handleWrite(const std::string&                                   unique_key,
                                 const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                 int64_t                                              deadline_ms) {
    RTP_LLM_LOG_INFO("P2PConnectorPrefill::handleWrite: deadline_ms: %lld", deadline_ms);
    std::shared_ptr<GenerateStream> stream;
    while (currentTimeMs() < deadline_ms) {
        stream = stream_store_->stealStream(unique_key);
        if (!stream) {
            RTP_LLM_LOG_INFO("P2PConnectorPrefill::handleWrite: stream is null, sleep 1ms");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        break;
    }

    RTP_LLM_LOG_INFO("P2PConnectorPrefill::handleWrite: stream: %p", stream.get());

    if (!stream) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill::handleWrite failed: stream is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "stream is null");
    }

    if (!scheduler_) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill::handleWrite failed: scheduler is null, only rank 0 can call this");
        return grpc::Status(grpc::StatusCode::INTERNAL, "scheduler is null");
    }

    // scheduler_->write 只需要 stream 和 decode_transfer_servers
    auto batch_kv_cache = stream->kvCache();
    if (batch_kv_cache.batchSize() == 0) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill::handleWrite failed: batch_kv_cache is empty");
        return grpc::Status(grpc::StatusCode::INTERNAL, "batch_kv_cache is empty");
    }
    auto resource = batch_kv_cache.resource(0);
    if (resource.layerBlockIds().size() == 0 || resource.cacheKeys().size() == 0) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill::handleWrite failed: resource layer block ids or cache keys is empty");
        return grpc::Status(grpc::StatusCode::INTERNAL, "resource layer block ids or cache keys is empty");
    }

    // TODO: optimize this copy
    std::shared_ptr<KVCacheResourceV1> resource_ptr = std::make_shared<KVCacheResourceV1>(resource);
    return scheduler_->write(resource_ptr,
                             stream->getPdSeparationUniqueKey(),
                             stream->streamId(),
                             decode_transfer_servers,
                             stream->getDeadlineMs());
}

void P2PConnectorPrefill::addStream(const std::string& unique_key, GenerateStreamPtr stream) {
    RTP_LLM_LOG_INFO("P2PConnectorPrefill::addStream: unique_key: %s", unique_key.c_str());
    stream_store_->addStream(unique_key, stream);
}

std::shared_ptr<TPBroadcastService::Callback> P2PConnectorPrefill::makeCallback() {
    if (!worker_) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefill::makeCallback: worker is null");
        return nullptr;
    }
    return std::make_shared<P2PConnectorPrefillWorkerTPCallback>(worker_);
}

}  // namespace rtp_llm