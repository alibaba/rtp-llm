#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeWorker.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/LayerBlockConvertorImpl.h"
#include <map>

namespace rtp_llm {

P2PConnectorDecodeWorker::P2PConnectorDecodeWorker(const GptInitParameter&                  gpt_init_parameter,
                                                   DeviceBase*                              device_base,
                                                   const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator):
    gpt_init_parameter_(gpt_init_parameter), device_base_(device_base), kv_cache_allocator_(kv_cache_allocator) {}

P2PConnectorDecodeWorker::~P2PConnectorDecodeWorker() = default;

bool P2PConnectorDecodeWorker::init() {
    if (!kv_cache_allocator_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker init failed: kv_cache_allocator is null");
        return false;
    }

    // init layer block converter
    auto layer_block_convertor = std::make_shared<LayerBlockConvertorImpl>(kv_cache_allocator_);

    // init transfer server
    transfer_server_ = std::make_shared<TransferServer>(layer_block_convertor, device_base_);
    if (!transfer_server_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker init failed: transfer_server is null");
        return false;
    }

    if (!transfer_server_->init(gpt_init_parameter_.cache_store_rdma_mode_,
                                gpt_init_parameter_.cache_store_listen_port_,
                                gpt_init_parameter_.cache_store_config.messager_io_thread_count,
                                gpt_init_parameter_.cache_store_config.messager_worker_thread_count,
                                gpt_init_parameter_.cache_store_config.messager_io_thread_count,
                                gpt_init_parameter_.cache_store_config.messager_worker_thread_count,
                                2,  // TODO: gpt_init_parameter_.cache_store_config.rdma_connections_per_host,
                                gpt_init_parameter_.cache_store_config.rdma_connect_timeout_ms)) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker init failed: transfer_server init failed");
        return false;
    }

    // init layer cache buffer task store
    layer_cache_buffer_task_store_ = transfer_server_->getLayerCacheBufferTaskStore();
    if (!layer_cache_buffer_task_store_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker init failed: get layer_cache_buffer_task_store failed");
        return false;
    }

    // register buffers
    auto buffers = kv_cache_allocator_->getAllBuffers();
    for (auto& [buffer, size] : buffers) {
        if (!transfer_server_->registerUserMr(buffer, size)) {
            RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker init failed: register user mr failed, buffer: %p, size: %ld",
                              buffer->data(),
                              size);
            return false;
        }
    }
    RTP_LLM_LOG_INFO("P2PConnectorDecodeWorker init success");
    return true;
}

bool P2PConnectorDecodeWorker::read(int64_t                                               request_id,
                                    const std::string&                                    unique_key,
                                    int64_t                                               deadline_ms,
                                    const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers) {
    if (!layer_cache_buffer_task_store_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeWorker read failed: layer_cache_buffer_task_store is null");
        return false;
    }

    if (layer_cache_buffers.empty()) {
        // empty layer cache buffers, just return true
        return true;
    }

    // 将 vector 转换为 map
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffer_map;
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        if (layer_cache_buffer) {
            layer_cache_buffer_map[layer_cache_buffer->getLayerId()] = layer_cache_buffer;
        }
    }

    auto layer_cache_buffer_task =
        layer_cache_buffer_task_store_->addTask(unique_key, layer_cache_buffer_map, deadline_ms);
    RTP_LLM_LOG_INFO("add task success, unique_key: %s, layer_cache_buffer_task: %p",
                     unique_key.c_str(),
                     layer_cache_buffer_task.get());
    if (!layer_cache_buffer_task) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecodeWorker read failed: layer_cache_buffer_task is null");
        return false;
    }

    // wait task done, maybe cancel or success
    layer_cache_buffer_task->waitDone();
    layer_cache_buffer_task_store_->stealTask(unique_key);  // remove task from task store

    // wait infligh loading done, should be fast
    layer_cache_buffer_task->waitLoadingDone();
    return layer_cache_buffer_task->success();
}

void P2PConnectorDecodeWorker::cancelRead(int64_t request_id, const std::string& unique_key) {
    if (!layer_cache_buffer_task_store_) {
        return;
    }

    auto layer_cache_buffer_task = layer_cache_buffer_task_store_->stealTask(unique_key);
    if (!layer_cache_buffer_task) {
        return;
    }
    layer_cache_buffer_task->setCancelled();
}

void P2PConnectorDecodeWorker::setLayerCacheBufferTaskStore(
    const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store) {
    layer_cache_buffer_task_store_ = layer_cache_buffer_task_store;
}

P2PConnectorDecodeWorkerTPCallback::P2PConnectorDecodeWorkerTPCallback(
    const std::shared_ptr<P2PConnectorDecodeWorker>& p2p_connector_decode_worker):
    p2p_connector_decode_worker_(p2p_connector_decode_worker) {}

bool P2PConnectorDecodeWorkerTPCallback::shouldProcess(const BroadcastTpRequestPB& request) {
    return request.has_p2p_request();
}

grpc::Status P2PConnectorDecodeWorkerTPCallback::onBroadcastTp(const BroadcastTpRequestPB& request,
                                                               BroadcastTpResponsePB&      response) {
    auto p2p_request = request.p2p_request();
    auto unique_key  = p2p_request.unique_key();
    auto request_id  = p2p_request.request_id();
    auto deadline_ms = p2p_request.deadline_ms();

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (const auto& layer_block_pb : p2p_request.layer_blocks()) {
        auto layer_id = layer_block_pb.layer_id();

        auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        auto cache_keys         = layer_block_pb.cache_keys();
        auto block_ids          = layer_block_pb.block_ids();
        if (cache_keys.size() != block_ids.size()) {
            RTP_LLM_LOG_WARNING("P2PConnectorDecodeTPCallback::onBroadcastTp: cache_keys and block_ids size mismatch");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cache_keys and block_ids size mismatch");
        }
        for (size_t i = 0; i < cache_keys.size(); i++) {
            layer_cache_buffer->addBlockId(cache_keys[i], block_ids[i]);
        }
        layer_cache_buffers.push_back(layer_cache_buffer);
    }

    bool success = p2p_connector_decode_worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);
    response.mutable_p2p_response()->set_success(success);
    return success ? grpc::Status::OK : grpc::Status(grpc::StatusCode::INTERNAL, "read failed");
}

}  // namespace rtp_llm
