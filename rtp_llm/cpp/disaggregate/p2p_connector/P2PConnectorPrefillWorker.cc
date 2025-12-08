#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillWorker.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/LayerBlockConvertorImpl.h"
#include <thread>
#include <chrono>
#include <algorithm>

namespace rtp_llm {

P2PConnectorPrefillWorker::P2PConnectorPrefillWorker(const GptInitParameter&                  gpt_init_parameter,
                                                     DeviceBase*                              device_base,
                                                     const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator):
    gpt_init_parameter_(gpt_init_parameter),
    device_base_(device_base),
    kv_cache_allocator_(kv_cache_allocator),
    asymmetric_tp_util_(std::make_shared<AsymmetricTpUtil>(gpt_init_parameter)),
    computed_buffers_(std::make_shared<ComputedLayerCacheBufferStore>()),
    load_contexts_(std::make_shared<PrefillWorkerLoadContextStore>()) {}

P2PConnectorPrefillWorker::~P2PConnectorPrefillWorker() {
    store_wait_thread_stop_ = true;
    if (store_wait_thread_.joinable()) {
        store_wait_thread_.join();
    }
}

bool P2PConnectorPrefillWorker::init() {
    if (!kv_cache_allocator_) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker init failed: kv_cache_allocator is null");
        return false;
    }

    // init layer block converter
    auto layer_block_converter = std::make_shared<LayerBlockConvertorImpl>(kv_cache_allocator_);

    // init transfer client
    transfer_client_ = std::make_shared<TransferClient>(layer_block_converter, device_base_);
    if (!transfer_client_) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker init failed: transfer_client is null");
        return false;
    }
    if (!transfer_client_->init(gpt_init_parameter_.cache_store_config.cache_store_rdma_mode,
                                gpt_init_parameter_.cache_store_config.messager_io_thread_count,
                                gpt_init_parameter_.cache_store_config.messager_io_thread_count,
                                gpt_init_parameter_.cache_store_config.messager_worker_thread_count)) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker init failed: transfer_client init failed");
        return false;
    }

    // register buffers
    auto buffers = kv_cache_allocator_->getAllBuffers();
    for (auto& [buffer, size] : buffers) {
        if (!transfer_client_->registerUserMr(buffer, size)) {
            RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker init failed: register user mr failed, buffer: %p, size: %ld",
                              buffer->data(),
                              size);
            return false;
        }
    }

    // init store wait thread
    store_wait_thread_ = std::thread([this]() { this->storeWaitThread(); });
    RTP_LLM_LOG_INFO("P2PConnectorPrefillWorker init success");
    return true;
}

bool P2PConnectorPrefillWorker::writeByLayer(int                                       layer_id,
                                             const std::shared_ptr<KVCacheResourceV1>& resource,
                                             int64_t                                   request_id,
                                             DeviceEventPtr                            event) {

    auto layer_cache_buffer = LayerCacheBufferUtil::convert(*resource, 0, layer_id);
    if (!layer_cache_buffer) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker writeByLayer failed: layer_cache_buffer is null");
        return false;
    }

    int64_t deadline_ms = currentTimeMs() + store_wait_timeout_ms_;
    {
        std::unique_lock<std::mutex> lock(store_wait_mutex_);
        store_wait_contexts_.emplace_back(request_id, event, layer_cache_buffer, deadline_ms);
    }
    RTP_LLM_LOG_INFO("P2PConnectorPrefillWorker writeByLayer end, request_id: %ld, layer_id: %d", request_id, layer_id);
    return true;
}

bool P2PConnectorPrefillWorker::write(int64_t                                              request_id,
                                      const std::string&                                   unique_key,
                                      int64_t                                              deadline_ms,
                                      const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    auto asymmetric_tp_contexts = asymmetric_tp_util_->handleAsymmetricTP(decode_transfer_servers);
    if (asymmetric_tp_contexts.empty()) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillWorker write: asymmetric_tp_contexts is empty");
        return false;
    }

    auto load_context = load_contexts_->addContext(
        request_id, unique_key, deadline_ms, asymmetric_tp_contexts, gpt_init_parameter_.num_layers_);

    while (currentTimeMs() < deadline_ms) {
        if (load_context->isCanceled() || load_context->isTimeout()) {
            RTP_LLM_LOG_INFO("P2PConnectorPrefillWorker write, request_id: %ld, load_context is canceled or timeout",
                             request_id);
            break;
        }

        if (load_context->isDone()) {
            RTP_LLM_LOG_INFO(
                "P2PConnectorPrefillWorker write, request_id: %ld, load_context is done, all transfers are done",
                request_id);
            break;
        }

        // get computed layer cache buffer
        // TODO: change to condition wait
        auto computed_layer_cache_buffer = computed_buffers_->getBuffer(request_id);
        if (!computed_layer_cache_buffer) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        for (auto [layer_id, layer_cache_buffer] : computed_layer_cache_buffer->layer_cache_buffers) {
            if (!load_context->needTransfer(layer_id)) {
                continue;
            }
            load_context->startTransfer(layer_id);
            auto& asymmetric_tp_contexts = load_context->asymmetricTPContexts();
            for (size_t i = 0; i < asymmetric_tp_contexts.size(); i++) {
                const auto& asymmetric_tp_context = asymmetric_tp_contexts[i];
                auto        id = layer_id * static_cast<int>(asymmetric_tp_contexts.size()) + static_cast<int>(i);
                RTP_LLM_LOG_INFO(
                    "P2PConnectorPrefillWorker write, request_id: %ld, layer_id: %d, transfer to remote, id: %d",
                    request_id,
                    layer_id,
                    id);
                transfer_client_->transfer(
                    asymmetric_tp_context.decode_ip,
                    asymmetric_tp_context.decode_port,
                    unique_key,
                    layer_cache_buffer,
                    static_cast<uint32_t>(asymmetric_tp_context.local_partition_count),
                    static_cast<uint32_t>(asymmetric_tp_context.local_partition_id),
                    static_cast<uint32_t>(asymmetric_tp_context.remote_partition_count),
                    static_cast<uint32_t>(asymmetric_tp_context.remote_partition_id),
                    [load_context, id](bool success) { load_context->notifyDone(id, success); },
                    deadline_ms - currentTimeMs());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // write task done , remove task from store
    load_contexts_->removeContext(request_id);

    // wait until all transfers are done
    while (!load_context->isAllTransfersDone()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return load_context->isAllSuccess() && load_context->isDone();
}

void P2PConnectorPrefillWorker::cancelWrite(int64_t request_id, const std::string& unique_key) {
    auto load_context = load_contexts_->getContext(request_id);
    if (load_context) {
        load_context->setCanceled();
    }
}

void P2PConnectorPrefillWorker::storeWaitThread() {
    while (!store_wait_thread_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        storeWaitThreadProcess();

        // clear expired computed layer cache buffers
        computed_buffers_->checkTimeout();

        // clear expired load context
        load_contexts_->checkTimeout();
    }
}

void P2PConnectorPrefillWorker::storeWaitThreadProcess() {
    std::unique_lock<std::mutex> lock(store_wait_mutex_);
    auto                         iter = store_wait_contexts_.begin();
    while (iter != store_wait_contexts_.end()) {
        auto& [request_id, device_event, computed_layer_cache_buffer, deadline_ms] = *iter;
        int64_t current_time_ms                                                    = currentTimeMs();
        if (current_time_ms >= deadline_ms) {
            RTP_LLM_LOG_WARNING("store wait timeout, request_id: %ld, deadline_ms: %ld, current_time_ms: %ld",
                                request_id,
                                deadline_ms,
                                current_time_ms);
            iter = store_wait_contexts_.erase(iter);
            continue;
        }

        // RTP_LLM_LOG_INFO("P2PConnectorPrefillWorker storeWaitThread device_event: %p, checkReadiness: %d",
        // device_event.get(), device_event->checkReadiness());
        if (device_event == nullptr || device_event->checkReadiness()) {
            computed_buffers_->addBuffer(request_id, computed_layer_cache_buffer, deadline_ms);
            iter = store_wait_contexts_.erase(iter);
            RTP_LLM_LOG_INFO(
                "P2PConnectorPrefillWorker storeWaitThread add computed_layer_cache_buffer, request_id: %ld",
                request_id);
        } else {
            ++iter;
        }
    }
}

P2PConnectorPrefillWorkerTPCallback::P2PConnectorPrefillWorkerTPCallback(
    const std::shared_ptr<P2PConnectorPrefillWorker>& p2p_connector_prefill_worker):
    p2p_connector_prefill_worker_(p2p_connector_prefill_worker) {}

bool P2PConnectorPrefillWorkerTPCallback::shouldProcess(const BroadcastTpRequestPB& request) {
    return request.has_p2p_request();
}

grpc::Status P2PConnectorPrefillWorkerTPCallback::onBroadcastTp(const BroadcastTpRequestPB& request,
                                                                BroadcastTpResponsePB&      response) {
    auto p2p_request  = request.p2p_request();
    auto request_id   = p2p_request.request_id();
    auto unique_key   = p2p_request.unique_key();
    auto deadline_ms  = p2p_request.deadline_ms();
    auto layer_blocks = p2p_request.layer_blocks();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& peer_worker : p2p_request.peer_workers()) {
        decode_transfer_servers.push_back(std::make_pair(peer_worker.ip(), peer_worker.cache_store_port()));
    }
    bool success = p2p_connector_prefill_worker_->write(request_id, unique_key, deadline_ms, decode_transfer_servers);

    RTP_LLM_LOG_INFO("P2PConnectorPrefillWorkerTPCallback::onBroadcastTp: write success: %d", success);
    response.mutable_p2p_response()->set_success(success);
    return success ? grpc::Status::OK : grpc::Status(grpc::StatusCode::INTERNAL, "write failed");
}

}  // namespace rtp_llm
