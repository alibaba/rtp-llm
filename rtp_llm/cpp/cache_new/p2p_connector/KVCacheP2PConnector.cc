#include "rtp_llm/cpp/cache_new/p2p_connector/KVCacheP2PConnector.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"
#include "autil/StringUtil.h"
#include <set>

using namespace rtp_llm::cache_store;

namespace rtp_llm {

KVCacheP2PLoadAsyncContext::KVCacheP2PLoadAsyncContext() {}

KVCacheP2PLoadAsyncContext::~KVCacheP2PLoadAsyncContext() = default;

void KVCacheP2PLoadAsyncContext::addTPBroadcastResult(const std::shared_ptr<TPBroadcastResult>& tp_broadcast_result) {
    tp_broadcast_results_.push_back(tp_broadcast_result);
}

bool KVCacheP2PLoadAsyncContext::success() const {
    return all_success_;
}

void KVCacheP2PLoadAsyncContext::cancel() {
    // TODO: not support cancel for now, u should wait till done
    // maybe add message to cancel decode load context id
    RTP_LLM_LOG_WARNING("not support to cancel");
    waitDone();
}

void KVCacheP2PLoadAsyncContext::waitDone() {
    for (auto& tp_broadcast_result : tp_broadcast_results_) {
        tp_broadcast_result->waitDone();
        all_success_ &= tp_broadcast_result->success();
    }
}

void KVCacheP2PLoadAsyncContext::setAllSuccess(bool all_success) {
    all_success_ = all_success;
}

KVCacheP2PConnector::KVCacheP2PConnector(const GptInitParameter&             gpt_init_params,
                                         CacheStoreConfig&                   cache_store_config,
                                         DeviceBase*                         device,
                                         const kmonitor::MetricsReporterPtr& metrics_reporter,
                                         KVCacheAllocatorPtr                 kv_cache_allocator,
                                         const std::shared_ptr<TPBroadcast>& tp_broadcast):
    gpt_init_params_(gpt_init_params),
    cache_store_config_(cache_store_config),
    device_(device),
    metrics_reporter_(metrics_reporter),
    kv_cache_allocator_(kv_cache_allocator),
    tp_broadcast_(tp_broadcast),
    tp_rank_(gpt_init_params_.tp_rank_),
    kvcache_reg_mr_(false) {}

KVCacheP2PConnector::~KVCacheP2PConnector() {
    deregUserMr();
}

bool KVCacheP2PConnector::init() {
    // TODO: memory_util_ = createMemoryUtilImpl(cache_store_config_.rdma_mode);

    regUserMr();

    tcp_client_ = std::make_shared<TcpClient>();
    if (!tcp_client_->init(gpt_init_params_.cache_store_config.messager_io_thread_count)) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init tcp client failed");
        return false;
    }

    tcp_server_ = std::make_shared<TcpServer>();
    if (!tcp_server_->init(gpt_init_params_.cache_store_config.messager_io_thread_count,
                           gpt_init_params_.cache_store_config.messager_worker_thread_count,
                           gpt_init_params_.cache_store_listen_port_,
                           true)) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init tcp server failed");
        return false;
    }

    for (auto& worker_addr : gpt_init_params_.worker_addrs_) {
        auto ip_parts = autil::StringUtil::split(worker_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_LOG_WARNING("invalid worker addr [%s]", worker_addr.c_str());
            return false;
        }
        local_worker_infos_.push_back(
            CacheStoreServerWorker(ip_parts[0],
                                   autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0),
                                   autil::StringUtil::strToInt32WithDefault(ip_parts[2].c_str(), 0)));
    }

    cache_store_client_ = std::make_shared<CacheStoreClient>(tcp_client_, tcp_server_);
    if (!cache_store_client_->init()) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init cache store client failed");
        return false;
    }

    cache_store_server_ = std::make_shared<CacheStoreServer>(
        tcp_client_, tcp_server_, gpt_init_params_.num_layers_, kv_cache_allocator_, local_worker_infos_);
    if (!cache_store_server_->init()) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init cache store server failed");
        return false;
    }

    if (!tcp_server_->start()) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : start tcp server failed");
        return false;
    }

    RTP_LLM_LOG_INFO("KVCacheP2PConnector init success");
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
KVCacheP2PConnector::asyncRead(const BatchKVCacheResourcePtr& resource, const std::string& ip, uint32_t port) {
    auto peer_worker_infos = cache_store_client_->getPeerWorkerInfo(ip, port);
    if (peer_worker_infos.empty()) {
        RTP_LLM_LOG_WARNING("peer_worker_info is null, cannot read cache from %s:%u", ip.c_str(), port);
        return nullptr;
    }

    if (peer_worker_infos.size() % local_worker_infos_.size() != 0
        && local_worker_infos_.size() % peer_worker_infos.size() != 0) {
        RTP_LLM_LOG_WARNING("peer_worker_infos size %zu and local_worker_infos size %zu are not divisible",
                            peer_worker_infos.size(),
                            local_worker_infos_.size());
        return nullptr;
    }

    // do not process duplicate read on same block from different batch for now
    // maybe we can compact same block_id and cache_key first
    auto load_async_context = std::make_shared<KVCacheP2PLoadAsyncContext>();
    for (int batch_id = 0; batch_id < resource->batchSize(); batch_id++) {
        auto cache_keys          = resource->cacheKeys(batch_id);
        auto layer_block_ids     = resource->layerBlockIds(batch_id);
        auto tp_broadcast_result = asyncReadOneBatch(cache_keys, layer_block_ids, peer_worker_infos);
        if (!tp_broadcast_result) {
            load_async_context->setAllSuccess(false);
            return load_async_context;
        }
        load_async_context->addTPBroadcastResult(tp_broadcast_result);
    }
    return load_async_context;
}

std::shared_ptr<TPBroadcastResult>
KVCacheP2PConnector::asyncReadOneBatch(const std::vector<int64_t>&                   cache_keys,
                                       const std::vector<std::shared_ptr<BlockIds>>& layer_block_ids,
                                       const std::vector<CacheStoreServerWorker>&    peer_worker_infos) {
    P2PConnectorLoadBroadcastInfoPB broadcast_info_pb;
    for (auto& cache_key : cache_keys) {
        broadcast_info_pb.add_cache_keys(cache_key);
    }
    for (int i = 0; i < layer_block_ids.size(); i++) {
        auto layer_block_id = layer_block_ids[i];

        auto layer_cache_block = broadcast_info_pb.add_layer_blocks();
        layer_cache_block->set_layer_id(i);

        for (auto& block_id : layer_block_id->block_indices) {
            layer_cache_block->add_block_ids(block_id);
        }
    }
    for (auto& peer_worker_info : peer_worker_infos) {
        auto worker_info = broadcast_info_pb.add_peer_workers();
        worker_info->set_ip(peer_worker_info.ip);
        worker_info->set_port(peer_worker_info.port);
        worker_info->set_rdma_port(peer_worker_info.rdma_port);
    }
    return tp_broadcast_->broadcast(broadcast_info_pb);
}

bool KVCacheP2PConnector::read(const std::vector<int64_t>&                cache_keys,
                               const std::map<int, std::vector<int>>&     layer_block_ids,
                               const std::vector<CacheStoreServerWorker>& peer_worker_infos) {
    std::vector<std::shared_ptr<CacheStoreClientLoadContext>> load_contexts;

    std::vector<std::set<void*>> layer_loading_buffer_set;
    layer_loading_buffer_set.resize(gpt_init_params_.num_layers_);

    if (local_worker_infos_.size() > peer_worker_infos.size()) {
        auto peer_partition_count = local_worker_infos_.size() / peer_worker_infos.size();
        auto peer_partition_id    = local_worker_infos_.size() % peer_worker_infos.size();
        auto peer_worker_addr     = peer_worker_infos[tp_rank_ / peer_partition_count];

        auto load_context = loadFromPeerWorker(cache_keys,
                                               layer_block_ids,
                                               peer_worker_addr,
                                               1,
                                               0,
                                               peer_partition_count,
                                               peer_partition_id,
                                               layer_loading_buffer_set);
        load_contexts.push_back(load_context);
    } else {
        auto local_partition_count = peer_worker_infos.size() / local_worker_infos_.size();
        for (int i = 0; i < local_partition_count; i++) {
            auto peer_worker_addr = peer_worker_infos[i + tp_rank_ * local_partition_count];
            auto load_context     = loadFromPeerWorker(cache_keys,
                                                   layer_block_ids,
                                                   peer_worker_addr,
                                                   local_partition_count,
                                                   i,
                                                   1,
                                                   0,
                                                   layer_loading_buffer_set);
            load_contexts.push_back(load_context);
        }
    }

    bool all_success = true;
    for (auto& load_context : load_contexts) {
        load_context->waitDone();
        if (!load_context->success()) {
            all_success = false;
        }
    }
    return all_success;
}

std::shared_ptr<CacheStoreClientLoadContext>
KVCacheP2PConnector::loadFromPeerWorker(const std::vector<int64_t>&            cache_keys,
                                        const std::map<int, std::vector<int>>& layer_block_ids,
                                        const CacheStoreServerWorker&          peer_worker_addr,
                                        int                                    local_partition_count,
                                        int                                    local_partition_id,
                                        int                                    peer_partition_count,
                                        int                                    peer_partition_id,
                                        std::vector<std::set<void*>>&          layer_loading_buffer_set) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (auto& [layer_id, block_ids] : layer_block_ids) {
        auto  layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        auto& loading_buffer_set = layer_loading_buffer_set[layer_id];
        for (int i = 0; i < cache_keys.size(); i++) {
            auto cache_key = cache_keys[i];
            auto block_id  = block_ids[i];
            auto kv_buffer = kv_cache_allocator_->convertIndexToBuffer(
                block_id, layer_id, local_partition_count, local_partition_id);
            if (kv_buffer.k_addr && loading_buffer_set.find(kv_buffer.k_addr->data()) != loading_buffer_set.end()) {
                layer_cache_buffer->addBlockCacheBuffer(cache_key, block_id, kv_buffer.k_addr, kv_buffer.v_addr);
                loading_buffer_set.insert(kv_buffer.k_addr->data());
            }
        }
        layer_cache_buffers.push_back(layer_cache_buffer);
    }
    return cache_store_client_->asyncLoad(layer_cache_buffers,
                                          100 /* TODO: gpt_init_params_.load_timeout_ms_ */,
                                          peer_worker_addr.ip,
                                          peer_worker_addr.port,
                                          peer_partition_count,
                                          peer_partition_id);
}

std::shared_ptr<KVCacheConnector::AsyncContext> KVCacheP2PConnector::asyncWrite(const BatchKVCacheResourcePtr& resource,
                                                                                DeviceEventPtr                 event) {

    throw std::runtime_error("not supported");
}

std::shared_ptr<KVCacheConnector::AsyncContext>
KVCacheP2PConnector::asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, DeviceEventPtr event) {
    for (int batch_id = 0; batch_id < resource->batchSize(); batch_id++) {
        auto cache_keys      = resource->cacheKeys(batch_id);
        auto layer_block_ids = resource->layerBlockIds(batch_id);
        auto block_ids       = layer_block_ids[layer_id]->block_indices;

        auto layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < block_ids.size(); i++) {
            auto block_id  = block_ids[i];
            auto cache_key = cache_keys[i];
            layer_cache_buffer->addBlockCacheBuffer(cache_key, block_id);
        }
        cache_store_server_->asyncStore(layer_cache_buffer, event, 100 /* TODO: gpt_init_params_.store_timeout_ms_ */);
    }
    // will not give context in this function
    return std::make_shared<KVCacheP2PStoreAsyncContext>();
}

void KVCacheP2PConnector::regUserMr() {
    if (!kv_cache_allocator_) {
        RTP_LLM_LOG_WARNING("kv_cache_allocator_ is null, cannot register user mr");
        return;
    }

    if (kvcache_reg_mr_) {
        return;
    }

    // 从 KVCacheAllocator 获取 layerCacheBase
    auto   buffers    = kv_cache_allocator_->cacheBuffers();
    size_t block_size = kv_cache_allocator_->blockSize();
    if (buffers.empty() || block_size == 0) {
        RTP_LLM_LOG_WARNING("buffers is empty or block_size is 0, cannot register user mr");
        return;
    }

    RTP_LLM_LOG_INFO("start to register user mr for %zu buffers, block_size: %zu", buffers.size(), block_size);

    auto start_time_us = currentTimeUs();
    for (const auto& buffer : buffers) {
        // TODO: maybe should align by k_size avoid alignment problem
        if (!memory_util_->regUserMr(buffer->data(), buffer->size(), true, block_size)) {
            RTP_LLM_LOG_ERROR("register user mr for buffer failed");
            return;
        }
    }
    kvcache_reg_mr_ = true;

    auto cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
    RTP_LLM_LOG_INFO(
        "register user mr for all buffers success: cost %ld ms, buffer_count=%zu", cost_time_ms, buffers.size());
}

void KVCacheP2PConnector::deregUserMr() {
    if (!kv_cache_allocator_ || !kvcache_reg_mr_) {
        return;
    }

    bool all_success = true;
    for (const auto& buffer : kv_cache_allocator_->cacheBuffers()) {
        if (!memory_util_->deregUserMr(buffer->data(), true)) {
            RTP_LLM_LOG_ERROR("deregister user mr for buffer failed");
            return;
        }
    }
    kvcache_reg_mr_ = false;
}

}  // namespace rtp_llm