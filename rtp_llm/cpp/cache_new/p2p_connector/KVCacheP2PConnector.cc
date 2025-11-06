#include "rtp_llm/cpp/cache_new/KVCacheP2PConnector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/InitParams.h"
#include "rtp_llm/cpp/core/Logging.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "autil/StringUtil.h"
#include <torch/torch.h>
#include <set>
#include <thread>

namespace rtp_llm {

KVCacheP2PConnector::KVCacheP2PConnector(const GptInitParameter&             gpt_init_params,
                                         CacheStoreConfig&                   cache_store_config,
                                         DeviceBase*                         device,
                                         const kmonitor::MetricsReporterPtr& metrics_reporter,
                                         KVCacheAllocatorPtr                 kv_cache_allocator):
    gpt_init_params_(gpt_init_params),
    cache_store_config_(cache_store_config),
    device_(device),
    metrics_reporter_(metrics_reporter),
    kv_cache_allocator_(kv_cache_allocator),
    kvcache_reg_mr_(false) {}

KVCacheP2PConnector::~KVCacheP2PConnector() {
    deregUserMr();
}

bool KVCacheP2PConnector::init() {
    memory_util_ = std::static_pointer_cast<NormalCacheStore>(cache_store_)->getMemoryUtil();
    RTP_LLM_LOG_INFO("NormalCacheStore initialized successfully");

    // TODO: MTP model id
    regUserMr(gpt_init_params_.model_id);

    tcp_client_ = std::make_shared<TcpClient>();
    if (!tcp_client_->init(gpt_init_params_.io_thread_count)) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init tcp client failed");
        return false;
    }

    tcp_server_ = std::make_shared<TcpServer>();
    if (!tcp_server_->init(gpt_init_params_.io_thread_count,
                           gpt_init_params_.worker_thread_count,
                           gpt_init_params_.cache_store_listen_port_,
                           true)) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init tcp server failed");
        return false;
    }

    std::vector<CacheStoreServerWorker> worker_addrs;
    for (auto& worker_addr : gpt_init_params_.worker_addrs_) {
        auto ip_parts = autil::StringUtil::split(peer_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_LOG_WARNING("invalid worker addr [%s]", worker_addr.c_str());
            return false;
        }
        worker_addrs.push_back(
            CacheStoreServerWorker(ip_parts[0],
                                   autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0),
                                   autil::StringUtil::strToInt32WithDefault(ip_parts[2].c_str(), 0)));
    }

    cache_store_client_ = std::make_shared<CacheStoreClient>(tcp_client_, tcp_server_);
    if (!cache_store_client_->init()) {
        RTP_LLM_LOG_ERROR("KVCacheP2PConnector init failed : init cache store client failed");
        return false;
    }

    cache_store_server_ =
        std::make_shared<CacheStoreServer>(tcp_client_, tcp_server_, gpt_init_params_.num_layers_, worker_addrs);
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

std::shared_ptr<AsyncContext>
KVCacheP2PConnector::asyncRead(const BatchKVCacheResourcePtr& resource, const std::string& ip, uint32_t port) {
    auto peer_worker_info = cache_store_client_->getPeerWorkerInfo();
    if (peer_worker_info == nullptr) {
        RTP_LLM_LOG_WARNING("peer_worker_info is null, cannot read cache from %s:%u", ip.c_str(), port);
        return nullptr;
    }

    // TODO: compute buffers to load by tp rank

    // TODO: notify local tps to load

    // TODO: wait local tps load done

    return std::make_shared<AsyncContext>();
}

std::shared_ptr<AsyncContext> KVCacheP2PConnector::asyncWrite(const BatchKVCacheResourcePtr& resource,
                                                              DeviceEventPtr                 event) {
    // TODO: notify local tps to write

    // TODO: wait local tps write done
}

std::shared_ptr<AsyncContext>
KVCacheP2PConnector::asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, DeviceEventPtr event) {
    // TODO: notify local tps to write by layer

    // TODO: wait local tps write by layer done
}

void KVCacheP2PConnector::regUserMr(size_t model_id) {
    if (!kv_cache_allocator_) {
        RTP_LLM_LOG_WARNING("kv_cache_allocator_ is null, cannot register user mr");
        return;
    }

    if (!cache_store_ || kvcache_reg_mr_) {
        return;
    }

    // 从 KVCacheAllocator 获取 layerCacheBase
    auto   buffers    = kv_cache_allocator_->cacheBuffers();
    size_t block_size = kv_cache_allocator_->blockSize();
    if (buffers.empty() || block_size == 0) {
        RTP_LLM_LOG_WARNING("buffers is empty or block_size is 0, cannot register user mr");
        return;
    }

    RTP_LLM_LOG_INFO("start to register user mr for %zu buffers, block_size: %zu, model_id: %zu",
                     buffers.size(),
                     block_size,
                     model_id);

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
    RTP_LLM_LOG_INFO("register user mr for all buffers success: cost %ld ms, buffer_count=%zu, model_id: %zu",
                     cost_time_ms,
                     buffers.size(),
                     model_id);
}

void KVCacheP2PConnector::deregUserMr() {
    if (!kv_cache_allocator_ || !kvcache_reg_mr_) {
        return;
    }

    RTP_LLM_LOG_INFO("start to deregister user mr for %zu layers", registered_cache_ptrs_.size());

    // 遍历注销已注册的缓存指针
    bool all_success = true;
    for (const auto& buffer : kv_cache_allocator_->cacheBuffers()) {
        if (!memory_util_->deregUserMr(buffer, true)) {
            RTP_LLM_LOG_ERROR("deregister user mr for buffer failed");
            return;
        }
    }
    kvcache_reg_mr_ = false;
}

}  // namespace rtp_llm