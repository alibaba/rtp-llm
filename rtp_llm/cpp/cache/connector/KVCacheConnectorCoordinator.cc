#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <pthread.h>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#ifdef USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#endif

namespace rtp_llm {
namespace {

class TieredEvictionMeta final : public Meta {
public:
    TieredEvictionMeta(bool enable_memory, bool enable_remote, std::string trace_id):
        enable_memory_(enable_memory), enable_remote_(enable_remote), trace_id_(std::move(trace_id)) {}

    bool enableMemoryCache() const override { return enable_memory_; }
    bool enableRemoteCache() const override { return enable_remote_; }
    const std::string& trace_id() const override { return trace_id_; }
    const std::string& unique_id() const override { return empty_string_; }
    const std::vector<int64_t>& tokens() const override { return empty_tokens_; }

private:
    bool                 enable_memory_{false};
    bool                 enable_remote_{false};
    std::string          trace_id_;
    std::string          empty_string_;
    std::vector<int64_t> empty_tokens_;
};

}  // namespace

KVCacheConnectorCoordinator::KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                                         const KVCacheConfig&                     kv_cache_config,
                                                         const RuntimeConfig&                     runtime_config,
                                                         const ParallelismConfig&                 parallelism_config,
                                                         const SpeculativeExecutionConfig&        sp_config,
                                                         const std::shared_ptr<KVCacheAllocator>& allocator,
                                                         const kmonitor::MetricsReporterPtr&      metrics_reporter,
                                                         const PDSepConfig&                       pd_sep_config,
                                                         const CacheStoreConfig&                  cache_store_config):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    parallelism_config_(parallelism_config),
    sp_config_(sp_config),
    allocator_(allocator),
    metrics_reporter_(metrics_reporter),
    pd_sep_config_(pd_sep_config),
    cache_store_config_(cache_store_config) {}

KVCacheConnectorCoordinator::~KVCacheConnectorCoordinator() {
    stop_.store(true);
    stopTieredEvictionWorker();
    // release all connectors to make sure all async context done
    memory_connector_.reset();
    connectors_.clear();
    // connectors already released, all async context should be done
    autil::ScopedTime2 timer;
    while (true) {
        if (timer.done_ms() > update_interval_ms_ * 2) {
            RTP_LLM_LOG_WARNING(
                "coordinator destructor timeout, read or write list not empty, timeout: %d ms, read list size: %zu, write list size: %zu",
                timer.done_ms(),
                fused_async_read_context_list_.size(),
                fused_async_write_context_list_.size());
            break;
        }
        {
            std::lock_guard<std::mutex> lock(update_mutex_);
            if (fused_async_read_context_list_.empty() && fused_async_write_context_list_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (update_thread_) {
        update_thread_->stop();
        update_thread_.reset();
    }
}

bool KVCacheConnectorCoordinator::hasActiveConnectors() const {
    return !connectors_.empty();
}

bool KVCacheConnectorCoordinator::hasP2PConnector() const {
    return p2p_connector_ != nullptr;
}

bool KVCacheConnectorCoordinator::init() {
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.device_cache_high_watermark_ratio >= 1
                                && kv_cache_config_.device_cache_high_watermark_ratio <= 100,
                            "DEVICE_CACHE_HIGH_WATERMARK_RATIO must be in [1, 100]");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_high_watermark_ratio >= 1
                                && kv_cache_config_.memory_cache_high_watermark_ratio <= 100,
                            "MEMORY_CACHE_HIGH_WATERMARK_RATIO must be in [1, 100]");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_remote_eviction_watermark_ratio >= 1
                                && kv_cache_config_.memory_cache_remote_eviction_watermark_ratio <= 100,
                            "MEMORY_CACHE_REMOTE_EVICTION_WATERMARK_RATIO must be in [1, 100]");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_remote_eviction_timeout_ms > 0,
                            "MEMORY_CACHE_REMOTE_EVICTION_TIMEOUT_MS must be positive");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_config_.memory_cache_remote_eviction_max_blocks > 0,
                            "MEMORY_CACHE_REMOTE_EVICTION_MAX_BLOCKS must be positive");
    if (kv_cache_config_.enable_memory_cache_remote_eviction) {
        RTP_LLM_CHECK_WITH_INFO(
            kv_cache_config_.memory_cache_remote_eviction_watermark_ratio
                <= kv_cache_config_.memory_cache_high_watermark_ratio,
            "MEMORY_CACHE_REMOTE_EVICTION_WATERMARK_RATIO must not exceed MEMORY_CACHE_HIGH_WATERMARK_RATIO");
        RTP_LLM_CHECK_WITH_INFO(cache_config_.groupNums() == 1,
                                "memory cache remote eviction only supports one cache group");
    }
    RTP_LLM_LOG_INFO("connector coordinator init, cache config: [%s], kv cache config: [%s], runtime config: [%s]",
                     cache_config_.debugString().c_str(),
                     kv_cache_config_.to_string().c_str(),
                     runtime_config_.to_string().c_str());
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_memory_cache) {
        memory_connector_ = initMemoryConnector();
        connectors_.emplace_back(memory_connector_);
    }
#ifdef USE_REMOTE_KV_CACHE
    if (kv_cache_config_.reuse_cache && kv_cache_config_.enable_remote_cache) {
        remote_connector_ = initRemoteConnector();
        connectors_.emplace_back(remote_connector_);
    }
#endif
    if (!initP2PConnectorInternal()) {
        RTP_LLM_LOG_WARNING("init P2P connector failed, P2P path disabled — engine continues without it");
    }
    initUpdateThread();
    initTieredEvictionWorker();
    return true;
}

void KVCacheConnectorCoordinator::initUpdateThread() {
    update_thread_ = autil::LoopThread::createLoopThread(
        [this]() { updateOnce(); }, update_interval_ms_ * 1000, "CoordinatorUpdateThread");
    RTP_LLM_CHECK_WITH_INFO(update_thread_ != nullptr, "init update thread failed");
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async read failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    // empty cache keys will not handled by coordinator.
    if (kvcache_resource.cacheKeys().empty()) {
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1 && !kvcache_resource.cacheKeysAreCpCanonical()) {
        CPSlotMapper mapper(cp_size - 1, cp_size, static_cast<int>(cache_config_.seq_size_per_block));
        ref_keys = mapper.canonicalCacheKeys(kvcache_resource.cacheKeys());
        // Short requests (< cp_size logical blocks) have no complete virtual
        // block, so the canonical last-rank-key namespace is empty by design.
        // Skip silently — connector activity for these is a no-op anyway.
        if (ref_keys.empty()) {
            return nullptr;
        }
        ref_resource = mapper.projectConnectorResource(kvcache_resource, cache_config_, ref_keys);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async read failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> match_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        match_contexts.at(i) = connectors_.at(i)->asyncMatch(resource, connector_context->meta());
    }

    auto fused_match_context = std::make_shared<FusedAsyncContext>(std::move(match_contexts));
    auto fused_read_context =
        std::make_shared<FusedAsyncReadContext>(fused_match_context, resource, connector_context->meta());
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_read_context_list_.push_back(fused_read_context);
    }
    return fused_read_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context) {
    if (stop_.load()) {
        return nullptr;
    }
    if (!connector_context) {
        RTP_LLM_LOG_WARNING("async write failed, connector context is null");
        return nullptr;
    }
    const auto& kvcache_resource = connector_context->kvCacheResource();
    if (kvcache_resource.cacheKeys().empty()) {
        RTP_LLM_LOG_DEBUG("async write failed, kvcache resource cache keys is empty, resource: [%s]",
                          kvcache_resource.debugString().c_str());
        return nullptr;
    }

    const int       cp_size      = cpSize();
    CacheKeysType   ref_keys     = kvcache_resource.cacheKeys();
    KVCacheResource ref_resource = kvcache_resource;
    if (cp_size > 1 && !kvcache_resource.cacheKeysAreCpCanonical()) {
        CPSlotMapper mapper(cp_size - 1, cp_size, static_cast<int>(cache_config_.seq_size_per_block));
        ref_keys = mapper.canonicalCacheKeys(kvcache_resource.cacheKeys());
        if (ref_keys.empty()) {
            return nullptr;  // request shorter than one virtual block — nothing to write
        }
        ref_resource = mapper.projectConnectorResource(kvcache_resource, cache_config_, ref_keys);
        ref_keys     = ref_resource.cacheKeys();
    }
    auto resource = allocator_->incrKVCacheRef(ref_resource, ref_keys, true);
    if (!resource) {
        RTP_LLM_LOG_WARNING("async write failed, incr kvcache ref failed, resource: [%s]",
                            kvcache_resource.debugString().c_str());
        return nullptr;
    }

    std::vector<std::shared_ptr<AsyncContext>> write_contexts(connectors_.size());
    for (int i = 0; i < connectors_.size(); i++) {
        write_contexts.at(i) = connectors_.at(i)->asyncWrite(resource, connector_context->meta());
    }

    auto fused_write_context = std::make_shared<FusedAsyncContext>(std::move(write_contexts));
    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        fused_async_write_context_list_.push_back(fused_write_context);
    }
    return fused_write_context;
}

std::shared_ptr<AsyncContext>
KVCacheConnectorCoordinator::asyncWriteByLayer(int                                                  layer_id,
                                               const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    if (!p2p_connector_) {
        return nullptr;
    }
    if (!layer_context) {
        RTP_LLM_LOG_WARNING("asyncWriteByLayer: layer_context is null, skip P2P write for layer %d", layer_id);
        return nullptr;
    }
    if (layer_id == 0) {
        RTP_LLM_LOG_INFO("asyncWriteByLayer [P2P]: dispatching layer_id=%d, request_id=%ld to P2PConnector",
                         layer_id,
                         layer_context->requestId());
    }
    return p2p_connector_->asyncWriteByLayer(layer_id, layer_context);
}

std::shared_ptr<KVCacheMemoryConnector> KVCacheConnectorCoordinator::initMemoryConnector() {
    auto memory_connector = std::make_shared<KVCacheMemoryConnector>(cache_config_,
                                                                     kv_cache_config_,
                                                                     parallelism_config_,
                                                                     allocator_,
                                                                     runtime_config_.worker_grpc_addrs,
                                                                     metrics_reporter_);
    RTP_LLM_CHECK_WITH_INFO(memory_connector->init(), "memory connector init failed");
    return memory_connector;
}

std::shared_ptr<RemoteConnector> KVCacheConnectorCoordinator::initRemoteConnector() {
#ifdef USE_REMOTE_KV_CACHE
    RTP_LLM_CHECK_WITH_INFO(!cache_config_.use_independent_block_pools,
                            "remote connector does not support independent KV cache block pools");
    const auto block_pool = allocator_->getBlockPool();
    RTP_LLM_CHECK_WITH_INFO(block_pool != nullptr, "remote connector requires a contiguous KV cache block pool");
    // TODO : get lora info map
    auto remote_connector_ = std::make_shared<RemoteConnector>(cache_config_,
                                                               kv_cache_config_,
                                                               runtime_config_,
                                                               parallelism_config_,
                                                               sp_config_,
                                                               block_pool->getBaseAddress(),
                                                               block_pool->getTotalSizeBytes(),
                                                               allocator_,
                                                               metrics_reporter_);

    remote_connector_->setMemoryConnector(memory_connector_);
    RTP_LLM_CHECK_WITH_INFO(remote_connector_->init(), "remote connector init failed");
    return remote_connector_;
#else
    RTP_LLM_LOG_ERROR("not RemoteConnector");
    return nullptr;
#endif
}

int KVCacheConnectorCoordinator::cpSize() const {
    const auto& cp_cfg = parallelism_config_.prefill_cp_config;
    if (!cp_cfg.kv_cache_sharded) {
        return 1;
    }
    if (parallelism_config_.tp_size > 1) {
        return static_cast<int>(parallelism_config_.tp_size);
    }
    if (parallelism_config_.role_type == RoleType::DECODE && cp_cfg.is_prefill_enabled()
        && cp_cfg.prefill_cp_size > 1) {
        return static_cast<int>(cp_cfg.prefill_cp_size);
    }
    return 1;
}

void KVCacheConnectorCoordinator::updateOnce() {
    RTP_LLM_PROFILE_FUNCTION();
    processReadContexts();
    processWriteContexts();
}

void KVCacheConnectorCoordinator::processReadContexts() {
    RTP_LLM_PROFILE_FUNCTION();
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_read_context_list_.begin(); it != fused_async_read_context_list_.end();) {
        auto fused_read_context = *it;
        if (fused_read_context->done()) {
            fused_read_context->notifyDone();
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        // 没有 done 但是有 read context, 或者 match context 还没 done, 或者 match 失败 (下一轮调度处理), 继续等待
        auto read_context  = fused_read_context->fusedReadContext();
        auto match_context = fused_read_context->fusedMatchContext();
        if (read_context || !match_context->done() || !match_context->success()) {
            it = std::next(it);
            continue;
        }
        // match success, start read
        asyncReadAfterMatch(fused_read_context);
        it = std::next(it);
    }
}

void KVCacheConnectorCoordinator::processWriteContexts() {
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_write_context_list_.begin(); it != fused_async_write_context_list_.end();) {
        auto fused_write_context = *it;
        if (fused_write_context->done()) {
            it = fused_async_write_context_list_.erase(it);
            continue;
        }
        it = std::next(it);
    }
}

// this function is called under lock
void KVCacheConnectorCoordinator::asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext> fused_read_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto match_contexts = fused_read_context->fusedMatchContext()->contexts();
    RTP_LLM_CHECK_WITH_INFO(
        match_contexts.size() == connectors_.size(),
        "match contexts size is not equal to connectors size, match contexts size: [%d], connectors size: [%d]",
        match_contexts.size(),
        connectors_.size());

    int                                        already_reuse_num = fused_read_context->resource()->reuseBlockNum();
    std::vector<std::shared_ptr<AsyncContext>> connector_read_contexts;
    for (int i = 0; i < match_contexts.size(); i++) {
        auto match_context = std::dynamic_pointer_cast<AsyncMatchContext>(match_contexts.at(i));
        if (!match_context) {
            continue;
        }
        const auto matched_num = match_context->matchedBlockCount();
        if (matched_num <= already_reuse_num) {
            continue;
        }
        auto connector_read_context = connectors_.at(i)->asyncRead(fused_read_context->resource(),
                                                                   fused_read_context->meta(),
                                                                   match_context,
                                                                   already_reuse_num,
                                                                   matched_num - already_reuse_num);
        if (connector_read_context) {
            connector_read_contexts.emplace_back(connector_read_context);
            already_reuse_num = matched_num;
        }
    }
    fused_read_context->setFusedReadContext(std::make_shared<FusedAsyncContext>(connector_read_contexts));
}

void KVCacheConnectorCoordinator::handleRead(const P2PConnectorStartLoadRequestPB& request,
                                             P2PConnectorStartLoadResponsePB&      response,
                                             std::function<bool()>                 is_cancelled) {
    if (!p2p_connector_) {
        RTP_LLM_LOG_WARNING("handleRead called but P2P connector not initialized");
        return;
    }
    p2p_connector_->handleRead(request, response, std::move(is_cancelled));
}

bool KVCacheConnectorCoordinator::executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_mem_request()) {
        RTP_LLM_CHECK(memory_connector_ != nullptr);
        return memory_connector_->copyCache(request.mem_request(), *(response.mutable_mem_response()));
    } else if (request.has_remote_request()) {
#ifdef USE_REMOTE_KV_CACHE
        RTP_LLM_CHECK(remote_connector_ != nullptr);
        return remote_connector_->copyCache(request.remote_request(), *(response.mutable_remote_response()));
#endif
        RTP_LLM_CHECK(false);
        return false;
    } else if (request.has_p2p_request()) {
        if (!p2p_connector_) {
            RTP_LLM_LOG_WARNING("executeFunction: p2p_request received but P2P connector not initialized");
            return false;
        }
        return p2p_connector_->executeFunction(request, response);
    } else {
        RTP_LLM_LOG_WARNING("execute function failed, request is invalid, request: [%s]",
                            request.DebugString().c_str());
        return false;
    }
}

bool KVCacheConnectorCoordinator::isPdInvertMode() const {
    return (pd_sep_config_.role_type == RoleType::PREFILL || pd_sep_config_.role_type == RoleType::DECODE)
           && pd_sep_config_.decode_entrance;
}

bool KVCacheConnectorCoordinator::initP2PConnectorInternal() {
    // TODO: P2P connector initialization is disabled until the next PR enables
    // scheduler async load cache support. Change to `#if 1` to activate.
#if 0
    if (!isPdInvertMode()) {
        return true;
    }
    const uint32_t layer_all_num         = static_cast<uint32_t>(cache_config_.layer_all_num);
    auto           layer_block_converter = std::make_shared<LayerBlockConverterImpl>(allocator_);

    auto p2p_config = P2PConnectorConfig::create(
        runtime_config_, cache_store_config_, parallelism_config_, pd_sep_config_, layer_all_num);
    auto p2p = std::make_shared<P2PConnector>(std::move(p2p_config), layer_block_converter, metrics_reporter_);
    if (!p2p->init()) {
        RTP_LLM_LOG_ERROR("P2PConnector init failed");
        p2p.reset();  // 显式释放，避免半初始化状态的 P2PConnector 意外使用
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(update_mutex_);
        p2p_connector_ = std::move(p2p);
        connectors_.emplace_back(p2p_connector_);
    }
    RTP_LLM_LOG_INFO("P2PConnector initialized successfully, total connectors: %zu", connectors_.size());
#endif
    return true;
}

void KVCacheConnectorCoordinator::initTieredEvictionWorker() {
    if (!kv_cache_config_.reuse_cache || !kv_cache_config_.enable_tiered_memory_cache
        || !kv_cache_config_.enable_memory_cache || !kv_cache_config_.enable_device_cache
        || !kv_cache_config_.enable_memory_cache_remote_eviction) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(tiered_eviction_mutex_);
        tiered_eviction_accepting_ = true;
        tiered_eviction_stopping_  = false;
    }
    // Metrics groups are registered lazily on the first report. Emit an initial
    // zero-valued sample so all tiered-eviction metrics are discoverable even
    // before memory pressure selects the first remote-eviction victim.
    if (metrics_reporter_) {
        RtpLLMMemoryRemoteEvictionMetricsCollector collector;
        metrics_reporter_->report<RtpLLMMemoryRemoteEvictionMetrics,
                                  RtpLLMMemoryRemoteEvictionMetricsCollector>(nullptr, &collector);
    }
    tiered_eviction_worker_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "MemRemoteEvict");
        tieredEvictionLoop();
    });
}

void KVCacheConnectorCoordinator::stopTieredEvictionWorker() {
    {
        std::lock_guard<std::mutex> lock(tiered_eviction_mutex_);
        tiered_eviction_accepting_ = false;
        tiered_eviction_stopping_  = true;
    }
    tiered_eviction_cv_.notify_all();
    if (tiered_eviction_worker_.joinable()) {
        tiered_eviction_worker_.join();
    }
}

void KVCacheConnectorCoordinator::enqueueTieredEviction(const std::string& trace_id) {
    {
        std::lock_guard<std::mutex> lock(tiered_eviction_mutex_);
        if (!tiered_eviction_accepting_ || tiered_eviction_stopping_) {
            return;
        }
        tiered_eviction_queue_.push_back(trace_id);
    }
    tiered_eviction_cv_.notify_one();
}

void KVCacheConnectorCoordinator::tieredEvictionLoop() {
    while (true) {
        std::string trace_id;
        {
            std::unique_lock<std::mutex> lock(tiered_eviction_mutex_);
            tiered_eviction_cv_.wait(lock, [this]() {
                return tiered_eviction_stopping_ || !tiered_eviction_queue_.empty();
            });
            if (tiered_eviction_queue_.empty()) {
                if (tiered_eviction_stopping_) {
                    return;
                }
                continue;
            }
            trace_id = std::move(tiered_eviction_queue_.front());
            tiered_eviction_queue_.pop_front();
        }
        try {
            runTieredEviction(trace_id);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("tiered cache eviction failed, trace_id=%s, error=%s", trace_id.c_str(), e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("tiered cache eviction failed, trace_id=%s, unknown error", trace_id.c_str());
        }
    }
}

size_t KVCacheConnectorCoordinator::blocksAboveHighWatermark(size_t total_blocks,
                                                               size_t free_blocks,
                                                               int    high_watermark_ratio) {
    const size_t effective_free = std::min(total_blocks, free_blocks);
    const size_t ratio          = static_cast<size_t>(high_watermark_ratio);
    const size_t min_free       = (total_blocks * (100 - ratio) + 99) / 100;
    return min_free > effective_free ? min_free - effective_free : 0;
}

size_t KVCacheConnectorCoordinator::projectedBlocksAboveHighWatermark(size_t total_blocks,
                                                                        size_t free_blocks,
                                                                        size_t incoming_blocks,
                                                                        int high_watermark_ratio) {
    const size_t effective_free = std::min(total_blocks, free_blocks);
    const size_t used           = total_blocks - effective_free;
    const size_t max_used       = total_blocks * static_cast<size_t>(high_watermark_ratio) / 100;
    if (used >= max_used) {
        return used - max_used + incoming_blocks;
    }
    const size_t headroom = max_used - used;
    return incoming_blocks > headroom ? incoming_blocks - headroom : 0;
}

size_t KVCacheConnectorCoordinator::deviceBlocksAboveHighWatermark() const {
    return blocksAboveHighWatermark(allocator_->totalBlocksNum(),
                                    allocator_->notInUseBlocksNum(),
                                    kv_cache_config_.device_cache_high_watermark_ratio);
}

size_t KVCacheConnectorCoordinator::memoryBlocksAboveHighWatermark(size_t incoming_blocks) const {
    if (!memory_connector_) {
        return 0;
    }
    return projectedBlocksAboveHighWatermark(memory_connector_->totalMemoryBlocks(),
                                             memory_connector_->freeMemoryBlocks(),
                                             incoming_blocks,
                                             kv_cache_config_.memory_cache_high_watermark_ratio);
}

size_t KVCacheConnectorCoordinator::memoryBlocksAboveRemoteEvictionWatermark(size_t incoming_blocks) const {
    if (!memory_connector_) {
        return 0;
    }
    return projectedBlocksAboveHighWatermark(memory_connector_->totalMemoryBlocks(),
                                             memory_connector_->freeMemoryBlocks(),
                                             incoming_blocks,
                                             kv_cache_config_.memory_cache_remote_eviction_watermark_ratio);
}

void KVCacheConnectorCoordinator::enforceMemoryHighWatermark(size_t incoming_blocks,
                                                              const std::string& trace_id) {
    const size_t need = memoryBlocksAboveHighWatermark(incoming_blocks);
    if (need == 0 || !memory_connector_) {
        return;
    }
    const size_t evicted = memory_connector_->evictMemoryImmediately(need);
    if (metrics_reporter_) {
        RtpLLMMemoryRemoteEvictionMetricsCollector collector;
        collector.memory_emergency_evict_block_count = evicted;
        metrics_reporter_->report<RtpLLMMemoryRemoteEvictionMetrics,
                                  RtpLLMMemoryRemoteEvictionMetricsCollector>(nullptr, &collector);
    }
    RTP_LLM_LOG_INFO(
        "tiered memory emergency eviction, trace_id=%s, hard_watermark_ratio=%d, memory_total_blocks=%zu, memory_free_blocks=%zu, requested=%zu, evicted=%zu, incoming=%zu",
        trace_id.c_str(),
        kv_cache_config_.memory_cache_high_watermark_ratio,
        memory_connector_->totalMemoryBlocks(),
        memory_connector_->freeMemoryBlocks(),
        need,
        evicted,
        incoming_blocks);
}

void KVCacheConnectorCoordinator::runTieredEviction(const std::string& trace_id) {
    bool   remote_task_started  = false;
    size_t estimated_d2h_blocks = deviceBlocksAboveHighWatermark();
    if (estimated_d2h_blocks == 0) {
        return;
    }

#ifdef USE_REMOTE_KV_CACHE
    const bool remote_spill_enabled = kv_cache_config_.enable_memory_cache_remote_eviction
                                      && kv_cache_config_.enable_remote_cache && remote_connector_ != nullptr
                                      && memory_connector_ != nullptr && cache_config_.groupNums() == 1;
    if (remote_spill_enabled) {
        const size_t requested_remote_evict_blocks =
            memoryBlocksAboveRemoteEvictionWatermark(estimated_d2h_blocks);
        const size_t capped_remote_evict_blocks = std::min(
            requested_remote_evict_blocks,
            static_cast<size_t>(kv_cache_config_.memory_cache_remote_eviction_max_blocks));
        RTP_LLM_LOG_INFO(
            "memory remote eviction decision, trace_id=%s, device_total_blocks=%zu, device_free_blocks=%zu, device_high_watermark_ratio=%d, estimated_d2h_blocks=%zu, memory_total_blocks=%zu, memory_free_blocks=%zu, memory_remote_eviction_watermark_ratio=%d, memory_high_watermark_ratio=%d, requested_remote_evict_blocks=%zu, capped_remote_evict_blocks=%zu, max_remote_evict_blocks=%d",
            trace_id.c_str(),
            allocator_->totalBlocksNum(),
            allocator_->notInUseBlocksNum(),
            kv_cache_config_.device_cache_high_watermark_ratio,
            estimated_d2h_blocks,
            memory_connector_->totalMemoryBlocks(),
            memory_connector_->freeMemoryBlocks(),
            kv_cache_config_.memory_cache_remote_eviction_watermark_ratio,
            kv_cache_config_.memory_cache_high_watermark_ratio,
            requested_remote_evict_blocks,
            capped_remote_evict_blocks,
            kv_cache_config_.memory_cache_remote_eviction_max_blocks);
        auto victims = memory_connector_->prepareRemoteEviction(capped_remote_evict_blocks);
        if (victims.empty()) {
            RTP_LLM_LOG_INFO(
                "memory remote eviction skipped, trace_id=%s, requested_remote_evict_blocks=%zu, capped_remote_evict_blocks=%zu, selected_victim_blocks=0",
                trace_id.c_str(),
                requested_remote_evict_blocks,
                capped_remote_evict_blocks);
        }
        if (!victims.empty()) {
            remote_task_started = true;
            if (metrics_reporter_) {
                RtpLLMMemoryRemoteEvictionMetricsCollector collector;
                collector.memory_remote_evict_inflight_blocks = victims.size();
                metrics_reporter_->report<RtpLLMMemoryRemoteEvictionMetrics,
                                          RtpLLMMemoryRemoteEvictionMetricsCollector>(nullptr, &collector);
            }
            CacheKeysType        cache_keys;
            std::vector<int32_t> memory_block_ids;
            cache_keys.reserve(victims.size());
            memory_block_ids.reserve(victims.size());
            size_t remote_evict_bytes = 0;
            for (const auto& victim : victims) {
                cache_keys.push_back(victim.cache_key);
                memory_block_ids.push_back(victim.block_index);
                remote_evict_bytes += victim.block_size;
                RTP_LLM_LOG_DEBUG(
                    "memory remote eviction victim, trace_id=%s, unique_id=%s, cache_key=%ld, memory_block_id=%d, block_size=%zu, generation=%lu",
                    trace_id.c_str(),
                    trace_id.c_str(),
                    victim.cache_key,
                    victim.block_index,
                    victim.block_size,
                    victim.generation);
            }
            RTP_LLM_LOG_INFO(
                "memory remote eviction started, trace_id=%s, requested_remote_evict_blocks=%zu, capped_remote_evict_blocks=%zu, selected_victim_blocks=%zu, bytes=%zu, inflight=%zu, memory_total_blocks=%zu, memory_free_blocks=%zu, estimated_d2h_blocks=%zu, timeout_ms=%d",
                trace_id.c_str(),
                requested_remote_evict_blocks,
                capped_remote_evict_blocks,
                victims.size(),
                remote_evict_bytes,
                victims.size(),
                memory_connector_->totalMemoryBlocks(),
                memory_connector_->freeMemoryBlocks(),
                estimated_d2h_blocks,
                kv_cache_config_.memory_cache_remote_eviction_timeout_ms);
            const auto remote_evict_started = std::chrono::steady_clock::now();
            bool remote_success = false;
            try {
                auto lease = std::make_shared<std::vector<KVCacheMemoryConnector::MemoryRemoteEvictionItem>>(victims);
                auto meta  = std::make_shared<TieredEvictionMeta>(false, true, trace_id);
                auto ctx   = remote_connector_->asyncWriteMemory(cache_keys, memory_block_ids, meta, lease);
                if (ctx) {
                    const auto started = std::chrono::steady_clock::now();
                    bool       timeout_logged = false;
                    while (!ctx->done()) {
                        if (!timeout_logged
                            && std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - started)
                                       .count()
                                   >= kv_cache_config_.memory_cache_remote_eviction_timeout_ms) {
                            RTP_LLM_LOG_WARNING(
                                "memory remote eviction exceeded timeout; waiting for safe buffer release, trace_id=%s, timeout_ms=%d",
                                trace_id.c_str(),
                                kv_cache_config_.memory_cache_remote_eviction_timeout_ms);
                            timeout_logged = true;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    remote_success = ctx->success();
                }
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("memory remote eviction failed with exception, trace_id=%s, error=%s",
                                    trace_id.c_str(), e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("memory remote eviction failed with unknown exception, trace_id=%s",
                                    trace_id.c_str());
            }
            // Detached entries must always leave the in-flight state. On failure they
            // are dropped according to the configured failure policy.
            memory_connector_->finishRemoteEviction(victims, remote_success);
            const auto remote_evict_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                     std::chrono::steady_clock::now() - remote_evict_started)
                                                     .count();
            if (metrics_reporter_) {
                RtpLLMMemoryRemoteEvictionMetricsCollector collector;
                collector.memory_remote_evict_qps                 = true;
                collector.memory_remote_evict_fail_qps            = !remote_success;
                collector.memory_remote_evict_block_count         = victims.size();
                collector.memory_remote_evict_success_block_count = remote_success ? victims.size() : 0;
                collector.memory_remote_evict_failed_block_count  = remote_success ? 0 : victims.size();
                collector.memory_remote_evict_latency_us          = remote_evict_latency_us;
                collector.memory_remote_evict_bytes               = remote_evict_bytes;
                collector.memory_remote_evict_inflight_blocks     = 0;
                metrics_reporter_->report<RtpLLMMemoryRemoteEvictionMetrics,
                                          RtpLLMMemoryRemoteEvictionMetricsCollector>(nullptr, &collector);
            }
            RTP_LLM_LOG_INFO(
                "memory remote eviction finished, trace_id=%s, attempted_blocks=%zu, success_blocks=%zu, failed_blocks=%zu, bytes=%zu, latency_us=%ld, inflight=0, success=%d",
                trace_id.c_str(),
                victims.size(),
                remote_success ? victims.size() : 0,
                remote_success ? 0 : victims.size(),
                remote_evict_bytes,
                remote_evict_latency_us,
                remote_success);
        }
    }
#endif

    const auto device_to_memory_started = std::chrono::steady_clock::now();
    // Task B rebuilds its Device eviction plan from current state after Task A.
    const size_t need_d2h_blocks = deviceBlocksAboveHighWatermark();
    if (need_d2h_blocks == 0) {
        return;
    }
    auto evicted_resource = allocator_->popBlocksFromCache(need_d2h_blocks);
    if (!evicted_resource || !evicted_resource->hasCacheKeys()) {
        RTP_LLM_LOG_INFO("tiered Device->Memory eviction no-op, trace_id=%s, requested=%zu",
                         trace_id.c_str(), need_d2h_blocks);
        return;
    }

    const auto& source_resource = evicted_resource->cacheResource(0);
    size_t actual_d2h_blocks = source_resource.cacheKeys().size();
    if (!source_resource.lastBlockAligned() && actual_d2h_blocks > 0) {
        --actual_d2h_blocks;
    }
    enforceMemoryHighWatermark(actual_d2h_blocks, trace_id);

    std::shared_ptr<KVCacheResource> connector_resource;
    std::shared_ptr<AsyncContext>    memory_ctx;
    try {
        connector_resource = allocator_->incrKVCacheRef(source_resource, source_resource.cacheKeys(), true);
        if (connector_resource) {
            auto meta  = std::make_shared<TieredEvictionMeta>(true, false, trace_id);
            memory_ctx = memory_connector_->asyncWrite(connector_resource, meta);
        } else {
            RTP_LLM_LOG_WARNING("tiered Device->Memory eviction failed to pin device blocks, trace_id=%s",
                                trace_id.c_str());
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tiered Device->Memory eviction failed with exception, trace_id=%s, error=%s",
                            trace_id.c_str(), e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("tiered Device->Memory eviction failed with unknown exception, trace_id=%s",
                            trace_id.c_str());
    }

    // Drop the Device BlockCache ownership even if pinning or asyncWrite fails.
    allocator_->blockCacheFree(evicted_resource);
    try {
        if (memory_ctx) {
            while (!memory_ctx->done()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("tiered Device->Memory wait failed with exception, trace_id=%s, error=%s",
                            trace_id.c_str(), e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("tiered Device->Memory wait failed with unknown exception, trace_id=%s",
                            trace_id.c_str());
    }
    connector_resource.reset();

    // Recheck with actual post-copy occupancy. This is the hard waterline guard.
    enforceMemoryHighWatermark(0, trace_id);
    const auto device_to_memory_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                 std::chrono::steady_clock::now() - device_to_memory_started)
                                                 .count();
    if (remote_task_started && metrics_reporter_) {
        RtpLLMMemoryRemoteEvictionMetricsCollector collector;
        collector.device_to_memory_after_remote_latency_us = device_to_memory_latency_us;
        metrics_reporter_->report<RtpLLMMemoryRemoteEvictionMetrics,
                                  RtpLLMMemoryRemoteEvictionMetricsCollector>(nullptr, &collector);
    }
    RTP_LLM_LOG_INFO(
        "tiered Device->Memory eviction finished, trace_id=%s, after_remote=%d, requested=%zu, actual=%zu, latency_us=%ld, hard_watermark_ratio=%d, memory_total_blocks=%zu, memory_free_blocks=%zu, success=%d",
        trace_id.c_str(),
        remote_task_started,
        need_d2h_blocks,
        actual_d2h_blocks,
        device_to_memory_latency_us,
        kv_cache_config_.memory_cache_high_watermark_ratio,
        memory_connector_->totalMemoryBlocks(),
        memory_connector_->freeMemoryBlocks(),
        memory_ctx ? memory_ctx->success() : 0);
}

std::vector<CacheKeyType> KVCacheConnectorCoordinator::memoryCacheKeys() const {
    if (!memory_connector_) {
        return {};
    }
    return memory_connector_->cacheKeys();
}

std::vector<CacheKeyType> KVCacheConnectorCoordinator::memoryCacheKeysForStatus() const {
    if (!memory_connector_) {
        return {};
    }
    return memory_connector_->cacheKeysForStatus();
}

}  // namespace rtp_llm
