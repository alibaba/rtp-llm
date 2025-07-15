#include "rtp_llm/cpp/cache/DistKvCache.h"

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"

using namespace rtp_llm::threefs;

namespace rtp_llm {

DistKvCache::DistKvCache(CacheManager*                       cache_manager,
                         const GptInitParameter&             gpt_init_params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), gpt_init_params_(gpt_init_params), metrics_reporter_(metrics_reporter) {}

DistKvCache::~DistKvCache() {}

bool DistKvCache::init(const DistKvCacheInitParams& init_params) {
    // TODO(LXQ): 打印 params
    init_params_ = init_params;

    // default 3fs planner
    if (!init_params_.storage_manager_params.init_params_3fs.has_value()) {
        RTP_LLM_LOG_WARNING("init failed, 3fs init params is empty");
        return false;
    }
    const auto& storage_3fs_init_params = init_params_.storage_manager_params.init_params_3fs.value();
    planner_                            = std::make_unique<DefaultDistKvCachePlanner>(
        cache_manager_, gpt_init_params_, storage_3fs_init_params, metrics_reporter_);

    storage_ = std::make_unique<DistStorageManager>(metrics_reporter_);
    if (!storage_->init(init_params_.storage_manager_params)) {
        RTP_LLM_LOG_WARNING("init failed, init storage failed");
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();
    RTP_LLM_LOG_INFO("dist kv cache init success");
    return true;
}

int32_t DistKvCache::matchForAllRank(const std::vector<int64_t>&        cache_keys,
                                     int64_t                            request_id,
                                     std::map<std::string, std::string> extra_metas) const {
    if (cache_keys.empty()) {
        return 0;
    }

    if (!planner_ || !storage_) {
        return 0;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markMatchBeginUs(metrics);

    auto match_len = cache_keys.size();
    for (int i = 0; i < gpt_init_params_.tp_size_; i++) {
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + match_len);
        extra_metas["TP_RANK"] = std::to_string(i);
        auto ret               = match(cache_keys_to_match, request_id, extra_metas);
        if (ret < match_len) {
            match_len = ret;
        }
        if (match_len == 0) {
            break;
        }
    }

    DistKvCacheMetrics::markMatchDoneUs(metrics);
    RTP_LLM_LOG_DEBUG("dist kv cache match cache for all rank, matched len %d, input cache keys len %d",
                      match_len,
                      cache_keys.size());
    return match_len;
}

int32_t DistKvCache::match(const std::vector<int64_t>&        cache_keys,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        extra_metas[key] = value;
    }
    extra_metas["SKIP_IOV"] = "1";

    int match_len = cache_keys.size();
    for (; match_len > 0; --match_len) {
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + match_len);
        auto                 layout_items = planner_->layout(cache_keys_to_match, {}, extra_metas);
        if (layout_items.empty()) {
            continue;
        }

        bool success = true;
        for (auto& item : layout_items) {
            if (!storage_->lookup(item)) {
                success = false;
                break;
            }
        }
        if (success) {
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("dist kv cache match cache for rank %d, matched len %d, input cache keys len %d",
                      gpt_init_params_.tp_rank_,
                      match_len,
                      cache_keys.size());
    return match_len;
}

bool DistKvCache::getForAllRank(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) {
    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalGetCacheBeginUs(metrics);

    std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin());
    std::vector<int32_t> block_indices_to_match(block_indices.begin(), block_indices.begin());
    bool result = syncCallAllRank(cache_keys_to_match, block_indices_to_match, request_id, extra_metas, OpType::OP_GET);

    DistKvCacheMetrics::markTotalGetCacheDoneUs(metrics);
    DistKvCacheMetrics::setGetCacheFailedQps(metrics, result == false);

    const auto  matched_len  = result ? static_cast<int32_t>(cache_keys.size()) : 0;
    const auto& cache_config = cache_manager_->cacheConfig();
    total_matched_len_.fetch_add(matched_len);
    int64_t total_matched_len = total_matched_len_.load();
    total_input_len_.fetch_add(std::stoi(extra_metas.at("SEQ_CACHE_KEY_NUM")));
    int64_t total_input_len = total_input_len_.load();

    DistKvCacheMetrics::setCacheReuseLength(metrics, matched_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setTotalCacheReuseLength(metrics, total_matched_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setTotalCacheInputLength(metrics, total_input_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setTotalCacheHitRate(metrics, total_matched_len * 100.0 / total_input_len);

    return result;
}

bool DistKvCache::get(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        extra_metas[key] = value;
    }
    extra_metas["GET"]     = "1";
    extra_metas["TP_RANK"] = std::to_string(gpt_init_params_.tp_rank_);

    auto layout_items = planner_->layout(cache_keys, block_indices, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_DEBUG("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markGetCacheBeginUs(metrics);

    for (auto& item : layout_items) {
        if (!storage_->get(item)) {
            RTP_LLM_LOG_WARNING("dist kv cache get cache, get cache failed, key: %s", item.key.c_str());
            DistKvCacheMetrics::markGetCacheDoneUs(metrics);
            return false;
        }
    }

    DistKvCacheMetrics::markGetCacheDoneUs(metrics);

    if (!planner_->verify(layout_items,
                          cache_keys,
                          block_indices,
                          extra_metas.empty() ? default_metas_ : extra_metas,
                          gpt_init_params_.tp_rank_)) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, verify cache failed, request key: %s", request_id);
        return false;
    }

    return true;
}

bool DistKvCache::putForAllRank(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalPutCacheBeginUs(metrics);

    bool result = syncCallAllRank(cache_keys, block_indices, request_id, extra_metas, OpType::OP_PUT);

    DistKvCacheMetrics::markTotalPutCacheDoneUs(metrics);
    DistKvCacheMetrics::setPutCacheFailedQps(metrics, result == false);
    return result;
}

bool DistKvCache::put(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        extra_metas[key] = value;
    }
    extra_metas["TP_RANK"] = std::to_string(gpt_init_params_.tp_rank_);

    auto layout_items = planner_->layout(cache_keys, block_indices, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_DEBUG("dist kv cache put cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markPutCacheBeginUs(metrics);

    for (auto& item : layout_items) {
        if (!storage_->put(item)) {
            RTP_LLM_LOG_WARNING("dist kv cache put cache, put cache failed, key: %s", item.key.c_str());
            DistKvCacheMetrics::markPutCacheDoneUs(metrics);
            return false;
        }
    }

    DistKvCacheMetrics::markPutCacheDoneUs(metrics);
    return true;
}

struct WorkerRpcContext {
    WorkerRpcContext() {
        client_context = std::make_shared<grpc::ClientContext>();
    }
    grpc::Status                         status;
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::string                          server_addr;
    DistKvCacheRequestPB                 request;
    DistKvCacheResponsePB                response;
    grpc::CompletionQueue                completion_queue;
};

// TODO: sync call all rank 的逻辑用到的地方比较多, 抽一下
bool DistKvCache::syncCallAllRank(const std::vector<int64_t>&        cache_keys,
                                  const std::vector<int32_t>&        block_indices,
                                  int64_t                            request_id,
                                  std::map<std::string, std::string> extra_metas,
                                  DistKvCache::OpType                op_type) const {
    const auto& grpc_workers = gpt_init_params_.worker_grpc_addrs_;
    if (grpc_workers.empty()) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, grpc workers is empty, request: %ld", request_id);
        return false;
    }
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, rpc pool is nullptr, request: %ld", request_id);
        return false;
    }

    const int                     worker_size = static_cast<int>(grpc_workers.size());
    std::vector<WorkerRpcContext> worker_rpc_contexts(worker_size);

    for (int rank = 0; rank < worker_size; ++rank) {
        RTP_LLM_LOG_DEBUG("send get cache rpc request to rank: %d, addr: %s", rank, grpc_workers[rank].c_str());
        const auto& worker_addr    = grpc_workers[rank];
        auto        connect_status = rpc_pool_->getConnection(worker_addr);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING("rpc get cache failed, get grpc connection failed for rank: %d, request: %ld, addr: %s",
                                rank,
                                request_id,
                                worker_addr.c_str());
            return false;
        }
        auto& rpc_context = worker_rpc_contexts[rank];
        auto& request     = rpc_context.request;

        request.set_request_id(request_id);
        for (const auto cache_key : cache_keys) {
            request.add_cache_keys(cache_key);
        }
        for (const auto block_index : block_indices) {
            request.add_block_ids(block_index);
        }
        request.set_op(op_type == OpType::OP_GET ? DistKvCacheOp::GET : DistKvCacheOp::PUT);
        for (const auto& extra_meta : extra_metas) {
            auto* meta = request.add_extra_metas();
            meta->set_key(extra_meta.first);
            meta->set_value(extra_meta.second);
        }

        rpc_context.stub        = connect_status.value().stub;
        rpc_context.server_addr = worker_addr;

        std::unique_ptr<::grpc::ClientAsyncResponseReader<DistKvCacheResponsePB>> reader(
            rpc_context.stub->AsyncDistKvCache(
                rpc_context.client_context.get(), request, &(rpc_context.completion_queue)));
        reader->Finish(&(rpc_context.response), &rpc_context.status, reinterpret_cast<void*>(rank));
    }

    std::vector<int> success_ranks(worker_size, 0);
    int              finished_count      = 0;
    bool             all_request_success = true;
    const int        timeout_ms          = 5000;  // TODO(LXQ): timeout parameter
    const auto       start_time          = std::chrono::steady_clock::now();

    while (true) {
        if (finished_count == worker_size) {
            break;
        }

        if (std::chrono::steady_clock::now() - start_time >= std::chrono::milliseconds(timeout_ms)) {
            std::string timeout_ranks_str;
            for (int rank = 0; rank < worker_size; ++rank) {
                if (success_ranks[rank] == 0) {
                    timeout_ranks_str += std::to_string(rank) + ",";
                }
            }
            RTP_LLM_LOG_WARNING("request timeout: %d ms, timeout rank: [%s], request: %ld",
                                timeout_ms,
                                timeout_ranks_str.c_str(),
                                request_id);
            all_request_success = false;
            break;
        }

        const int  once_timeout_ms = 10;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (uint32_t i = 0; i < worker_rpc_contexts.size(); i++) {
            if (success_ranks[i] == 1) {
                continue;
            }
            auto& rpc_context = worker_rpc_contexts.at(i);

            void*      got_tag     = nullptr;
            bool       ok          = false;
            const auto next_status = rpc_context.completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_LOG_WARNING(
                    "request failed, grpc completion queue failed, status: %d, request: %ld", next_status, request_id);
                all_request_success = false;
                break;
            }
            ++finished_count;

            const int   rank   = reinterpret_cast<intptr_t>(got_tag);
            const auto& status = rpc_context.status;
            if (!status.ok()) {
                RTP_LLM_LOG_WARNING("request failed, rank %d failed, error: %d(%s), addr: %s, request: %ld",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    rpc_context.server_addr.c_str(),
                                    request_id);
                all_request_success = false;
                break;
            }
            success_ranks[rank] = 1;
            RTP_LLM_LOG_DEBUG("rank %d request success, request: %ld", rank, request_id);
        }
        if (!all_request_success) {
            break;
        }
    }
    return all_request_success;
}

}  // namespace rtp_llm