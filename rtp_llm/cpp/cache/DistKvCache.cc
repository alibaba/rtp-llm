#include "rtp_llm/cpp/cache/DistKvCache.h"

namespace rtp_llm {

DistKvCache::DistKvCache(CacheManager*                       cache_manager,
                         const GptInitParameter&             params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), params_(params), metrics_reporter_(metrics_reporter) {}

DistKvCache::~DistKvCache() {}

bool DistKvCache::init(const DistKvCacheInitParams& init_params) {
    init_params_ = init_params;

    planner_.reset(new DefaultDistKvCachePlanner(cache_manager_, params_, metrics_reporter_));

    storage_ = std::make_unique<DistStorageManager>(metrics_reporter_);
    if (!storage_->init(init_params_.manager_params)) {
        RTP_LLM_LOG_WARNING("dist kv cache init, init storage failed");
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();
    RTP_LLM_LOG_INFO("dist kv cache init");
    return true;
}

int32_t DistKvCache::matchCacheForAllRank(const std::vector<int64_t>&        cache_keys,
                                          const std::vector<int32_t>&        block_indices,
                                          int64_t                            request_id,
                                          std::map<std::string, std::string> extra_metas) const {
    if (cache_keys.empty() || block_indices.empty()) {
        return 0;
    }

    if (!planner_ || !storage_) {
        return 0;
    }

    auto match_len = cache_keys.size();
    for (int i = 0; i < params_.tp_size_; i++) {
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + match_len);
        auto                 ret = matchCache(cache_keys_to_match, block_indices, request_id, extra_metas, i);
        if (ret < match_len) {
            match_len = ret;
        }
        if (match_len == 0) {
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("dist kv cache match cache for all rank, matched len %d, input cache keys len %d",
                      match_len,
                      cache_keys.size());
    return match_len;
}

int32_t DistKvCache::matchCache(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas,
                                int                                tp_rank) const {

    if (!extra_metas.empty()) {
        for (auto& [key, value] : default_metas_) {
            extra_metas[key] = value;
        }
    }
    int match_len = cache_keys.size();
    for (; match_len > 0; --match_len) {
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + match_len);

        auto layout_items = planner_->layout(
            cache_keys_to_match, block_indices, extra_metas.empty() ? default_metas_ : extra_metas, tp_rank, true);
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
                      params_.tp_rank_,
                      match_len,
                      cache_keys.size());
    return match_len;
}

bool DistKvCache::getCacheForAllRank(const std::vector<int64_t>&        cache_keys,
                                     const std::vector<int32_t>&        block_indices,
                                     int64_t                            request_id,
                                     std::map<std::string, std::string> extra_metas) const {
    auto match_len = matchCacheForAllRank(cache_keys, block_indices, request_id, extra_metas);
    if (match_len == 0) {
        return false;
    }
    std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + match_len);
    std::vector<int32_t> block_indices_to_match(block_indices.begin(), block_indices.begin() + match_len);
    return syncCallAllRank(cache_keys_to_match, block_indices_to_match, request_id, extra_metas, OpType::OP_GET);
}

bool DistKvCache::getCache(const std::vector<int64_t>&        cache_keys,
                           const std::vector<int32_t>&        block_indices,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    if (!extra_metas.empty()) {
        for (auto& [key, value] : default_metas_) {
            extra_metas[key] = value;
        }
    }

    auto layout_items = planner_->layout(
        cache_keys, block_indices, extra_metas.empty() ? default_metas_ : extra_metas, params_.tp_rank_);
    if (layout_items.empty()) {
        RTP_LLM_LOG_DEBUG("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    for (auto& item : layout_items) {
        if (!storage_->get(item)) {
            RTP_LLM_LOG_WARNING("dist kv cache get cache, get cache failed, key: %s", item.key.c_str());
            return false;
        }
    }
    if (!planner_->verify(layout_items,
                          cache_keys,
                          block_indices,
                          extra_metas.empty() ? default_metas_ : extra_metas,
                          params_.tp_rank_)) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, verify cache failed, request key: %s", request_id);
        return false;
    }

    return true;
}

bool DistKvCache::putForAllRank(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    return syncCallAllRank(cache_keys, block_indices, request_id, extra_metas, OpType::OP_PUT);
}

bool DistKvCache::putCache(const std::vector<int64_t>&        cache_keys,
                           const std::vector<int32_t>&        block_indices,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    if (!extra_metas.empty()) {
        for (auto& [key, value] : default_metas_) {
            extra_metas[key] = value;
        }
    }

    auto layout_items = planner_->layout(
        cache_keys, block_indices, extra_metas.empty() ? default_metas_ : extra_metas, params_.tp_rank_);
    if (layout_items.empty()) {
        RTP_LLM_LOG_DEBUG("dist kv cache put cache, layout iovs is empty");
        return false;
    }

    for (auto& item : layout_items) {
        if (!storage_->put(item)) {
            RTP_LLM_LOG_WARNING("dist kv cache put cache, put cache failed, key: %s", item.key.c_str());
            return false;
        }
    }
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
    const auto& grpc_workers = params_.worker_grpc_addrs_;
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