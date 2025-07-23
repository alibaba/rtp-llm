#include "rtp_llm/cpp/cache/DistKvCache.h"

#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/DistKvCacheMetrics.h"

using namespace rtp_llm::threefs;

namespace rtp_llm {

inline std::size_t hashString(const std::string& str) {
    std::hash<std::string> hasher;
    return hasher(str);
}

DistKvCache::DistKvCache(CacheManager*                       cache_manager,
                         const GptInitParameter&             gpt_init_params,
                         const kmonitor::MetricsReporterPtr& metrics_reporter):
    cache_manager_(cache_manager), gpt_init_params_(gpt_init_params), metrics_reporter_(metrics_reporter) {}

DistKvCache::~DistKvCache() {
    RTP_LLM_LOG_INFO("DistKvCache destructor");
    rpc_pool_.reset();
    storage_.reset();
    planner_.reset();
    metrics_reporter_.reset();
    cache_manager_ = nullptr;
}

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

    if (!initDefaultMetas(storage_3fs_init_params)) {
        RTP_LLM_LOG_WARNING("init failed, init default metas failed");
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();
    RTP_LLM_LOG_INFO("dist kv cache init success");
    return true;
}

bool DistKvCache::initDefaultMetas(const DistStorage3FSInitParams& storage_3fs_init_params) {
    const auto& cache_config             = cache_manager_->cacheConfig();
    default_metas_["SEQ_SIZE_PER_BLOCK"] = std::to_string(cache_config.seq_size_per_block);
    default_metas_["DTYPE"]              = std::to_string(static_cast<int>(cache_config.dtype));
    default_metas_["USE_MLA"]            = std::to_string(static_cast<int>(cache_config.use_mla));
    default_metas_["TP_SIZE"]            = std::to_string(gpt_init_params_.tp_size_);
    default_metas_["TP_RANK"]            = std::to_string(gpt_init_params_.tp_rank_);

    auto biz_name = autil::EnvUtil::getEnv("BIZ_NAME", std::string(""));
    if (biz_name.empty()) {
        RTP_LLM_LOG_WARNING("init metas failed, biz name is empty");
        return false;
    }
    default_metas_["BIZ_NAME"] = biz_name;

    const auto ckpt_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
    if (ckpt_path.empty()) {
        RTP_LLM_LOG_WARNING("init metas failed, ckpt path is empty");
        return false;
    }
    default_metas_["CKPT_PATH"]      = std::to_string(hashString(ckpt_path));
    default_metas_["LORA_CKPT_PATH"] = "no_lora";

    std::ostringstream oss;
    for (const auto& [key, value] : default_metas_) {
        oss << key << ":" << value << ", ";
    }
    RTP_LLM_LOG_INFO("dist kv cache default metas: [%s]", oss.str().c_str());

    return true;
}

int32_t DistKvCache::matchForAllRank(const std::vector<int64_t>&        cache_keys,
                                     int64_t                            request_id,
                                     std::map<std::string, std::string> extra_metas) {
    if (cache_keys.empty() || !planner_ || !storage_) {
        return 0;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markMatchBeginUs(metrics);

    int32_t match_len = static_cast<int32_t>(cache_keys.size());
    for (int i = 0; i < gpt_init_params_.tp_size_; i++) {
        extra_metas["TP_RANK"] = std::to_string(i);
        auto ret               = match(cache_keys, request_id, extra_metas);
        if (ret < match_len) {
            match_len = ret;
        }
        if (match_len == 0) {
            break;
        }
    }

    DistKvCacheMetrics::markMatchDoneUs(metrics);

    total_match_len_.fetch_add(match_len);
    int64_t       total_match_len = total_match_len_.load();
    const int32_t input_len       = static_cast<int32_t>(cache_keys.size());
    total_input_len_.fetch_add(input_len);
    int64_t total_input_len = total_input_len_.load();

    const auto& cache_config = cache_manager_->cacheConfig();
    DistKvCacheMetrics::setMatchQps(metrics, true);
    DistKvCacheMetrics::setMatchSuccessQps(metrics, match_len > 0);
    DistKvCacheMetrics::setCacheInputLength(metrics, input_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setCacheMatchLength(metrics, match_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setCacheHitRate(metrics, match_len * 100.0 / input_len);
    DistKvCacheMetrics::setTotalCacheMatchLength(metrics, total_match_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setTotalCacheInputLength(metrics, total_input_len * cache_config.seq_size_per_block);
    DistKvCacheMetrics::setTotalCacheHitRate(metrics, total_match_len * 100.0 / total_input_len);

    return match_len;
}

int32_t DistKvCache::match(const std::vector<int64_t>&        cache_keys,
                           int64_t                            request_id,
                           std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    int32_t match_len         = 0;
    int32_t already_match_len = 0;
    if (const auto it = extra_metas.find("IGNORE_CACHE_KEY_NUM"); it != extra_metas.end()) {
        already_match_len = std::stoi(it->second);
    }

    for (int len = cache_keys.size(); len > already_match_len; --len) {
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + len);
        auto                 layout_items = planner_->layout(cache_keys_to_match, {}, extra_metas, true);
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
            match_len = len;
            break;
        }
    }

    RTP_LLM_LOG_DEBUG("request: %ld, already match len: %d, dist kv match len: %d, cache keys: [%zu|%s]",
                      request_id,
                      already_match_len,
                      match_len,
                      cache_keys.size(),
                      vectorToString(cache_keys).c_str());
    return match_len;
}

bool DistKvCache::getForAllRank(const std::vector<int64_t>&        cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "get cache for all rank failed, cache key size %zu not equal to block index size %zu, request id: %lld",
            cache_keys.size(),
            block_indices.size(),
            request_id);
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalGetCacheBeginUs(metrics);

    bool result = syncCallAllRank(cache_keys, block_indices, request_id, extra_metas, OpType::OP_GET);

    DistKvCacheMetrics::markTotalGetCacheDoneUs(metrics);
    DistKvCacheMetrics::setGetCacheFailedQps(metrics, result == false);
    if (result) {
        int32_t local_match_len = 0;
        if (const auto it = extra_metas.find("IGNORE_CACHE_KEY_NUM"); it != extra_metas.end()) {
            local_match_len = std::stoi(it->second);
        }
        auto        remote_match_len = cache_keys.size() - local_match_len;
        const auto& cache_config     = cache_manager_->cacheConfig();
        DistKvCacheMetrics::setCacheGetLength(metrics, remote_match_len * cache_config.seq_size_per_block);
    }

    return result;
}

bool DistKvCache::get(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }
    extra_metas["GET"] = "1";

    auto layout_items = planner_->layout(cache_keys, block_indices, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markGetCacheBeginUs(metrics);

    for (auto& item : layout_items) {
        if (!storage_->get(item)) {
            RTP_LLM_LOG_WARNING("get cache failed, rank: %d, key: %s", gpt_init_params_.tp_rank_, item.key.c_str());
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
    auto tmp_extra_metas = extra_metas;
    for (auto& [key, value] : default_metas_) {
        if (tmp_extra_metas.count(key) == 0) {
            tmp_extra_metas[key] = value;
        }
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalPutCacheBeginUs(metrics);

    bool exist = true;
    for (int i = 0; i < gpt_init_params_.tp_size_; i++) {
        tmp_extra_metas["TP_RANK"] = std::to_string(i);
        auto layout_items          = planner_->layout(cache_keys, block_indices, tmp_extra_metas, true);
        for (auto& item : layout_items) {
            if (!storage_->lookup(item)) {
                exist = false;
                break;
            }
        }
        if (!exist) {
            break;
        }
    }

    if (exist) {
        DistKvCacheMetrics::markTotalPutCacheDoneUs(metrics);
        return true;
    }

    bool result = syncCallAllRank(cache_keys, block_indices, request_id, extra_metas, OpType::OP_PUT);

    const auto& cache_config = cache_manager_->cacheConfig();
    DistKvCacheMetrics::markTotalPutCacheDoneUs(metrics);
    DistKvCacheMetrics::setPutCacheFailedQps(metrics, result == false);
    DistKvCacheMetrics::setCachePutLength(metrics, cache_keys.size() * cache_config.seq_size_per_block);

    return result;
}

bool DistKvCache::put(const std::vector<int64_t>&        cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    auto layout_items = planner_->layout(cache_keys, block_indices, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache put cache, layout iovs is empty");
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
    const int        timeout_ms =
        op_type == OpType::OP_GET ? init_params_.rpc_get_cache_timeout_ms : init_params_.rpc_put_cache_timeout_ms;
    const auto start_time = std::chrono::steady_clock::now();

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