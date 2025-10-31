#include "rtp_llm/cpp/cache/DistKvCache.h"

#include <future>
#include <thread>
#include <algorithm>

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
    if (io_thread_pool_) {
        io_thread_pool_->stop();
        io_thread_pool_->waitFinish();
        io_thread_pool_.reset();
    }
    if (wait_match_thread_pool_) {
        wait_match_thread_pool_->stop();
        wait_match_thread_pool_->waitFinish();
        wait_match_thread_pool_.reset();
    }
    rpc_pool_.reset();
    storage_.reset();
    planner_.reset();
    metrics_reporter_.reset();
    cache_manager_ = nullptr;
}

bool DistKvCache::init(const DistKvCacheInitParams& init_params) {
    RTP_LLM_LOG_INFO("dist kvcache init params: [%s]", init_params.toString().c_str());
    init_params_         = init_params;
    auto& storage_params = init_params_.storage_manager_params;

    // default 3fs planner
    if (!storage_params.init_params_3fs.has_value()) {
        RTP_LLM_LOG_WARNING("init failed, 3fs init params is empty");
        return false;
    }
    const auto& init_params_3fs = storage_params.init_params_3fs.value();
    planner_                    = std::make_unique<DefaultDistKvCachePlanner>(
        cache_manager_, gpt_init_params_, init_params_3fs, metrics_reporter_);

    storage_params.lookup_timeout_ms = init_params.match_timeout_ms;
    storage_params.get_timeout_ms    = init_params.rpc_get_cache_timeout_ms;
    storage_params.put_timeout_ms    = init_params.rpc_put_cache_timeout_ms;

    storage_ = std::make_unique<DistStorageManager>(metrics_reporter_);
    if (!storage_->init(storage_params)) {
        RTP_LLM_LOG_WARNING("init failed, init storage failed");
        return false;
    }

    if (!initDefaultMetas()) {
        RTP_LLM_LOG_WARNING("init failed, init default metas failed");
        return false;
    }

    wait_match_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(thread_num_, queue_size_, nullptr, "WaitMatchThreadPool");
    if (!wait_match_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start wait match thread pool failed, thread num: %zu, queue size: %zu",
                            thread_num_,
                            queue_size_);
        return false;
    }

    // parallel IO pool for 3FS get/put of multiple items
    io_thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(io_thread_num_, io_queue_size_, nullptr, "DistKvCacheIOThreadPool");
    if (!io_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("init failed, start io thread pool failed, thread num: %zu, queue size: %zu",
                            io_thread_num_,
                            io_queue_size_);
        return false;
    }

    rpc_pool_ = std::make_shared<rtp_llm::RPCPool>();
    RTP_LLM_LOG_INFO("dist kv cache init success");
    return true;
}

bool DistKvCache::initDefaultMetas() {
    const auto& cache_config             = cache_manager_->cacheConfig();
    default_metas_["SEQ_SIZE_PER_BLOCK"] = std::to_string(cache_config.seq_size_per_block);
    default_metas_["DTYPE"]              = std::to_string(static_cast<int>(cache_config.dtype));
    default_metas_["USE_MLA"]            = std::to_string(static_cast<int>(cache_config.use_mla));
    default_metas_["TP_SIZE"]            = std::to_string(gpt_init_params_.tp_size_);
    default_metas_["TP_RANK"]            = std::to_string(gpt_init_params_.tp_rank_);
    default_metas_["LAYOUT_VERSION"]     = "v2";

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

int32_t DistKvCache::matchForAllRank(const std::vector<size_t>&         cache_keys,
                                     size_t                             ignore_block_num,
                                     int64_t                            request_id,
                                     std::map<std::string, std::string> extra_metas) {
    if (cache_keys.empty() || !planner_ || !storage_) {
        return 0;
    }

    if (wait_match_thread_pool_->isFull()) {
        RTP_LLM_LOG_WARNING("match for all rank failed, wait match thread pool is full, something maybe wrong");
        return 0;
    }

    int32_t       match_len    = 0;
    const auto&   cache_config = cache_manager_->cacheConfig();
    const int32_t input_len    = static_cast<int32_t>(cache_keys.size());

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markMatchBeginUs(metrics);

    auto stop = std::make_shared<std::atomic<bool>>(false);
    auto task =
        [weak_this = weak_from_this(), cache_keys, ignore_block_num, request_id, extra_metas, stop]() -> int32_t {
        if (weak_this.expired()) {
            RTP_LLM_LOG_WARNING("match for all rank failed, dist kv cache has been expired, request_id: %ld",
                                request_id);
            return 0;
        }
        auto shared_this = weak_this.lock();

        int32_t match_len = static_cast<int32_t>(cache_keys.size());
        auto    metas     = extra_metas;
        for (int i = 0; i < shared_this->gpt_init_params_.tp_size_; i++) {
            metas["TP_RANK"] = std::to_string(i);
            auto ret         = shared_this->match(cache_keys, ignore_block_num, request_id, metas, stop);
            if (ret < match_len) {
                match_len = ret;
            }
            if (match_len == 0) {
                break;
            }
        }
        return match_len;
    };

    auto future = wait_match_thread_pool_->async(task);
    if (future.wait_for(std::chrono::milliseconds(init_params_.match_timeout_ms)) == std::future_status::ready) {
        match_len = future.get();
    } else {
        RTP_LLM_LOG_WARNING(
            "match for all rank timeout, timeout: %d ms, request: %ld, input block num: %zu, ignore block num: %zu",
            init_params_.match_timeout_ms,
            request_id,
            cache_keys.size(),
            ignore_block_num);
        if (stop) {
            stop->store(true);
        }
    }

    DistKvCacheMetrics::markMatchDoneUs(metrics);
    DistKvCacheMetrics::setMatchQps(metrics, true);
    DistKvCacheMetrics::setMatchFailedQps(metrics, match_len <= 0);
    DistKvCacheMetrics::setCacheInputLength(metrics, input_len * cache_config.seq_size_per_block);

    if (match_len > 0) {
        total_match_len_.fetch_add(match_len);
        int64_t total_match_len = total_match_len_.load();
        total_input_len_.fetch_add(input_len);
        int64_t total_input_len = total_input_len_.load();

        DistKvCacheMetrics::setCacheMatchLength(metrics, match_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setCacheHitRate(metrics, match_len * 100.0 / input_len);
        DistKvCacheMetrics::setTotalCacheMatchLength(metrics, total_match_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setTotalCacheInputLength(metrics, total_input_len * cache_config.seq_size_per_block);
        DistKvCacheMetrics::setTotalCacheHitRate(metrics, total_match_len * 100.0 / total_input_len);
    }
    return match_len;
}

int32_t DistKvCache::match(const std::vector<size_t>&                cache_keys,
                           size_t                                    ignore_block_num,
                           int64_t                                   request_id,
                           std::map<std::string, std::string>        extra_metas,
                           const std::shared_ptr<std::atomic<bool>>& stop) const {
    if (stop && stop->load()) {
        return 0;
    }
    if (cache_keys.empty()) {
        return 0;
    }

    RTP_LLM_LOG_DEBUG("match cache_keys size: %zu", cache_keys.size());
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    RTP_LLM_LOG_DEBUG("cache_keys size: %zu", cache_keys.size());
    RTP_LLM_LOG_DEBUG("ignore_block_num: %zu", ignore_block_num);
    RTP_LLM_LOG_DEBUG("extra_metas: %s", extra_metas.at("TP_RANK").c_str());

    // get layout at the granularity of items i.e. blocks collection
    auto layout_items = planner_->layout(cache_keys, {}, ignore_block_num, extra_metas);

    if (layout_items.empty()) {
        return 0;
    }

    if (storage_->lookup(layout_items.back())) {
        RTP_LLM_LOG_DEBUG("request[%ld] match all", request_id);
        return cache_keys.size();
    }

    int32_t left      = 0;
    int32_t right     = static_cast<int32_t>(layout_items.size()) - 2;
    int32_t mid       = 0;
    int32_t match_len = 0;

    while (left <= right) {
        if (stop && stop->load()) {
            return 0;
        }
        mid = left + (right - left) / 2;
        if (storage_->lookup(layout_items[mid])) {
            RTP_LLM_LOG_DEBUG("request[%ld] match index: %d", request_id, mid);
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    match_len = (right + 1) * init_params_.max_block_size_per_item;
    DistStorage::Item item;
    // deep search partial block
    item       = DistStorage::Item();
    item.type  = DistStorage::ST_3FS;
    item.metas = extra_metas;

    int start_key = (right + 1) * init_params_.max_block_size_per_item;
    int end_key   = start_key + init_params_.max_block_size_per_item - 1;
    end_key       = std::min(end_key, static_cast<int>(cache_keys.size()) - 1);
    for (int i = end_key; i >= start_key; i--) {
        item.key               = std::to_string(cache_keys[start_key]) + "_" + std::to_string(cache_keys[i]);
        item.metas["ITEM_KEY"] = item.key;
        if (storage_->lookup(item)) {
            RTP_LLM_LOG_DEBUG(
                "request[%ld] deep match item: %s, len: %d", request_id, item.key.c_str(), i - start_key + 1);
            match_len += (i - start_key + 1);
            break;
        }
    }

    return match_len;
}

bool DistKvCache::getForAllRank(const std::vector<size_t>&         cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                size_t                             ignore_block_num,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);

    const auto input_len        = static_cast<int32_t>(cache_keys.size());
    const auto remote_match_len = input_len - ignore_block_num;
    if (remote_match_len != block_indices.size()) {
        DistKvCacheMetrics::setGetCacheFailedQps(metrics, true);
        RTP_LLM_LOG_WARNING(
            "get cache for all rank failed, cache key size %d not equal to block size %zu, request id: %lld",
            remote_match_len,
            block_indices.size(),
            request_id);
        return false;
    }

    DistKvCacheMetrics::markTotalGetCacheBeginUs(metrics);

    bool result = syncCallAllRank(cache_keys, block_indices, ignore_block_num, request_id, extra_metas, OpType::OP_GET);

    DistKvCacheMetrics::markTotalGetCacheDoneUs(metrics);
    DistKvCacheMetrics::setGetCacheFailedQps(metrics, result == false);
    if (result) {
        const auto& cache_config = cache_manager_->cacheConfig();
        DistKvCacheMetrics::setCacheGetLength(metrics, remote_match_len * cache_config.seq_size_per_block);
    }

    return result;
}

bool DistKvCache::get(const std::vector<size_t>&         cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      size_t                             ignore_block_num,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    auto layout_items = planner_->layout(cache_keys, block_indices, ignore_block_num, extra_metas);
    for (auto& item : layout_items) {
        RTP_LLM_LOG_DEBUG("layout item: %s", item.key.c_str());
    }

    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markGetCacheBeginUs(metrics);

    std::vector<autil::ThreadPoolBase::Future<bool>> get_futures;
    get_futures.reserve(layout_items.size());
    for (auto& it : layout_items) {
        auto get_task = [this, it]() mutable -> bool { return storage_->get(it); };
        get_futures.emplace_back(io_thread_pool_->async(get_task));
    }

    bool any_failed = false;
    for (auto& future : get_futures) {
        if (future.valid() && !future.get()) {
            any_failed = true;
            break;
        }
    }
    if (any_failed) {
        DistKvCacheMetrics::markGetCacheDoneUs(metrics);
        return false;
    }

    DistKvCacheMetrics::markGetCacheDoneUs(metrics);

    if (!planner_->verify(layout_items, cache_keys, block_indices, extra_metas)) {
        RTP_LLM_LOG_WARNING("dist kv cache get cache, verify cache failed, request key: %s", request_id);
        return false;
    }

    return true;
}

bool DistKvCache::putForAllRank(const std::vector<size_t>&         cache_keys,
                                const std::vector<int32_t>&        block_indices,
                                size_t                             ignore_block_num,
                                int64_t                            request_id,
                                std::map<std::string, std::string> extra_metas) const {
    auto rpc_extra_metas = extra_metas;  // 拷贝一份用于rpc传输, 否则rpc可能会传比较多冗余信息
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markTotalPutCacheBeginUs(metrics);

    bool result =
        syncCallAllRank(cache_keys, block_indices, ignore_block_num, request_id, rpc_extra_metas, OpType::OP_PUT);

    const auto& cache_config = cache_manager_->cacheConfig();
    DistKvCacheMetrics::markTotalPutCacheDoneUs(metrics);
    DistKvCacheMetrics::setPutCacheFailedQps(metrics, result == false);
    DistKvCacheMetrics::setCachePutLength(metrics, cache_keys.size() * cache_config.seq_size_per_block);

    return result;
}

bool DistKvCache::put(const std::vector<size_t>&         cache_keys,
                      const std::vector<int32_t>&        block_indices,
                      size_t                             ignore_block_num,
                      int64_t                            request_id,
                      std::map<std::string, std::string> extra_metas) const {
    for (auto& [key, value] : default_metas_) {
        if (extra_metas.count(key) == 0) {
            extra_metas[key] = value;
        }
    }

    auto layout_items = planner_->layout(cache_keys, block_indices, ignore_block_num, extra_metas);
    if (layout_items.empty()) {
        RTP_LLM_LOG_WARNING("dist kv cache put cache, layout iovs is empty");
        return false;
    }

    auto metrics = DistKvCacheMetricsFactory::createMetrics(metrics_reporter_);
    DistKvCacheMetrics::markPutCacheBeginUs(metrics);

    std::vector<autil::ThreadPoolBase::Future<bool>> put_futures;
    put_futures.reserve(layout_items.size());
    for (auto& it : layout_items) {
        auto put_task = [this, it]() mutable -> bool { return storage_->putIfNotExist(it); };
        put_futures.emplace_back(io_thread_pool_->async(put_task));
    }

    bool any_failed = false;
    for (auto& future : put_futures) {
        if (future.valid() && !future.get()) {
            any_failed = true;
            break;
        }
    }
    if (any_failed) {
        DistKvCacheMetrics::markPutCacheDoneUs(metrics);
        return false;
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
bool DistKvCache::syncCallAllRank(const std::vector<size_t>&         cache_keys,
                                  const std::vector<int32_t>&        block_indices,
                                  size_t                             ignore_block_num,
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
        request.set_ignore_block_num(ignore_block_num);
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