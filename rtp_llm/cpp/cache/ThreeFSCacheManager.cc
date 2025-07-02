#include "rtp_llm/cpp/cache/ThreeFSCacheManager.h"

#include "rtp_llm/cpp/cache/ThreeFSBlockCache.h"
#include "rtp_llm/cpp/cache/ThreeFSMetrics.h"
#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm::threefs {

ThreeFSCacheManager::~ThreeFSCacheManager() {
    RTP_LLM_LOG_INFO("3fs cache manager destructor");
    rpc_pool_.reset();
    threefs_block_cache_.reset();
}

bool ThreeFSCacheManager::init() {
    auto threefs_block_cache =
        std::make_shared<ThreeFSBlockCache>(k_cache_, v_cache_, cache_config_, metrics_reporter_);
    if (!threefs_block_cache->init()) {
        RTP_LLM_LOG_WARNING("init failed, 3fs block cache init failed");
        return false;
    }
    threefs_block_cache_ = threefs_block_cache;
    rpc_pool_            = std::make_shared<rtp_llm::RPCPool>();
    return true;
}

int32_t ThreeFSCacheManager::matchCache(const std::vector<int64_t>& cache_keys) const {
    if (cache_keys.empty()) {
        RTP_LLM_LOG_WARNING("match cache failed, cache keys is empty");
        return 0;
    }

    if (!threefs_block_cache_) {
        RTP_LLM_LOG_WARNING("match cache failed, 3fs block cache is nullptr");
        return 0;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
    ThreeFSMetrics::markMatchBeginUs(metrics);

    int32_t matched_len = 0;
    for (int i = cache_keys.size() - 1; i >= 0; --i) {
        const auto           kvcache_key = constructKvCacheKey(cache_keys[i]);
        std::vector<int64_t> cache_keys_to_match(cache_keys.begin(), cache_keys.begin() + i + 1);
        if (threefs_block_cache_->matchCache(kvcache_key, cache_keys_to_match)) {
            matched_len = i + 1;
            break;
        }
    }
    ThreeFSMetrics::markMatchDoneUs(metrics);

    RTP_LLM_LOG_DEBUG("3fs matched len: %d, input cache keys: [%zu|%s]",
                      matched_len,
                      cache_keys.size(),
                      vectorToString(cache_keys).c_str());
    return matched_len;
}

bool ThreeFSCacheManager::getCacheForRank(const std::vector<int64_t>& cache_keys,
                                          const std::vector<int32_t>& block_indices,
                                          int64_t                     request_id) const {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "get cache for rank failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu, request: %ld",
            cache_keys.size(),
            block_indices.size(),
            request_id);
        return false;
    }
    if (!threefs_block_cache_) {
        RTP_LLM_LOG_WARNING("get cache for rank failed, 3fs block cache is nullptr, request: %ld", request_id);
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
    ThreeFSMetrics::markGetCacheBeginUs(metrics);

    const auto kvcache_key = constructKvCacheKey(cache_keys.back());
    auto       result      = threefs_block_cache_->getCache(kvcache_key, cache_keys, block_indices);
    ThreeFSMetrics::markGetCacheDoneUs(metrics);

    if (!result) {
        RTP_LLM_LOG_WARNING(
            "get cache for rank failed, kvcache key: %s, request: %ld", kvcache_key.c_str(), request_id);
    }
    return result;
}

bool ThreeFSCacheManager::putCacheForRank(const std::vector<int64_t>& cache_keys,
                                          const std::vector<int32_t>& block_indices,
                                          int64_t                     request_id) const {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "put cache for rank failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu, request: %ld",
            cache_keys.size(),
            block_indices.size(),
            request_id);
        return false;
    }
    if (!threefs_block_cache_) {
        RTP_LLM_LOG_WARNING("put cache for rank failed, 3fs block cache is nullptr, request: %ld", request_id);
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
    ThreeFSMetrics::markPutCacheBeginUs(metrics);

    const auto kvcache_key = constructKvCacheKey(cache_keys.back());
    auto       result      = threefs_block_cache_->putCache(kvcache_key, cache_keys, block_indices);
    ThreeFSMetrics::markPutCacheDoneUs(metrics);

    if (!result) {
        RTP_LLM_LOG_WARNING(
            "put cache for rank failed, kvcache key: %s, request: %ld", kvcache_key.c_str(), request_id);
        return false;
    }
    RTP_LLM_LOG_DEBUG("put cache for rank success, kvcache key: %s, cache keys: [%lu|%s], block indices: [%lu|%s]",
                      kvcache_key.c_str(),
                      cache_keys.size(),
                      vectorToString(cache_keys).c_str(),
                      block_indices.size(),
                      vectorToString(block_indices).c_str());
    return true;
}

bool ThreeFSCacheManager::getCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                             const std::vector<int32_t>& block_indices,
                                             int32_t                     input_len,
                                             int64_t                     request_id) {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "get cache for all rank failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu",
            cache_keys.size(),
            block_indices.size());
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
    ThreeFSMetrics::markTotalGetCacheBeginUs(metrics);

    bool result = checkCacheIn3FS(cache_keys, false);
    if (result) {
        result = rpcGetCacheForAllRank(cache_keys, block_indices, request_id);
        if (!result) {
            RTP_LLM_LOG_WARNING(
                "get cache for all rank failed, request: %ld, cache key count: %zu", request_id, cache_keys.size());
        }
    }

    ThreeFSMetrics::markTotalGetCacheDoneUs(metrics);
    ThreeFSMetrics::setGetCacheFailedQps(metrics, result == false);

    const auto matched_len       = result ? static_cast<int32_t>(cache_keys.size()) : 0;
    int64_t    total_matched_len = 0;
    int64_t    total_input_len   = 0;
    {
        std::unique_lock<std::mutex> lock(cache_key_num_mutex_);
        cache_key_num_.first += matched_len;
        cache_key_num_.second += input_len;
        total_matched_len = cache_key_num_.first;
        total_input_len   = cache_key_num_.second;
    }

    const auto seq_size_per_block = cache_config_.seq_size_per_block;
    ThreeFSMetrics::setCacheReuseLength(metrics, matched_len * seq_size_per_block);
    ThreeFSMetrics::setTotalCacheReuseLength(metrics, total_matched_len * seq_size_per_block);
    ThreeFSMetrics::setTotalCacheInputLength(metrics, total_input_len * seq_size_per_block);
    ThreeFSMetrics::setTotalCacheHitRate(metrics, total_matched_len * 100.0 / total_input_len);
    return result;
}

bool ThreeFSCacheManager::rpcGetCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                                const std::vector<int32_t>& block_indices,
                                                int64_t                     request_id) const {
    const auto& grpc_workers = params_.worker_grpc_addrs_;
    if (grpc_workers.empty()) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, grpc workers is empty, request: %ld", request_id);
        return false;
    }
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("rpc get cache failed, rpc pool is nullptr, request: %ld", request_id);
        return false;
    }

    const int                                worker_size = static_cast<int>(grpc_workers.size());
    std::vector<WorkerRpcContext>            worker_rpc_contexts(worker_size);
    std::vector<BroadcastGetCacheResponsePB> responses(worker_size);

    const uint32_t                       cq_size = worker_size % 2 == 0 ? worker_size / 2 : worker_size / 2 + 1;
    std::vector<::grpc::CompletionQueue> completion_queues(cq_size);
    std::vector<int>                     request_num_per_cq(cq_size, 0);

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

        BroadcastGetCacheRequestPB get_cache_request;
        get_cache_request.set_request_id(request_id);
        for (const auto cache_key : cache_keys) {
            get_cache_request.add_cache_keys(cache_key);
        }
        for (const auto block_index : block_indices) {
            get_cache_request.add_block_ids(block_index);
        }

        auto& rpc_context       = worker_rpc_contexts[rank];
        rpc_context.stub        = connect_status.value().stub;
        rpc_context.server_addr = worker_addr;
        const int queue_index   = rank % cq_size;

        std::unique_ptr<::grpc::ClientAsyncResponseReader<BroadcastGetCacheResponsePB>> reader(
            rpc_context.stub->AsyncRemoteGetCache(
                rpc_context.client_context.get(), get_cache_request, &completion_queues[queue_index]));
        reader->Finish(&responses[rank], &rpc_context.status, reinterpret_cast<void*>(rank));
        ++request_num_per_cq[queue_index];
    }

    return waitAllReuqestDone(worker_rpc_contexts, completion_queues, request_num_per_cq, request_id);
}

bool ThreeFSCacheManager::putCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                             const std::vector<int32_t>& block_indices,
                                             int64_t                     request_id) const {
    if (cache_keys.empty() && block_indices.empty()) {
        return true;
    }
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_WARNING(
            "put cache for all rank failed, cache key size not equal to block index size, cache key size: %lu, block index size: %lu, request: %ld",
            cache_keys.size(),
            block_indices.size(),
            request_id);
        return false;
    }

    auto metrics = ThreeFSMetricsFactory::createMetrics(metrics_reporter_);
    ThreeFSMetrics::markTotalPutCacheBeginUs(metrics);

    bool result = checkCacheIn3FS(cache_keys, true);
    if (!result) {
        result = rpcPutCacheForAllRank(cache_keys, block_indices, request_id);
        if (!result) {
            RTP_LLM_LOG_WARNING(
                "put cache for all rank failed, rpc put cache failed, request: %ld, cache key count: %zu",
                request_id,
                cache_keys.size());
        }
    }

    ThreeFSMetrics::markTotalPutCacheDoneUs(metrics);
    ThreeFSMetrics::setPutCacheFailedQps(metrics, result == false);
    return result;
}

bool ThreeFSCacheManager::rpcPutCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                                const std::vector<int32_t>& block_indices,
                                                int64_t                     request_id) const {
    const auto& grpc_workers = params_.worker_grpc_addrs_;
    if (grpc_workers.empty()) {
        RTP_LLM_LOG_WARNING("rpc put cache failed, grpc workers is empty, request: %ld", request_id);
        return false;
    }
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("rpc put cache failed, rpc pool is nullptr, request: %ld", request_id);
        return false;
    }

    const int                                worker_size = static_cast<int>(grpc_workers.size());
    std::vector<WorkerRpcContext>            worker_rpc_contexts(worker_size);
    std::vector<BroadcastPutCacheResponsePB> responses(worker_size);

    const uint32_t                       cq_size = worker_size % 2 == 0 ? worker_size / 2 : worker_size / 2 + 1;
    std::vector<::grpc::CompletionQueue> completion_queues(cq_size);
    std::vector<int>                     request_num_per_cq(cq_size, 0);

    for (int rank = 0; rank < worker_size; ++rank) {
        RTP_LLM_LOG_DEBUG("send put cache rpc request to rank: %d, addr: %s, request: %ld",
                          rank,
                          grpc_workers[rank].c_str(),
                          request_id);
        const auto& worker_addr    = grpc_workers[rank];
        auto        connect_status = rpc_pool_->getConnection(worker_addr);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING(
                "rpc put cache failed, get grpc connection failed for rank: %d, request: %ld, worker addr: %s",
                rank,
                request_id,
                worker_addr.c_str());
            return false;
        }

        BroadcastPutCacheRequestPB put_cache_request;
        put_cache_request.set_request_id(request_id);
        for (const auto cache_key : cache_keys) {
            put_cache_request.add_cache_keys(cache_key);
        }
        for (const auto block_index : block_indices) {
            put_cache_request.add_block_ids(block_index);
        }

        auto& rpc_context       = worker_rpc_contexts[rank];
        rpc_context.stub        = connect_status.value().stub;
        rpc_context.server_addr = worker_addr;
        const int queue_index   = rank % cq_size;

        std::unique_ptr<::grpc::ClientAsyncResponseReader<BroadcastPutCacheResponsePB>> reader(
            rpc_context.stub->AsyncRemotePutCache(
                rpc_context.client_context.get(), put_cache_request, &completion_queues[queue_index]));
        reader->Finish(&responses[rank], &rpc_context.status, reinterpret_cast<void*>(rank));
        ++request_num_per_cq[queue_index];
    }

    bool success = waitAllReuqestDone(worker_rpc_contexts, completion_queues, request_num_per_cq, request_id);
    if (!success) {
        for (int rank = 0; rank < worker_size; ++rank) {
            const auto kvcache_key = constructKvCacheKey(cache_keys.back(), rank);
            if (threefs_block_cache_) {
                threefs_block_cache_->removeCache(kvcache_key);
            }
        }
        return false;
    }

    RTP_LLM_LOG_DEBUG(
        "rpc put cache success, cache keys: [%lu|%s]", cache_keys.size(), vectorToString(cache_keys).c_str());
    return true;
}

bool ThreeFSCacheManager::waitAllReuqestDone(const std::vector<WorkerRpcContext>&  worker_rpc_contexts,
                                             std::vector<::grpc::CompletionQueue>& completion_queues,
                                             std::vector<int>&                     unfinished_count_per_queue,
                                             int64_t                               request_id) const {
    const int        worker_size = static_cast<int>(worker_rpc_contexts.size());
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
        for (uint32_t i = 0; i < completion_queues.size(); i++) {
            if (unfinished_count_per_queue[i] == 0) {
                continue;
            }

            void*      got_tag     = nullptr;
            bool       ok          = false;
            const auto next_status = completion_queues[i].AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_LOG_WARNING(
                    "request failed, grpc completion queue failed, status: %d, request: %ld", next_status, request_id);
                all_request_success = false;
                break;
            }
            --unfinished_count_per_queue[i];
            ++finished_count;

            const int   rank   = reinterpret_cast<intptr_t>(got_tag);
            const auto& status = worker_rpc_contexts[rank].status;
            // const auto& response         = worker_rpc_contexts[rank].response;
            if (!status.ok()) {
                RTP_LLM_LOG_WARNING("request failed, rank %d failed, error: %d(%s), addr: %s, request: %ld",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    worker_rpc_contexts[rank].server_addr.c_str(),
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

bool ThreeFSCacheManager::checkCacheIn3FS(const std::vector<int64_t>& cache_keys, bool for_put) const {
    if (!threefs_block_cache_) {
        RTP_LLM_LOG_WARNING("check cache failed, 3fs block cache is nullptr");
        return false;
    }

    const auto rank_num = params_.worker_grpc_addrs_.size();
    if (rank_num == 0) {
        RTP_LLM_LOG_WARNING("check cache failed, rank num is 0");
        return false;
    }

    std::vector<int64_t> all_file_size(rank_num);
    int64_t              file_age = -1;
    for (int rank = 0; rank < rank_num; ++rank) {
        const auto kvcache_key             = constructKvCacheKey(cache_keys.back(), rank);
        auto [cur_file_size, cur_file_age] = threefs_block_cache_->getFileSizeAndAge(kvcache_key);
        if (cur_file_size == -1) {
            return false;
        }
        all_file_size[rank] = cur_file_size;
        if (file_age == -1) {
            file_age = cur_file_age;
        }
    }

    if (for_put) {
        const int32_t file_age_threshold_sec = 10;
        if (file_age < file_age_threshold_sec) {
            return true;
        }
    }

    const auto file_size = all_file_size[0];
    bool       all_file_size_equal =
        std::all_of(all_file_size.begin(), all_file_size.end(), [file_size](int64_t val) { return val == file_size; });
    if (all_file_size_equal && file_size != 0) {
        return true;
    }
    RTP_LLM_LOG_DEBUG("check cache failed, file size not equal, all file size: [%zu|%s]",
                      all_file_size.size(),
                      vectorToString(all_file_size).c_str());
    return false;
}

std::string ThreeFSCacheManager::constructKvCacheKey(int64_t last_cache_key, int32_t rank) const {
    // 3fs use the following string as filename:
    // kv_<model_name>_<layer_num>_<local_head_num_kv>_<size_per_head>_<seq_size_per_block>_<dtype>_<last_cache_key>_<rank>
    if (rank == -1) {
        rank = static_cast<int32_t>(params_.tp_rank_);
    }
    std::ostringstream oss;
    oss << "kv_" << params_.model_name_ << "_" << cache_config_.layer_num << "_" << cache_config_.local_head_num_kv
        << "_" << cache_config_.size_per_head << "_" << cache_config_.seq_size_per_block << "_"
        << static_cast<int>(cache_config_.dtype) << "_" << last_cache_key << "_" << rank;
    return oss.str();
}

}  // namespace rtp_llm::threefs