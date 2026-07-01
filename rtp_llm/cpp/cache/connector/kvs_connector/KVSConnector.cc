#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSConnector.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/allocator/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::kvs {
namespace {

std::string sanitizeKeyPart(std::string value) {
    for (char& ch : value) {
        if (ch == '/' || ch == ' ' || ch == '\t' || ch == '\n') {
            ch = '_';
        }
    }
    return value;
}

std::string hashString(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex << hash;
    return oss.str();
}

template <typename T>
std::string joinValues(const std::vector<T>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

size_t totalBytes(const std::vector<BlockInfo>& iovs) {
    size_t bytes = 0;
    for (const auto& iov : iovs) {
        if (iov.size_bytes > std::numeric_limits<size_t>::max() - bytes) {
            return std::numeric_limits<size_t>::max();
        }
        bytes += iov.size_bytes;
    }
    return bytes;
}

bool isValidBlockInfo(const BlockInfo& block_info) {
    return block_info.addr != nullptr && block_info.size_bytes > 0;
}

}  // namespace

bool KVSConnectorState::done() const {
    auto cur = state();
    return cur == State::SUCCESS || cur == State::ERROR || cur == State::THREADPOOL_FULL || cur == State::CLIENT_ERROR;
}

bool KVSConnectorState::success() const {
    return state() == State::SUCCESS;
}

void KVSConnectorState::set(State state) {
    state_.store(state, std::memory_order_release);
}

KVSConnectorState::State KVSConnectorState::state() const {
    return state_.load(std::memory_order_acquire);
}

bool KVSAsyncMatchContext::done() const {
    return state_.done();
}

KVSAsyncMatchContext::~KVSAsyncMatchContext() {
    if (read_session_ && !read_session_->lease_id.empty() && client_) {
        client_->release(read_session_->lease_id);
    }
}

bool KVSAsyncMatchContext::success() const {
    return state_.success();
}

size_t KVSAsyncMatchContext::matchedBlockCount() const {
    return matched_block_count_;
}

bool KVSAsyncContext::done() const {
    return state_.done();
}

bool KVSAsyncContext::success() const {
    return state_.success();
}

KVSConnector::KVSConnector(const CacheConfig&                 cache_config,
                           const KVCacheConfig&               kv_cache_config,
                           const RuntimeConfig&               runtime_config,
                           const ParallelismConfig&           parallelism_config,
                           std::shared_ptr<KVCacheAllocator>  allocator,
                           const kmonitor::MetricsReporterPtr metrics_reporter,
                           std::shared_ptr<KVSClient>         client):
    state_(std::make_shared<KVSConnectorSharedState>()) {
    state_->cache_config       = cache_config;
    state_->kv_cache_config    = kv_cache_config;
    state_->runtime_config     = runtime_config;
    state_->parallelism_config = parallelism_config;
    state_->allocator          = std::move(allocator);
    state_->metrics_reporter   = metrics_reporter;
    state_->client             = client ? std::move(client) : std::make_shared<KVSClient>();
}

KVSConnector::~KVSConnector() {
    if (thread_pool_) {
        // Async tasks capture shared state instead of this, so queued/running tasks do
        // not depend on KVSConnector object lifetime during shutdown.
        thread_pool_->stop();
        thread_pool_->waitFinish();
        thread_pool_.reset();
    }
}

bool KVSConnector::init() {
    state_->deployment_id = buildDeploymentId(state_);
    KVSClientConfig config;
    config.v6d_url         = state_->kv_cache_config.kvs_v6d_url;
    config.v6d_socket_path = state_->kv_cache_config.kvs_v6d_socket_path;
    config.timeout_ms      = state_->kv_cache_config.kvs_timeout_ms;
    config.lease_term_sec  = state_->kv_cache_config.kvs_lease_term_sec;
    if (!state_->client->init(config)) {
        RTP_LLM_LOG_ERROR("KVSConnector init failed: KVSClient init failed");
        return false;
    }
    thread_pool_ = std::make_unique<autil::ThreadPool>(state_->kv_cache_config.reco_asyncwrapper_thread_num,
                                                       state_->kv_cache_config.reco_asyncwrapper_queue_size,
                                                       nullptr,
                                                       "KVSThreadPool",
                                                       /*stopIfHasException=*/true);
    if (!thread_pool_->start("")) {
        RTP_LLM_LOG_ERROR("KVSConnector init failed: start thread pool failed");
        return false;
    }
    RTP_LLM_LOG_INFO("KVSConnector initialized, deployment_id[%s]", state_->deployment_id.c_str());
    return true;
}

std::shared_ptr<AsyncMatchContext> KVSConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                            const std::shared_ptr<Meta>&            meta) {
    if (!meta || !meta->enableRemoteCache() || !resource || !thread_pool_) {
        return nullptr;
    }
    auto async_context = std::make_shared<KVSAsyncMatchContext>(resource->reuseBlockNum(), state_->client);
    auto ec            = thread_pool_->pushTask(
        [state = state_, resource, meta, async_context]() {
            async_context->state_.set(KVSConnectorState::State::START);
            asyncMatchTask(state, resource, meta, async_context);
        },
        false);
    if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
        async_context->state_.set(KVSConnectorState::State::THREADPOOL_FULL);
        return nullptr;
    }
    return async_context;
}

std::shared_ptr<AsyncContext> KVSConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                      const std::shared_ptr<Meta>&              meta,
                                                      const std::shared_ptr<AsyncMatchContext>& match_context,
                                                      int                                       start_read_block_index,
                                                      int                                       read_block_num) {
    (void)meta;
    if (!resource || read_block_num <= 0) {
        return nullptr;
    }
    auto kvs_match_context = std::dynamic_pointer_cast<KVSAsyncMatchContext>(match_context);
    if (!kvs_match_context) {
        return nullptr;
    }
    auto async_context = std::make_shared<KVSAsyncContext>();
    auto ec            = thread_pool_->pushTask(
        [state = state_, resource, start_read_block_index, read_block_num, async_context, kvs_match_context]() {
            async_context->state_.set(KVSConnectorState::State::START);
            asyncReadTask(state, resource, start_read_block_index, read_block_num, async_context, kvs_match_context);
        },
        false);
    if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
        async_context->state_.set(KVSConnectorState::State::THREADPOOL_FULL);
        return nullptr;
    }
    return async_context;
}

std::shared_ptr<AsyncContext> KVSConnector::asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                       const std::shared_ptr<Meta>&            meta) {
    if (!meta || !meta->enableRemoteCache() || !resource || !thread_pool_) {
        return nullptr;
    }
    auto async_context = std::make_shared<KVSAsyncContext>();
    auto ec            = thread_pool_->pushTask(
        [state = state_, resource, meta, async_context]() {
            async_context->state_.set(KVSConnectorState::State::START);
            asyncWriteTask(state, resource, meta, async_context);
        },
        false);
    if (ec != autil::ThreadPoolBase::ERROR_TYPE::ERROR_NONE) {
        async_context->state_.set(KVSConnectorState::State::THREADPOOL_FULL);
        return nullptr;
    }
    return async_context;
}

std::shared_ptr<AsyncContext>
KVSConnector::asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    (void)layer_context;
    RTP_LLM_LOG_DEBUG("KVSConnector asyncWriteByLayer is disabled in v1, layer_id[%d]", layer_id);
    return nullptr;
}

void KVSConnector::asyncMatchTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                                  const std::shared_ptr<KVCacheResource>&         resource,
                                  const std::shared_ptr<Meta>&                    meta,
                                  const std::shared_ptr<KVSAsyncMatchContext>&    async_context) {
    auto keys = resource->cacheKeys();
    // Match only sealed/full blocks. The current tail can still be growing, so it
    // is intentionally excluded from remote reuse candidates.
    if (!keys.empty()) {
        keys.pop_back();
    }
    const size_t prev_reuse = async_context->prev_reuse_blocks_num_;
    if (prev_reuse >= keys.size()) {
        async_context->matched_block_count_ = prev_reuse;
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        return;
    }

    auto                     block_objects = buildBlockObjects(state, resource, keys);
    std::vector<std::string> object_keys;
    for (size_t i = prev_reuse; i < block_objects.size(); ++i) {
        for (const auto& object : block_objects[i].objects) {
            object_keys.push_back(object.object_key);
        }
    }
    if (object_keys.empty()) {
        RTP_LLM_LOG_WARNING("KVSConnector match has no readable objects, prev_reuse[%zu], candidate_blocks[%zu]",
                            prev_reuse,
                            keys.size());
        async_context->matched_block_count_ = prev_reuse;
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        return;
    }
    std::sort(object_keys.begin(), object_keys.end());
    object_keys.erase(std::unique(object_keys.begin(), object_keys.end()), object_keys.end());

    RTP_LLM_LOG_INFO(
        "KVSConnector match start, trace_id[%s], prev_reuse[%zu], candidate_blocks[%zu], object_count[%zu]",
        meta->trace_id().c_str(),
        prev_reuse,
        block_objects.size(),
        object_keys.size());
    auto session = state->client->acquireForRead(object_keys, "kvs_match_" + meta->trace_id());
    if (!session) {
        RTP_LLM_LOG_WARNING("KVSConnector match failed, degrade to miss, trace_id[%s]", meta->trace_id().c_str());
        async_context->matched_block_count_ = prev_reuse;
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        return;
    }

    size_t                       matched = prev_reuse;
    std::vector<KVSBlockObjects> matched_blocks;
    for (size_t i = prev_reuse; i < block_objects.size(); ++i) {
        bool hit = true;
        for (const auto& object : block_objects[i].objects) {
            if (session->handles.find(object.object_key) == session->handles.end()) {
                hit = false;
                break;
            }
        }
        if (!hit) {
            break;
        }
        matched++;
        matched_blocks.push_back(block_objects[i]);
    }

    if (matched <= prev_reuse && !session->lease_id.empty()) {
        state->client->release(session->lease_id);
        session->lease_id.clear();
    }
    async_context->matched_block_count_ = matched;
    async_context->matched_blocks_      = std::move(matched_blocks);
    async_context->read_session_        = std::move(session);
    RTP_LLM_LOG_INFO("KVSConnector match done, trace_id[%s], matched_blocks[%zu], prev_reuse[%zu], lease[%s]",
                     meta->trace_id().c_str(),
                     matched,
                     prev_reuse,
                     async_context->read_session_ ? async_context->read_session_->lease_id.c_str() : "");
    async_context->state_.set(KVSConnectorState::State::SUCCESS);
}

void KVSConnector::asyncReadTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                                 const std::shared_ptr<KVCacheResource>&         resource,
                                 int                                             start_read_block_index,
                                 int                                             read_block_num,
                                 const std::shared_ptr<KVSAsyncContext>&         async_context,
                                 const std::shared_ptr<KVSAsyncMatchContext>&    match_context) {
    if (!match_context->read_session_) {
        async_context->state_.set(KVSConnectorState::State::ERROR);
        return;
    }
    std::vector<KVSObjectBuffer> dst_buffers;
    const int                    end_block_index = start_read_block_index + read_block_num;
    for (const auto& block : match_context->matched_blocks_) {
        if (static_cast<int>(block.block_index) < start_read_block_index
            || static_cast<int>(block.block_index) >= end_block_index) {
            continue;
        }
        for (const auto& object : block.objects) {
            dst_buffers.push_back(KVSObjectBuffer{object.object_key, object.iovs});
        }
    }
    if (dst_buffers.empty()) {
        RTP_LLM_LOG_WARNING("KVSConnector read has no destination buffers, start_block[%d], block_num[%d]",
                            start_read_block_index,
                            read_block_num);
        state->client->release(match_context->read_session_->lease_id);
        match_context->read_session_.reset();
        async_context->state_.set(KVSConnectorState::State::CLIENT_ERROR);
        return;
    }
    RTP_LLM_LOG_INFO("KVSConnector read start, start_block[%d], block_num[%d], object_count[%zu]",
                     start_read_block_index,
                     read_block_num,
                     dst_buffers.size());
    bool ok = state->client->load(*match_context->read_session_, dst_buffers);
    state->client->release(match_context->read_session_->lease_id);
    match_context->read_session_.reset();
    if (ok) {
        resource->setRemoteReuseBlockNum(start_read_block_index + read_block_num);
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        RTP_LLM_LOG_INFO("KVSConnector read done, remote_reuse_blocks[%d]", start_read_block_index + read_block_num);
    } else {
        RTP_LLM_LOG_WARNING("KVSConnector read failed, start_block[%d], block_num[%d], object_count[%zu]",
                            start_read_block_index,
                            read_block_num,
                            dst_buffers.size());
        async_context->state_.set(KVSConnectorState::State::CLIENT_ERROR);
    }
}

void KVSConnector::asyncWriteTask(const std::shared_ptr<KVSConnectorSharedState>& state,
                                  const std::shared_ptr<KVCacheResource>&         resource,
                                  const std::shared_ptr<Meta>&                    meta,
                                  const std::shared_ptr<KVSAsyncContext>&         async_context) {
    auto keys = resource->cacheKeys();
    if (!keys.empty() && !resource->lastBlockAligned()) {
        keys.pop_back();
    }
    if (keys.empty()) {
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        return;
    }
    auto                         block_objects = buildBlockObjects(state, resource, keys);
    std::vector<KVSObjectBuffer> src_buffers;
    for (const auto& block : block_objects) {
        for (const auto& object : block.objects) {
            if (!object.iovs.empty() && totalBytes(object.iovs) > 0) {
                src_buffers.push_back(KVSObjectBuffer{object.object_key, object.iovs});
            }
        }
    }
    RTP_LLM_LOG_INFO("KVSConnector write start, trace_id[%s], blocks[%zu], object_count[%zu]",
                     meta->trace_id().c_str(),
                     block_objects.size(),
                     src_buffers.size());
    if (state->client->store(src_buffers, "kvs_store_" + meta->trace_id())) {
        async_context->state_.set(KVSConnectorState::State::SUCCESS);
        RTP_LLM_LOG_INFO(
            "KVSConnector write done, trace_id[%s], object_count[%zu]", meta->trace_id().c_str(), src_buffers.size());
    } else {
        RTP_LLM_LOG_WARNING(
            "KVSConnector write failed, trace_id[%s], object_count[%zu]", meta->trace_id().c_str(), src_buffers.size());
        async_context->state_.set(KVSConnectorState::State::CLIENT_ERROR);
    }
}

std::vector<KVSBlockObjects> KVSConnector::buildBlockObjects(const std::shared_ptr<KVSConnectorSharedState>& state,
                                                             const std::shared_ptr<KVCacheResource>&         resource,
                                                             const std::vector<CacheKeyType>&                keys) {
    std::vector<KVSBlockObjects> result;
    result.reserve(keys.size());
    for (size_t block_index = 0; block_index < keys.size(); ++block_index) {
        KVSBlockObjects block_objects;
        block_objects.block_index = block_index;
        for (int group_id = 0; group_id < resource->groupNums(); ++group_id) {
            const auto& block_ids = resource->groupBlocks().at(group_id)->blocks();
            if (block_index >= block_ids.size() || isNullBlockIdx(block_ids[block_index])) {
                continue;
            }
            std::vector<BlockInfo> iovs;
            for (int layer_id : state->cache_config.layerIdsForGroup(static_cast<size_t>(group_id))) {
                auto block_infos = state->allocator->convertIndexToBuffer(layer_id, group_id, block_ids[block_index]);
                for (const auto& block_info : block_infos) {
                    if (!isValidBlockInfo(block_info)) {
                        iovs.clear();
                        break;
                    }
                    iovs.push_back(block_info);
                }
                if (iovs.empty()) {
                    break;
                }
            }
            if (!iovs.empty()) {
                block_objects.objects.push_back(KVSBlockObject{
                    block_index, group_id, buildObjectKey(state, keys[block_index], group_id), std::move(iovs)});
            }
        }
        result.push_back(std::move(block_objects));
    }
    return result;
}

std::string KVSConnector::buildObjectKey(const std::shared_ptr<KVSConnectorSharedState>& state,
                                         CacheKeyType                                    cache_key,
                                         int32_t                                         group_id) {
    std::ostringstream oss;
    oss << sanitizeKeyPart(state->kv_cache_config.kvs_object_namespace) << "/v"
        << sanitizeKeyPart(state->kv_cache_config.kvs_cache_key_version) << "/" << state->deployment_id << "/tp-"
        << state->parallelism_config.tp_rank << "/group-" << group_id << "/block-" << cache_key;
    return oss.str();
}

std::string KVSConnector::buildDeploymentId(const std::shared_ptr<KVSConnectorSharedState>& state) {
    // The deployment id is model-layout scoped. CHECKPOINT_PATH separates
    // checkpoint variants when available, while the cache layout fields below
    // protect same-name deployments with different block object sizes.
    const auto         checkpoint_path = autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string(""));
    std::ostringstream raw;
    raw << "model=" << state->runtime_config.model_name << ";ckpt=" << checkpoint_path
        << ";layers=" << state->cache_config.layer_num << "/" << state->cache_config.layer_all_num
        << ";dtype=" << getDataTypeStr(state->cache_config.dtype) << ";seq=" << state->cache_config.seq_size_per_block
        << ";kernel_seq=" << state->cache_config.kernel_seq_size_per_block
        << ";kv_block=" << state->cache_config.kv_block_size_bytes
        << ";kv_scale=" << state->cache_config.kv_scale_size_bytes
        << ";block=" << state->cache_config.block_size_bytes
        << ";kv_stride=" << state->cache_config.kv_block_stride_bytes
        << ";scale_stride=" << state->cache_config.kv_scale_stride_bytes
        << ";group_seq=" << joinValues(state->cache_config.group_seq_size_per_block)
        << ";tp=" << state->parallelism_config.tp_size << ";fp8=" << state->kv_cache_config.fp8_kv_cache
        << ";int8=" << state->kv_cache_config.int8_kv_cache << ";mla=" << state->cache_config.use_mla
        << ";groups=" << state->cache_config.groupNums() << ";typed=" << state->cache_config.use_typed_cache_regions;
    return sanitizeKeyPart(state->runtime_config.model_name.empty() ? "model" : state->runtime_config.model_name) + "-"
           + hashString(raw.str());
}

}  // namespace rtp_llm::kvs
