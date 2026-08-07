#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <torch/extension.h>

using namespace std;

namespace rtp_llm {

class StreamConnectorContext: public KVCacheConnectorReadWriteContext {
public:
    StreamConnectorContext(std::shared_ptr<BatchKVCacheResource> resource, std::shared_ptr<Meta> meta):
        resource_(std::move(resource)), meta_(std::move(meta)) {}

    const KVCacheResource& kvCacheResource() const override {
        return resource_->cacheResource(0);
    }
    const std::shared_ptr<Meta>& meta() const override {
        return meta_;
    }

private:
    std::shared_ptr<BatchKVCacheResource> resource_;
    std::shared_ptr<Meta>                 meta_;
};

class MetaImpl: public Meta {
public:
    MetaImpl(bool enable_memory_cache, bool enable_remote_cache, std::string trace_id):
        enable_memory_cache_(enable_memory_cache),
        enable_remote_cache_(enable_remote_cache),
        trace_id_(std::move(trace_id)) {}
    ~MetaImpl() override = default;

    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }
    GenerateStream* generateStream() const override {
        return generate_stream_;
    }
    void reportError(ErrorCode code, const std::string& message) override {
        if (generate_stream_) {
            generate_stream_->reportError(code, message);
        }
    }
    std::optional<P2PRoutingContext> p2pRouting() const override {
        return routing_ctx_;
    }
    void fillRoutingContext(GenerateStream* stream) {
        if (stream && !routing_ctx_.has_value()) {
            auto& routing           = routing_ctx_.emplace();
            routing.request_id      = stream->streamId();
            routing.unique_key      = stream->uniqueKey();
            routing.deadline_ms     = stream->deadlineMs();
            routing.prefill_addr    = stream->prefillAddr();
            routing.prefill_tp_size = stream->getPrefillTpSize();
            routing.prefill_cp_size = stream->getPrefillCpSize();
        }
    }

    GenerateStream*                  generate_stream_ = nullptr;
    std::optional<P2PRoutingContext> routing_ctx_;

private:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_;
    std::vector<int64_t> tokens_;
};

namespace {

torch::Tensor buildFirstTokenUpdateTokens(GenerateStream* stream, int32_t first_token_id) {
    const auto next_batch_size = static_cast<int64_t>(stream->nextBatchSize());
    if (!stream->hasNumBeams()) {
        auto tokens = torch::empty({next_batch_size, 1}, torch::kInt32);
        std::fill(tokens.data_ptr<int32_t>(), tokens.data_ptr<int32_t>() + next_batch_size, first_token_id);
        return tokens;
    }
    const auto seq_len         = static_cast<int64_t>(stream->seqLength());
    auto       tokens          = torch::zeros({next_batch_size, seq_len + 1}, torch::kInt32);
    const auto current_batches = std::max<int64_t>(1, static_cast<int64_t>(stream->currentBatchSize()));
    for (int64_t batch = 0; batch < next_batch_size; ++batch) {
        const auto source = static_cast<int>(std::min(batch, current_batches - 1));
        const auto prefix = stream->completeTokenIdsVec(source);
        auto*      row    = tokens.data_ptr<int32_t>() + batch * tokens.size(1);
        std::copy(prefix.begin(), prefix.end(), row);
        row[seq_len] = first_token_id;
    }
    return tokens;
}

const P2PSideChannelPayload* p2pSideChannel(const std::shared_ptr<AsyncContext>& context) {
    auto p2p = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(context);
    return p2p ? p2p->sideChannelPayload() : nullptr;
}

void applyP2PSideChannel(const P2PSideChannelPayload& payload, GenerateStream* stream) {
    if (payload.has_first_token) {
        stream->setIsContextStream(false);
        stream->step();
        stream->updateWithoutLock({.new_tokens =
                                       buildFirstTokenUpdateTokens(stream, static_cast<int32_t>(payload.first_token_id)),
                                   .num_new_tokens = 1,
                                   .hidden_states = {},
                                   .logits = {},
                                   .softmax_probs = {},
                                   .cum_log_probs = {},
                                   .all_probs = {},
                                   .loss = {},
                                   .src_batch_indices = {},
                                   .all_hidden_states = {},
                                   .update_remote_generate = false,
                                   .force_update_info = false});
    }

    if (payload.total_reuse_len > 0) {
        stream->setPrefillReuseLength(
            payload.total_reuse_len, payload.local_reuse_len, payload.remote_reuse_len, payload.memory_reuse_len);
    }

    const bool has_probs = payload.propose_probs.shape_size() > 0 || !payload.propose_probs.fp16_data().empty()
                           || !payload.propose_probs.bf16_data().empty()
                           || !payload.propose_probs.fp32_data().empty();
    const bool has_hidden = payload.propose_hidden.shape_size() > 0 || !payload.propose_hidden.fp16_data().empty()
                            || !payload.propose_hidden.bf16_data().empty()
                            || !payload.propose_hidden.fp32_data().empty();
    if (!payload.propose_tokens.empty() || has_probs || has_hidden) {
        if (payload.propose_tokens.size() < 2) {
            stream->setContainProposeToken(false);
            stream->setProposeToken({});
            stream->setSPOutputBuffer(nullptr);
            if (!stream->disableSpRun()) {
                stream->reportErrorWithoutLock(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                                                "P2P side channel has invalid speculative tokens");
            }
        } else {
            auto probs = has_probs ? TensorPbConvert::pbToTorch(payload.propose_probs) :
                                     torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32));
            auto hidden = has_hidden ? TensorPbConvert::pbToTorch(payload.propose_hidden) :
                                       torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16));
            stream->setReuseLength(stream->seqLength() - 1);
            stream->setSpEditRun(false);
            stream->setMtpTokenIndex(stream->seqLength() - 1);
            stream->setContainProposeToken(true);
            stream->setProposeToken(payload.propose_tokens);
            auto output    = std::make_shared<SpeculativeExecutorStreamOutput>();
            output->tokens = torch::zeros({1, static_cast<int64_t>(payload.propose_tokens.size())}, torch::kInt32);
            memcpy(output->tokens.data_ptr<int>(),
                   payload.propose_tokens.data(),
                   payload.propose_tokens.size() * sizeof(int));
            output->all_probs     = probs;
            output->hidden_states = hidden;
            output->tensors_holder.emplace_back(std::move(probs));
            output->tensors_holder.emplace_back(std::move(hidden));
            stream->setSPOutputBuffer(output);
        }
    }

    if (!payload.position_ids.empty()) {
        stream->setContextPositionIds(
            torch::tensor(payload.position_ids, torch::dtype(torch::kInt32).device(torch::kCPU)));
    }
}

std::shared_ptr<const CacheTopology> warmupCacheTopology() {
    static const auto topology = []() {
        constexpr auto kWarmupCacheTag = "__warmup__";
        auto           spec            = std::make_shared<MHAKVCacheSpec>();
        spec->tag                      = kWarmupCacheTag;

        GroupBase group;
        group.tag                       = kWarmupCacheTag;
        group.spec                      = std::move(spec);
        group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.layer_ids                 = {0};
        group.seq_size_per_block        = 1;
        group.kernel_seq_size_per_block = 1;

        return CacheTopology::create({std::move(group)}, {{0, {kWarmupCacheTag}}});
    }();
    return topology;
}

}  // namespace

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    const auto topology = resource_context_.cache_manager ?
                              resource_context_.cache_manager->cacheConfig().topologyPtr() :
                              warmupCacheTopology();
    batch_kv_cache_resource_->initGroups(topology);
    resource_released_ = false;
}

void StreamCacheResource::releaseResource() {
    RTP_LLM_PROFILE_FUNCTION();
    if (!resource_context_.cache_manager) {
        return;
    }
    // Check against double release
    if (resource_released_) {
        RTP_LLM_LOG_ERROR("=== DOUBLE RELEASE CACHE RESOURCE DETECTED ===");
        RTP_LLM_LOG_ERROR("  stream_ ptr:                   %p", static_cast<void*>(stream_));
        RTP_LLM_LOG_ERROR("  stream alive (magic check):    %s",
                          stream_->isStreamAlive() ? "YES" : "NO (stream already destroyed!)");
        if (stream_->isStreamAlive()) {
            RTP_LLM_LOG_ERROR("  stream id:                     %ld", stream_->streamId());
            RTP_LLM_LOG_ERROR("  stream state:                  %s",
                              StreamStateToString(stream_->generate_status_->status).c_str());
            RTP_LLM_LOG_ERROR("  stream hasError:                %d", stream_->hasError());
            RTP_LLM_LOG_ERROR("  stream hasNumBeams:            %d", stream_->hasNumBeams());
        }
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource_ use_count: %ld", batch_kv_cache_resource_.use_count());
        RTP_LLM_LOG_ERROR("  curBlocksNum:                  %d", curBlocksNum());
        RTP_LLM_LOG_ERROR("  need_release_resource:         %d", need_release_resource_);
        RTP_LLM_LOG_ERROR("  fake_inited:                   %d", fake_inited_);
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource:       %s", batch_kv_cache_resource_->debugString().c_str());
        RTP_LLM_LOG_ERROR("  thread id:                     %lu",
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));
        abort();
    }
    if (allocator_load_context_) {
        if (!resource_context_.cache_manager->abortPendingLoad(allocator_load_context_)) {
            RTP_LLM_LOG_DEBUG("allocator load was already committed or completed before release");
        }
        allocator_load_context_.reset();
    }
    if (p2p_load_context_) {
        resource_context_.cache_manager->cancelP2PLoad(p2p_load_context_);
        p2p_load_context_.reset();
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->hasError())) {
        return;
    }
    RTP_LLM_LOG_DEBUG("releaseResource: stream=%ld, curBlocksNum=%d, pd_kvcache_ref=%p",
                      stream_->streamId(),
                      curBlocksNum(),
                      pd_kvcache_ref_.get());
    tryReleaseKVBlock(curBlocksNum());
    batch_kv_cache_resource_->clearBlocks();
    resource_released_ = true;
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);

    if (fake_inited_) {
        int max_blocks_num = curBlocksNum();
        int batch_size     = batch_kv_cache_resource_->batchSize();
        batch_kv_cache_resource_->clearBlocks();
        batch_kv_cache_resource_->resetBatchSize(batch_size);
        fake_inited_ = false;
        return max_blocks_num;
    }

    // NOTE: Currently only support releasing all blocks
    // Partial release (shrink) is not supported yet
    int total_blocks = curBlocksNum();
    RTP_LLM_CHECK(nums == total_blocks);

    if (total_blocks > 0) {
        if (reuseCache() && !stream_->hasError() && stream_->getStatus() == StreamState::FINISHED) {
            const Tier target_tier = storeTarget();
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, storing cache, curBlocksNum=%d, target_tier=%s",
                              stream_->streamId(),
                              total_blocks,
                              tierName(target_tier));
            if (target_tier != Tier::NONE) {
                InsertInfo insert_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, target_tier};
                resource_context_.cache_manager->insertIntoCache(insert_info);
            }
        } else {
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, NOT storing cache, reuseCache=%d, hasError=%d, status=%s",
                              stream_->streamId(),
                              reuseCache(),
                              stream_->hasError(),
                              StreamStateToString(stream_->getStatus()).c_str());
        }

        FreeInfo free_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr()};
        free_info.request_id = stream_->streamId();

        resource_context_.cache_manager->free(free_info);
    }

    return total_blocks;
}

// TODO, 等待删除。
int StreamCacheResource::singleBatchNeedBlocks(int seq_len, int reserve_step) const {
    return resource_context_.cache_manager->singleBatchNeedBlocks(batch_kv_cache_resource_, seq_len, reserve_step);
}

int StreamCacheResource::estimatePeakNeedBlocks(
    int seq_len, int common_seq_len, int remaining_tokens, int reserve_step, int target_batch_size) const {
    return resource_context_.cache_manager->estimatePeakNeedBlocks(batch_kv_cache_resource_,
                                                                   seq_len,
                                                                   common_seq_len,
                                                                   remaining_tokens,
                                                                   reserve_step,
                                                                   reuseCache(),
                                                                   target_batch_size);
}

void StreamCacheResource::publishReuseLengths(int total, int host, int disk, int backend) {
    stream_->setReuseLength(total);
    stream_->setMtpTokenIndex(total);
    stream_->setInitialReuseLength(total);
    stream_->setLocalReuseLength(total - backend);
    stream_->setRemoteReuseLength(backend);
    stream_->setHostReuseLength(host);
    stream_->setDiskReuseLength(disk);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::Status StreamCacheResource::initKVBlock() {
    RTP_LLM_PROFILE_FUNCTION();
    // Decode side: first malloc should NOT use device cache, regardless of runtime config.
    // Follow-up allocations (incrKVBlock) will respect reuseCache() && enableDeviceCache().
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_kv_cache_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;

    const bool disable_first_malloc_reuse =
        resource_context_.cache_manager->cacheConfig().disable_decode_first_malloc_device_reuse;
    const bool is_decode_role  = (resource_context_.role_type == RoleType::DECODE);
    const bool is_first_malloc = (batch_kv_cache_resource_->curBlocksNum() == 0);

    if (disable_first_malloc_reuse && is_decode_role && is_first_malloc) {
        malloc_info.reuse_cache         = false;
        malloc_info.enable_cache_lookup = false;
    } else {
        malloc_info.reuse_cache         = reuseCache();
        malloc_info.enable_cache_lookup = enableCacheLookup();
    }
    malloc_info.enable_remove_skipped_blocks = false;

    MallocResult result = resource_context_.cache_manager->malloc(malloc_info);
    recordCacheReuseMallocResult(result);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    const bool load_pending = result.async_context != nullptr;
    publishReuseLengths(result.reuse_len,
                        load_pending ? 0 : result.host_reuse_len,
                        load_pending ? 0 : result.disk_reuse_len,
                        0);
    allocator_load_context_ = std::move(result.async_context);
    return absl::OkStatus();
}

void StreamCacheResource::clearCacheReuseState() {
    stream_->setReuseLength(0);
    stream_->setMtpTokenIndex(0);
    stream_->setInitialReuseLength(0);
    stream_->setHostReuseLength(0);
    stream_->setDiskReuseLength(0);
    stream_->setLocalReuseLength(0);
    stream_->setRemoteReuseLength(0);
}

void StreamCacheResource::recordCacheReuseMallocResult(const MallocResult& result) {
    cache_reuse_metrics_.block_aligned_input_length = result.block_aligned_input_length;
    cache_reuse_metrics_.match_latency_us           = result.match_cost_time_us;
    cache_reuse_metrics_.report_match_latency       = result.match_end_time_us > 0;
    malloc_begin_time_us_                           = result.malloc_begin_time_us;
    if (!result.load_attempted) {
        if (result.match_end_time_us > 0) {
            cache_reuse_metrics_.match_to_ready_latency_us =
                std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
            cache_reuse_metrics_.report_match_to_ready_latency = true;
        }
        return;
    }

    cache_reuse_metrics_.load_prepare_latency_us = result.load_prepare_latency_us;
    load_wait_begin_time_us_                     = currentTimeUs();
    if (!result.success) {
        clearCacheReuseState();
        cache_reuse_metrics_.load_success        = false;
        cache_reuse_metrics_.report_load_metrics = true;
        cache_reuse_metrics_.match_to_ready_latency_us =
            std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
}

absl::Status StreamCacheResource::finalizeAllocatorLoad() {
    const bool load_success = allocator_load_context_->success();
    const auto error        = allocator_load_context_->errorInfo();
    if (load_success) {
        const auto context            = std::static_pointer_cast<LoadAsyncContext>(allocator_load_context_);
        const size_t total   = context->matchedBlocks();
        const size_t local   = context->localMatchedBlocks();
        const size_t host    = context->matchedBlocks(Tier::HOST);
        const size_t disk    = context->matchedBlocks(Tier::DISK);
        const size_t backend = total - local;
        auto&        resource = batch_kv_cache_resource_->cacheResource(0);
        resource.setDeviceReuseBlockNum(local - host - disk);
        resource.setMemoryReuseBlockNum(host);
        resource.setDiskReuseBlockNum(disk);
        resource.setStorageBackendReuseBlockNum(backend);
        const int tokens = reuseBlockTokens();
        publishReuseLengths(total * tokens, host * tokens, disk * tokens, backend * tokens);
    } else if (resource_context_.role_type == RoleType::PREFILL) {
        stream_->setHostReuseLength(0);
        stream_->setDiskReuseLength(0);
    } else {
        clearCacheReuseState();
    }
    if (!cache_reuse_metrics_.report_load_metrics) {
        const int64_t load_end_time_us = currentTimeUs();
        cache_reuse_metrics_.load_success                 = load_success;
        cache_reuse_metrics_.report_load_metrics          = true;
        cache_reuse_metrics_.report_load_wait_latency     = true;
        cache_reuse_metrics_.load_wait_latency_us =
            std::max<int64_t>(load_end_time_us - load_wait_begin_time_us_, 0);
        cache_reuse_metrics_.match_to_ready_latency_us =
            std::max<int64_t>(load_end_time_us - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
    allocator_load_context_.reset();

    if (load_success || resource_context_.role_type == RoleType::PREFILL) {
        return absl::OkStatus();
    }
    const std::string error_text = error.ToString();
    return absl::InternalError(error_text.empty() ? "allocator load failed" : "allocator load failed: " + error_text);
}

void StreamCacheResource::reportCacheReuseMetrics() {
    if (stream_->metrics_reporter_ == nullptr || !reuseCache()) {
        return;
    }
    const int64_t input_length       = stream_->inputLength();
    const int64_t total_reuse_length = stream_->initialReuseLength();
    const auto    reuse_rate         = [input_length](int64_t reuse_length) {
        return input_length > 0 ? static_cast<float>(reuse_length * 100.0 / input_length) : 0.0f;
    };
    cache_reuse_metrics_.kv_cache_reuse_length = total_reuse_length;
    cache_reuse_metrics_.device_reuse_length   = stream_->deviceReuseLength();
    cache_reuse_metrics_.host_reuse_length     = stream_->hostReuseLength();
    cache_reuse_metrics_.disk_reuse_length     = stream_->diskReuseLength();
    cache_reuse_metrics_.remote_reuse_length   = stream_->remoteReuseLength();
    cache_reuse_metrics_.kv_cache_hit_rate     = reuse_rate(total_reuse_length);
    cache_reuse_metrics_.device_hit_rate       = reuse_rate(cache_reuse_metrics_.device_reuse_length);
    cache_reuse_metrics_.host_hit_rate         = reuse_rate(cache_reuse_metrics_.host_reuse_length);
    cache_reuse_metrics_.disk_hit_rate         = reuse_rate(cache_reuse_metrics_.disk_reuse_length);
    cache_reuse_metrics_.report_reuse_metrics  = true;
    kmonitor::MetricsTags tags;
    stream_->metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(
        &tags, &cache_reuse_metrics_);
}

absl::Status StreamCacheResource::waitForAllocatorLoad() {
    if (!allocator_load_context_) {
        return absl::OkStatus();
    }
    allocator_load_context_->waitDone();
    if (!allocator_load_context_->done()) {
        return absl::InternalError("allocator load context is non-terminal after waitDone");
    }
    return finalizeAllocatorLoad();
}

absl::Status StreamCacheResource::incrKVBlock() {
    RTP_LLM_PROFILE_FUNCTION();
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource      = batch_kv_cache_resource_;
    malloc_info.complete_token_ids           = stream_->completeTokenIdsPtr();
    malloc_info.request_id                   = stream_->streamId();
    malloc_info.verbose                      = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.reuse_cache                  = reuseCache();
    malloc_info.enable_cache_lookup          = enableCacheLookup();
    malloc_info.enable_remove_skipped_blocks = true;

    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    if (result.reuse_len > 0) {
        publishReuseLengths(result.reuse_len, result.host_reuse_len, result.disk_reuse_len, 0);
    }
    if (result.async_context) {
        const bool aborted = resource_context_.cache_manager->abortPendingLoad(result.async_context);
        if (!aborted) {
            RTP_LLM_LOG_DEBUG("incremental allocator load was already committed before rejection");
        }
        return absl::FailedPreconditionError("async incremental KV block allocation is unsupported");
    }

    return absl::OkStatus();
}

bool StreamCacheResource::asyncLoadCache() {
    RTP_LLM_PROFILE_FUNCTION();
    if (!p2p_load_context_ && resource_context_.cache_manager && resource_context_.cache_manager->hasP2PConnector()) {
        auto meta = std::make_shared<MetaImpl>(
            reuseCache() && enableMemoryCache(), reuseCache() && enableRemoteCache(), stream_->traceId());
        meta->generate_stream_ = stream_;
        meta->fillRoutingContext(stream_);
        auto context = std::make_shared<StreamConnectorContext>(batch_kv_cache_resource_, std::move(meta));
        p2p_load_context_ = resource_context_.cache_manager->asyncLoadCache(context);
    }
    return allocator_load_context_ != nullptr || p2p_load_context_ != nullptr;
}

bool StreamCacheResource::loadCacheDone() {
    if (allocator_load_context_) {
        if (!allocator_load_context_->done()) {
            return false;
        }
        const ErrorInfo error   = allocator_load_context_->errorInfo();
        const bool      success = allocator_load_context_->success();
        const auto      status  = finalizeAllocatorLoad();
        if (!success) {
            RTP_LLM_LOG_WARNING(
                "block tree load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
        }
        if (!status.ok()) {
            stream_->reportEventWithoutLock(
                StreamEvents::Error, ErrorCode::LOAD_CACHE_TIMEOUT, "block tree cache load failed");
        }
    }

    if (p2p_load_context_) {
        if (!p2p_load_context_->done()) {
            return false;
        }
        if (p2p_load_context_->success()) {
            if (const auto* payload = p2pSideChannel(p2p_load_context_)) {
                applyP2PSideChannel(*payload, stream_);
            }
            p2p_load_context_.reset();
        } else {
            // P2P always represents a mandatory request lifecycle once a
            // context exists. This includes Tree-full-hit/no-transfer, where
            // StartLoad still returns side-channel state and releases Prefill
            // resources, and Prefill registration itself. None of these
            // failures is a benign cache miss.
            const auto error = p2p_load_context_->errorInfo();
            p2p_load_context_.reset();
            RTP_LLM_LOG_WARNING(
                "P2P cache load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
            stream_->reportEventWithoutLock(StreamEvents::Error, error.code(), error.ToString());
        }
    }
    return true;
}

// TODO, delete it soon
int StreamCacheResource::curBlocksNum() const {
    return batch_kv_cache_resource_->curBlocksNum();
}

bool StreamCacheResource::isContextStream() const {
    RTP_LLM_CHECK_WITH_INFO(stream_ != nullptr, "StreamCacheResource::isContextStream called with null stream");
    return stream_->isContextStream();
}

const BatchKVCacheResource& StreamCacheResource::kvCache() const {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

BatchKVCacheResource& StreamCacheResource::kvCacheMutable() {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    *batch_kv_cache_resource_ = kv_cache_resource;
}

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    return resource_context_.cache_manager->updateKVBlock(
        batch_kv_cache_resource_, block_src_batch, copy_last_block, block_update_mapping_);
}

bool StreamCacheResource::hasCacheKeys() const {
    return batch_kv_cache_resource_->hasCacheKeys();
}

const CacheKeysType& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    return batch_kv_cache_resource_->cacheKeys(batch_id);
}

void StreamCacheResource::fakeInitKVBlock(size_t reserved_blocks) {
    fake_inited_ = true;
    batch_kv_cache_resource_->resetBatchSize(stream_->maxBatchSize());
    const auto topology = resource_context_.cache_manager ?
                              resource_context_.cache_manager->cacheConfig().topologyPtr() :
                              warmupCacheTopology();
    batch_kv_cache_resource_->initGroups(topology);

    reserved_blocks = std::max(1ul, reserved_blocks);
    batch_kv_cache_resource_->resizeBlocks(reserved_blocks, 0);
}

int StreamCacheResource::mallocFailedTimes() const {
    return malloc_failed_times_;
}

bool StreamCacheResource::reuseCache() const {
    return resource_context_.reuse_cache && stream_->reuseCache();
}

bool StreamCacheResource::enableHostCache() const {
    return resource_context_.enable_host_cache && stream_->enableHostCache();
}

bool StreamCacheResource::enableDeviceCache() const {
    return resource_context_.enable_device_cache && stream_->enableDeviceCache();
}

bool StreamCacheResource::enableDiskCache() const {
    return resource_context_.enable_disk_cache && stream_->enableDiskCache();
}

bool StreamCacheResource::enableCacheLookup() const {
    const bool any_global_tier = resource_context_.enable_device_cache || resource_context_.enable_host_cache
                                 || resource_context_.enable_disk_cache || resource_context_.enable_remote_cache;
    return reuseCache() && any_global_tier;
}

Tier StreamCacheResource::storeTarget() const {
    if (!reuseCache()) {
        return Tier::NONE;
    }
    if (enableDeviceCache()) {
        return Tier::DEVICE;
    }
    if (enableHostCache()) {
        return Tier::HOST;
    }
    if (enableDiskCache()) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

void StreamCacheResource::swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs) {
    if (rhs == lhs) {
        return;
    }

    auto type_list = resource_context_.cache_manager->cacheConfig().groupTypesSnapshot();

    for (size_t i = 0; i < type_list.size(); i++) {
        if (type_list[i] == CacheGroupType::LINEAR) {
            batch_kv_cache_resource_->swapBlocks(batch_id, i, rhs, lhs);
        }
    }
}

void StreamCacheResource::holdKVCacheForPDSep() {
    auto&       resource   = batch_kv_cache_resource_->cacheResource(0);
    const auto& cache_keys = resource.cacheKeys();
    auto        ref = resource_context_.cache_manager->incrKVCacheRef(resource, cache_keys, /*is_connector=*/true);
    if (ref) {
        pd_kvcache_ref_ = std::move(ref);
    }
}

void StreamCacheResource::releaseKVCacheForPDSep() {
    pd_kvcache_ref_.reset();
}
}  // namespace rtp_llm
