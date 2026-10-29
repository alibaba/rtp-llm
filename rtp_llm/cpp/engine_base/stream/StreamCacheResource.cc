#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <algorithm>
#include <limits>
#include <thread>

using namespace std;

namespace rtp_llm {

namespace {

std::shared_ptr<const CacheTopology> warmupCacheTopology() {
    static const auto topology = []() {
        constexpr auto kWarmupCacheTag  = "__warmup__";
        auto           spec             = std::make_shared<MHAKVCacheSpec>();
        spec->tag                       = kWarmupCacheTag;
        spec->seq_size_per_block        = 1;
        spec->kernel_seq_size_per_block = 1;

        GroupBase group;
        group.tag    = kWarmupCacheTag;
        group.spec   = std::move(spec);
        group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
        return CacheTopology::create({std::move(group)}, {{0, {kWarmupCacheTag}}});
    }();
    return topology;
}

}  // namespace

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    // cache manager is null when warmup
    const auto topology = resource_context_.cache_manager ?
                              resource_context_.cache_manager->cacheConfig().topologyPtr() :
                              warmupCacheTopology();
    batch_kv_cache_resource_->initGroups(topology);
    resource_released_ = false;
}

void StreamCacheResource::releaseResource() {
    decode_reuse_use_state_.adopt({}, 0);
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
            RTP_LLM_LOG_ERROR("  stream id:                     %s", stream_->streamLogTag().c_str());
            RTP_LLM_LOG_ERROR("  stream state:                  %s",
                              StreamStateToString(stream_->generate_status_->status).c_str());
            RTP_LLM_LOG_ERROR("  stream hasError:                %d", stream_->hasErrorWithoutLock());
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
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->hasErrorWithoutLock())) {
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
    RTP_LLM_LOG_DEBUG("stream [%s] try release [%lu] blocks", stream_->streamLogTag().c_str(), nums);

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
        if (reuseCache() && !stream_->hasErrorWithoutLock() && stream_->getStatus() == StreamState::FINISHED) {
            const Tier target_tier = storeTarget();
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, storing cache, curBlocksNum=%d, target_tier=%s",
                              stream_->streamId(),
                              total_blocks,
                              tierName(target_tier));
            if (remoteCacheDecodePublicationEnabled() && resource_context_.role_type == RoleType::DECODE
                && !autil::EnvUtil::getEnv("RTP_LLM_STREAM_ASYNC", false)
                && stream_->completeTokenIdsPtr()->seqLength() > 0) {
                (void)resource_context_.cache_manager->captureCompletedDecode(
                    *batch_kv_cache_resource_,
                    static_cast<size_t>(stream_->completeTokenIdsPtr()->seqLength() - 1),
                    true);
            }
            if (target_tier != Tier::NONE) {
                InsertInfo insert_info{
                    batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, target_tier, stream_->streamId()};
                size_t     resident_prefix_length = 0;
                resource_context_.cache_manager->insertIntoCache(insert_info, resident_prefix_length);
            }
        } else {
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, NOT storing cache, reuseCache=%d, hasError=%d, status=%s",
                              stream_->streamId(),
                              reuseCache(),
                              stream_->hasErrorWithoutLock(),
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
    // Follow-up allocations (incrKVBlock) use enableCacheLookup().
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
    malloc_info.enable_storage_backend       = !skip_remote_cache_;

    // The allocator performs lookup and page preparation before backend.read.
    // Keep an attempt-local steady timestamp; report only after a real remote
    // read succeeds and the resulting GPU pages are adopted below.
    remote_restore_attempt_started_at_.reset();
    if (resource_context_.role_type == RoleType::PREFILL && remoteCacheDecodePublicationEnabled()
        && malloc_info.enable_storage_backend && malloc_info.enable_cache_lookup) {
        remote_restore_attempt_started_at_ = RemoteRestoreTiming::Clock::now();
    }
    MallocResult result = resource_context_.cache_manager->malloc(malloc_info);
    recordCacheReuseMallocResult(result);
    if (!result.success) {
        malloc_failed_times_++;
        switch (result.status) {
            case MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED:
                return absl::UnavailableError("kv cache is temporarily unavailable");
            case MallocStatus::PERMANENT_RESOURCE_EXHAUSTED:
                return absl::ResourceExhaustedError("request exceeds usable kv cache capacity");
            case MallocStatus::INTERNAL_ERROR:
                return absl::InternalError("malloc failed");
            case MallocStatus::NONE:
                RTP_LLM_LOG_ERROR("malloc returned failure without an error status, request_id=%ld",
                                  malloc_info.request_id);
                return absl::InternalError("malloc failed without an error status");
        }
        RTP_LLM_LOG_ERROR("malloc returned failure with unknown status=%d, request_id=%ld",
                          static_cast<int>(result.status),
                          malloc_info.request_id);
        return absl::InternalError("malloc failed with unknown status");
    }

    // HOST/DISK reuse is published after the asynchronous load completes.
    publishReuseLengths(result.reuse_len, 0, 0, 0);
    adoptDecodeReuse(std::move(result.decode_sources), result.reuse_len);
    allocator_load_context_ = std::move(result.async_context);
    return absl::OkStatus();
}

void StreamCacheResource::clearCacheReuseState() {
    decode_reuse_use_state_.adopt({}, 0);  // Never reset a consumed logical request on retry.
    remote_restore_lifecycle_.reset();
    remote_restore_attempt_started_at_.reset();
    stream_->setReuseLength(0);
    stream_->setMtpTokenIndex(0);
    stream_->setInitialReuseLength(0);
    stream_->setHostReuseLength(0);
    stream_->setDiskReuseLength(0);
    stream_->setLocalReuseLength(0);
    stream_->setRemoteReuseLength(0);
}

void StreamCacheResource::adoptDecodeReuse(DecodeReuseSourcesPtr sources, int64_t adopted_end) {
    if (resource_context_.role_type == RoleType::PREFILL && stream_->metrics_reporter_
        && remoteCacheDecodePublicationEnabled())
        decode_reuse_use_state_.adopt(std::move(sources), adopted_end);
}

void StreamCacheResource::reportDecodeCacheUse(std::optional<RemoteRestoreTiming::Clock::time_point> gpu_ready_at) {
    if (resource_context_.role_type != RoleType::PREFILL || !stream_->metrics_reporter_ || !gpu_ready_at)
        return;
    try {
        const auto counts = decode_reuse_use_state_.consume(true);
        if (!counts)
            return;
        for (size_t tier = 0; tier < counts->by_tier.size(); ++tier) {
            if (counts->by_tier[tier] <= 0)
                continue;
            RtpLLMDecodeCacheTokenMetricsCollector collector;
            collector.stage          = "used";
            collector.tier           = tierName(static_cast<Tier>(tier));
            collector.logical_tokens = counts->by_tier[tier];
            stream_->metrics_reporter_->report<RtpLLMDecodeCacheTokenMetrics>(nullptr, &collector);
        }
        RtpLLMRemoteCacheOutcomeMetricsCollector outcome;
        outcome.stage   = "use";
        outcome.outcome = counts->total() > 0 ? "used" : "no_decode_interval";
        stream_->metrics_reporter_->report<RtpLLMRemoteCacheOutcomeMetrics>(nullptr, &outcome);
    } catch (const std::bad_alloc&) {
        decode_reuse_use_state_.adopt({}, 0);
        RTP_LLM_LOG_WARNING("optional Decode cache use attribution unavailable");
    }
}

void StreamCacheResource::recordCacheReuseMallocResult(const MallocResult& result) {
    decode_reuse_use_state_.adopt({}, 0);
    remote_restore_lifecycle_.reset();
    cache_reuse_metrics_.device_input_length    = result.device_input_length;
    cache_reuse_metrics_.match_cost_time_us     = result.match_cost_time_us;
    cache_reuse_metrics_.report_match_cost_time = result.match_end_time_us > 0;
    malloc_begin_time_us_                       = result.malloc_begin_time_us;
    cache_reuse_metrics_.load_attempted         = result.load_attempted;
    if (!result.load_attempted) {
        if (result.match_end_time_us > 0) {
            cache_reuse_metrics_.match_to_ready_latency_us =
                std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
            cache_reuse_metrics_.report_match_to_ready_latency = true;
        }
        return;
    }

    // Metrics describe the current load attempt. A retry must not inherit the
    // previous attempt's terminal result or latency.
    cache_reuse_metrics_.load_success                  = false;
    cache_reuse_metrics_.report_load_metrics           = false;
    cache_reuse_metrics_.report_load_wait_latency      = false;
    cache_reuse_metrics_.load_wait_latency_us          = 0;
    cache_reuse_metrics_.report_match_to_ready_latency = false;
    cache_reuse_metrics_.match_to_ready_latency_us     = 0;
    cache_reuse_metrics_.load_prepare_latency_us       = result.load_prepare_latency_us;
    load_wait_begin_time_us_                           = currentTimeUs();
    if (!result.success) {
        clearCacheReuseState();
        cache_reuse_metrics_.load_success              = false;
        cache_reuse_metrics_.report_load_metrics       = true;
        cache_reuse_metrics_.match_to_ready_latency_us = std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
}

void StreamCacheResource::publishAllocatorReuse(const LoadAsyncContext& context, bool local_only) {
    const size_t total = local_only ? context.localMatchedBlocks() : context.matchedBlocks();
    const size_t local = context.localMatchedBlocks();
    const size_t host  = context.matchedBlocks(Tier::HOST);
    const size_t disk  = context.matchedBlocks(Tier::DISK);
    RTP_LLM_CHECK_WITH_INFO(total >= local && host <= local && disk <= local - host,
                            "invalid allocator reuse counts: total=%zu local=%zu host=%zu disk=%zu",
                            total,
                            local,
                            host,
                            disk);
    const int tokens = reuseBlockTokens();
    RTP_LLM_CHECK_WITH_INFO(tokens > 0 && total <= static_cast<size_t>(std::numeric_limits<int>::max()) / tokens,
                            "allocator reuse token count exceeds int range: blocks=%zu tokens=%d",
                            total,
                            tokens);
    const size_t backend  = total - local;
    auto&        resource = batch_kv_cache_resource_->cacheResource(0);
    resource.setDeviceReuseBlockNum(local - host - disk);
    resource.setMemoryReuseBlockNum(host);
    resource.setDiskReuseBlockNum(disk);
    resource.setStorageBackendReuseBlockNum(backend);
    publishReuseLengths(total * tokens, host * tokens, disk * tokens, backend * tokens);
}

absl::Status StreamCacheResource::finalizeAllocatorLoad() {
    const bool load_success  = allocator_load_context_->success();
    const auto error         = allocator_load_context_->errorInfo();
    const auto load_context  = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
    auto       malloc_status = load_context == nullptr ? MallocStatus::NONE : load_context->mallocStatus();
    if (load_success) {
        RTP_LLM_CHECK(load_context != nullptr);
        publishAllocatorReuse(*load_context, false);
        try {
            adoptDecodeReuse(load_context->decodeSources(), stream_->reuseLength());
        } catch (const std::bad_alloc&) {
            decode_reuse_use_state_.adopt({}, 0);
        }
        if (resource_context_.role_type == RoleType::PREFILL && stream_->metrics_reporter_
            && stream_->remoteReuseLength() > 0) {
            const auto timing = load_context->remoteRestoreTiming();
            if (timing && remote_restore_attempt_started_at_
                && remote_restore_lifecycle_.adopt(*timing, remote_restore_attempt_started_at_)) {
                reportRemoteRestoreLatency("checkpoint_gpu_ready", *remote_restore_lifecycle_.checkpointReadyUs());
                reportRemoteRestoreLatency("read_to_checkpoint_gpu_ready",
                                           *remote_restore_lifecycle_.readToCheckpointUs());
            }
        }
    } else if (resource_context_.role_type == RoleType::PREFILL && malloc_status == MallocStatus::NONE) {
        const bool backend_timeout = load_context && load_context->backendTimedOut();
        skip_remote_cache_         = skip_remote_cache_ || backend_timeout;
        const bool reuse_local     = load_context && load_context->localLoadSucceeded();
        const int  reuse_length    = reuse_local ?
                                         static_cast<int>(load_context->localMatchedBlocks()) * reuseBlockTokens() :
                                         stream_->reuseLength();
        if (!resource_context_.cache_manager->detachSharedBlocksForRecompute(batch_kv_cache_resource_, reuse_length)) {
            // Re-enter admission with no request refs if detaching the writable
            // suffix cannot fit. Never compute into cache/other-request blocks.
            resource_context_.cache_manager->free({batch_kv_cache_resource_, stream_->completeTokenIdsPtr()});
            clearCacheReuseState();
            malloc_status = MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
        } else if (reuse_local) {
            publishAllocatorReuse(*load_context, true);
            try {
                adoptDecodeReuse(load_context->decodeSources(true), stream_->reuseLength());
            } catch (const std::bad_alloc&) {
                decode_reuse_use_state_.adopt({}, 0);
            }
        } else {
            stream_->setHostReuseLength(0);
            stream_->setDiskReuseLength(0);
        }
    } else {
        clearCacheReuseState();
    }
    if (!cache_reuse_metrics_.report_load_metrics) {
        const int64_t load_end_time_us                = currentTimeUs();
        cache_reuse_metrics_.load_success             = load_success;
        cache_reuse_metrics_.report_load_metrics      = true;
        cache_reuse_metrics_.report_load_wait_latency = true;
        cache_reuse_metrics_.load_wait_latency_us = std::max<int64_t>(load_end_time_us - load_wait_begin_time_us_, 0);
        cache_reuse_metrics_.match_to_ready_latency_us = std::max<int64_t>(load_end_time_us - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
    allocator_load_context_.reset();

    if (load_success) {
        return absl::OkStatus();
    }
    if (malloc_status == MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED) {
        return absl::UnavailableError("allocator load materialization is temporarily out of KV blocks");
    }
    if (malloc_status == MallocStatus::PERMANENT_RESOURCE_EXHAUSTED) {
        return absl::ResourceExhaustedError("allocator load materialization exceeds KV cache capacity");
    }
    if (malloc_status == MallocStatus::INTERNAL_ERROR) {
        return absl::InternalError("allocator load materialization failed");
    }
    if (resource_context_.role_type == RoleType::PREFILL) {
        return absl::OkStatus();
    }
    const std::string error_text = error.ToString();
    if (error.code() == ErrorCode::DEADLINE_EXCEEDED) {
        return absl::DeadlineExceededError(error_text.empty() ? "allocator load timed out" : error_text);
    }
    return absl::InternalError(error_text.empty() ? "allocator load failed" : "allocator load failed: " + error_text);
}

void StreamCacheResource::reportCacheReuseMetrics() {
    if (stream_->metrics_reporter_ == nullptr || !reuseCache()) {
        return;
    }
    cache_reuse_metrics_.kv_cache_reuse_length            = stream_->initialReuseLength();
    cache_reuse_metrics_.stream_cache_device_reuse_length = stream_->deviceReuseLength();
    cache_reuse_metrics_.stream_cache_memory_reuse_length = stream_->hostReuseLength();
    cache_reuse_metrics_.stream_cache_disk_reuse_length   = stream_->diskReuseLength();
    cache_reuse_metrics_.stream_cache_remote_reuse_length = stream_->remoteReuseLength();
    cache_reuse_metrics_.report_reuse_metrics             = true;
    kmonitor::MetricsTags tags;
    stream_->metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(
        &tags, &cache_reuse_metrics_);
}

void StreamCacheResource::reportRemoteContextReady(std::optional<RemoteRestoreTiming::Clock::time_point> gpu_ready_at) {
    const auto latency =
        remote_restore_lifecycle_.completeContextForward(gpu_ready_at.value_or(RemoteRestoreTiming::Clock::now()));
    if (!latency || !stream_->metrics_reporter_) {
        return;
    }
    // The existing context forward can fuse historical tail with new prompt or
    // tool tokens. No source/tail identity is available here yet, so do not call
    // this tail-only restore, actual D-token use or TTFT. The GPU-ready mode
    // requires the explicit producer collective + existing output-stream fence;
    // arbitrary CPU updates remain observations without that qualification.
    reportRemoteRestoreLatency(gpu_ready_at ? "fused_context_gpu_ready" : "root_context_update_observed", *latency);
}

void StreamCacheResource::reportRemoteRestoreLatency(const char* mode, int64_t latency_us) {
    if (!stream_->metrics_reporter_)
        return;
    try {
        RtpLLMRemoteRestoreLatencyMetricsCollector collector;
        collector.mode       = mode;
        collector.latency_us = latency_us;
        stream_->metrics_reporter_
            ->report<RtpLLMRemoteRestoreLatencyMetrics, RtpLLMRemoteRestoreLatencyMetricsCollector>(nullptr,
                                                                                                    &collector);
    } catch (...) {
        RTP_LLM_LOG_WARNING("remote restore telemetry failed; inference result is retained");
    }
}

std::optional<absl::Status> StreamCacheResource::pollAllocatorLoad() {
    if (!allocator_load_context_) {
        return absl::OkStatus();
    }
    if (!allocator_load_context_->done()) {
        return std::nullopt;
    }
    return finalizeAllocatorLoad();
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

absl::Status StreamCacheResource::incrKVBlock(int seq_len_override) {
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
    malloc_info.incr_seq_len_override        = seq_len_override;

    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    if (result.reuse_len > 0) {
        publishReuseLengths(result.reuse_len, 0, 0, 0);
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
    return allocator_load_context_ != nullptr;
}

bool StreamCacheResource::loadCacheDone() {
    if (allocator_load_context_) {
        if (!allocator_load_context_->done()) {
            return false;
        }
        const ErrorInfo error   = allocator_load_context_->errorInfo();
        const bool      success = allocator_load_context_->success();
        const auto      status  = finalizeAllocatorLoad();
        if (!success && !absl::IsUnavailable(status)) {
            RTP_LLM_LOG_WARNING(
                "block tree load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
        }
        if (absl::IsUnavailable(status)) {
            ++malloc_failed_times_;
            stream_->generate_status_->clearLoadInitiated();
            reportMallocRetry();
        } else if (!status.ok()) {
            stream_->reportEventWithoutLock(StreamEvents::Error,
                                            absl::IsDeadlineExceeded(status) ? ErrorCode::LOAD_CACHE_TIMEOUT :
                                                                               ErrorCode::MALLOC_FAILED,
                                            std::string(status.message()));
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

void StreamCacheResource::reportMallocRetry() const {
    if (!stream_->metrics_reporter_) {
        return;
    }
    RtpLLMCacheOperationMetricsCollector collector;
    collector.operation_type = RtpLLMCacheOperationMetricsCollector::OpType::MALLOC_RETRY;
    stream_->metrics_reporter_->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(nullptr,
                                                                                                          &collector);
}

bool StreamCacheResource::reuseCache() const {
    // AND logic: global REUSE_CACHE=1 AND per-request reuse_cache both must be true.
    // Per-request field flows frontend → FlexLB → engine via protobuf.
    return !stream_->generateConfig()->isPrefillOnly() && resource_context_.reuse_cache
           && (resource_context_.ignore_request_cache_switches || stream_->reuseCache());
}

bool StreamCacheResource::enableMemoryCache() const {
    // Local tiers are deployment policy; request fields remain wire-compatible only.
    return resource_context_.enable_memory_cache;
}

bool StreamCacheResource::enableDeviceCache() const {
    return resource_context_.enable_device_cache;
}

bool StreamCacheResource::enableDiskCache() const {
    return resource_context_.enable_disk_cache;
}

bool StreamCacheResource::enableCacheLookup() const {
    const bool any_global_tier = resource_context_.enable_device_cache || resource_context_.enable_memory_cache
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
    if (enableMemoryCache()) {
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

    for (const auto& group : resource_context_.cache_manager->cacheConfig().topology().groups()) {
        if (group.policy.group_type == CacheGroupType::LINEAR) {
            batch_kv_cache_resource_->swapBlocks(batch_id, group.tag, rhs, lhs);
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
