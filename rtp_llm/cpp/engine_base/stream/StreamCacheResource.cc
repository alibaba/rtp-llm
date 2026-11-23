#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/LinearPrefillTrace.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <algorithm>
#include <limits>
#include <thread>
#include <unordered_set>

using namespace std;

namespace rtp_llm {

namespace {

bool hasLinearCheckpointHole(const BatchKVCacheResource& resource, const CacheConfig& config, int prefix, int length) {
    for (const auto& group : config.groups()) {
        const auto& policy = group.policy;
        if (policy.group_type != CacheGroupType::LINEAR || !policy.enable_prefix_reuse || policy.shared_chunk_state) {
            continue;
        }
        const size_t span = group.seqSizePerBlock();
        const size_t end  = (static_cast<size_t>(length) + span - 1) / span;
        // The prefix's last page is read by the first recompute chunk.
        const size_t first = std::min(end, static_cast<size_t>(std::max(prefix - 1, 0)) / span);
        for (int row = 0; row < resource.batchSize(); ++row) {
            const auto& ids = resource.cacheResource(row).blockIds(group.tag);
            const auto& blocks = ids.blocks();
            if (ids.linearPrefillPlan()) {
                // Other sparse slots are intentional. The replan allocates any
                // missing destination states, while the prefix must already exist.
                if (prefix > 0 && (first >= blocks.size() || isNullBlockIdx(blocks[first]))) {
                    return true;
                }
            } else if (blocks.size() < end
                       || std::any_of(blocks.begin() + first, blocks.begin() + end, isNullBlockIdx)) {
                return true;
            }
        }
    }
    return false;
}

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

void traceLinearPrefillAllocation(const BatchKVCacheResource& resource, int64_t request_id, int configured_n) {
    if (!linear_prefill_trace::enabled()) {
        return;
    }
    for (int batch = 0; batch < resource.batchSize(); ++batch) {
        for (const auto& tag : resource.cacheResource(batch).groupTags()) {
            const auto& ids = resource.cacheResource(batch).blockIds(tag);
            const auto& plan = ids.linearPrefillPlan();
            if (!plan) {
                continue;
            }
            std::unordered_set<BlockIdxType> owned;
            for (const auto block : ids.blocks()) {
                if (block > 0) {
                    owned.insert(block);
                }
            }
            std::ostringstream fields;
            fields << "\"request_id\":" << request_id << ",\"batch\":" << batch
                   << ",\"group_tag\":" << linear_prefill_trace::quote(tag)
                   << ",\"configured_tail_blocks\":" << configured_n
                   << ",\"prompt_tokens\":" << plan->promptTokens()
                   << ",\"initial_reuse_tokens\":" << plan->reusedTokens()
                   << ",\"page_tokens\":" << plan->pageTokens()
                   << ",\"checkpoint_tokens\":" << plan->checkpointTokens()
                   << ",\"planned_new_blocks\":" << plan->newPositions().size()
                   << ",\"owned_state_blocks\":" << owned.size()
                   << ",\"borrowed_block\":" << ids.blocks()[plan->borrowedPosition()]
                   << ",\"has_separate_tail\":" << (plan->hasSeparateTail() ? "true" : "false");
            linear_prefill_trace::emit("linear_prefill_allocated", fields.str());
        }
    }
}

void traceLinearCacheLoaded(const LoadAsyncContext& context, const KVCacheManager& manager,
                            int64_t request_id, int reuse_tokens) {
    if (!linear_prefill_trace::enabled() || !manager.blockTreeCache()) {
        return;
    }
    const auto& mapper = manager.cpSlotMapper();
    const int interval = mapper ? mapper->reuseBlockTokens(manager.cacheConfig()) : manager.cacheConfig().seq_size_per_block;
    const auto& sets = manager.blockTreeCache()->groupSets();
    for (const auto& desc : context.loadDescs()) {
        const auto tokens = (desc.path_index + 1) * static_cast<size_t>(interval);
        if (tokens > static_cast<size_t>(reuse_tokens) || desc.group_set_id >= sets.size()) {
            continue;
        }
        const auto& tags = sets[desc.group_set_id]->groupTags();
        for (size_t member = 0; member < tags.size(); ++member) {
            if (!manager.cacheConfig().group(tags[member]).policy.prefill_borrow_tail
                || member >= desc.target_blocks.size()) {
                continue;
            }
            std::ostringstream fields;
            fields << "\"request_id\":" << request_id
                   << ",\"group_tag\":" << linear_prefill_trace::quote(tags[member])
                   << ",\"checkpoint_tokens\":" << tokens
                   << ",\"source_tier\":" << linear_prefill_trace::quote(tierName(desc.source_tier))
                   << ",\"device_block\":" << desc.target_blocks[member];
            linear_prefill_trace::emit("linear_cache_loaded", fields.str());
        }
    }
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
    pending_linear_history_committed_tokens_ = 0;
}

bool StreamCacheResource::copyChunkKernelBlocks(int batch, const std::string& tag, int32_t* dst) const {
    if (stream_->isFakeStream() && stream_->warmup_chunk_window_) {
        const auto& group = resource_context_.cache_manager->cacheConfig().group(tag);
        if (group.policy.group_type != CacheGroupType::LINEAR) {
            return false;
        }
        const auto span   = group.seqSizePerBlock();
        const int  prefix = stream_->prefixLength();
        const int  end    = prefix + stream_->contextLength();
        const auto count  = batch_kv_cache_resource_->kernelBlocks(batch, tag).size();
        RTP_LLM_CHECK_WITH_INFO(end > prefix && static_cast<size_t>(end - 1) / span < count,
                                "warmup LINEAR state table does not cover the chunk");
        // Two slots per request: readers of the prefix cannot race checkpoint
        // writers. Intermediate checkpoints are unnecessary during warmup.
        const int initial = 1 + 2 * batch + stream_->warmup_chunk_index_ % 2;
        const int output  = 1 + 2 * batch + (stream_->warmup_chunk_index_ + 1) % 2;
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(std::max(initial, output)) < group.block_num,
                                "warmup LINEAR pool needs two state slots per request");
        std::fill_n(dst, count, NULL_BLOCK_IDX);
        if (prefix > 0) {
            dst[static_cast<size_t>(prefix - 1) / span] = initial;
        }
        dst[static_cast<size_t>(end - 1) / span] = output;
        return true;
    }
    if (!stream_->useChunkWindow() || stream_->isFakeStream()) {
        return false;
    }
    const auto& context = resource_context_;
    const auto& config  = context.cache_manager->cacheConfig();
    const auto& group   = config.group(tag);
    const auto  tail    = static_cast<size_t>(stream_->seqLength() - 1) / group.seqSizePerBlock();
    const auto& ids     = batch_kv_cache_resource_->cacheResource(batch).blockIds(tag);
    if (ids.linearPrefillPlan()) {
        return ids.copyLinearPrefillKernelBlocks(stream_->prefixLength(), stream_->currentChunkLen(), dst);
    }
    if (group.policy.shared_chunk_state) {
        // HCA has no overlap. Aligned chunks start a new 128-token compression interval.
        RTP_LLM_CHECK_WITH_INFO(stream_->seqSizePerBlock() % 128 == 0 && stream_->prefixLength() % 128 == 0
                                    && group.seqSizePerBlock() % 128 == 0,
                                "shared HCA state requires compression-aligned chunks");
        // Completed intervals use this forward's raw KV; only the tail needs persistent state.
        const auto chunk_tail =
            static_cast<size_t>(stream_->prefixLength() + stream_->currentChunkLen() - 1) / group.seqSizePerBlock();
        ids.copySharedKernelBlocks(tail, chunk_tail, dst);
        return true;
    }
    // A whole-prompt prefill needs no cyclic mapping. Short multimodal requests
    // may reach the scheduler without having reserved chunk banks at admission.
    if (stream_->prefixLength() == 0 && !stream_->isMiddleChunk()) {
        return false;
    }
    const bool reuse = reuseCache();
    const int  group_reserve =
        config.isLinearReplayGroup(static_cast<int>(config.topology().groupIdForTag(tag))) ? 0 : stream_->reserveStep();
    const auto banks =
        group.chunkCacheBlockCount(stream_->seqLength(), stream_->maxChunkSize(), reuse, group_reserve);
    if (banks == 0) {
        return false;
    }
    RTP_LLM_CHECK_WITH_INFO(
        stream_->prefixLength() % group.seqSizePerBlock() == 0 && stream_->currentChunkLen() <= stream_->maxChunkSize()
            && (!stream_->isMiddleChunk()
                || (stream_->prefixLength() + stream_->currentChunkLen()) % group.seqSizePerBlock() == 0),
        "cyclic chunk cache requires aligned prefixes and bounded grants");
    ids.copyCyclicKernelBlocks(tail, banks, dst);
    return true;
}

void StreamCacheResource::completeLinearPrefillChunk(int completed_tokens) {
    if (resource_context_.role_type != RoleType::PREFILL || !reuseCache()
        || resource_context_.cache_manager->cacheConfig().reuse_cache_tail_blocks == std::numeric_limits<int>::max()) {
        return;
    }
    for (int batch = 0; batch < batch_kv_cache_resource_->batchSize(); ++batch) {
        for (const auto& tag : batch_kv_cache_resource_->cacheResource(batch).groupTags()) {
            auto& ids = batch_kv_cache_resource_->mutableBlockIds(batch, tag);
            if (linear_prefill_trace::enabled() && ids.linearPrefillPlan()) {
                const auto round = ids.linearPrefillPlan()->round(stream_->prefixLength(), stream_->currentChunkLen());
                std::ostringstream fields;
                fields << "\"request_id\":" << stream_->streamId() << ",\"batch\":" << batch
                       << ",\"group_tag\":" << linear_prefill_trace::quote(tag)
                       << ",\"prefix_tokens\":" << stream_->prefixLength()
                       << ",\"end_tokens\":" << completed_tokens
                       << ",\"input_block\":" << (round.input_owner_position ? ids.blocks()[*round.input_owner_position] : 0)
                       << ",\"output_block\":" << ids.blocks()[round.output_owner_position]
                       << ",\"writes\":[";
                for (size_t i = 0; i < round.writes.size(); ++i) {
                    const auto& write = round.writes[i];
                    fields << (i ? "," : "") << "{\"block\":" << ids.blocks()[write.owner_position]
                           << ",\"tokens\":" << write.tokens
                           << ",\"retained\":" << (write.retained ? "true" : "false") << "}";
                }
                fields << ']';
                linear_prefill_trace::emit("linear_prefill_round_completed", fields.str());
            }
            ids.completeLinearPrefillChunk(completed_tokens);
        }
    }
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
    if (allocator_load_context_ && !deferAllocatorLoadRelease()) {
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
    stream_->clearLinearReplayWindow();
    linear_replay_.lease.reset();
    linear_replay_.clearInitialState();
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
            const Tier target_tier = insertTarget();
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, inserting cache, curBlocksNum=%d, target_tier=%s",
                              stream_->streamId(),
                              total_blocks,
                              tierName(target_tier));
            if (target_tier != Tier::NONE) {
                InsertInfo insert_info{
                    batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, target_tier, stream_->streamId()};
                insert_info.pd_decode_linear_tail_only = pd_decode_linear_tail_only_;
                size_t resident_prefix_length = 0;
                resource_context_.cache_manager->insertIntoCache(insert_info, resident_prefix_length);
            }
        } else {
            RTP_LLM_LOG_DEBUG(
                "tryReleaseKVBlock: stream=%ld, NOT inserting cache, reuseCache=%d, hasError=%d, status=%s",
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

int StreamCacheResource::estimatePeakNeedBlocks(int                seq_len,
                                                int                common_seq_len,
                                                int                remaining_tokens,
                                                int                reserve_step,
                                                int                target_batch_size,
                                                const std::string& pool_tag) const {
    return resource_context_.cache_manager->estimatePeakNeedBlocks(batch_kv_cache_resource_,
                                                                   seq_len,
                                                                   common_seq_len,
                                                                   remaining_tokens,
                                                                   reserve_step,
                                                                   reuseCache(),
                                                                   target_batch_size,
                                                                   linearCachePolicy(target_batch_size),
                                                                   pool_tag,
                                                                   resource_context_.role_type == RoleType::PREFILL
                                                                        && stream_->isContextStream()
                                                                        && stream_->chunkedPrefillEnabled()
                                                                        && !stream_->isFakeStream() ?
                                                                        stream_->inputLength() : 0);
}

LinearCachePolicy StreamCacheResource::linearCachePolicy(int target_batch_size) const {
    // A verified P->D handoff owns a tail-only layout even while the stream is
    // still marked as context/Fusion. Admission must match its initial allocator,
    // and committed-history publication must never run on that sparse layout.
    if (pd_decode_linear_tail_only_) {
        return LinearCachePolicy::TAIL_ONLY;
    }
    return canReclaimLinearHistory(target_batch_size) ? LinearCachePolicy::RECLAIM_COMMITTED :
                                                        LinearCachePolicy::KEEP_HISTORY;
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
    if (stream_->isPerfTest() && !isContextStream() && stream_->seqLength() > 1
        && batch_kv_cache_resource_->curBlocksNum() == 0
        && resource_context_.cache_manager->cacheConfig().linear_replay.enabled()) {
        // Fixed Decode treats the prompt's last token as its first anchor.
        // Initialize the processed prefix, then grow to the unchanged sequence
        // in initMalloc's existing second phase without pruning that state.
        malloc_info.complete_token_ids =
            std::make_shared<CompleteTokenIds>(*malloc_info.complete_token_ids, /*share=*/true);
        malloc_info.complete_token_ids->setSeqLength(stream_->seqLength() - 1);
        malloc_info.complete_token_ids->setReserveStep(stream_->reserveStep());
        malloc_info.incr_seq_len_override = stream_->seqLength();
    }
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
    malloc_info.enable_storage_backend     = !skip_remote_cache_;
    malloc_info.pd_decode_linear_tail_only  = pd_decode_linear_tail_only_;
    malloc_info.max_chunk_tokens =
        stream_->chunkedPrefillEnabled() && stream_->isContextStream() ? stream_->maxChunkSize() : 0;
    if (resource_context_.role_type == RoleType::PREFILL && stream_->isContextStream()
        && stream_->chunkedPrefillEnabled() && !stream_->isFakeStream()) {
        malloc_info.linear_prefill_prompt_tokens = stream_->inputLength();
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
    allocator_load_context_ = std::move(result.async_context);
    if (!allocator_load_context_) {
        traceLinearPrefillAllocation(*batch_kv_cache_resource_, stream_->streamId(),
                                      resource_context_.cache_manager->cacheConfig().reuse_cache_tail_blocks);
    }
    return allocator_load_context_ ? absl::OkStatus() : prepareLinearReplayResources();
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
    cache_reuse_metrics_.device_input_length    = result.device_input_length;
    cache_reuse_metrics_.match_cost_time_us     = result.match_cost_time_us;
    cache_reuse_metrics_.report_match_cost_time = result.match_end_time_us > 0;
    malloc_begin_time_us_                       = result.malloc_begin_time_us;
    cache_reuse_metrics_.load_attempted         = result.load_attempted;

    // A retry must not inherit the previous attempt's metrics, even when it no longer needs a load.
    cache_reuse_metrics_.load_success                  = false;
    cache_reuse_metrics_.report_load_metrics           = false;
    cache_reuse_metrics_.report_load_wait_latency      = false;
    cache_reuse_metrics_.load_wait_latency_us          = 0;
    cache_reuse_metrics_.report_match_to_ready_latency = false;
    cache_reuse_metrics_.match_to_ready_latency_us     = 0;
    cache_reuse_metrics_.load_prepare_latency_us       = result.load_prepare_latency_us;
    if (!result.load_attempted) {
        if (result.match_end_time_us > 0) {
            cache_reuse_metrics_.match_to_ready_latency_us =
                std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
            cache_reuse_metrics_.report_match_to_ready_latency = true;
        }
        return;
    }

    load_wait_begin_time_us_ = currentTimeUs();
    if (!result.success) {
        clearCacheReuseState();
        cache_reuse_metrics_.load_success              = false;
        cache_reuse_metrics_.report_load_metrics       = true;
        cache_reuse_metrics_.match_to_ready_latency_us = std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
}

void StreamCacheResource::publishAllocatorReuse(const LoadAsyncContext& context, bool local_only) {
    const size_t local = local_only ? context.deviceReadyBlocks() : context.localMatchedBlocks();
    const size_t total = local_only ? local : context.matchedBlocks();
    const size_t host  = context.matchedBlocks(Tier::HOST, local);
    const size_t disk  = context.matchedBlocks(Tier::DISK, local);
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
    const size_t backend = total - local;
    // Initial allocation shares the matched prefix across all batches.
    for (int batch = 0; batch < batch_kv_cache_resource_->batchSize(); ++batch) {
        auto& resource = batch_kv_cache_resource_->cacheResource(batch);
        resource.setDeviceReuseBlockNum(local - host - disk);
        resource.setMemoryReuseBlockNum(host);
        resource.setDiskReuseBlockNum(disk);
        resource.setStorageBackendReuseBlockNum(backend);
    }
    publishReuseLengths(total * tokens, host * tokens, disk * tokens, backend * tokens);
}

absl::Status StreamCacheResource::finalizeAllocatorLoad() {
    const bool load_success  = allocator_load_context_->success();
    const auto error         = allocator_load_context_->errorInfo();
    const auto load_context  = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
    auto       malloc_status = load_context == nullptr ? MallocStatus::NONE : load_context->mallocStatus();
    const bool timed_out     = load_context && load_context->deadlineExceeded();
    if (timed_out) {
        deferAllocatorLoadRelease();
    }

    bool linear_checkpoint_hole = false;
    if (load_success) {
        RTP_LLM_CHECK(load_context != nullptr);
        publishAllocatorReuse(*load_context, false);
    } else if (!timed_out && resource_context_.role_type == RoleType::PREFILL && malloc_status == MallocStatus::NONE) {
        const bool backend_timeout = load_context && load_context->backendTimedOut();
        skip_remote_cache_         = skip_remote_cache_ || backend_timeout;
        const int ready_length =
            load_context ? static_cast<int>(load_context->deviceReadyBlocks()) * reuseBlockTokens() : 0;
        const bool reuse_local =
            load_context && (load_context->localLoadSucceeded() || ready_length > stream_->reuseLength());
        const int   reuse_length = reuse_local ? ready_length : stream_->reuseLength();
        const auto& manager      = resource_context_.cache_manager;
        linear_checkpoint_hole =
            isContextStream() && stream_->chunkedPrefillEnabled() && reuseCache()
            && hasLinearCheckpointHole(
                *batch_kv_cache_resource_, manager->cacheConfig(), reuse_length, stream_->seqLength());
        if (linear_checkpoint_hole) {
            clearCacheReuseState();
        } else if (!manager->detachSharedBlocksForRecompute(batch_kv_cache_resource_, reuse_length)) {
            manager->free({batch_kv_cache_resource_, stream_->completeTokenIdsPtr()});
            clearCacheReuseState();
            malloc_status = MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
        } else if (reuse_local) {
            publishAllocatorReuse(*load_context, true);
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

    if (timed_out) {
        return absl::DeadlineExceededError("match-to-ready deadline exceeded");
    }
    if (malloc_status == MallocStatus::NONE) {
        traceLinearPrefillAllocation(*batch_kv_cache_resource_, stream_->streamId(),
                                      resource_context_.cache_manager->cacheConfig().reuse_cache_tail_blocks);
        if (load_context && (load_success || load_context->localLoadSucceeded())) {
            traceLinearCacheLoaded(*load_context, *resource_context_.cache_manager,
                                    stream_->streamId(), stream_->initialReuseLength());
        }
    }
    if (load_success) {
        return prepareLinearReplayResources();
    }
    if (linear_checkpoint_hole) {
        return absl::FailedPreconditionError(
            "chunked prefill LINEAR checkpoint missing after async cache load failure");
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

bool StreamCacheResource::deferAllocatorLoadRelease() {
    auto context = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
    if (!context || !context->needBackendMatch()) {
        return false;
    }
    auto replacement = std::make_shared<BatchKVCacheResource>();
    replacement->resetBatchSize(batch_kv_cache_resource_->batchSize());
    replacement->initGroups(resource_context_.cache_manager->cacheConfig().topologyPtr());
    context->deferRelease(batch_kv_cache_resource_);
    batch_kv_cache_resource_ = std::move(replacement);
    allocator_load_context_.reset();
    return true;
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

absl::Status StreamCacheResource::incrKVBlock(int seq_len_override, bool preserve_inflight) {
    RTP_LLM_PROFILE_FUNCTION();
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }
    // The scheduler owns host block-table mutation. In particular, an async
    // output worker may have committed tokens while the previous gather still
    // read this table. Its watermark is safe to consume here, before the next
    // allocation/gather, and is independent of a speculative allocation bound.
    drainLinearHistory();
    if (!preserve_inflight && batch_kv_cache_resource_->curBlocksNum() > 0) {
        const auto status = prepareLinearReplayResources(/*prepare_tails=*/false);
        if (!status.ok()) {
            return status;
        }
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource      = batch_kv_cache_resource_;
    malloc_info.complete_token_ids           = stream_->completeTokenIdsPtr();
    malloc_info.request_id                   = stream_->streamId();
    malloc_info.verbose                      = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.reuse_cache                  = reuseCache();
    malloc_info.enable_cache_lookup          = enableCacheLookup();
    malloc_info.enable_remove_skipped_blocks = !stream_->isContextStream();
    malloc_info.preserve_inflight            = preserve_inflight;
    malloc_info.incr_seq_len_override        = seq_len_override;
    malloc_info.pd_decode_linear_tail_only   = pd_decode_linear_tail_only_;
    // One predicate with the estimators: only a request that actually owns the
    // committed-watermark retirement keeps its linear history across decode.
    malloc_info.linear_cache_policy = linearCachePolicy(std::max(1, batch_kv_cache_resource_->batchSize()));

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

    // GC has already kept the committed tail and future reservations. Existing
    // replay destinations may still be in use, so do not replace their bindings.
    return preserve_inflight ? absl::OkStatus() : prepareLinearReplayResources();
}

void StreamCacheResource::publishPrefillCache(int processed_length) {
    const auto& manager     = resource_context_.cache_manager;
    const auto  target_tier = insertTarget();
    if (fake_inited_ || !manager || !manager->cacheConfig().linear_replay.enabled() || !reuseCache()
        || target_tier == Tier::NONE || processed_length <= 0) {
        return;
    }
    // Called by completed Prefill bookkeeping before Decode can prune the
    // immutable checkpoints. Truncate logical keys before the allocator applies CP stride.
    auto         prefix           = std::make_shared<BatchKVCacheResource>(*batch_kv_cache_resource_);
    const size_t completed_blocks = static_cast<size_t>(processed_length) / seqSizePerBlock();
    for (int batch = 0; batch < prefix->batchSize(); ++batch) {
        const auto& keys = batch_kv_cache_resource_->cacheKeys(batch);
        prefix->setBatchCacheKeys(batch,
                                  CacheKeysType(keys.begin(), keys.begin() + std::min(keys.size(), completed_blocks)));
    }
    prefix->setLastBlockAligned(true);
    InsertInfo insert_info{prefix, stream_->completeTokenIdsPtr(), false, target_tier};
    insert_info.pd_decode_linear_tail_only = pd_decode_linear_tail_only_;
    size_t resident_prefix_length = 0;
    manager->insertIntoCache(insert_info, resident_prefix_length);
}

bool StreamCacheResource::reserveLinearReplaySlot(bool* newly_reserved) {
    std::lock_guard<std::mutex> lock(*stream_->mutex_);
    if (newly_reserved != nullptr) {
        *newly_reserved = false;
    }
    auto& manager = resource_context_.cache_manager;
    if (!manager || !manager->cacheConfig().linear_replay.enabled() || fake_inited_ || stream_->isFakeStream()
        || stream_->generateConfig()->isPrefillOnly()) {
        return true;
    }
    if (resource_released_ || stream_->getStatus() == StreamState::FINISHED || stream_->hasErrorWithoutLock()) {
        return false;
    }
    if (!linear_replay_.lease) {
        linear_replay_.lease = manager->acquireLinearReplaySlot();
        if (newly_reserved != nullptr) {
            *newly_reserved = linear_replay_.lease != nullptr;
        }
    }
    return linear_replay_.lease != nullptr;
}

void StreamCacheResource::releaseUnusedLinearReplaySlot() {
    std::lock_guard<std::mutex> lock(*stream_->mutex_);
    if (stream_->getStatus() == StreamState::WAITING && linear_replay_.initial_block_ids.empty()
        && !linear_replay_.window && curBlocksNum() == 0 && !allocator_load_context_
        && !stream_->hasEventWithoutLock(StreamEvents::LoadInitiated)) {
        linear_replay_.lease.reset();
    }
}

absl::Status StreamCacheResource::prepareLinearReplayResources(bool prepare_tails) {
    auto& manager = resource_context_.cache_manager;
    if (!manager || !manager->cacheConfig().linear_replay.enabled() || isContextStream() || fake_inited_) {
        return absl::OkStatus();
    }
    const auto& config = manager->cacheConfig();
    if (!linear_replay_.lease) {
        linear_replay_.lease = manager->acquireLinearReplaySlot();
        if (!linear_replay_.lease) {
            return absl::UnavailableError("LINEAR replay log slots are temporarily unavailable");
        }
    }
    if (linear_replay_.initial_block_ids.empty() && !linear_replay_.window) {
        const int processed = std::max(0, stream_->seqLength() - 1);
        linear_replay_.initial_block_ids.assign(config.groupNums(), -1);
        if (processed > 0) {
            linear_replay_.initial_state_hold =
                manager->holdLinearReplayBlocks(batch_kv_cache_resource_, processed - 1);
            for (const int group_id : config.linear_replay.group_ids) {
                const auto& group = config.groups().at(static_cast<size_t>(group_id));
                const size_t position = static_cast<size_t>(processed - 1) / group.seqSizePerBlock();
                const auto& blocks = batch_kv_cache_resource_->blocks(0, group.tag);
                if (position >= blocks.size() || blocks[position] <= 0) {
                    linear_replay_.lease.reset();
                    linear_replay_.clearInitialState();
                    return absl::FailedPreconditionError("LINEAR replay canonical state is missing");
                }
                linear_replay_.initial_block_ids[static_cast<size_t>(group_id)] = blocks[position];
            }
        }
        manager->markLinearReplayStarted(batch_kv_cache_resource_, processed);
    }
    if (prepare_tails) {
        const std::vector<std::shared_ptr<LinearReplayBlockHold>> request_holds = {
            linear_replay_.initial_state_hold, stream_->linearReplayStateBlockHold()};
        if (!manager->makeLinearReplayTailsPrivate(
                batch_kv_cache_resource_, std::max(0, stream_->seqLength() - 1), request_holds)) {
            return absl::ResourceExhaustedError("LINEAR replay tail copy-on-write allocation failed");
        }
    }
    return absl::OkStatus();
}

void StreamCacheResource::reclaimLinearHistory(int committed_tokens) {
    if (!resource_context_.cache_manager || !resource_context_.cache_manager->cacheConfig().enable_linear_cache_reclaim
        || committed_tokens <= 0) {
        return;
    }
    pending_linear_history_committed_tokens_ = std::max(pending_linear_history_committed_tokens_, committed_tokens);
}

bool StreamCacheResource::canReclaimLinearHistory(int target_batch_size) const {
    if (!resource_context_.cache_manager || !resource_context_.cache_manager->cacheConfig().enable_linear_cache_reclaim
        || fake_inited_ || resource_released_ || !need_release_resource_ || allocator_load_context_
        || stream_->hasErrorWithoutLock() || stream_->hasNumBeams() || batch_kv_cache_resource_->batchSize() != 1
        || target_batch_size != 1 || stream_->maxBatchSize() != 1 || !block_update_mapping_.empty()
        || pd_decode_linear_tail_only_ || resource_context_.role_type != RoleType::PDFUSION || stream_->queryPdSep()
        || pd_kvcache_ref_) {
        return false;
    }
    // DEVICE publication acquires tree references synchronously. Lower-tier
    // stores only queue a transfer, so retain the request references there.
    if (reuseCache() && !enableDeviceCache()) {
        return false;
    }
    const auto& config = resource_context_.cache_manager->cacheConfig();
    if (!config.linear_replay.enabled()) {
        return true;
    }
    // Replay retires its own checkpoints. Only legacy LINEAR groups need the
    // scheduler to wait for the host reclamation watermark.
    for (int group_id = 0; group_id < config.groupNums(); ++group_id) {
        if (config.isLegacyLinearGroup(group_id)) {
            return true;
        }
    }
    return false;
}

void StreamCacheResource::drainLinearHistory() {
    // The scheduler owns block-table mutation, so the pending watermark drains
    // here, at the next allocation — context streams included: the linear
    // attention only needs the retained tail, so committed checkpoints are
    // reclaimable even while the chunked prefill is still running.
    if (pending_linear_history_committed_tokens_ <= 0 || !canReclaimLinearHistory(stream_->maxBatchSize())
        || curBlocksNum() == 0) {
        return;
    }
    const bool reuse_cache = reuseCache();
    const int committed_tokens =
        std::min(pending_linear_history_committed_tokens_, std::max(0, stream_->seqLength() - 1));
    updateCacheKeys(batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), seqSizePerBlock());
    resource_context_.cache_manager->reclaimLinearHistory(batch_kv_cache_resource_, committed_tokens, reuse_cache);
    pending_linear_history_committed_tokens_ = 0;
}

std::shared_ptr<LinearReplayBlockHold> StreamCacheResource::holdLinearReplayBlocks() {
    // While bookkeeping is pending, GPU accept lengths can select a page
    // beyond the host seqLength. Retain every writable tail page until the
    // verify's last-use event completes; the allocator may otherwise reuse it.
    // With current host state, retain only the exact destination as before.
    const int position = stream_->hasPendingAsyncBookkeeping() ? -1 : std::max(0, stream_->seqLength() - 1);
    return resource_context_.cache_manager->holdLinearReplayBlocks(batch_kv_cache_resource_, position);
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
        const ErrorInfo error        = allocator_load_context_->errorInfo();
        const bool      success      = allocator_load_context_->success();
        const auto      load_context = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
        const int64_t   completed_us = load_context ? load_context->completionTimeUs() : 0;
        if (completed_us > 0) {
            stream_->endCacheExposedWaitWithoutLock(completed_us);
        }
        const auto status = finalizeAllocatorLoad();
        const bool retry_prefill_without_remote =
            absl::IsDeadlineExceeded(status) && load_context && load_context->deadlineExceeded()
            && resource_context_.role_type == RoleType::PREFILL && isContextStream();
        if (status.ok()) {
            // Successful pages were ready when the transfer settled, even if the
            // scheduler only observes them after another batch finishes.
            stream_->recordKvReadyTime(success && completed_us > 0 ? completed_us : currentTimeUs());
        }
        if (!success && !absl::IsUnavailable(status)) {
            RTP_LLM_LOG_WARNING(
                "block tree load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
        }
        if (absl::IsUnavailable(status) || retry_prefill_without_remote) {
            // On an overall deadline, finalization deferred the old allocation.
            // Re-admit with fresh blocks and allow local cache lookup.
            skip_remote_cache_ = skip_remote_cache_ || retry_prefill_without_remote;
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
    pending_linear_history_committed_tokens_ = 0;
}

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch,
                                        bool                    copy_last_block,
                                        int                     valid_tail_tokens) {
    const bool success = resource_context_.cache_manager->updateKVBlock(
        batch_kv_cache_resource_, block_src_batch, copy_last_block, block_update_mapping_);
    if (success && copy_last_block) {
        // Bind the live prefix to this ownership transaction, before later steps
        // can change the stream length. Unknown callers retain full-copy (zero).
        for (auto& mapping : block_update_mapping_) {
            mapping.valid_tokens = valid_tail_tokens;
        }
    }
    return success;
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

Tier StreamCacheResource::insertTarget() const {
    if (!reuseCache()) {
        return Tier::NONE;
    }
    if (enableDeviceCache()) {
        return Tier::DEVICE;
    }
    return Tier::NONE;
}

void StreamCacheResource::swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs) {
    if (rhs == lhs) {
        return;
    }

    const auto& config = resource_context_.cache_manager->cacheConfig();
    for (size_t group_id = 0; group_id < config.groups().size(); ++group_id) {
        const auto& group = config.groups()[group_id];
        if (config.isLegacyLinearGroup(static_cast<int>(group_id))) {
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
