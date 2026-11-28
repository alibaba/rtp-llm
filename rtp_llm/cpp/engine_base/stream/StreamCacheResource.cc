#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include <algorithm>
#include <limits>
#include <cstring>
#include <thread>
#include <torch/extension.h>

using namespace std;

namespace rtp_llm {

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
            auto& routing                   = routing_ctx_.emplace();
            routing.request_id              = stream->streamId();
            routing.unique_key              = stream->uniqueKey();
            routing.deadline_ms             = stream->deadlineMs();
            routing.prefill_addr            = stream->prefillAddr();
            routing.prefill_tp_size         = stream->getPrefillTpSize();
            routing.prefill_cp_size         = stream->getPrefillCpSize();
            routing.prefill_pipeline_layout = stream->prefillPipelineLayout();
            routing.decode_needs_compute    = stream->needsDecodeAfterPrefill();
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
    if (!stream->usesBeamSearchTokenLayoutForCurrentStep()) {
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

bool p2pLoadSafeToRelease(const std::shared_ptr<AsyncContext>& context) {
    if (!context || !context->done()) {
        return context == nullptr;
    }
    const auto p2p = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(context);
    return !p2p || !p2p->resourceHoldPending();
}

void applyP2PSideChannel(const P2PSideChannelPayload& payload, GenerateStream* stream) {
#if defined(USING_CUDA)
    auto generator = stream->getGenerator();
    if (stream->generateConfig()->random_seed.has_value() || payload.has_rng_state) {
        if (!payload.has_rng_state || !generator.defined() || !stream->generateConfig()->random_seed.has_value()) {
            stream->reportErrorWithoutLock(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                                           "P2P handoff is missing or has unexpected request RNG state");
            return;
        }
        try {
            // The request seed is already installed. Restore before replay and first Decode sampling.
            std::lock_guard<std::mutex> lock(generator.mutex());
            generator.set_offset(payload.rng_offset);
        } catch (const std::exception& error) {
            stream->reportErrorWithoutLock(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                                           std::string("failed to restore P2P RNG offset: ") + error.what());
            return;
        }
    }
#endif
    // StartLoad transfers newly computed Prefill KV; it does not create a Decode cache hit.
    // Preserve the reuse counters established by Decode cache lookup for every pool layout.
    stream->generateInput()->handoff_multimodal_lengths = payload.multimodal_lengths;
    stream->mergePrefillFrontendContextMetricsWithoutLock(payload.frontend_context_token_num,
                                                          payload.frontend_context_token_num_with_cache,
                                                          payload.frontend_context_execute_time_us,
                                                          payload.frontend_context_execute_time_with_cache_us);
    if (payload.has_first_token) {
        const auto output_tensor = [&](const char* name) {
            const auto it = payload.first_token_tensors.find(name);
            return it == payload.first_token_tensors.end() ? torch::Tensor{} : TensorPbConvert::pbToTorch(it->second);
        };
        std::optional<PromptLogitsOutput> prompt_logits;
        if (payload.prompt_logits.has_value()) {
            const auto& pb                = *payload.prompt_logits;
            prompt_logits                 = PromptLogitsOutput{};
            prompt_logits->topk_logprobs  = TensorPbConvert::pbToTorch(pb.topk_logprobs());
            prompt_logits->topk_token_ids = TensorPbConvert::pbToTorch(pb.topk_token_ids());
            if (pb.has_target_logprobs())
                prompt_logits->target_logprobs = TensorPbConvert::pbToTorch(pb.target_logprobs());
            prompt_logits->start_pos = pb.start_pos();
            prompt_logits->end_pos   = pb.end_pos();
        }
        stream->setIsContextStream(false);
        stream->step();
        stream->updateWithoutLock(
            {.new_tokens        = buildFirstTokenUpdateTokens(stream, static_cast<int32_t>(payload.first_token_id)),
             .num_new_tokens    = 1,
             .hidden_states     = output_tensor("first_token_hidden_states"),
             .logits            = output_tensor("first_token_logits"),
             .softmax_probs     = output_tensor("first_token_softmax_probs"),
             .cum_log_probs     = output_tensor("first_token_cum_log_probs"),
             .all_probs         = output_tensor("first_token_all_probs"),
             .loss              = output_tensor("first_token_loss"),
             .src_batch_indices = {},
             .all_hidden_states = output_tensor("first_token_all_hidden_states"),
             .update_remote_generate = false,
             .force_update_info      = false,
             .prompt_logits          = std::move(prompt_logits),
             .generation_prefill_cuda_graph_status =
                 static_cast<GenerationPrefillCudaGraphStatus>(payload.generation_prefill_cuda_graph_status),
             .token_logprobs        = output_tensor("first_token_logprobs"),
             .top_logprob_token_ids = output_tensor("first_token_top_logprob_token_ids"),
             .top_logprobs          = output_tensor("first_token_top_logprobs")});
        // DSpARK resumes from the prompt tail without a Prefill proposal.
        stream->initSpeculativeHandoffPositions();
    }

    {
        stream->setPrefillReuseLength(payload.total_reuse_len,
                                      payload.local_reuse_len,
                                      payload.remote_reuse_len,
                                      payload.memory_reuse_len,
                                      payload.disk_reuse_len,
                                      payload.prefill_use_independent_block_pools);
    }

    torch::Tensor probs;
    torch::Tensor hidden;
    try {
        // A shape such as [0] describes an empty tensor, not a proposal.
        // Decode before checking numel so malformed shapes/bytes still fail.
        probs  = TensorPbConvert::pbToTorch(payload.propose_probs);
        hidden = TensorPbConvert::pbToTorch(payload.propose_hidden);
    } catch (const std::exception& error) {
        stream->reportErrorWithoutLock(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                                       std::string("invalid P2P speculative tensor: ") + error.what());
        return;
    }
    const bool has_probs  = probs.numel() > 0;
    const bool has_hidden = hidden.numel() > 0;
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
            if (!has_probs) {
                probs = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32));
            }
            if (!has_hidden) {
                hidden = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16));
            }
            stream->initSpeculativeHandoffPositions();
            stream->setContainProposeToken(true);
            stream->setProposeToken(payload.propose_tokens);
            auto output    = std::make_shared<SpeculativeExecutorStreamOutput>();
            output->tokens = torch::zeros({1, static_cast<int64_t>(payload.propose_tokens.size())}, torch::kInt32);
            memcpy(output->tokens.data_ptr<int>(),
                   payload.propose_tokens.data(),
                   payload.propose_tokens.size() * sizeof(int));
            // Wire snapshots are CPU-owned; speculative execution consumes device tensors.
            output->all_probs     = probs.to(torch::kCUDA);
            output->hidden_states = hidden.to(torch::kCUDA);
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
            const auto& blocks = resource.blocks(row, group.tag);
            if (blocks.size() < end || std::any_of(blocks.begin() + first, blocks.begin() + end, isNullBlockIdx)) {
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
    if (!stream_->useChunkWindow() || stream_->isFakeStream()) {
        return false;
    }
    const auto& context = resource_context_;
    const auto& config  = context.cache_manager->cacheConfig();
    const auto& group   = config.group(tag);
    const auto  tail    = static_cast<size_t>(stream_->seqLength() - 1) / group.seqSizePerBlock();
    const auto& ids     = batch_kv_cache_resource_->cacheResource(batch).blockIds(tag);
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

bool StreamCacheResource::p2pReleasePending() const {
    return !p2pLoadSafeToRelease(p2p_load_context_) || !p2pLoadSafeToRelease(released_p2p_load_context_.lock());
}

void StreamCacheResource::trackReleasedP2PLoad() {
    released_p2p_load_context_ = p2p_load_context_;
    auto p2p                   = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(p2p_load_context_);
    p2p_load_context_.reset();
    if (p2p) {
        p2p->onTargetSafeToRelease([weak_stream = stream_->weak_from_this()] {
            if (auto stream = weak_stream.lock()) {
                stream->notifyLocalCleanupProgress();
            }
        });
    }
}

void StreamCacheResource::failPrefillPublication(const ErrorInfo& error) {
    if (resource_context_.role_type == RoleType::PREFILL && stream_->queryPdSep() && resource_context_.cache_manager) {
        resource_context_.cache_manager->failP2PPublication(
            stream_->streamId(), stream_->uniqueKey(), stream_->deadlineMs(), error);
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
    if (p2p_load_context_) {
        resource_context_.cache_manager->cancelP2PLoad(p2p_load_context_);
        trackReleasedP2PLoad();
    }
    allocator_load_error_ = absl::OkStatus();
    p2p_cancel_requested_ = false;
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->hasErrorWithoutLock())) {
        return;
    }
    RTP_LLM_LOG_DEBUG("releaseResource: stream=%ld, curBlocksNum=%d", stream_->streamId(), curBlocksNum());
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
        const bool side_channel_only = resource_context_.role_type == RoleType::DECODE && stream_->queryPdSep()
                                       && !stream_->needsDecodeAfterPrefill();
        if (!side_channel_only && reuseCache() && !stream_->hasErrorWithoutLock()
            && stream_->getStatus() == StreamState::FINISHED) {
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

        const bool log_release = resource_context_.role_type == RoleType::DECODE && stream_->queryPdSep()
                                 && Logger::getEngineLogger().isLevelEnabled(alog::LOG_LEVEL_INFO);
        const auto released_blocks = log_release ? batch_kv_cache_resource_->debugString() : std::string{};
        resource_context_.cache_manager->free(free_info);
        if (log_release) {
            RTP_LLM_LOG_INFO("P2P KV event=stream_release request_id=%ld unique_key=%s event_time_us=%ld blocks=%s",
                             stream_->streamId(),
                             stream_->uniqueKey().c_str(),
                             currentTimeUs(),
                             released_blocks.c_str());
        }
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
                                                                   pool_tag);
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
        || p2pReleasePending()) {
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
    // Async tree allocation publishes its final physical table on completion.
    // P2P must hold that committed table, including newly materialized targets.
    if (allocator_load_context_) {
        return true;
    }
    if (stream_->queryPdSep() && !p2p_load_context_ && resource_context_.cache_manager
        && resource_context_.cache_manager->hasP2PConnector()) {
        auto meta              = std::make_shared<MetaImpl>(reuseCache() && enableMemoryCache(),
                                               reuseCache() && resource_context_.enable_remote_cache,
                                               stream_->traceId());
        meta->generate_stream_ = stream_;
        meta->fillRoutingContext(stream_);
        if (resource_context_.role_type == RoleType::DECODE && stream_->needsDecodeAfterPrefill()
            && Logger::getEngineLogger().isLevelEnabled(alog::LOG_LEVEL_INFO)) {
            RTP_LLM_LOG_INFO("P2P KV event=target_bound request_id=%ld unique_key=%s event_time_us=%ld blocks=%s",
                             stream_->streamId(),
                             stream_->uniqueKey().c_str(),
                             currentTimeUs(),
                             batch_kv_cache_resource_->cacheResource(0).debugString().c_str());
        }
        p2p_load_context_ = resource_context_.cache_manager->asyncLoadCache(
            batch_kv_cache_resource_->cacheResource(0), meta, treeCoveredBlockNum());
    }
    return allocator_load_context_ != nullptr || p2p_load_context_ != nullptr;
}

size_t StreamCacheResource::treeCoveredBlockNum() const {
    const auto& resource = batch_kv_cache_resource_->cacheResource(0);
    if (!allocator_load_context_) {
        // After materialization all restored tiers reside in the physical table.
        return resource.deviceReuseBlockNum() + resource.memoryReuseBlockNum() + resource.diskReuseBlockNum()
               + resource.storageBackendReuseBlockNum();
    }
    const auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
    if (!load_context) {
        RTP_LLM_LOG_WARNING("allocator load context is not LoadAsyncContext, stream=%ld", stream_->streamId());
        return resource.deviceReuseBlockNum();
    }
    return load_context->matchedBlocks();
}

bool StreamCacheResource::loadCacheDone() {
    bool allocator_completed = false;
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
        allocator_completed = status.ok();
        if (!success && !absl::IsUnavailable(status)) {
            RTP_LLM_LOG_WARNING(
                "block tree load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
        }
        if ((absl::IsUnavailable(status) || retry_prefill_without_remote) && !p2p_load_context_) {
            // On an overall deadline, finalization deferred the old allocation.
            // Re-admit with fresh blocks and allow local cache lookup.
            skip_remote_cache_ = skip_remote_cache_ || retry_prefill_without_remote;
            ++malloc_failed_times_;
            stream_->generate_status_->clearLoadInitiated();
            reportMallocRetry();
            return true;
        }
        if (!status.ok()) {
            // Preserve the allocator cause while cancellation drains P2P writes.
            allocator_load_error_ = status;
            if (p2p_load_context_ && !p2p_cancel_requested_) {
                resource_context_.cache_manager->cancelP2PLoad(p2p_load_context_);
                p2p_cancel_requested_ = true;
            }
        }
    }

    if (!allocator_load_error_.ok()) {
        // Tree failure terminates the whole load. Keep the Stream in
        // LOADING_CACHE until P2P cancellation reaches a terminal state and
        // its Decode target lease confirms that no transfer can still write.
        if (!p2pLoadSafeToRelease(p2p_load_context_)) {
            return false;
        }
        p2p_load_context_.reset();
        p2p_cancel_requested_ = false;
        stream_->reportEventWithoutLock(
            StreamEvents::Error,
            absl::IsDeadlineExceeded(allocator_load_error_) ? ErrorCode::LOAD_CACHE_TIMEOUT : ErrorCode::MALLOC_FAILED,
            std::string(allocator_load_error_.message()));
        allocator_load_error_ = absl::OkStatus();
        return true;
    }

    if (allocator_completed && !p2p_load_context_) {
        // Resume a P2P start deferred until tree allocation supplied its blocks.
        asyncLoadCache();
    }
    if (p2p_load_context_) {
        if (!p2p_load_context_->done()) {
            return false;
        }
        if (p2p_load_context_->success()) {
            if (const auto p2p = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(p2p_load_context_)) {
                if (const auto* payload = p2p->sideChannelPayload()) {
                    try {
                        applyP2PSideChannel(*payload, stream_);
                        if (!stream_->hasErrorWithoutLock()) {
                            stream_->p2p_handoff_ready_time_us_.store(p2p->handoffReadyTimeUs());
                        }
                    } catch (const std::exception& error) {
                        stream_->reportErrorWithoutLock(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                                                        std::string("invalid P2P side-channel payload: ")
                                                            + error.what());
                    }
                }
            }
            if (stream_->generateInput()->request_deadline_is_handoff_only && !stream_->hasErrorWithoutLock()) {
                stream_->p2p_handoff_completed_.store(true);
            }
            p2p_load_context_.reset();
        } else {
            // P2P always represents a mandatory request lifecycle once a
            // context exists. This includes Tree-full-hit/no-transfer, where
            // StartLoad still returns side-channel state and releases Prefill
            // resources, and Prefill registration itself. None of these
            // failures is a benign cache miss.
            const auto error = p2p_load_context_->errorInfo();
            // The scheduler may still retain the target lease after logical failure.
            // Keep its physical-completion proof visible to cancellation cleanup.
            trackReleasedP2PLoad();
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

}  // namespace rtp_llm
