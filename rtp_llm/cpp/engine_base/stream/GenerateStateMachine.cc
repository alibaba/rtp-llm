#include "rtp_llm/cpp/engine_base/stream/GenerateStateMachine.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>

using namespace std;

namespace rtp_llm {
namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
}

using CacheBlockLayout = std::vector<std::tuple<int, std::string, BlockIndicesType, BlockIndicesType>>;

CacheBlockLayout cacheBlockLayout(const BatchKVCacheResource& cache) {
    CacheBlockLayout layout;
    for (int batch = 0; batch < cache.batchSize(); ++batch) {
        const auto& resource = cache.cacheResource(batch);
        for (const auto& tag : resource.groupTags()) {
            layout.emplace_back(batch, tag, resource.blocks(tag), resource.kernelBlocks(tag));
        }
    }
    return layout;
}

}  // namespace
// ============================================================================
// GenerateStateMachine method implementations
// ============================================================================

StreamState GenerateStateMachine::moveToNext(bool coalesce_mtp_lookahead) {
    // Error 最高优先级，任何状态下直接终止
    if (events_.has(StreamEvents::Error)) {
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
        return StreamState::FINISHED;
    }

    switch (status.load(std::memory_order_acquire)) {
        case StreamState::WAITING:
            handleWaiting();
            break;
        case StreamState::LOADING_CACHE:
            handleLoading();
            break;
        case StreamState::RUNNING:
            handleRunning(coalesce_mtp_lookahead);
            break;
        case StreamState::FINISHED:
            break;
        default:
            RTP_LLM_LOG_ERROR("Error: Unrecognized Generate State");
            if (error_info.ok()) {
                error_info = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "Error: Unrecognized Generate State");
            }
            status.store(StreamState::FINISHED, std::memory_order_release);
            releaseResource();
            break;
    }
    return status.load(std::memory_order_acquire);
}

void GenerateStateMachine::handleWaiting() {
    if (!events_.has(StreamEvents::CanRun)) {
        return;
    }
    // LoadInitiated 未设置时，必须先执行 initKVBlock 和 asyncLoadCache
    if (!events_.has(StreamEvents::LoadInitiated)) {
        auto stream = stream_cache_resource_->stream();
        if (stream != nullptr) {
            stream->recordWaitLatency();
        }
        auto result = stream_cache_resource_->initKVBlock();
        if (!result.ok()) {
            const auto role_type              = stream_cache_resource_->resourceContext().role_type;
            const bool retryable_prefill_init = (role_type == RoleType::PDFUSION || role_type == RoleType::PREFILL)
                                                && stream_cache_resource_->isContextStream();
            if (absl::IsUnavailable(result) && retryable_prefill_init) {
                stream_cache_resource_->reportMallocRetry();
                return;
            }
            error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
            status.store(StreamState::FINISHED, std::memory_order_release);
            releaseResource();
            return;
        }
        bool ret = stream_cache_resource_->asyncLoadCache();
        // 设置 LoadInitiated 标志，表示已尝试asyncLoadCache. 当前实现即便asyncLoadCache失败也不再重试
        reportEvent(StreamEvents::LoadInitiated);
        if (ret) {
            if (stream != nullptr) {
                stream->recordLoadingCacheStartTime();
            }
            status.store(StreamState::LOADING_CACHE, std::memory_order_release);
        } else if (stream_cache_resource_->resourceContext().role_type != RoleType::DECODE) {
            // Loading cache 失败或不需要loading，直接触发重计算
            // 当前decodeRpcServer会调用moveToNext，判断role type避免decodeRpcServer在enqueue前提早走到running状态
            transitionToRunning();
        }
        return;
    }

    // A PREFILL role normally runs only the context pass, so it must not call
    // incrKVBlock while the stream is still a context stream. With PD fallback,
    // the same role can continue into decode after GenerateStream::update()
    // flips isContextStream() to false; from that point block tables must grow
    // exactly like a decode stream.
    if (stream_cache_resource_->resourceContext().role_type == RoleType::PREFILL
        && stream_cache_resource_->isContextStream()) {
        transitionToRunning();
        return;
    }

    // Decode streams, including PREFILL-role streams after fallback, must keep
    // cache block tables aligned with the growing sequence length.
    auto result = stream_cache_resource_->incrKVBlock();
    if (!result.ok()) {
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
        return;
    }
    transitionToRunning();
}

void GenerateStateMachine::handleLoading() {
    if (stream_cache_resource_->loadCacheDone()) {
        auto stream = stream_cache_resource_->stream();
        if (stream != nullptr) {
            stream->recordLoadingCacheDoneTime();
        }
        status.store(StreamState::WAITING, std::memory_order_release);
    }
}

void GenerateStateMachine::transitionToRunning() {
    auto stream = stream_cache_resource_->stream();
    if (stream != nullptr) {
        stream->recordRunningTime();
    }
    status.store(StreamState::RUNNING, std::memory_order_release);
}

void GenerateStateMachine::handleRunning(bool coalesce_mtp_lookahead) {
    // in pd sep case，kvcache could be released after remote load done.
    if (events_.has(StreamEvents::GenerateDone)) {
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
        return;
    }
    if (stream_cache_resource_->resourceContext().role_type == RoleType::PREFILL
        && stream_cache_resource_->isContextStream()) {
        return;
    }
    // Keep a valid MTP page-table snapshot until its reserved capacity is
    // exhausted. Drain committed history together with growth, rather than
    // invalidating the same continuation again on the following round.
    int                             seq_len_override   = -1;
    int                             allocation_seq_len = -1;
    GenerateStream* stream           = stream_cache_resource_->stream();
    std::optional<CacheBlockLayout> committed_cache_layout;
    if (stream != nullptr) {
        const bool  pending   = stream->hasPendingAsyncBookkeeping();
        const auto  mtp_state = stream->getMtpAsyncDeviceState();
        const int   reservation_seq_len = pending ? mtp_state.next_seq_len_upper_bound : stream->seqLength();
        int64_t     coalesced_target_len = -1;
        if (coalesce_mtp_lookahead && !pending && GenerateStream::hasMtpCacheSnapshot(mtp_state) && mtp_state.batch
            && reserve_step_ > 1) {
            const int64_t committed = stream->seqLength();
            const int64_t block_sz  = static_cast<int64_t>(stream->seqSizePerBlock());
            const int64_t reserve   = static_cast<int64_t>(reserve_step_);
            int64_t       max_useful_seq = static_cast<int64_t>(stream->maxSeqLen());
            if (stream->generateConfig() && stream->generateConfig()->max_new_tokens > 0) {
                max_useful_seq = std::min<int64_t>(
                    max_useful_seq,
                    static_cast<int64_t>(stream->inputLength())
                        + static_cast<int64_t>(stream->generateConfig()->max_new_tokens));
            }
            if (block_sz > reserve) {
                const int64_t target = std::min<int64_t>(committed + block_sz, max_useful_seq) - reserve;
                if (target > committed) {
                    coalesced_target_len = target;
                }
            }
        }
        if (GenerateStream::hasMtpCacheSnapshot(mtp_state) && reservation_seq_len > 0
            && stream_cache_resource_->singleBatchNeedBlocks(reservation_seq_len, static_cast<int>(reserve_step_))
                   == 0) {
            if (coalesced_target_len < 0
                || stream_cache_resource_->singleBatchNeedBlocks(static_cast<int>(coalesced_target_len),
                                                                 static_cast<int>(reserve_step_))
                       == 0) {
                return;
            }
        }

        // Once bookkeeping catches up, the host table represents the same
        // committed layout as the device snapshot. Allocation/reclamation may
        // still change it; retain the cached mappings only if neither table
        // changes. Value copies are required because BlockIds owners alias.
        if (!pending && GenerateStream::hasMtpCacheSnapshot(mtp_state)) {
            committed_cache_layout = cacheBlockLayout(stream_cache_resource_->kvCache());
            if (mtp_state.batch && reserve_step_ > 1) {
                // Host commit can lag one speculative round. Reserve the next
                // round as well, so FULL and LINEAR groups grow together. Bound
                // the lookahead to one page and the model context limit; this
                // also keeps the current LINEAR tail inside its retained range.
                const int64_t committed  = stream->seqLength();
                const int64_t next_round = std::max<int64_t>(committed, mtp_state.next_seq_len_upper_bound)
                                           + static_cast<int64_t>(reserve_step_);
                const int64_t desired =
                    coalesced_target_len > 0 ? std::max(next_round, coalesced_target_len) : next_round;
                const int64_t lookahead = std::min<int64_t>(
                    {desired, committed + stream->seqSizePerBlock(), static_cast<int64_t>(stream->maxSeqLen())});
                if (lookahead > committed) {
                    allocation_seq_len = static_cast<int>(lookahead);
                }
            }
        }

        // Only use publish-time overrides while the host commit is still in
        // flight.  Once it finishes, complete_token_ids is authoritative and
        // avoids carrying a conservative upper bound into later rounds.
        if (pending) {
            const int normal_override = stream->getNormalAsyncDeviceState().next_real_seq_len;
            if (normal_override > 0) {
                seq_len_override = normal_override;
            } else if (mtp_state.next_seq_len_upper_bound > 0) {
                seq_len_override = mtp_state.next_seq_len_upper_bound;
            }
        }
        if (asyncDebugEnabled() && stream->hasPendingAsyncBookkeeping()) {
            RTP_LLM_LOG_WARNING("[async-debug] handleRunning while async bookkeeping pending: stream=%ld pd_sep=%d "
                                "status=%s published_seq_upper=%d normal_last_real=%d normal_next_real=%d "
                                "mtp_next_real=%d override=%d",
                                stream->streamId(),
                                stream->queryPdSep(),
                                StreamStateToString(status.load(std::memory_order_acquire)).c_str(),
                                mtp_state.next_seq_len_upper_bound,
                                stream->getNormalAsyncDeviceState().last_real_seq_len,
                                stream->getNormalAsyncDeviceState().next_real_seq_len,
                                mtp_state.next_seq_len_upper_bound,
                                seq_len_override);
        }
    }
    // An unfinished host commit cannot be used to validate device page tables.
    if (stream != nullptr && !committed_cache_layout) {
        stream->clearMtpCacheSnapshot(/*retain_batch=*/true);
    }
    auto result = stream_cache_resource_->incrKVBlock(allocation_seq_len >= 0 ? allocation_seq_len : seq_len_override);
    if (!result.ok() && allocation_seq_len >= 0) {
        // Lookahead is optional under memory pressure. The allocator rolls
        // back a failed growth; retry only the original required capacity.
        result = stream_cache_resource_->incrKVBlock(seq_len_override);
    }
    if (committed_cache_layout
        && (!result.ok() || *committed_cache_layout != cacheBlockLayout(stream_cache_resource_->kvCache()))) {
        stream->clearMtpCacheSnapshot(/*retain_batch=*/result.ok());
    }
    if (!result.ok()) {
        // Report Error event so moveToNext() won't be called again on this stream
        reportEvent(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "incrKVBlock failed: LACK MEM");
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
    }
}

void GenerateStateMachine::releaseResource() {
    if (stream_cache_resource_->isResourceReleased()) {
        return;
    }
    // releaseResource runs under GenerateStream::mutex_; do not wait here.
    // If a worker still owns KV blocks, mark deferred and let its dec path
    // perform the release after the pending count drains.
    GenerateStream* stream = stream_cache_resource_->stream();
    if (stream != nullptr && stream->hasPendingAsyncBookkeeping()) {
        stream->markDeferredRelease();
        return;
    }
    stream_cache_resource_->releaseResource();
}
}  // namespace rtp_llm
