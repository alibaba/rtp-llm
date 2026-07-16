#include "rtp_llm/cpp/engine_base/stream/GenerateStateMachine.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include <cstdlib>
#include <string>

using namespace std;

namespace rtp_llm {
namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
}

}  // namespace
// ============================================================================
// GenerateStateMachine method implementations
// ============================================================================

StreamState GenerateStateMachine::moveToNext() {
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
            handleRunning();
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
        auto result = stream_cache_resource_->initKVBlock(reserve_step_);
        if (!result.ok()) {
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
            if (stream != nullptr) {
                stream->recordRunningTime();
            }
            status.store(StreamState::RUNNING, std::memory_order_release);
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
        auto stream = stream_cache_resource_->stream();
        if (stream != nullptr) {
            stream->recordRunningTime();
        }
        status.store(StreamState::RUNNING, std::memory_order_release);
        return;
    }

    // Decode streams, including PREFILL-role streams after fallback, must keep
    // cache block tables aligned with the growing sequence length.
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_);
    if (!result.ok()) {
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
        return;
    }
    auto stream = stream_cache_resource_->stream();
    if (stream != nullptr) {
        stream->recordRunningTime();
    }
    status.store(StreamState::RUNNING, std::memory_order_release);
    return;
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

void GenerateStateMachine::handleRunning() {
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
    // Use the publish-time seqLength so incrKVBlock doesn't race the async
    // worker's update() — a stale read skips the block-boundary allocation.
    // Prefer Normal state; fall back to MTP state if the stream is MTP.
    int             seq_len_override = -1;
    GenerateStream* stream           = stream_cache_resource_->stream();
    if (stream != nullptr) {
        const int normal_override = stream->getNormalAsyncDeviceState().next_real_seq_len;
        if (normal_override > 0) {
            seq_len_override = normal_override;
        } else {
            const auto& mtp_state = stream->getMtpAsyncDeviceState();
            const int   mtp_override = mtp_state.next_real_seq_len;
            if (mtp_override > 0) {
                seq_len_override = mtp_override;
            }
        }
        if (asyncDebugEnabled() && stream->hasPendingAsyncBookkeeping()) {
            RTP_LLM_LOG_WARNING("[async-debug] handleRunning while async bookkeeping pending: stream=%ld pd_sep=%d "
                                "status=%s seq_len=%d normal_last_real=%d normal_next_real=%d "
                                "mtp_next_real=%d override=%d",
                                stream->streamId(),
                                stream->queryPdSep(),
                                StreamStateToString(status.load(std::memory_order_acquire)).c_str(),
                                stream->seqLength(),
                                stream->getNormalAsyncDeviceState().last_real_seq_len,
                                stream->getNormalAsyncDeviceState().next_real_seq_len,
                                stream->getMtpAsyncDeviceState().next_real_seq_len,
                                seq_len_override);
        }
    }
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_, seq_len_override);
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
