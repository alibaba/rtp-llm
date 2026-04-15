#include "rtp_llm/cpp/engine_base/stream/GenerateStateMachine.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

using namespace std;

namespace rtp_llm {
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
            status.store(StreamState::LOADING_CACHE, std::memory_order_release);
        } else if (stream_cache_resource_->resourceContext().role_type != RoleType::DECODE) {
            // Loading cache 失败或不需要loading，直接触发重计算
            // 当前decodeRpcServer会调用moveToNext，判断role type避免decodeRpcServer在enqueue前提早走到running状态
            status.store(StreamState::RUNNING, std::memory_order_release);
        }
        return;
    }

    // Prefill 角色在 LoadInitiated 后不需要 incrKVBlock。
    // Prefill 端只需要一次 initKVBlock 分配所有 block，如果调用 incrKVBlock
    // 会导致 cache manager 对不完整的最后一个 block 执行 pop_back，
    // 破坏已分配的 block 结构。
    if (stream_cache_resource_->resourceContext().role_type == RoleType::PREFILL) {
        status.store(StreamState::RUNNING, std::memory_order_release);
        return;
    }

    // 绕过incrKVBlock at prefill
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_);
    if (!result.ok()) {
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
        return;
    }
    status.store(StreamState::RUNNING, std::memory_order_release);
    return;
}

void GenerateStateMachine::handleLoading() {
    if (stream_cache_resource_->loadCacheDone()) {
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
    if (stream_cache_resource_->resourceContext().role_type == RoleType::PREFILL) {
        return;
    }
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_);
    if (!result.ok()) {
        // Report Error event so moveToNext() won't be called again on this stream
        reportEvent(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "incrKVBlock failed: LACK MEM");
        status.store(StreamState::FINISHED, std::memory_order_release);
        releaseResource();
    }
}

void GenerateStateMachine::releaseResource() {
    if (!stream_cache_resource_->isResourceReleased()) {
        stream_cache_resource_->releaseResource();
    }
}
}  // namespace rtp_llm
