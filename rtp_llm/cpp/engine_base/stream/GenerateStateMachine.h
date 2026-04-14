#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class StreamCacheResource;  // forward declaration

// Stream 生命周期状态机，将原先分散在 FIFOScheduler 中的状态转移逻辑集中管理。
// 状态转移路径: WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED
// 每次调度轮调用 moveToNext() 驱动状态转移，由 FIFOScheduler::evaluateAndUpdateStreams 统一调用。
// 外部通过 reportEvent() 投递事件（替代原先分散的 reportXX 接口），moveToNext() 消费累积事件后决策转移。
// 线程安全说明：GenerateStateMachine 本身不提供同步机制，外部调用者需保证 reportEvent() 和 moveToNext()
// 的调用串行化（通常通过 GenerateStream::mutex_ 保护）。
struct GenerateStateMachine {
public:
    GenerateStateMachine(std::shared_ptr<StreamCacheResource> stream_cache_resource):
        stream_cache_resource_(stream_cache_resource) {}

    // 统一的事件上报接口
    // 注意：此方法非线程安全，外部应当仅通过GenerateStream在持锁路径下调用
    void reportEvent(StreamEvents::EventType event,
                     ErrorCode               error_code = ErrorCode::NONE_ERROR,
                     const std::string&      error_msg  = "") {
        if (error_info.ok() && event == StreamEvents::Error) {
            error_info = ErrorInfo(error_code, error_msg);
        }
        events_.append(event);
    }

    // 检查是否包含指定事件
    bool hasEvent(StreamEvents::EventType event) const {
        return events_.has(event);
    }

    StreamState moveToNext();

    StreamState getStatus() const {
        return status.load(std::memory_order_acquire);
    }

    void setReserveStep(size_t reserve_step) {
        reserve_step_ = reserve_step;
    }

    // 公开的状态和错误信息，GenerateStream 等外部代码直接访问
    // status 使用 atomic 保证线程安全：moveToNext() 在 mutex_ 下写入，getStatus() 无锁读取
    std::atomic<StreamState> status = StreamState::WAITING;
    ErrorInfo                error_info;

private:
    void handleWaiting();
    void handleLoading();
    void handleRunning();
    void releaseResource();

    StreamEvents events_;

    std::shared_ptr<StreamCacheResource> stream_cache_resource_ = nullptr;
    size_t                               reserve_step_          = 0;
};

}  // namespace rtp_llm
