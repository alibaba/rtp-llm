#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class StreamCacheResource;  // forward declaration

// Stream lifecycle state holder. Lifecycle transitions (WAITING -> RUNNING ->
// FINISHED, cache loading, error handling) are now driven directly by
// GenerateStream's lifecycle methods (prepare/isReady/activate/advance/finish).
// This struct retains event accumulation, status storage, and error reporting.
// Thread safety: callers must serialise reportEvent() and status writes
// (typically via GenerateStream::mutex_).
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

    StreamState getStatus() const {
        return status.load(std::memory_order_acquire);
    }

    void setReserveStep(size_t reserve_step) {
        reserve_step_ = reserve_step;
    }

    // 公开的状态和错误信息，GenerateStream 等外部代码直接访问
    // status 使用 atomic 保证线程安全：lifecycle methods 在 mutex_ 下写入，getStatus() 无锁读取
    std::atomic<StreamState> status = StreamState::WAITING;
    ErrorInfo                error_info;

private:
    StreamEvents events_;

    std::shared_ptr<StreamCacheResource> stream_cache_resource_ = nullptr;
    size_t                               reserve_step_          = 0;
};

}  // namespace rtp_llm
