#pragma once

#include <memory>
#include <string>
#include "autil/TimeUtility.h"
#include "maga_transformer/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {

class LoadStatus {
public:
    enum class State {
        LOADING,
        SUCCESS,
        FAILED
    };
    LoadStatus(int load_timeout_ms, grpc::ServerContext* server_context) 
            : load_timeout_ms(load_timeout_ms), server_context(server_context) {}

    void waitDone() {
        std::unique_lock<std::mutex> lock(mutex);
        auto once_time_ms = 30;
        auto start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        while (true) {
            auto cost_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us) / 1000;
            if (cost_time_ms >= load_timeout_ms) {
                auto error_code = ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
                error_info = ErrorInfo(error_code, ErrorCodeToString(error_code));
                break;
            }
            if (server_context->IsCancelled()) {
                auto error_code = ErrorCode::CANCELLED;
                error_info = ErrorInfo(error_code, ErrorCodeToString(error_code));
                break;
            }
            if (cond.wait_for(lock, std::chrono::milliseconds(once_time_ms),
                        [this] { return state != State::LOADING; })) {
                break;
            }
        }
    }

    void updateResult(bool success, CacheStoreErrorCode ec) {
        std::lock_guard<std::mutex> lock(mutex);
        if (success) {
            state = State::SUCCESS;
        } else {
            state = State::FAILED;
        }
        auto error_code = transCacheStoreErrorCode(ec);
        error_info = ErrorInfo(error_code, ErrorCodeToString(error_code));
        cond.notify_all();
    }

    bool ok() {
        std::unique_lock<std::mutex> lock(mutex);
        return error_info.ok();
    }

    std::string toString() {
        std::unique_lock<std::mutex> lock(mutex);
        return error_info.ToString();   
    }

    const ErrorInfo& errorInfo() const {
        return error_info;
    }

public:
    State                   state = State::LOADING;
    ErrorInfo               error_info;
    std::condition_variable cond;
    std::mutex              mutex;
    int                     load_timeout_ms;
    grpc::ServerContext*    server_context;
};

}  // namespace rtp_llm
