#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class AsyncContext {
public:
    using DoneCallback = std::function<void(ErrorInfo)>;

    AsyncContext()          = default;
    virtual ~AsyncContext() = default;

public:
    virtual void      waitDone()      = 0;
    virtual void      onDone(DoneCallback callback) = 0;
    virtual bool      done() const    = 0;
    virtual bool      success() const = 0;
    virtual ErrorInfo errorInfo() const {
        return ErrorInfo::OkStatus();
    }
};

// Immutable context for an already-completed operation.
class CompletedAsyncContext final: public AsyncContext {
public:
    explicit CompletedAsyncContext(ErrorInfo error_info);
    ~CompletedAsyncContext() override = default;

    void      waitDone() override;
    void      onDone(DoneCallback callback) override;
    bool      done() const override;
    bool      success() const override;
    ErrorInfo errorInfo() const override;

private:
    ErrorInfo error_info_;
};

class FusedAsyncContext: public AsyncContext {
public:
    FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts);
    ~FusedAsyncContext() override = default;

public:
    void      waitDone() override;
    void      onDone(DoneCallback callback) override;
    bool      done() const override;
    bool      success() const override;
    ErrorInfo errorInfo() const override;

    const std::vector<std::shared_ptr<AsyncContext>>& contexts() const {
        return contexts_;
    }

private:
    std::vector<std::shared_ptr<AsyncContext>> contexts_;
};

}  // namespace rtp_llm
