#pragma once

#include <functional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class TransferAsyncContext {
public:
    using DoneCallback = std::function<void(TransferResult)>;

    virtual ~TransferAsyncContext()                      = default;
    virtual void           waitDone()                    = 0;
    virtual void           onDone(DoneCallback callback) = 0;
    virtual bool           done() const                  = 0;
    virtual bool           success() const               = 0;
    virtual TransferResult result() const                = 0;

    ErrorInfo errorInfo() const {
        return result().status;
    }
};

class CompletedTransferAsyncContext final: public TransferAsyncContext {
public:
    explicit CompletedTransferAsyncContext(TransferResult result, size_t block_count = 0): result_(std::move(result)) {
        result_.expandBlockErrors(block_count);
    }

    void waitDone() override {}
    void onDone(DoneCallback callback) override {
        if (callback) {
            callback(result_);
        }
    }
    bool done() const override {
        return true;
    }
    bool success() const override {
        return result_.ok();
    }
    TransferResult result() const override {
        return result_;
    }

private:
    TransferResult result_;
};

}  // namespace rtp_llm
