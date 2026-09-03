#pragma once

#include <memory>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm::legacy::p2p {

class AsyncContext {
public:
    virtual ~AsyncContext() = default;

    virtual void      waitDone()      = 0;
    virtual bool      done() const    = 0;
    virtual bool      success() const = 0;
    virtual ErrorInfo errorInfo() const {
        return ErrorInfo::OkStatus();
    }
};

class AsyncMatchContext: public AsyncContext {
public:
    ~AsyncMatchContext() override = default;

    virtual size_t matchedBlockCount() const = 0;
};

}  // namespace rtp_llm::legacy::p2p
