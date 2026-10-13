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

}  // namespace rtp_llm::legacy::p2p

namespace rtp_llm {

class AsyncMatchContext: public ::rtp_llm::legacy::p2p::AsyncContext {
public:
    ~AsyncMatchContext() override = default;

    virtual size_t matchedBlockCount() const = 0;
};

}  // namespace rtp_llm
