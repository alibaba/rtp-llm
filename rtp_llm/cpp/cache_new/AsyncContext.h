#pragma once

namespace rtp_llm {

class AsyncContext {
public:
    AsyncContext()          = default;
    virtual ~AsyncContext() = default;

public:
    virtual void waitDone()      = 0;
    virtual void cancel()        = 0;
    virtual bool done() const    = 0;
    virtual bool success() const = 0;
};

}  // namespace rtp_llm