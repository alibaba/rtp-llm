#pragma once

#include <string>

namespace rtp_llm {

// Transport seam used by KVCMPublisher. Production uses the built-in HTTP
// reporter; tests inject a deterministic implementation through this interface.
class KVCacheEventReporter {
public:
    virtual ~KVCacheEventReporter() = default;

    // Returns true only when the transport completed with a successful HTTP
    // status. KVCMPublisher owns response parsing and protocol recovery. A
    // snapshot upload can overlap heartbeat traffic, so implementations must
    // allow concurrent post() calls (or serialize them internally) without
    // sharing response state between calls.
    virtual bool post(const std::string& route, const std::string& request, std::string& response) noexcept = 0;

    // Reporters with interruptible I/O override this hook so publisher
    // shutdown never has to wait for an in-flight request timeout. It may be
    // called concurrently with post() and must be thread-safe and non-blocking.
    virtual void cancel() noexcept {}
};

}  // namespace rtp_llm
