#pragma once

#include <string>

namespace rtp_llm {

// Transport seam used by KVCMPublisher. Production uses the built-in HTTP
// reporter; tests inject a deterministic implementation through this interface.
class KVCacheEventReporter {
public:
    virtual ~KVCacheEventReporter() = default;

    virtual bool post(const std::string& route, const std::string& request, std::string& response) noexcept = 0;
};

}  // namespace rtp_llm
