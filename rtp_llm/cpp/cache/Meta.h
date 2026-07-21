#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Forward declaration for type-safe generateStream() return type
class GenerateStream;

class Meta {
public:
    virtual ~Meta() = default;

public:
    virtual bool                        enableMemoryCache() const = 0;
    virtual bool                        enableRemoteCache() const = 0;
    virtual const std::string&          trace_id() const          = 0;
    virtual const std::string&          unique_id() const         = 0;
    virtual const std::vector<int64_t>& tokens() const            = 0;

    // P2P read extension: returns GenerateStream pointer for type safety.
    // Non-P2P scenarios can return nullptr by default.
    virtual GenerateStream* generateStream() const {
        return nullptr;
    }

    // Set stream to stop with error code and message (delegates to GenerateStream::setStop)
    virtual void setStop(ErrorCode error_code, const std::string& error_msg) {
        // Default: no-op (non-P2P scenario)
    }

    // P2P routing context: encapsulates routing metadata extracted from GenerateStream
    // Constructed once during Meta creation, then read-only for P2P connector paths
    struct P2PRoutingContext {
        int64_t                          request_id = 0;
        std::string                      unique_key;
        int64_t                          deadline_ms = 0;
        std::pair<std::string, uint32_t> prefill_addr;  // {ip, port}
        int                              prefill_tp_size = 0;
    };

    // Returns P2P routing context if this is a P2P load Meta
    // Default: nullopt (non-P2P scenario)
    virtual std::optional<P2PRoutingContext> p2pRouting() const {
        return std::nullopt;
    }
};

}  // namespace rtp_llm
