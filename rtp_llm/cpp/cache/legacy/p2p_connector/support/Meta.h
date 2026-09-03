#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
class GenerateStream;
}

namespace rtp_llm::legacy::p2p {

class Meta {
public:
    virtual ~Meta() = default;

    virtual bool                        enableMemoryCache() const = 0;
    virtual bool                        enableRemoteCache() const = 0;
    virtual const std::string&          trace_id() const          = 0;
    virtual const std::string&          unique_id() const         = 0;
    virtual const std::vector<int64_t>& tokens() const            = 0;

    virtual ::rtp_llm::GenerateStream* generateStream() const {
        return nullptr;
    }

    virtual void setStop(ErrorCode error_code, const std::string& error_msg) {}

    struct P2PRoutingContext {
        int64_t                          request_id = 0;
        std::string                      unique_key;
        int64_t                          deadline_ms = 0;
        std::pair<std::string, uint32_t> prefill_addr;
        int                              prefill_tp_size = 0;
    };

    virtual std::optional<P2PRoutingContext> p2pRouting() const {
        return std::nullopt;
    }
};

}  // namespace rtp_llm::legacy::p2p
