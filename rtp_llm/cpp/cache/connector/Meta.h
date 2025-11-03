#pragma once

#include <utility>
#include <vector>

namespace rtp_llm {

class Meta {
public:
    virtual ~Meta() = default;

public:
    virtual bool                        enableMemoryCache() const = 0;
    virtual bool                        enableRemoteCache() const = 0;
    virtual const std::string&          trace_id() const          = 0;
    virtual const std::string&          unique_id() const         = 0;
    virtual const std::vector<int64_t>& tokens() const            = 0;
};

}  // namespace rtp_llm
