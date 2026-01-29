#pragma once

#include <utility>

namespace rtp_llm {

class Meta {
public:
    virtual ~Meta() = default;

public:
    virtual bool enableMemoryCache() const = 0;
};

}  // namespace rtp_llm
