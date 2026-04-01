#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

class IGenerateStream;

class Meta {
public:
    virtual ~Meta() = default;

public:
    virtual bool                        enableMemoryCache() const = 0;
    virtual bool                        enableRemoteCache() const = 0;
    virtual const std::string&          trace_id() const          = 0;
    virtual const std::string&          unique_id() const         = 0;
    virtual const std::vector<int64_t>& tokens() const            = 0;

    // P2P read 扩展：非纯虚，非P2P场景下默认返回安全值，不影响已有实现
    virtual std::shared_ptr<IGenerateStream> generateStream() const {
        return nullptr;
    }
};

}  // namespace rtp_llm
