#pragma once

#include <functional>
#include <memory>
#include <string_view>

namespace rtp_llm::media {

class ContentDecoder {
public:
    using Checkpoint = std::function<void()>;
    using Sink       = std::function<void(std::string_view)>;
    // Null means urllib3 would pass the body through unchanged.
    static std::unique_ptr<ContentDecoder>
    create(std::string_view header, Sink output, Checkpoint checkpoint, size_t byte_limit);
    virtual ~ContentDecoder()                  = default;
    virtual void write(std::string_view input) = 0;
    virtual void finish()                      = 0;
};
}  // namespace rtp_llm::media
