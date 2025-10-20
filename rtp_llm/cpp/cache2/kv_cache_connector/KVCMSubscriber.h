#pragma once

#include <vector>
#include <string>

namespace rtp_llm {
class KVCMSubscriber {
public:
    virtual ~KVCMSubscriber()                                            = default;
    virtual bool init(const std::vector<std::string>& domains)           = 0;
    virtual bool getAddresses(std::vector<std::string>& addresses) const = 0;
};
}  // namespace rtp_llm