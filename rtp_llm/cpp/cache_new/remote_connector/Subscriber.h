#pragma once

#include <vector>
#include <string>

namespace rtp_llm {
namespace remote_connector {

class Subscriber {
public:
    virtual ~Subscriber()                                                = default;
    virtual bool init(const std::vector<std::string>& domains)           = 0;
    virtual bool getAddresses(std::vector<std::string>& addresses) const = 0;
};

}  // namespace remote_connector
}  // namespace rtp_llm