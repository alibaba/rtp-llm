#pragma once
#include "Subscriber.h"

namespace rtp_llm {
namespace remote_connector {

class DirectSubscriber: public Subscriber {
public:
    virtual bool init(const std::vector<std::string>& domains) override {
        addresses_ = domains;
        return true;
    }
    virtual bool getAddresses(std::vector<std::string>& addresses) const override {
        addresses = addresses_;
        return true;
    }

private:
    std::vector<std::string> addresses_;
};

}  // namespace remote_connector
}  // namespace rtp_llm