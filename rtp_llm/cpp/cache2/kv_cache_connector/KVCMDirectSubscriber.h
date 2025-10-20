#pragma once
#include "KVCMSubscriber.h"

namespace rtp_llm {
class KVCMDirectSubscriber: public KVCMSubscriber {
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
}  // namespace rtp_llm