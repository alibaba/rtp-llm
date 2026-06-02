#pragma once
#include <mutex>
#include <memory>
#include "Subscriber.h"

namespace rtp_llm {
namespace remote_connector {

class VIPServerSubscriber: public Subscriber {
public:
    ~VIPServerSubscriber() override = default;
    virtual bool init(const std::vector<std::string>& domains) override;
    virtual bool getAddresses(std::vector<std::string>& addresses) const override;

private:
    class VIPServerDestructor;
    std::mutex                           destructor_mutex_;
    std::shared_ptr<VIPServerDestructor> destructor_;
    std::string                          jmenv_domain_;
    std::vector<std::string>             domains_;
};

}  // namespace remote_connector
}  // namespace rtp_llm