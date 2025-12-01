#include "VIPServerSubscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#ifdef KVCM_INTERNAL
#include "autil/EnvUtil.h"
#include "option.h"
#include "iphost.h"
#include "vipclient.h"
using namespace middleware::vipclient;
#endif

namespace rtp_llm {
namespace remote_connector {

#ifdef KVCM_INTERNAL
class VIPServerSubscriber::VIPServerDestructor {
public:
    ~VIPServerDestructor() {
        if (inited_) {
            VipClientApi::UnInit();
            VipClientApi::DestoryApi();
            inited_ = true;
        }
    }
    void set_inited() {
        inited_ = true;
    }

    bool inited() const {
        return inited_;
    }

private:
    bool inited_ = false;
};

#endif
bool VIPServerSubscriber::init(const std::vector<std::string>& domains) {
#ifdef KVCM_INTERNAL
    std::unique_lock<std::mutex> lock(destructor_mutex_);
    if (destructor_ == nullptr) {
        destructor_ = std::make_shared<VIPServerDestructor>();
    }
    if (destructor_->inited()) {
        RTP_LLM_LOG_INFO("VIPServerSubscriber has been inited");
        return true;
    }
    jmenv_domain_ = autil::EnvUtil::getEnv("KVCM_VIP_JMENV", std::string("jmenv.tbsite.net"));
    VipClientApi::CreateApi();
    Option option;
    option.set_failover_path(".");
    option.set_log_path(".");
    option.set_cache_path(".");
    if (!VipClientApi::Init(jmenv_domain_.c_str(), option)) {
        RTP_LLM_LOG_ERROR("init failed jmenvDom: [%s], error: [%s]\n", jmenv_domain_.c_str(), strerror(errno));
        VipClientApi::DestoryApi();
        return false;
    }
    RTP_LLM_LOG_INFO("init sucess jmenvDom: [%s]", jmenv_domain_.c_str());
    domains_ = domains;
    destructor_->set_inited();
    return true;
#else
    RTP_LLM_LOG_ERROR("not support vipserver");
    return false;
#endif
}

bool VIPServerSubscriber::getAddresses(std::vector<std::string>& addresses) const {
#ifdef KVCM_INTERNAL
    addresses.clear();
    for (const auto& domain : domains_) {
        if (domain.empty()) {
            RTP_LLM_LOG_WARNING("domain empty");
            continue;
        }
        IPHostArray hosts;
        if (!VipClientApi::QueryAllIp(domain.c_str(), &hosts, 10 * 1000)) {
            RTP_LLM_LOG_WARNING("QueryAllIp failed, domain: [%s]", domain.c_str());
            continue;
        }
        for (unsigned int i = 0; i < hosts.size(); ++i) {
            const auto& host    = hosts.get(i);
            std::string address = std::string(host.ip()) + ":" + std::to_string(host.port());
            if (host.valid() && host.weight() > 0) {
                RTP_LLM_LOG_DEBUG("kvcm server address [%s] valid", address.c_str());
                addresses.push_back(std::move(address));
            } else {
                RTP_LLM_LOG_DEBUG("kvcm server address [%s] invalid", address.c_str());
            }
        }
    }
    if (addresses.empty()) {
        RTP_LLM_LOG_ERROR("not get any valid ip!");
        return false;
    }
    return true;
#else
    RTP_LLM_LOG_ERROR("not support vipserver");
    return false;
#endif
}

}  // namespace remote_connector
}  // namespace rtp_llm