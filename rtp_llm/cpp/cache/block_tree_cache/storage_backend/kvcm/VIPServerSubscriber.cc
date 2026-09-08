#include "VIPServerSubscriber.h"
#include "rtp_llm/cpp/utils/Logger.h"
#if defined(KVCM_INTERNAL) || defined(RECO_INTERNAL)
#include "autil/EnvUtil.h"
#include "option.h"
#include "iphost.h"
#include "vipclient.h"
using namespace middleware::vipclient;
#endif

namespace rtp_llm {
namespace kvcm {

#if defined(KVCM_INTERNAL) || defined(RECO_INTERNAL)
namespace {

// The process owns the VIP API. Subscriber destruction must never tear down
// global state while another wrapper is discovering addresses.
class VIPServerApi {
public:
    static VIPServerApi& instance() {
        static VIPServerApi api;
        return api;
    }

    bool init() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (inited_) {
            return true;
        }
        const auto domain = autil::EnvUtil::getEnv(
            "KVCM_VIP_JMENV", autil::EnvUtil::getEnv("RECO_VIP_JMENV", std::string("jmenv.tbsite.net")));
        VipClientApi::CreateApi();
        Option option;
        option.set_failover_path(".");
        option.set_log_path(".");
        option.set_cache_path(".");
        if (!VipClientApi::Init(domain.c_str(), option)) {
            RTP_LLM_LOG_ERROR("VIPServer initialization failed: %s", strerror(errno));
            VipClientApi::DestoryApi();
            return false;
        }
        inited_ = true;
        return true;
    }

    ~VIPServerApi() {
        if (inited_) {
            VipClientApi::UnInit();
            VipClientApi::DestoryApi();
        }
    }

private:
    VIPServerApi() = default;
    std::mutex mutex_;
    bool       inited_ = false;
};

}  // namespace
#endif

bool VIPServerSubscriber::init(const std::vector<std::string>& domains) {
#if defined(KVCM_INTERNAL) || defined(RECO_INTERNAL)
    if (!VIPServerApi::instance().init()) {
        return false;
    }
    domains_ = domains;
    return true;
#else
    RTP_LLM_LOG_ERROR("not support vipserver");
    return false;
#endif
}

bool VIPServerSubscriber::getAddresses(std::vector<std::string>& addresses) const {
#if defined(KVCM_INTERNAL) || defined(RECO_INTERNAL)
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

}  // namespace kvcm
}  // namespace rtp_llm
