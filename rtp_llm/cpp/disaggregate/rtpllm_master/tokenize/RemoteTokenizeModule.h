#pragma once
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/common/TaskDescription.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"

namespace rtp_llm {
namespace rtp_llm_master {

class RemoteTokenizeModule {
public:
    RemoteTokenizeModule() = default;
    // maybe should use seperate subscribe config;
    bool                                             init(std::shared_ptr<PrefillLoadBalancer> load_balancer);
    absl::StatusOr<std::shared_ptr<TaskDescription>> encodeRequest(const std::string& request,
                                                                   const std::string& biz_name);

private:
    std::shared_ptr<PrefillLoadBalancer>           load_balancer_;
    std::shared_ptr<http_server::SimpleHttpClient> http_client_;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm
