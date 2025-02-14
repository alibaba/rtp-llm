#pragma once
#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/MasterInitParameter.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/MasterHttpServer.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/tokenize/RemoteTokenizeModule.h"
namespace rtp_llm {
namespace rtp_llm_master {

class RtpLLMMasterEntry {
public:
    RtpLLMMasterEntry();
    ~RtpLLMMasterEntry();
    bool init(const MasterInitParameter& param);
protected:
    bool initLoadBalancer(const MasterInitParameter& param);
    bool initHttpServer();    
    bool initTokenizeService();
    LoadBalancerInitParams createLoadBalancerInitParams(const MasterInitParameter& param);
    EstimatorConfig createEstimatorConfig(const PyEstimatorConfig& py_config);    

private:
    MasterInitParameter param_;
    std::shared_ptr<PrefillLoadBalancer> load_balancer_;
    std::shared_ptr<RemoteTokenizeModule> tokenize_service_;
    std::shared_ptr<MasterHttpServer> http_server_;
    std::string biz_name_;
};

void registerRtpLLMMasterEntry(py::module m);

}  // namespace rtp_llm_master
}  // namespace rtp_llm
