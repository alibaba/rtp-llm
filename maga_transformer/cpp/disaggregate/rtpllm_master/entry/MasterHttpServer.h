#pragma once
#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "maga_transformer/cpp/api_server/common/HealthService.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/entry/RandomRequestIdGenerator.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/tokenize/RemoteTokenizeModule.h"

namespace rtp_llm {
namespace rtp_llm_master {

class MasterHttpServer {
public:
    MasterHttpServer(std::shared_ptr<RemoteTokenizeModule> tokenize_service,
                     std::shared_ptr<PrefillLoadBalancer>  load_balancer,
                     const std::string&                    biz_name,
                     int64_t                               port):
        tokenize_service_(tokenize_service),
        load_balancer_(load_balancer),
        port_(port),
        address_("tcp:127.0.0.1:" + std::to_string(port_)),
        biz_name_(biz_name) {
        request_id_counter_.store(autil::TimeUtility::currentTimeInMilliSeconds());
        FT_LOG_INFO("set request_id_counter_ init value to %ld", request_id_counter_.load());
    }
    ~MasterHttpServer();
    bool start();
    void stop();

protected:
    bool isStoped() const;
    bool registerServices();
    bool registerHealthService();
    bool registerHandleService();
    void handleRequest(const http_server::HttpRequest&                  request,
                       std::unique_ptr<http_server::HttpResponseWriter> writer);
    void handleError(const http_server::HttpRequest&                  request,
                     std::unique_ptr<http_server::HttpResponseWriter> writer,
                     const std::string&                               error_msg);

private:
    std::atomic<int64_t>                     request_id_counter_{0};
    std::atomic_bool                         is_stopped_{true};
    std::shared_ptr<HealthService>           health_service_;    
    std::unique_ptr<http_server::HttpServer> http_server_;
    std::shared_ptr<RemoteTokenizeModule>    tokenize_service_;
    std::shared_ptr<PrefillLoadBalancer>     load_balancer_;
    int64_t                                  port_;
    std::string                              address_;
    std::string                              biz_name_;
    RandomStringGenerator                    generator_;
};

}  // namespace rtp_llm_master

}  // namespace rtp_llm
