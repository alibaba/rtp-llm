#include "rtp_llm/cpp/disaggregate/rtpllm_master/tokenize/RemoteTokenizeModule.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/common/UserRequest.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <exception>

namespace rtp_llm {
namespace rtp_llm_master {

std::string TOKENIZE_PATH = "/tokenize";

bool RemoteTokenizeModule::init(std::shared_ptr<PrefillLoadBalancer> load_balancer) {
    if (!load_balancer) {
        RTP_LLM_LOG_WARNING("load balancer is null, RemoteTokenizeModule init failed");
        return false;
    }
    load_balancer_ = load_balancer;
    http_client_   = std::make_shared<http_server::SimpleHttpClient>();
    return true;
}

//TODO: request timeout
absl::StatusOr<std::shared_ptr<TaskDescription>> RemoteTokenizeModule::encodeRequest(const std::string& request,
                                                                                const std::string& biz_name) {
    if (!http_client_) {
        return absl::InternalError("http client is null, encodeRequest failed");
    }
    if (!load_balancer_) {
        return absl::InternalError("load balancer is null, encodeRequest failed");
    }
    auto host = load_balancer_->getRandomHost(biz_name);
    if (!host) {
        auto err = autil::StringUtil::formatString("choose host failed, biz:%s", biz_name.c_str());
        return absl::InternalError(err);
    }
    const std::string url = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
    bool                             success          = false;
    std::string                      error_msg;
    std::shared_ptr<TaskDescription> task_description = std::make_shared<TaskDescription>();
    autil::Notifier                  notifier;
    http_server::HttpCallBack        http_call_back =
        [this, &success, &notifier, &task_description, &error_msg](bool ok, const std::string& response_body) {
            if (!ok) {
                error_msg = response_body;
                success = false;
                notifier.notify();
                return;
            }
            TokenizeResponse tokenize_response;
            try {
                autil::legacy::FromJsonString(tokenize_response, response_body);
            } catch (const std::exception& e) {
                error_msg = e.what();
                success = false;
                notifier.notify();
                return;
            }
            task_description->input_length  = tokenize_response.token_ids.size();
            task_description->prefix_length = 0;
            std::swap(task_description->token_ids, tokenize_response.token_ids);
            success = true;
            notifier.notify();
        };
    if (!http_client_->post(url, TOKENIZE_PATH, request, std::move(http_call_back))) {
        return absl::InternalError(autil::StringUtil::formatString("http post request failed, url: %s", url.c_str()));
    }
    notifier.waitNotification();
    if (!success) {
        return absl::InternalError(autil::StringUtil::formatString("http post request get error, url: %s, error msg: %s", url.c_str(), error_msg.c_str()));
    }
    return task_description;
#undef CREATE_ERROR_MSG
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm