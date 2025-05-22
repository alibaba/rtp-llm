#include "rtp_llm/cpp/embedding_engine/arpc/ArpcServerWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void ArpcServerWrapper::start() {
    arpc_server_transport_.reset(new anet::Transport(2, anet::SHARE_THREAD));
    arpc_server_.reset(new arpc::ANetRPCServer(arpc_server_transport_.get(), 10, 20));
    RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->start(), "arpc server start public transport failed");
    arpc_server_transport_->setName("ARPC SERVER");
    std::string spec("tcp:0.0.0.0:" + std::to_string(port_));
    RTP_LLM_CHECK_WITH_INFO(arpc_server_->Listen(spec), "arpc listen on %s failed", spec.c_str());
    // auto wrapper = std::make_shared<ServiceWrapper>(*this, rpcService, compatibleInfo);
    arpc_server_->RegisterService(service_.get());
    RTP_LLM_LOG_INFO("ARPC Server listening on %s", spec.c_str());
}

void ArpcServerWrapper::stop() {
    if (arpc_server_) {
        arpc_server_->Close();
        arpc_server_->StopPrivateTransport();
    }
    if (arpc_server_transport_) {
        RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->stop(), "transport stop failed");
        RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->wait(), "transport wait failed");
    }
    RTP_LLM_LOG_INFO("ARPC Server stopped");
}

} // namespace rtp_llm