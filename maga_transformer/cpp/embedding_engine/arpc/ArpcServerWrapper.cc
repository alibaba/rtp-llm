#include "maga_transformer/cpp/embedding_engine/arpc/ArpcServerWrapper.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void ArpcServerWrapper::start() {
    arpc_server_transport_.reset(new anet::Transport(2, anet::SHARE_THREAD));
    arpc_server_.reset(new arpc::ANetRPCServer(arpc_server_transport_.get(), 4, 10));
    FT_CHECK_WITH_INFO(arpc_server_transport_->start(), "arpc server start public transport failed");
    arpc_server_transport_->setName("ARPC SERVER");
    std::string spec("tcp:0.0.0.0:" + std::to_string(port_));
    FT_CHECK_WITH_INFO(arpc_server_->Listen(spec), "arpc listen on %s failed", spec.c_str());
    // auto wrapper = std::make_shared<ServiceWrapper>(*this, rpcService, compatibleInfo);
    arpc_server_->RegisterService(service_.get());
    FT_LOG_INFO("ARPC Server listening on %s", spec.c_str());
}

void ArpcServerWrapper::stop() {
    if (arpc_server_) {
        arpc_server_->Close();
        arpc_server_->StopPrivateTransport();
    }
    if (arpc_server_transport_) {
        FT_CHECK_WITH_INFO(arpc_server_transport_->stop(), "transport stop failed");
        FT_CHECK_WITH_INFO(arpc_server_transport_->wait(), "transport wait failed");
    }
    FT_LOG_INFO("ARPC Server stopped");
}

} // namespace rtp_llm