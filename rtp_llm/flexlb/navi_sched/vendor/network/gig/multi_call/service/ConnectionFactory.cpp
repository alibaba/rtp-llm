#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionFactory.h"

#include <grpc++/generic/generic_stub.h>

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ConnectionFactory);

void GrpcConnectionFactory::initChannelArgs(GrpcChannelInitParams params) {
    _channelArgs.reset(new grpc::ChannelArguments());
    _channelArgs->SetMaxReceiveMessageSize(-1);
    _channelArgs->SetMaxSendMessageSize(-1);
    if (params.keepAliveInterval > 0) {
        // ref https://github.com/grpc/grpc/blob/master/doc/keepalive.md
        AUTIL_LOG(INFO, "enable grpc keep-alive with interval[%ld] timeout[%ld]",
                  params.keepAliveInterval, params.keepAliveTimeout);
        _channelArgs->SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, params.keepAliveInterval);
        _channelArgs->SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, params.keepAliveTimeout);
    }
    if (_secureConfig.pemRootCerts.empty()) {
        _channelCredentials = grpc::InsecureChannelCredentials();
    } else {
        _channelCredentials = grpc::SslCredentials(getSslOptions());
        _channelArgs->SetSslTargetNameOverride(_secureConfig.targetName);
    }
}

grpc::SslCredentialsOptions GrpcConnectionFactory::getSslOptions() const {
    grpc::SslCredentialsOptions option;
    option.pem_root_certs = _secureConfig.pemRootCerts;
    option.pem_private_key = _secureConfig.pemPrivateKey;
    option.pem_cert_chain = _secureConfig.pemCertChain;
    return option;
}

} // namespace multi_call
