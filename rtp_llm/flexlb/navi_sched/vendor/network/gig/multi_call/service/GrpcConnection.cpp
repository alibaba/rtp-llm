#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcConnection.h"

#include <grpc++/generic/generic_stub.h>
#include <unistd.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcConnection);



GrpcConnection::GrpcConnection(const GrpcClientWorkerPtr &grpcWorker, const std::string &spec,
                               const std::shared_ptr<grpc::ChannelCredentials> &channelCredentials,
                               const std::shared_ptr<grpc::ChannelArguments> &channelArgs,
                               ProtocolType type)
    : Connection(spec, type, 0, 0, 0)
    , _grpcWorker(grpcWorker)
    , _channelCredentials(channelCredentials)
    , _channelArgs(channelArgs) {
}

GrpcConnection::~GrpcConnection() {
}

GrpcChannelPtr GrpcConnection::getChannel() {
    {
        ScopedReadWriteLock lock(_channelLock, 'r');
        if (_channel) {
            return _channel;
        }
    }
    {
        ScopedReadWriteLock lock(_channelLock, 'w');
        if (_channel) {
            return _channel;
        }
        if (!_channelArgs) {
            _channel = grpc::CreateChannel(_spec, _channelCredentials);
        } else {
            _channel = grpc::CreateCustomChannel(_spec, _channelCredentials, *_channelArgs);
        }
        return _channel;
    }
}



} // namespace multi_call
