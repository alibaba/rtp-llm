#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcStreamConnectionPool.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionFactory.h"

using namespace std;
using namespace autil::legacy;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcStreamConnectionPool);

GrpcStreamConnectionPool::GrpcStreamConnectionPool() {
}

GrpcStreamConnectionPool::~GrpcStreamConnectionPool() {
}

void GrpcStreamConnectionPool::stop() {
    if (_worker) {
        _worker->stop();
    }
    _worker.reset();
    ConnectionPool::stop();
    AUTIL_LOG(INFO, "GrpcStreamConnectionPool[%p] stopped", this);
}

bool GrpcStreamConnectionPool::init(const ProtocolConfig &config) {
    AUTIL_LOG(INFO, "GrpcStreamConnectionPool init, config [%s]", ToJsonString(config).c_str());
    _worker.reset(new GrpcClientWorker(config.threadNum));
    auto factory = new GrpcStreamConnectionFactory(_worker, config.secureConfig);
    GrpcChannelInitParams params;
    params.keepAliveInterval = config.keepAliveInterval;
    params.keepAliveTimeout = config.keepAliveTimeout;
    factory->initChannelArgs(params);
    _connectionFactory.reset(factory);
    return true;
}

} // namespace multi_call
