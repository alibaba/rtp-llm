#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionPoolFactory.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcStreamConnectionPool.h"


using namespace std;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, ConnectionPoolFactory);

ConnectionPoolPtr ConnectionPoolFactory::createConnectionPool(ProtocolType type) {
    switch (type) {
    case MC_PROTOCOL_GRPC_STREAM:
        return ConnectionPoolPtr(new GrpcStreamConnectionPool());
    default:
        AUTIL_LOG(ERROR, "create ConnectionPool failed, unknown type [%u]", type);
        return ConnectionPoolPtr();
    }
}

} // namespace multi_call
