#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionManager.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionPoolFactory.h"
#include "autil/ThreadPool.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ConnectionManager);

ConnectionManager::ConnectionManager() : _callBackThreadPool(nullptr) {
}

ConnectionManager::~ConnectionManager() {
    stop();
}

void ConnectionManager::stop() {
    for (size_t i = 0; i < MC_PROTOCOL_UNKNOWN; i++) {
        if (_poolVector[i]) {
            _poolVector[i]->stop();
            _poolVector[i].reset();
        }
    }
    if (_callBackThreadPool) {
        _callBackThreadPool->stop();
        DELETE_AND_SET_NULL(_callBackThreadPool);
    }
}

bool ConnectionManager::init(const ConnectionConfig &config, const MiscConfig &miscConf) {
    if (config.empty()) {
        AUTIL_LOG(WARN, "no connection type configured");
        return true;
    }
    auto callbackThreadNum = miscConf.callbackThreadNum;
    if (callbackThreadNum > 0) {
        _callBackThreadPool = new autil::LockFreeThreadPool(
            miscConf.callbackThreadNum, 1000, autil::WorkItemQueueFactoryPtr(), "GigConnMgrCB");
        if (!_callBackThreadPool->start()) {
            AUTIL_LOG(ERROR, "start callback thread pool failed");
            return false;
        }
    }
    for (const auto &conf : config) {
        const auto &typeStr = conf.first;
        auto type = convertProtocolType(typeStr);
        if (MC_PROTOCOL_UNKNOWN == type) {
            AUTIL_LOG(ERROR, "unknown config type [%s]", typeStr.c_str());
            return false;
        }
        if (_poolVector[type]) {
            AUTIL_LOG(ERROR, "duplicate config for type [%s]", typeStr.c_str());
            return false;
        }
        auto connectionPool = createConnectionPool(type);
        if (!connectionPool) {
            return false;
        }
        if (!connectionPool->init(conf.second)) {
            return false;
        }
        _poolVector[type] = connectionPool;
    }
    return true;
}

ConnectionPoolPtr ConnectionManager::createConnectionPool(ProtocolType type) {
    return ConnectionPoolFactory::createConnectionPool(type);
}

ConnectionPtr ConnectionManager::getConnection(const std::string &spec, ProtocolType type) {
    const auto &connectionPool = getConnectionPool(type);
    if (!connectionPool) {
        return ConnectionPtr();
    }
    auto con = connectionPool->getConnection(spec);
    if (con) {
        con->setCallBackThreadPool(_callBackThreadPool);
    }
    return con;
}

void ConnectionManager::syncConnection(const set<string> &keepSpecs) {
    
    
    
    
    if (_poolVector[MC_PROTOCOL_GRPC_STREAM]) {
        _poolVector[MC_PROTOCOL_GRPC_STREAM]->syncConnection(keepSpecs);
    }
    
    
}

bool ConnectionManager::supportProtocol(ProtocolType type) const {
    return _poolVector[type] != nullptr;
}

ConnectionPoolPtr ConnectionManager::getConnectionPool(ProtocolType type) {
    if (type >= MC_PROTOCOL_UNKNOWN) {
        AUTIL_LOG(ERROR, "invalid type [%u]", type);
        return ConnectionPoolPtr();
    }
    auto &connectionPool = _poolVector[type];
    if (!connectionPool) {
        AUTIL_LOG(ERROR, "type [%u] not enabled", type);
        return ConnectionPoolPtr();
    }
    return connectionPool;
}

} // namespace multi_call
