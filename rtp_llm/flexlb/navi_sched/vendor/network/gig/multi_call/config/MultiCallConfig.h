#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/legacy/jsonizable.h"

namespace multi_call {

struct TopoNode;

class VipConfig : public autil::legacy::Jsonizable
{
public:
    VipConfig() : grpcPortDiff(0), supportHeartbeat(false), queryFullApi(false) {
    }
    ~VipConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool validate(bool &enabled) const;
    bool operator==(const VipConfig &rhs) const;

public:
    std::string jmenvDom;
    std::vector<std::string> clusters;
    std::string serviceKey;
    int32_t grpcPortDiff;
    bool supportHeartbeat;
    bool queryFullApi;

private:
    AUTIL_LOG_DECLARE();
};

class SubscribeConfig : public autil::legacy::Jsonizable
{
public:
    SubscribeConfig() : allowEmptySub(true) {
    }
    ~SubscribeConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    // virtual for ut
    virtual bool validate(bool &vipEnabled) const;
    bool operator==(const SubscribeConfig &rhs) const;

public:
    VipConfig vipConfig; // deprecated
    bool allowEmptySub;

private:
    AUTIL_LOG_DECLARE();
};

class SecureConfig : public autil::legacy::Jsonizable
{
public:
    SecureConfig() {
    }
    ~SecureConfig() {
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const SecureConfig &rhs) const;

public:
    std::string pemRootCerts;
    std::string pemPrivateKey;
    std::string pemCertChain;
    std::string targetName;
};

struct GigRpcServerConfig : public autil::legacy::Jsonizable {
    GigRpcServerConfig()
        : heartbeatThreadNum(5) // default
        , heartbeatQueueSize(1000)
        , enableAgentStat(true) {
    }

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const GigRpcServerConfig &rhs) const;

    size_t heartbeatThreadNum;
    size_t heartbeatQueueSize;
    std::string logPrefix;
    std::string snapshotLogDir;
    bool enableAgentStat;
};

struct GrpcServerDescription {
    bool valid() const {
        // todo
        return ioThreadNum > 0;
    }
    size_t ioThreadNum;
    std::string port;
    SecureConfig secureConfig;
};

class ArpcServerDescription
{
public:
    ArpcServerDescription()
        : threadNum(1)
        , queueSize(256)
        , ioThreadNum(1)
        , enableArpcMetric(false)
        , enableANetMetric(false) {
    }

    std::string name;
    std::string port;
    uint32_t threadNum;
    uint32_t queueSize;
    uint32_t ioThreadNum;
    bool enableArpcMetric;
    bool enableANetMetric;
};

class HttpArpcServerDescription
{
public:
    HttpArpcServerDescription()
        : threadNum(8)
        , queueSize(10000)
        , ioThreadNum(5)
        , decodeUri(false)
        , haCompatible(false)
        , passThrough(false) {
    }

    std::string name;
    std::string port;
    uint32_t threadNum;
    uint32_t queueSize;
    uint32_t ioThreadNum;
    bool decodeUri;
    bool haCompatible;
    bool passThrough;
};

class ProtocolConfig : public autil::legacy::Jsonizable
{
public:
    ProtocolConfig()
        : threadNum(DEFAULT_THREAD_NUM)
        , queueSize(DEFAULT_QUEUE_SIZE)
        , queueTimeout(RPC_CONNECTION_TIMEOUT)
        , queueConsumeTimeout(DEFAULT_QUEUE_CONSUME_TIMEOUT_US)
        , anetLoopTimeout(DEFAULT_TIMEOUT_LOOP_INTERVAL) {
    }
    ~ProtocolConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const ProtocolConfig &rhs) const;

public:
    uint32_t threadNum;
    uint32_t queueSize;
    uint32_t queueTimeout;
    int64_t queueConsumeTimeout;
    int64_t anetLoopTimeout;
    int64_t keepAliveInterval = 4000; // ms
    int64_t keepAliveTimeout = 2000;  // ms
    SecureConfig secureConfig;

private:
    AUTIL_LOG_DECLARE();
};

typedef std::map<std::string, ProtocolConfig> ConnectionConfig;

class MiscConfig : public autil::legacy::Jsonizable
{
public:
    MiscConfig()
        : snapshotLogCount(2000)
        , snapshotInterval(0)
        , virtualNodeCount(3000)
        , metricReportSamplingRate(0)
        , heartbeatTimeout(DEFAULT_TIMEOUT)
        , heartbeatThreadCount(DEFAULT_HEARTBEAT_THREAD_NUM)
        , heartbeatUnhealthInterval(DEFAULT_HEARTBEAT_UNHEALTH_INTERVAL)
        , callbackThreadNum(DEFAULT_CALLBACK_THREAD_NUM)
        , callProcessInterval(PROCESS_INTERVAL)
        , grpcChannelPollIntervalMs(DEFAULT_GRPC_CHANNEL_POLL_INTERVAL_MS)
        , weightUpdateStep(MIN_WEIGHT_UPDATE_STEP_FLOAT)
        , enableHeartbeat(false)
        , checkNeedUpdateSnapshot(false)
        , disableBizNotExistLog(false)
        , disableMetricReport(false)
        , simplifyMetricReport(false)
        , updateTimeInterval(UPDATE_TIME_INTERVAL)
        , callDelegationQueueSize(CALL_DELEGATION_QUEUE_SIZE) {
    }
    ~MiscConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const MiscConfig &rhs) const;
    bool enableConsistentHash() const;
    HashPolicy getHashPolicy() const;

public:
    size_t snapshotLogCount;
    size_t snapshotInterval;
    size_t virtualNodeCount;
    int32_t metricReportSamplingRate;
    int32_t heartbeatTimeout;
    int32_t heartbeatThreadCount;
    int32_t heartbeatUnhealthInterval;
    int32_t callbackThreadNum;
    int64_t callProcessInterval;
    int64_t grpcChannelPollIntervalMs;
    float weightUpdateStep;
    std::string logPrefix;
    std::string snapshotLogDir;
    bool enableHeartbeat;
    bool checkNeedUpdateSnapshot;
    bool disableBizNotExistLog;
    bool disableMetricReport;
    bool simplifyMetricReport;
    int64_t updateTimeInterval;
    std::string hashPolicy;
    size_t callDelegationQueueSize;
    int64_t cooldownAfterResetUs = 0;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(MiscConfig);

class MultiCallConfig : public autil::legacy::Jsonizable
{
public:
    MultiCallConfig() {
    }
    ~MultiCallConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json);
    bool operator==(const MultiCallConfig &rhs) const;

public:
    SubscribeConfig subConf;
    ConnectionConfig connectConf;
    MiscConfig miscConf;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(MultiCallConfig);

} // namespace multi_call
