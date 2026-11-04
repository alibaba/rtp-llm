#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "autil/StringUtil.h"

using namespace std;
using namespace autil;
using namespace autil::legacy;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, VipConfig);
AUTIL_LOG_SETUP(multi_call, SubscribeConfig);
AUTIL_LOG_SETUP(multi_call, ProtocolConfig);
AUTIL_LOG_SETUP(multi_call, MiscConfig);
AUTIL_LOG_SETUP(multi_call, MultiCallConfig);

void VipConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("jmenv_dom", jmenvDom, jmenvDom);
    json.Jsonize("services", clusters, clusters);
    json.Jsonize("service_key", serviceKey, serviceKey);
    json.Jsonize("grpc_port_diff", grpcPortDiff, grpcPortDiff);
    json.Jsonize("support_heartbeat", supportHeartbeat, supportHeartbeat);
    json.Jsonize("query_full_api", queryFullApi, queryFullApi);
}

bool VipConfig::operator==(const VipConfig &rhs) const {
    return jmenvDom == rhs.jmenvDom && clusters == rhs.clusters && serviceKey == rhs.serviceKey &&
           grpcPortDiff == rhs.grpcPortDiff && supportHeartbeat == rhs.supportHeartbeat &&
           queryFullApi == rhs.queryFullApi;
}

bool VipConfig::validate(bool &enabled) const {
    enabled = !jmenvDom.empty() || !clusters.empty();
    if (!enabled) {
        return true;
    }
    if (jmenvDom.empty()) {
        AUTIL_LOG(ERROR, "vip jmenvDom is empty");
        return false;
    }
    for (size_t i = 0; i < clusters.size(); i++) {
        if (clusters[i].empty()) {
            AUTIL_LOG(ERROR, "the %luth vip cluster name is empty", i);
            return false;
        }
    }
    return true;
}

void SubscribeConfig::Jsonize(Jsonizable::JsonWrapper &json) {
    json.Jsonize("vip", vipConfig, vipConfig);
    json.Jsonize("allow_empty_sub", allowEmptySub, allowEmptySub);
}

bool SubscribeConfig::operator==(const SubscribeConfig &rhs) const {
    return vipConfig == rhs.vipConfig && allowEmptySub == rhs.allowEmptySub;
}

bool SubscribeConfig::validate(bool &vipEnabled) const {
    return vipConfig.validate(vipEnabled);
}

void MultiCallConfig::Jsonize(Jsonizable::JsonWrapper &json) {
    json.Jsonize("subscribe_config", subConf, subConf);
    json.Jsonize("connection_config", connectConf, connectConf);
    json.Jsonize("misc_config", miscConf, miscConf);
}

bool MultiCallConfig::operator==(const MultiCallConfig &rhs) const {
    return subConf == rhs.subConf && connectConf == rhs.connectConf && miscConf == rhs.miscConf;
}

void SecureConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("pem_root_certs", pemRootCerts, pemRootCerts);
    json.Jsonize("pem_private_key", pemPrivateKey, pemPrivateKey);
    json.Jsonize("pem_cert_chain", pemCertChain, pemCertChain);
    json.Jsonize("target_name", targetName, targetName);
}

bool SecureConfig::operator==(const SecureConfig &rhs) const {
    return pemRootCerts == rhs.pemRootCerts && pemPrivateKey == rhs.pemPrivateKey &&
           pemCertChain == rhs.pemCertChain && targetName == rhs.targetName;
}

void ProtocolConfig::Jsonize(Jsonizable::JsonWrapper &json) {
    json.Jsonize("thread_num", threadNum, threadNum);
    json.Jsonize("queue_size", queueSize, queueSize);
    json.Jsonize("queue_timeout", queueTimeout, queueTimeout);
    json.Jsonize("queue_consume_timeout", queueConsumeTimeout, queueConsumeTimeout);
    json.Jsonize("anet_loop_timeout", anetLoopTimeout, anetLoopTimeout);
    json.Jsonize("keep_alive_interval", keepAliveInterval, keepAliveInterval);
    json.Jsonize("keep_alive_timeout", keepAliveTimeout, keepAliveTimeout);
    json.Jsonize("secure_config", secureConfig, secureConfig);
}

bool ProtocolConfig::operator==(const ProtocolConfig &rhs) const {
    return threadNum == rhs.threadNum && queueSize == rhs.queueSize &&
           queueTimeout == rhs.queueTimeout && queueConsumeTimeout == rhs.queueConsumeTimeout &&
           anetLoopTimeout == rhs.anetLoopTimeout && keepAliveInterval == rhs.keepAliveInterval &&
           keepAliveTimeout == rhs.keepAliveTimeout && secureConfig == rhs.secureConfig;
}

void MiscConfig::Jsonize(Jsonizable::JsonWrapper &json) {
    json.Jsonize("snapshot_log_count", snapshotLogCount, snapshotLogCount);
    json.Jsonize("snapshot_interval", snapshotInterval, snapshotInterval);
    json.Jsonize("consistent_hash_virtual_point", virtualNodeCount, virtualNodeCount);
    json.Jsonize("heartbeat_timeout", heartbeatTimeout, heartbeatTimeout);
    json.Jsonize("heartbeat_thread_num", heartbeatThreadCount, heartbeatThreadCount);
    json.Jsonize("callback_thread_num", callbackThreadNum, callbackThreadNum);
    json.Jsonize("call_process_interval", callProcessInterval, callProcessInterval);
    json.Jsonize("grpc_channel_poll_interval_ms", grpcChannelPollIntervalMs,
                 grpcChannelPollIntervalMs);
    json.Jsonize("weight_update_step", weightUpdateStep, weightUpdateStep);
    json.Jsonize("snapshot_log_prefix", logPrefix, logPrefix);
    json.Jsonize("snapshot_log_dir", snapshotLogDir, snapshotLogDir);
    json.Jsonize("enable_heartbeat", enableHeartbeat, enableHeartbeat);
    json.Jsonize("check_need_update_snapshot", checkNeedUpdateSnapshot, checkNeedUpdateSnapshot);
    json.Jsonize("disable_biz_not_exist_log", disableBizNotExistLog, disableBizNotExistLog);
    json.Jsonize("disable_metric_report", disableMetricReport, disableMetricReport);
    json.Jsonize("simplify_metric_report", simplifyMetricReport, simplifyMetricReport);
    json.Jsonize("metric_report_sampling_rate", metricReportSamplingRate, metricReportSamplingRate);
    json.Jsonize("manager_update_time_interval", updateTimeInterval, updateTimeInterval);
    json.Jsonize("hash_policy", hashPolicy, hashPolicy);
    json.Jsonize("call_delegation_queue_size", callDelegationQueueSize, callDelegationQueueSize);
    int64_t cooldownAfterResetMs = cooldownAfterResetUs / 1000;
    json.Jsonize("cooldown_after_reset_ms", cooldownAfterResetMs, cooldownAfterResetMs);
    cooldownAfterResetUs = cooldownAfterResetMs * 1000;
}

bool MiscConfig::operator==(const MiscConfig &rhs) const {
    return snapshotLogCount == rhs.snapshotLogCount && virtualNodeCount == rhs.virtualNodeCount &&
           heartbeatTimeout == rhs.heartbeatTimeout &&
           heartbeatThreadCount == rhs.heartbeatThreadCount &&
           callbackThreadNum == rhs.callbackThreadNum &&
           grpcChannelPollIntervalMs == rhs.grpcChannelPollIntervalMs &&
           weightUpdateStep == rhs.weightUpdateStep && logPrefix == rhs.logPrefix &&
           snapshotLogDir == rhs.snapshotLogDir && enableHeartbeat == rhs.enableHeartbeat &&
           checkNeedUpdateSnapshot == rhs.checkNeedUpdateSnapshot &&
           simplifyMetricReport == rhs.simplifyMetricReport &&
           metricReportSamplingRate == rhs.metricReportSamplingRate && hashPolicy == rhs.hashPolicy;
}

bool MiscConfig::enableConsistentHash() const {
    return virtualNodeCount != 0;
}

HashPolicy MiscConfig::getHashPolicy() const {
    if (virtualNodeCount != 0) {
        return HashPolicy::CONSISTENT_HASH;
    }
    if (hashPolicy == "consistent_hash") {
        return HashPolicy::CONSISTENT_HASH;
    } else if (hashPolicy == "random_hash") {
        return HashPolicy::RANDOM_HASH;
    }
    return HashPolicy::RANDOM_WEIGHTED_HASH;
}

void GigRpcServerConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("heartbeat_thread_num", heartbeatThreadNum, heartbeatThreadNum);
    json.Jsonize("heartbeat_queue_size", heartbeatQueueSize, heartbeatQueueSize);
    json.Jsonize("enable_agent_stat", enableAgentStat, enableAgentStat);
    json.Jsonize("snapshot_log_dir", snapshotLogDir, snapshotLogDir);
}

bool GigRpcServerConfig::operator==(const GigRpcServerConfig &rhs) const {
    return heartbeatThreadNum == rhs.heartbeatThreadNum &&
           heartbeatQueueSize == rhs.heartbeatQueueSize && enableAgentStat == rhs.enableAgentStat &&
           snapshotLogDir == rhs.snapshotLogDir;
}





} // namespace multi_call
