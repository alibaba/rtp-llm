#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "autil/Lock.h"
#include "autil/legacy/jsonizable.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace navi_sched {

struct AgentSchedulerConfig : public autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper &json) override;
    bool valid() const;
    std::string validationError() const;
    LatencyResidualModel::RuntimeConfig residualRuntimeConfig() const;

    static AgentSchedulerConfig fromServiceConfig(
        const NaviSchedConfig &config, double timeCompression = 1.0);

    int64_t batchTimeoutMs = 10;
    int64_t closedBatchKeepMs = 60 * 1000;
    size_t latencyFitBatchSize = 4;
    size_t latencyFitInterval = 0;
    double latencyLearningRate = 0.05;
    double latencyOnlineLearningRateScale = 1.0;
    double latencyCurrentSampleWeight = 4.0;
    std::vector<double> latencyParameterLearningRateScales;
    double latencyNonlinearScale = 1500.0;
    double queueLearningRate = 0.02;
    bool latencyCheckGradient = false;
    size_t latencyColdStartFitEpochs = 100;
    size_t latencyHistoryLimit = 2000;
    size_t latencyHistoryBucketCapacity = 64;
    size_t latencyPersistInterval = 256;
    int64_t latencyPersistIntervalMs = 10 * 1000;
    size_t latencyWarmupSampleCount = 10;
    int64_t latencyModelTrustMinRevision = 256;
    double latencyZeroAnchorMaxMs = 256.0;
    size_t latencyZeroAnchorMinSamples = 8;
    bool latencyValidationEnabled = false;
    double residualLearningRate = 0.0;
    double residualUpdateErrorFloorMs = 100.0;
    double residualUpdateErrorRatio = 0.1;
    int64_t residualIdleDecayStartMs = 2 * 60 * 1000;
    int64_t residualIdleDecayEndMs = 5 * 60 * 1000;
    double timeCompression = 1.0;
};

class AgentSchedulerConfigR : public navi::Resource {
public:
    enum class ApplyResult {
        APPLIED,
        STALE,
        INVALID,
    };

    struct Snapshot {
        AgentSchedulerConfig config;
        uint64_t generation = 0;
        std::string configHash;
        std::string configJson;
    };

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    Snapshot get() const;
    ApplyResult apply(uint64_t generation, const std::string &configHash,
                      const std::string &configJson, Snapshot &snapshot,
                      std::string &errorMessage);
    bool subscribe(const std::string &subscriberId,
                   const navi::AsyncPipePtr &notificationPipe);
    void unsubscribe(const std::string &subscriberId,
                     const navi::AsyncPipePtr &notificationPipe);

    static bool parse(const std::string &configJson, AgentSchedulerConfig &config,
                      std::string &errorMessage);
    static std::string toJson(const AgentSchedulerConfig &config);
    static std::string hashConfig(const AgentSchedulerConfig &config);
    static std::string hashJson(const std::string &canonicalConfigJson);

public:
    static const std::string RESOURCE_ID;

private:
    mutable autil::ThreadMutex _lock;
    Snapshot _snapshot;
    std::map<std::string, std::weak_ptr<navi::AsyncPipe>> _subscribers;
    std::string _initialConfigJson;
};

NAVI_TYPEDEF_PTR(AgentSchedulerConfigR);

} // namespace navi_sched
