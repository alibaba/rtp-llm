#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/LatencyModelGlobalData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/scheduler/ScheduleContextInternal.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class NaviSchedResourceTest;

namespace navi_sched {

class MasterLatencyModelR : public navi::Resource {
public:
    enum class RejectReason {
        NONE,
        INVALID_MODEL,
        SOURCE_MISMATCH,
        TRUSTED_REGRESSION,
        STALE_REVISION,
    };

    struct RecordResult {
        bool accepted = false;
        bool forcedUntrusted = false;
        uint32_t consecutiveRejectedCount = 0;
        RejectReason rejectReason = RejectReason::NONE;
        std::shared_ptr<const NaviSchedLatencyModel> model;
    };

    // NodeState and the live-model pool share this slot so model publication
    // and removal cannot leave two independently mutable model copies.
    class LiveModelSlot {
    public:
        virtual ~LiveModelSlot() = default;
        virtual std::shared_ptr<const NaviSchedLatencyModel> getModel() const = 0;
    };
    using LiveModelSlotPtr = std::shared_ptr<LiveModelSlot>;

    struct Stats {
        size_t fingerprintCount = 0;
        size_t candidateCount = 0;
        uint64_t acceptedUpdateCount = 0;
        uint64_t rejectedUpdateCount = 0;
        uint64_t warmupResponseCount = 0;
        uint64_t warmupNotificationCount = 0;
        uint64_t warmupNotificationFailureCount = 0;
        size_t warmupSubscriberCount = 0;
        uint64_t persistSuccessCount = 0;
        uint64_t persistFailureCount = 0;
        size_t globalZeroPointLiveNodeCount = 0;
        size_t globalZeroPointGroupCount = 0;
        uint64_t globalZeroPointRefreshCount = 0;
    };
    enum class GlobalZeroPointInvalidReason {
        NONE = 0,
        NOT_REFRESHED = 1,
        GROUP_NOT_FOUND = 2,
        MASTER_EPOCH_NOT_SET = 3,
        VERSION_NOT_SET = 4,
        FORMULA_MISMATCH = 5,
        BINARY_VERSION_INVALID = 6,
        PARAMETER_COUNT_INVALID = 7,
        ZERO_POINT_NON_FINITE = 8,
        ZERO_POINT_NEGATIVE = 9,
    };
    struct GlobalZeroPointDiagnostics {
        LatencyModelGlobalData data;
        GlobalZeroPointInvalidReason invalidReason =
            GlobalZeroPointInvalidReason::NOT_REFRESHED;
        size_t groupLiveNodeCount = 0;
        size_t groupTrustedLiveNodeCount = 0;

        bool available() const {
            return invalidReason == GlobalZeroPointInvalidReason::NONE;
        }
    };

    ~MasterLatencyModelR() override;

    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool activateAgent(const std::string &agentId,
                       const std::string &sourceId);
    LiveModelSlotPtr getLiveModelSlot(const std::string &agentId,
                                      const std::string &sourceId = "") const;
    bool record(const std::string &agentId, const std::string &fingerprint,
                const AgentLatencyToMasterData &latency, bool trusted,
                int64_t eventTimeUs, const std::string &sourceId = "",
                bool *liveUpdateAccepted = nullptr, bool warmStarted = false,
                RecordResult *recordResult = nullptr,
                bool rampFinished = true);
    bool updateWarmupSubscriber(
        const std::string &agentId, const std::string &sourceId,
        const std::string &fingerprint, const AgentLatencyToMasterData &latency,
        bool trusted, const navi::AsyncPipePtr &notificationPipe);
    void removeWarmupSubscriber(const std::string &sourceId);
    void removeAgent(const std::string &agentId,
                     const std::string &sourceId = "");
    LatencyModelGlobalData getGlobalZeroPoint(int32_t formula, int64_t binaryVersion,
                                              size_t parameterCount) const;
    GlobalZeroPointDiagnostics getGlobalZeroPointDiagnostics(
        int32_t formula, int64_t binaryVersion, size_t parameterCount) const;
    MasterLatencyModelWarmupToAgentDataPtr makeWarmup(
        const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
        size_t parameterCount);
    MasterLatencyModelWarmupToAgentDataPtr makeWarmupIfNew(
        const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
        size_t parameterCount, uint64_t lastCandidateVersion,
        uint64_t &candidateVersion);
    uint64_t getWarmupCandidateVersion(const std::string &fingerprint,
                                       int32_t formula,
                                       size_t parameterCount) const;
    bool save();
    Stats getStats() const;
    static const char *getRejectReasonName(RejectReason reason);
    navi::SummaryErrorCode getSummary(
        const std::string &query, navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    friend class ::NaviSchedResourceTest;

    struct ModelKey {
        std::string fingerprint;
        int32_t formula = 0;
        int64_t binaryVersion = 0;
        uint32_t parameterCount = 0;

        bool operator<(const ModelKey &other) const;
    };

    struct StoredCandidate {
        std::string agentId;
        int64_t sourceRevision = 0;
        int64_t eventTimeUs = 0;
        std::vector<double> parameters;
        NaviSchedLatencyOptimizerState optimizerState;
        int32_t residualVersion = 0;
        std::vector<double> residualParameters;
        int64_t residualLastUpdateTimeUs = 0;
    };

    using CandidateMap = std::map<std::string, StoredCandidate>;

    struct WarmupKey {
        std::string fingerprint;
        int32_t formula = 0;
        uint32_t parameterCount = 0;

        bool operator<(const WarmupKey &other) const;
    };

    struct ModelSignature {
        int32_t formula = 0;
        int64_t binaryVersion = 0;
        uint32_t parameterCount = 0;

        bool operator<(const ModelSignature &other) const;
    };

    struct LiveModel : NaviSchedLatencyModel {
        ModelSignature signature;
        ModelKey modelKey;
        int64_t eventTimeUs = 0;
        double zeroPoint = 0.0;
        NaviSchedLatencyOptimizerState optimizerState;
    };

    struct LiveEntry : LiveModelSlot {
        std::shared_ptr<const NaviSchedLatencyModel> getModel() const override
        {
            return std::atomic_load_explicit(&model, std::memory_order_acquire);
        }

        std::string sourceId;
        std::atomic<bool> active{true};
        uint32_t consecutiveRejectedCount = 0;
        std::shared_ptr<const LiveModel> model;
    };

    using GlobalZeroPointMap = std::map<ModelSignature, LatencyModelGlobalData>;
    using LiveEntryList = std::vector<std::shared_ptr<LiveEntry>>;

    struct WarmupSubscriber {
        std::string agentId;
        std::weak_ptr<navi::AsyncPipe> notificationPipe;
    };

    struct WarmupGroup {
        uint64_t version = 0;
        std::map<std::string, WarmupSubscriber> subscribers;
    };

    bool loadLocked();
    bool saveLocked();
    void pruneLocked(CandidateMap &candidates);
    void publishLiveEntriesLocked();
    void refreshGlobalZeroPoint();
    void removeWarmupSubscriberLocked(const std::string &sourceId);
    void removeLiveCandidateLocked(const std::string &agentId);
    void pruneLiveCandidatesLocked(const ModelKey &key,
                                   CandidateMap &candidates);
    void markRejectedModelLocked(const std::string &agentId,
                                 const std::shared_ptr<LiveEntry> &entry,
                                 RejectReason reason,
                                 RecordResult *recordResult);
    MasterLatencyModelWarmupToAgentDataPtr makeWarmupLocked(
        const std::string &fingerprint, int32_t formula, int64_t binaryVersion,
        size_t parameterCount, uint64_t lastCandidateVersion,
        uint64_t &candidateVersion);
    void notifyWarmupSubscribers(
        const std::vector<navi::AsyncPipePtr> &notificationPipes);
    static bool validFingerprint(const std::string &fingerprint);

private:
    mutable autil::ThreadMutex _lock;
    std::map<ModelKey, CandidateMap> _models;
    std::map<ModelKey, CandidateMap> _liveCandidates;
    std::map<std::string, ModelKey> _liveCandidateKeys;
    std::map<WarmupKey, WarmupGroup> _warmupGroups;
    std::map<std::string, WarmupKey> _warmupSubscriberKeys;
    std::map<std::string, std::shared_ptr<LiveEntry>> _liveModels;
    std::shared_ptr<const LiveEntryList> _liveEntries;
    std::shared_ptr<const GlobalZeroPointMap> _globalZeroPoint;
    autil::LoopThreadPtr _globalZeroPointThread;
    std::string _stateFile;
    // Trust admission is separate from the stricter sharing/persistence gate.
    int64_t _trustMinRevision = 256;
    int64_t _sourceMinRevision = 1024;
    size_t _maxCandidates = 4;
    int64_t _persistIntervalMs = 10000;
    int64_t _globalZeroPointRefreshIntervalMs = 1000;
    int64_t _lastPersistTimeUs = 0;
    int64_t _masterEpoch = 0;
    uint64_t _nextWarmupVersion = 1;
    std::atomic<uint64_t> _nextGlobalZeroPointVersion{1};
    std::atomic<uint64_t> _globalZeroPointRefreshCount{0};
    bool _dirty = false;
    Stats _stats;
};

NAVI_TYPEDEF_PTR(MasterLatencyModelR);

} // namespace navi_sched
