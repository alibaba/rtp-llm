#pragma once

#include "aios/navi_sched/data/NaviSchedKVPreparation.h"
#include "aios/navi_sched/data/NaviSchedExpectedKVWorkload.h"
#include "aios/navi_sched/resource/MasterKVPreparationModel.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace navi_sched {
// Owned by one Master NodeState and guarded by its existing runtime lock.
// Generation, duration prediction and start features are frozen together.
class MasterKVPreparationForecasts {
public:
    struct Estimate {
        uint64_t requestId=0, generation=0;
        int64_t receivedMonoUs=0;
        double observedPreparationMs=0.0;
        std::optional<double> totalMs;
        double weight(int64_t nowMonoUs,double queueMs) const {
            if (!totalMs) return 1.0; // Unknown duration retains complete planning load.
            const double elapsed=std::max<int64_t>(0,nowMonoUs-receivedMonoUs)/1000.0;
            if(observedPreparationMs>=*totalMs) return 1.0;
            const double remaining=*totalMs-observedPreparationMs;
            if(elapsed>=remaining || queueMs>=remaining-elapsed) return 1.0;
            return std::min(1.0,observedPreparationMs/ *totalMs+elapsed/ *totalMs+queueMs/ *totalMs);
        }
    };

    struct AuditRecord {
        enum class Phase { START, COMPLETE, FAILED };
        Phase phase=Phase::START;
        uint64_t session=0, revision=0, modelSamplesBefore=0;
        Estimate estimate;
        MasterKVPreparationModel::Features features{};
        double actualPreparationMs=-1.0;
        bool metadataKnown=false, observedPreparing=false, trained=false;
    };

    void beginSession(uint64_t session) {
        if (_session==session) return;
        _session=session;_revision=0;_loads.clear();_metadata.clear();_model={};
    }
    bool reserve(uint64_t requestId,int64_t readTokens,double initialQueueMs) {
        if(!_session || !requestId || readTokens<0 || !std::isfinite(initialQueueMs) || initialQueueMs<0.0) return false;
        const Metadata metadata{readTokens,initialQueueMs};
        auto at=_metadata.find(requestId);
        if(at!=_metadata.end()) return at->second.readTokens==readTokens && at->second.initialQueueMs==initialQueueMs;
        _metadata.emplace(requestId,metadata);return true;
    }
    void forget(uint64_t requestId) { _metadata.erase(requestId);_loads.erase(requestId); }
    void releaseReservation(uint64_t requestId) { _metadata.erase(requestId); }
    void clearActive() { _metadata.clear();_loads.clear(); }

    bool observe(uint64_t session,uint64_t revision,
                 const std::vector<NaviSchedKVPreparation> &requests,int64_t nowMonoUs,
                 std::vector<AuditRecord> *audit=nullptr) {
        if(audit) audit->clear();
        if(!_session || session!=_session || !revision || revision<=_revision || nowMonoUs<0) return false;
        // New starts in one report all use the model from before this report;
        // vector iteration order cannot expose same-report completion labels.
        double earlierReadTokens=0.0;
        for(const auto &request:requests) {
            auto at=_loads.find(request.requestId);
            if(request.state==NativeKVExecutionState::PREPARING && at!=_loads.end() &&
               at->second.estimate.generation==request.loadGeneration &&
               at->second.state==NativeKVExecutionState::PREPARING && at->second.metadataKnown)
                earlierReadTokens+=at->second.features[1]*4096.0;
        }
        for(const auto &request:requests) {
            if(!request.loadGeneration) continue;
            auto at=_loads.find(request.requestId);
            if(at!=_loads.end() && at->second.estimate.generation>request.loadGeneration) {
                at->second.seenRevision=revision;continue;
            }
            if(at==_loads.end() || at->second.estimate.generation!=request.loadGeneration) {
                if(at!=_loads.end()) _loads.erase(at);
                Entry entry;entry.estimate.requestId=request.requestId;entry.estimate.generation=request.loadGeneration;
                entry.estimate.receivedMonoUs=nowMonoUs;
                entry.estimate.observedPreparationMs=std::max(0.0,request.preparationElapsedMs);
                entry.seenRevision=revision;entry.state=request.state;
                auto metadata=_metadata.find(request.requestId);
                entry.metadataKnown=metadata!=_metadata.end();
                if(entry.metadataKnown) entry.features={1.0,metadata->second.readTokens/4096.0,earlierReadTokens/4096.0,metadata->second.initialQueueMs/1000.0};
                entry.observedPreparing=request.state==NativeKVExecutionState::PREPARING && request.preparationElapsedMs>=0.0;
                if(entry.metadataKnown && entry.observedPreparing) entry.estimate.totalMs=_model.predict(entry.features);
                if(audit && request.state==NativeKVExecutionState::PREPARING)
                    audit->push_back({AuditRecord::Phase::START,_session,revision,
                        _model.samples(),entry.estimate,entry.features,-1.0,
                        entry.metadataKnown,entry.observedPreparing,false});
                if(audit && request.state==NativeKVExecutionState::FAILED)
                    audit->push_back({AuditRecord::Phase::FAILED,_session,revision,
                        _model.samples(),entry.estimate,entry.features,-1.0,
                        entry.metadataKnown,entry.observedPreparing,false});
                _loads.emplace(request.requestId,std::move(entry));
            } else {
                at->second.seenRevision=revision;
                if(request.state==NativeKVExecutionState::PREPARING && at->second.state==NativeKVExecutionState::PREPARING) {
                    auto &estimate=at->second.estimate;
                    const auto extrapolated=estimate.observedPreparationMs+std::max<int64_t>(0,nowMonoUs-estimate.receivedMonoUs)/1000.0;
                    estimate.observedPreparationMs=std::max(extrapolated,request.preparationElapsedMs);
                    estimate.receivedMonoUs=nowMonoUs;
                }
            }
        }
        // Every start forecast is frozen before any of these outcomes trains it.
        for(const auto &request:requests) {
            auto at=_loads.find(request.requestId);
            if(at==_loads.end() || at->second.estimate.generation!=request.loadGeneration) continue;
            auto &entry=at->second;
            if(request.state==NativeKVExecutionState::FAILED) {
                if(audit && entry.state!=NativeKVExecutionState::FAILED)
                    audit->push_back({AuditRecord::Phase::FAILED,_session,revision,
                        _model.samples(),entry.estimate,entry.features,-1.0,
                        entry.metadataKnown,entry.observedPreparing,false});
                entry.observedPreparing=false;
            }
            const auto samplesBefore=_model.samples();
            if(!entry.trainingRecorded && entry.observedPreparing && entry.metadataKnown &&
               request.preparationElapsedMs>=0.0 &&
               (request.state==NativeKVExecutionState::READY || request.state==NativeKVExecutionState::EXECUTING)) {
                entry.trainingRecorded=_model.record(entry.features,request.preparationElapsedMs);
            }
            if(!entry.completionAudited && request.preparationElapsedMs>=0.0 &&
               (request.state==NativeKVExecutionState::READY || request.state==NativeKVExecutionState::EXECUTING)) {
                if(audit) audit->push_back({AuditRecord::Phase::COMPLETE,_session,revision,
                    samplesBefore,entry.estimate,entry.features,request.preparationElapsedMs,
                    entry.metadataKnown,entry.observedPreparing,entry.trainingRecorded});
                entry.completionAudited=true;
            }
            if(request.state!=NativeKVExecutionState::PREPARING) entry.state=request.state;
        }
        for(auto at=_loads.begin();at!=_loads.end();) {
            if(at->second.seenRevision!=revision) at=_loads.erase(at);else ++at;
        }
        _revision=revision;return true;
    }

    std::vector<Estimate> snapshot(const std::vector<NaviSchedKVPreparation> &requests) const {
        std::vector<Estimate> result;
        for(const auto &request:requests) {
            if(request.state!=NativeKVExecutionState::PREPARING) continue;
            auto at=_loads.find(request.requestId);
            if(at!=_loads.end() && at->second.estimate.generation==request.loadGeneration &&
               at->second.state==NativeKVExecutionState::PREPARING) result.push_back(at->second.estimate);
            else result.push_back({request.requestId,request.loadGeneration,0,0.0,{}});
        }
        return result;
    }
    const MasterKVPreparationModel &model() const { return _model; }
    size_t activeLoads() const { return _loads.size(); }
    size_t reservations() const { return _metadata.size(); }

private:
    struct Metadata { int64_t readTokens;double initialQueueMs; };
    struct Entry {
        Estimate estimate;
        MasterKVPreparationModel::Features features{};
        bool metadataKnown=false,observedPreparing=false,trainingRecorded=false,completionAudited=false;
        uint64_t seenRevision=0;
        NativeKVExecutionState state=NativeKVExecutionState::UNKNOWN;
    };
    uint64_t _session=0,_revision=0;
    std::unordered_map<uint64_t,Metadata> _metadata;
    std::unordered_map<uint64_t,Entry> _loads;
    MasterKVPreparationModel _model;
};
} // namespace navi_sched
