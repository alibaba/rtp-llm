#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/Spec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MiscUtil.h"

namespace multi_call {

class GeneratorBase
{
public:
    GeneratorBase(const std::string &clusterName, const std::string &bizName,
                  const std::string &strategy, const std::string &requestId, SourceIdTy sourceId,
                  VersionTy version, VersionTy preferVersion);
    ~GeneratorBase();

private:
    GeneratorBase(const GeneratorBase &);
    GeneratorBase &operator=(const GeneratorBase &);

public:
    void setClusterName(const std::string &clusterName) {
        _clusterName = clusterName;
    }
    void setBizName(const std::string &bizName) {
        _bizName = bizName;
    }
    std::string getBizName() const {
        return MiscUtil::createBizName(_clusterName, _bizName);
    }
    const std::string &getClusterName() const {
        return _clusterName;
    }
    void setFlowControlStrategy(const std::string &strategy) {
        _strategy = strategy;
    }
    const std::string &getFlowControlStrategy() const {
        if (!_strategy.empty()) {
            return _strategy;
        } else {
            return _bizName;
        }
    }
    void setRequestId(const std::string &requestId) {
        _requestId = requestId;
    }
    const std::string &getRequestId() const {
        return _requestId;
    }
    void setSourceId(SourceIdTy sourceId) {
        _sourceId = sourceId;
    }
    SourceIdTy getSourceId() const {
        return _sourceId;
    }
    VersionTy getVersion() const {
        return _version;
    }
    void setVersion(VersionTy version) {
        _version = version;
    }
    VersionTy getPreferVersion() const {
        return _preferVersion;
    }
    void setPreferVersion(VersionTy preferVersion) {
        _preferVersion = preferVersion;
    }
    void setUserRequestType(RequestType type) {
        _requestType = type;
    }
    RequestType getUserRequestType() const {
        return _requestType;
    }
    void setSpec(const Spec &spec) {
        _spec = spec;
    }
    const Spec &getSpec() const {
        return _spec;
    }
    void setDisableRetry(bool disable) {
        _disableRetry = disable;
    }
    bool getDisableRetry() const {
        return _disableRetry || _disableProbe;
    }
    void setDisableProbe(bool disable) {
        _disableProbe = disable;
    }
    bool getDisableProbe() const {
        return _disableProbe;
    }
    void setDisableDegrade(bool disable) {
        _disableDegrade = disable;
    }
    bool getDisableDegrade() const {
        return _disableDegrade;
    }
    void setIgnoreWeightLabelInConsistentHash(bool ignore) {
        _ignoreWeightLabelInConsistentHash = ignore;
    }
    bool getIgnoreWeightLabelInConsistentHash() const {
        return _ignoreWeightLabelInConsistentHash;
    }
    void addMatchTag(const std::string &tag, TagMatchType type) {
        if (!_tags) {
            _tags = std::make_shared<MatchTagMap>();
        }
        TagMatchInfo info;
        info.type = type;
        (*_tags)[tag] = info;
    }
    bool hasMatchTag(const std::string &tag) const {
        return _tags && (_tags->end() != _tags->find(tag));
    }
    void deleteMatchTag(const std::string &tag) {
        if (_tags) {
            _tags->erase(tag);
        }
    }
    void setMatchTags(const MatchTagMapPtr &tags) {
        _tags = tags;
    }
    MatchTagMapPtr cloneMatchTags() const {
        if (!_tags) {
            return nullptr;
        }
        return std::make_shared<MatchTagMap>(*_tags);
    }

protected:
    std::string _clusterName;
    std::string _bizName;
    std::string _strategy;
    std::string _requestId;
    SourceIdTy _sourceId;
    VersionTy _version;
    VersionTy _preferVersion;
    RequestType _requestType;
    Spec _spec;
    bool _disableRetry                      : 1;
    bool _disableProbe                      : 1;
    bool _disableDegrade                    : 1;
    bool _ignoreWeightLabelInConsistentHash : 1;
    MatchTagMapPtr _tags;
};

MULTI_CALL_TYPEDEF_PTR(GeneratorBase);

} // namespace multi_call

