#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/GeneratorBase.h"

using namespace std;

namespace multi_call {

GeneratorBase::GeneratorBase(const std::string &clusterName, const std::string &bizName,
                             const std::string &strategy, const std::string &requestId,
                             SourceIdTy sourceId, VersionTy version, VersionTy preferVersion)
    : _clusterName(clusterName)
    , _bizName(bizName)
    , _strategy(strategy)
    , _requestId(requestId)
    , _sourceId(sourceId)
    , _version(version)
    , _preferVersion(preferVersion)
    , _requestType(RT_NORMAL)
    , _disableRetry(false)
    , _disableProbe(false)
    , _disableDegrade(false)
    , _ignoreWeightLabelInConsistentHash(false) {
}

GeneratorBase::~GeneratorBase() {
}

} // namespace multi_call
