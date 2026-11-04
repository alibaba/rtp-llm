#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/RequestGenerator.h"

namespace multi_call {

class CachedRequestGenerator
{
public:
    CachedRequestGenerator(const RequestGeneratorPtr &generator) : _generator(generator) {
    }

public:
    void generate(PartIdTy partCnt, PartRequestMap &requestMap);
    std::string toString() const;

    inline const std::shared_ptr<RequestGenerator> getGenerator() const {
        return _generator;
    }

    inline void setClearCacheAfterGet(bool v) {
        _clearCacheAfterGet = v;
    }

private:
    RequestGeneratorPtr _generator;
    bool _clearCacheAfterGet {false};
    std::map<PartIdTy, PartRequestMap> _cachedPartRequestMap;
};

} // namespace multi_call
