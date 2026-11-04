#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CachedRequestGenerator.h"

namespace multi_call {

void CachedRequestGenerator::generate(PartIdTy partCnt, PartRequestMap &requestMap) {
    auto iter = _cachedPartRequestMap.find(partCnt);
    if (iter == _cachedPartRequestMap.end()) {
        _generator->generate(partCnt, requestMap);
        if (!_clearCacheAfterGet) {
            _cachedPartRequestMap.emplace(partCnt, requestMap);
        }
    } else {
        if (!_clearCacheAfterGet) {
            requestMap = iter->second;
        } else {
            requestMap.swap(iter->second);
            _cachedPartRequestMap.erase(iter);
        }
    }
}

std::string CachedRequestGenerator::toString() const {
    std::stringstream stream;
    for (auto &iter : _cachedPartRequestMap) {
        stream << "p:" << iter.first << ",r:" << iter.second.size() << ";";
    }
    return stream.str();
}

} // namespace multi_call
