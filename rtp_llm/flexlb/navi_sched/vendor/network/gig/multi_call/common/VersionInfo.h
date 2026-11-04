#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

struct VersionInfo {
public:
    VersionInfo() {
        clear();
    }
    ~VersionInfo() {
    }

public:
    bool complete() const {
        return partCount > 0 && partCount == healthCount;
    }
    bool heartbeatComplete() const {
        return partCount > 0 && partCount == heartbeatHealthCount;
    }
    bool subscribeComplete() const {
        return partCount > 0 && partCount == subscribeCount;
    }
    bool hasSubscribe() const {
        return partCount > 0 && subscribeCount > 0;
    }
    bool hasStopped() const {
        return stopCount > 0;
    }
    bool isMetaSame(const std::map<PartIdTy, MetaMapPtr> &rhs) const {
        if (metas.size() != rhs.size()) {
            return false;
        }
        for (const auto &[partId, metaMapPtr] : metas) {
            auto it = rhs.find(partId);
            if (it == rhs.end()) {
                return false;
            }
            if (!metaMapPtr) {
                if (!it->second) {
                } else {
                    return false;
                }
            } else {
                if (!it->second || *metaMapPtr != *it->second) {
                    return false;
                }
            }
        }
        return true;
    }
    bool operator==(const VersionInfo &rhs) const {
        return isMetaSame(rhs.metas) && partCount == rhs.partCount &&
               subscribeCount == rhs.subscribeCount && stopCount == rhs.stopCount &&
               healthCount == rhs.healthCount && heartbeatHealthCount == rhs.heartbeatHealthCount &&
               minHeartbeatHealthCount == rhs.minHeartbeatHealthCount;
    }

public:
    void clear() {
        partCount = 0;
        subscribeCount = 0;
        stopCount = 0;
        healthCount = 0;
        normalProviderCount = 0;
        heartbeatHealthCount = 0;
        minHeartbeatHealthCount = 0;
    }

public:
    size_t partCount;
    size_t subscribeCount;
    size_t stopCount;
    size_t healthCount;
    size_t normalProviderCount;

    size_t heartbeatHealthCount;
    size_t minHeartbeatHealthCount;

    std::map<PartIdTy, MetaMapPtr> metas;

private:
    AUTIL_LOG_DECLARE();
};

struct BizMetaInfo {
    std::string bizName;
    std::map<VersionTy, VersionInfo> versions;
};

MULTI_CALL_TYPEDEF_PTR(VersionInfo);

} // namespace multi_call

