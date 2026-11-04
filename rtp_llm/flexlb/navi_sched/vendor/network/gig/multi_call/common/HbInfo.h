#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/Heartbeat.pb.h"
#include "autil/Lock.h"

namespace multi_call {

struct HbMetaInfo {
    Meta meta;
    // VersionTy version;
};

MULTI_CALL_TYPEDEF_PTR(HbMetaInfo);

struct HbInfo {
public:
    HbInfo() : heartbeatProtocol(MC_PROTOCOL_UNKNOWN), heartbeatSpec(""), lastUpdateTime(0) {
    }

public:
    ProtocolType heartbeatProtocol;
    std::string heartbeatSpec;

public:
    void updateHbMeta(const HbMetaInfoPtr &hbMetaPtr);
    HbMetaInfoPtr getHbMeta(); // return copy of HbMetaInfoPtr
    void updateTime();
    void updateTime(int64_t time);
    int64_t getTime() {
        return lastUpdateTime;
    }

private:
    int64_t lastUpdateTime;
    autil::SpinLock _lock;
    HbMetaInfoPtr hbMeta;
};

MULTI_CALL_TYPEDEF_PTR(HbInfo);

} // namespace multi_call

