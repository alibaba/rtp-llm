#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

struct BizTopoInfo {
    std::string bizName;
    PartIdTy partCnt;
    PartIdTy partId;
    VersionTy version;
    WeightTy targetWeight;
    VersionTy protocolVersion;
    int32_t gigRpcPort;
    std::shared_ptr<FlowControlConfig> flowControlConfig;
};

class TopoInfoBuilder
{
public:
    TopoInfoBuilder();
    explicit TopoInfoBuilder(int32_t grpcPort);
    ~TopoInfoBuilder();

private:
    TopoInfoBuilder(const TopoInfoBuilder &);
    TopoInfoBuilder &operator=(const TopoInfoBuilder &);

public:
    void addBiz(const std::string &bizName, PartIdTy partCnt, PartIdTy partId, VersionTy version,
                WeightTy targetWeight, VersionTy protocolVersion, int32_t gigRpcPort,
                const std::shared_ptr<FlowControlConfig> &serverGigFlowConfig = nullptr);
    void addBiz(const std::string &bizName, PartIdTy partCnt, PartIdTy partId, VersionTy version,
                WeightTy targetWeight, VersionTy protocolVersion);
    void flushAllBizVersion(VersionTy version);
    std::string build() const;
    const std::vector<BizTopoInfo> &getBizTopoInfo() const;
    bool parseTopoInfo(const std::string &topoInfoStr, BizTopoInfo &topoInfo);

private:
    int32_t _grpcPort;
    std::vector<BizTopoInfo> _infos;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(TopoInfoBuilder);

} // namespace multi_call
