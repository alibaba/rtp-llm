#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/TopoInfoBuilder.h"

#include <sstream>

#include "autil/StringConvertor.h"
#include "autil/StringTokenizer.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, TopoInfoBuilder);

TopoInfoBuilder::TopoInfoBuilder() : _grpcPort(INVALID_PORT) {
}

TopoInfoBuilder::TopoInfoBuilder(int32_t grpcPort) : _grpcPort(grpcPort) {
}

TopoInfoBuilder::~TopoInfoBuilder() {
}

void TopoInfoBuilder::addBiz(const std::string &bizName, PartIdTy partCnt, PartIdTy partId,
                             VersionTy version, WeightTy targetWeight, VersionTy protocolVersion,
                             int32_t gigRpcPort,
                             const std::shared_ptr<FlowControlConfig> &serverGigFlowConfig) {
    BizTopoInfo bizInfo;
    bizInfo.bizName = bizName;
    bizInfo.partCnt = partCnt;
    bizInfo.partId = partId;
    bizInfo.version = version;
    bizInfo.targetWeight = targetWeight;
    bizInfo.protocolVersion = protocolVersion;
    bizInfo.gigRpcPort = gigRpcPort;
    bizInfo.flowControlConfig = serverGigFlowConfig;

    _infos.emplace_back(std::move(bizInfo));
}

void TopoInfoBuilder::addBiz(const std::string &bizName, PartIdTy partCnt, PartIdTy partId,
                             VersionTy version, WeightTy targetWeight, VersionTy protocolVersion) {
    addBiz(bizName, partCnt, partId, version, targetWeight, protocolVersion, _grpcPort, nullptr);
}

void TopoInfoBuilder::flushAllBizVersion(VersionTy version) {
    for (auto &bizInfo : _infos) {
        bizInfo.version = version;
    }
}

std::string TopoInfoBuilder::build() const {
    std::stringstream ss;
    for (const auto &bizInfo : _infos) {
        ss << bizInfo.bizName << ":" << bizInfo.partCnt << ":" << bizInfo.partId << ":"
           << bizInfo.version << ":" << bizInfo.targetWeight << ":" << bizInfo.protocolVersion
           << ":" << bizInfo.gigRpcPort << ":false:" << bizInfo.gigRpcPort
           << "|"; // support heartbeat
    }
    auto ret = ss.str();
    if (ret.empty()) {
        ss << "invalid_topo_info:0:0:0:false:" << _grpcPort;
        return ss.str();
    } else {
        return ret;
    }
}

bool TopoInfoBuilder::parseTopoInfo(const std::string &topoInfoStr, BizTopoInfo &topoInfo) {
    const auto &topoInfoVec = autil::StringTokenizer::constTokenize(
        autil::StringView(topoInfoStr.data(), topoInfoStr.length()), ":");
    if (topoInfoVec.size() < 3) {
        AUTIL_LOG(ERROR, "error topo info [%s]", topoInfoStr.c_str());
        return false;
    }
    topoInfo.bizName = topoInfoVec[0].to_string();
    topoInfo.partCnt =
        autil::StringConvertor::atoi<PartIdTy>(topoInfoVec[1].data(), topoInfoVec[1].length());
    topoInfo.partId =
        autil::StringConvertor::atoi<PartIdTy>(topoInfoVec[2].data(), topoInfoVec[2].length());
    if (topoInfoVec.size() > 4) {
        topoInfo.version =
            autil::StringConvertor::atoi<VersionTy>(topoInfoVec[3].data(), topoInfoVec[3].length());
    }
    if (topoInfoVec.size() > 5) {
        topoInfo.targetWeight =
            autil::StringConvertor::atoi<WeightTy>(topoInfoVec[4].data(), topoInfoVec[4].length());
    }
    if (topoInfoVec.size() > 6) {
        topoInfo.protocolVersion =
            autil::StringConvertor::atoi<VersionTy>(topoInfoVec[5].data(), topoInfoVec[5].length());
    }
    if (topoInfoVec.size() > 7) {
        topoInfo.gigRpcPort =
            autil::StringConvertor::atoi<int32_t>(topoInfoVec[6].data(), topoInfoVec[6].length());
    }
    return true;
}

const std::vector<BizTopoInfo> &TopoInfoBuilder::getBizTopoInfo() const {
    return _infos;
}
} // namespace multi_call
