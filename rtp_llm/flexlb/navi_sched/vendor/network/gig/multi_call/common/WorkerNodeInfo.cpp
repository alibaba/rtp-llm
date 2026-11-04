#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "autil/StringUtil.h"

using namespace std;
using namespace autil;
using namespace autil::legacy;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, TopoNodeMeta);

TopoNodeMeta::TopoNodeMeta() {
    clear();
}

TopoNodeMeta::~TopoNodeMeta() {
}

void TopoNodeMeta::clear() {
    targetWeight = MAX_WEIGHT;
    attributes.clear();
}

void TopoNodeMeta::init(const std::string &metaStr) {
    if (metaStr.empty()) {
        return;
    }
    try {
        FromJsonString(*this, metaStr);
    } catch (const ExceptionBase &e) {
        AUTIL_LOG(WARN, "jsonize node meta failed [%s], error [%s]",
                  ToJsonString(metaStr, true).c_str(), e.what());
        clear();
    }
}

void TopoNodeMeta::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("target_weight", targetWeight, targetWeight);
    json.Jsonize("attributes", attributes, attributes);
}

bool TopoNodeMeta::operator==(const TopoNodeMeta &rhs) const {
    return targetWeight == rhs.targetWeight && attributes == rhs.attributes &&
           metaEnv == rhs.metaEnv;
}

void TopoNodeMeta::setMetaEnv(const MetaEnv &env) {
    metaEnv = env;
}

TopoNode::TopoNode()
    : partCnt(INVALID_PART_COUNT)
    , partId(INVALID_PART_ID)
    , version(INVALID_VERSION_ID)
    , weight(MAX_WEIGHT)
    , supportHeartbeat(false)
    , isValid(true)
    , ssType(ST_UNKNOWN)
    , clientInfo(nullptr) {
}

bool TopoNode::operator==(const TopoNode &rhs) const {
    return nodeId == rhs.nodeId && isValid == rhs.isValid && meta == rhs.meta;
}

bool TopoNode::operator<(const TopoNode &rhs) const {
    return nodeId < rhs.nodeId;
}

bool TopoNode::generateNodeId() {
    int32_t maxLength = spec.ip.size() + bizName.size() + ((4 + MC_PROTOCOL_UNKNOWN) * 64) + 12;
    StringAppender appender(maxLength);
    appender.appendString(spec.ip)
        .appendChar('_')
        .appendString(bizName)
        .appendChar('_')
        .appendInt64(partCnt)
        .appendChar('_')
        .appendInt64(partId)
        .appendChar('_')
        .appendInt64(version);
    auto realWeight = getNodeWeight();
    if (realWeight < MIN_WEIGHT) {
        // copy provider is different from normal
        appender.appendChar('_').appendInt64(realWeight);
    }
    for (size_t i = 0; i < MC_PROTOCOL_UNKNOWN; i++) {
        appender.appendChar('_').appendInt64(spec.ports[i]);
    }
    appender.copyToString(nodeId);
    return validate();
}

bool TopoNode::validate() const {
    return partCnt > 0 && partId >= 0 && partCnt > partId;
}

WeightTy TopoNode::getNodeWeight() const {
    if (meta.targetWeight < MIN_WEIGHT) {
        return meta.targetWeight;
    } else {
        return weight;
    }
}



std::ostream &operator<<(std::ostream &os, const TopoNode &node) {
    os << node.spec << '_' << node.bizName << '_' << node.partCnt << '_' << node.partId << '_'
       << node.version;
    return os;
}

} // namespace multi_call
