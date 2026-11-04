#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/BizTopo.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/BizStat.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/TopoInfoBuilder.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/StringTable.h"
#include "autil/HashAlgorithm.h"
#include "autil/StringConvertor.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, BizTopo);

BizTopo::BizTopo(uint64_t publishId, PublishGroupTy group) : _group(group) {
    _signature.publishId = publishId;
}

BizTopo::~BizTopo() {
}

bool BizTopo::init(const ServerBizTopoInfo &info, StringTable &stringTable,
                   const std::shared_ptr<BizStat> &bizStat) {
    _bizStat = bizStat;
    _info = info;
    initFlowControl();
    initTopoSignature();
    initMetaSignature();
    initTagSignature();
    initHash(stringTable);
    return true;
}

void BizTopo::initFlowControl() {
    if (!_info.flowControlConfig) {
        return;
    }
    _info.metas[GIG_HEARTBEAT_FLOW_CONTROL_META_KEY] = ToJsonString(*_info.flowControlConfig, true);
}

void BizTopo::initTopoSignature() {
    int32_t maxLength = _info.bizName.size() + 64 * 4 + 8;
    autil::StringAppender appender(maxLength);
    appender.appendString(_info.bizName)
        .appendChar('_')
        .appendInt64(_info.partCount)
        .appendChar('_')
        .appendInt64(_info.partId)
        .appendChar('_')
        .appendInt64(_info.version)
        .appendChar('_')
        .appendInt64(_info.protocolVersion)
        .appendChar('_')
        .appendInt64(_info.isDefaultBiz);
    if (_info.targetWeight < 0) {
        appender.appendChar('_').appendInt64(_info.targetWeight);
    }
    appender.copyToString(_id);
    auto topoSig = autil::HashAlgorithm::hashString64(_id.c_str(), _id.size());
    while (0 == topoSig) {
        _id += "_suffix";
        topoSig = autil::HashAlgorithm::hashString64(_id.c_str(), _id.size());
    }
    _signature.topo = topoSig;
}

void BizTopo::initMetaSignature() {
    int32_t maxLength = 1024;
    autil::StringAppender appender(maxLength);
    for (const auto &pair : _info.metas) {
        appender.appendString(pair.first).appendChar('@').appendString(pair.second).appendChar('$');
    }
    std::string id;
    appender.copyToString(id);
    _signature.meta = autil::HashAlgorithm::hashString64(id.c_str(), id.size());
}

void BizTopo::initTagSignature() {
    int32_t maxLength = 256;
    autil::StringAppender appender(maxLength);
    for (const auto &pair : _info.tags) {
        appender.appendString(pair.first).appendChar('@').appendInt64(pair.second).appendChar('$');
    }
    std::string id;
    appender.copyToString(id);
    _signature.tag = autil::HashAlgorithm::hashString64(id.c_str(), id.size());
}

void BizTopo::initHash(StringTable &stringTable) {
    _topoInfoHash.bizName = collectString(_info.bizName, stringTable);
    for (const auto &pair : _info.metas) {
        uint64_t key = collectString(pair.first, stringTable);
        uint64_t value = collectString(pair.second, stringTable);
        _topoInfoHash.metaVec.emplace_back(key, value);
    }
}

uint64_t BizTopo::collectString(const std::string &value, StringTable &stringTable) {
    std::shared_ptr<std::string> strPtr;
    auto hash = stringTable.addStr(value, &strPtr);
    _strVec.push_back(strPtr);
    return hash;
}

void BizTopo::fillBizTopoDiff(const BizTopoSignature &clientSideSignature, bool supportHash,
                              BizTopoDef *topoDef) const {
    topoDef->set_target_weight(getTargetWeight());
    fillSignature(topoDef->mutable_signature());
    if (_signature.topo != clientSideSignature.topo) {
        fillTopoKey(supportHash, topoDef->mutable_key());
    }
    if (_signature.meta != clientSideSignature.meta) {
        fillMeta(supportHash, topoDef->mutable_metas());
    }
    if (_signature.tag != clientSideSignature.tag) {
        fillTag(topoDef->mutable_tags());
    }
}

void BizTopo::fillBizTopo(bool supportHash, BizTopoDef *topoDef) const {
    topoDef->set_target_weight(getTargetWeight());
    fillSignature(topoDef->mutable_signature());
    fillTopoKey(supportHash, topoDef->mutable_key());
    fillMeta(supportHash, topoDef->mutable_metas());
    fillTag(topoDef->mutable_tags());
}

WeightTy BizTopo::getTargetWeight() const {
    auto targetWeight = _info.targetWeight;
    if (_bizStat) {
        targetWeight = _bizStat->getTargetWeight(autil::TimeUtility::currentTime(), targetWeight);
    } else {
        if (isStopped()) {
            targetWeight = MIN_WEIGHT;
        }
    }
    return targetWeight;
}

void BizTopo::fillSignature(TopoSignatureDef *signatureDef) const {
    signatureDef->set_topo(_signature.topo);
    signatureDef->set_meta(_signature.meta);
    signatureDef->set_tag(_signature.tag);
    signatureDef->set_publish_id(_signature.publishId);
}

void BizTopo::fillTopoKey(bool supportHash, BizTopoKeyDef *keyDef) const {
    if (supportHash && _topoInfoHash.bizName != 0) {
        keyDef->set_biz_name_hash(_topoInfoHash.bizName);
    } else {
        keyDef->set_biz_name(_info.bizName);
    }
    keyDef->set_part_count(_info.partCount);
    keyDef->set_part_id(_info.partId);
    keyDef->set_version(_info.version);
    keyDef->set_protocol_version(_info.protocolVersion);
    keyDef->set_is_default_biz(_info.isDefaultBiz);
}

void BizTopo::fillMeta(bool supportHash, BizMetasDef *metaDef) const {
    if (supportHash) {
        for (const auto &pair : _topoInfoHash.metaVec) {
            auto singleMetaDef = metaDef->add_hash_metas();
            singleMetaDef->set_key(pair.first);
            singleMetaDef->set_value(pair.second);
        }
    } else {
        for (const auto &pair : _info.metas) {
            auto singleMetaDef = metaDef->add_metas();
            singleMetaDef->set_key(pair.first);
            singleMetaDef->set_value(pair.second);
        }
    }
}

void BizTopo::fillTag(BizTagsDef *tagsDef) const {
    autil::ScopedReadWriteLock lock(_tagLock, 'r');
    for (const auto &pair : _info.tags) {
        auto singleTagDef = tagsDef->add_tags();
        singleTagDef->set_tag(pair.first);
        singleTagDef->set_version(pair.second);
    }
}

void BizTopo::fillPropagationStat(BizTopoDef *topoDef) const {
    if (!_bizStat) {
        return;
    }
    auto propStat = topoDef->mutable_propagation_stat();
    _bizStat->fillThisPropagationStat(propStat);
}

void BizTopo::updateVolatileInfo(const BizVolatileInfo &info) {
    _info.targetWeight = info.targetWeight;
    autil::ScopedReadWriteLock lock(_tagLock, 'w');
    _info.tags = info.tags;
    initTagSignature();
}

void BizTopo::start() {
    _stopped = false;
}

void BizTopo::stop() {
    _stopped = true;
}

bool BizTopo::isStopped() const {
    return _stopped;
}

void BizTopo::addToBuilder(TopoInfoBuilder &builder) const {
    builder.addBiz(_info.bizName, _info.partCount, _info.partId, _info.version, _info.targetWeight,
                   _info.protocolVersion);
}

void BizTopo::toString(std::string &ret) const {
    ret += "  " + _id;
    ret += ", topo_sig: " + autil::StringUtil::toString(_signature.topo);
    ret += ", meta_sig: " + autil::StringUtil::toString(_signature.meta);
    ret += ", tag_sig: " + autil::StringUtil::toString(_signature.tag);
    ret += ", started: " + autil::StringUtil::toString(!isStopped());
    ret += ", default: " + autil::StringUtil::toString(_info.isDefaultBiz);
    ret += "\n    target_weight: " + autil::StringUtil::toString(getTargetWeight());
    ret += "\n    publish_id: " + autil::StringUtil::toString(_signature.publishId);
    ret += "\n    publish_group: " + autil::StringUtil::toString(_group);
    ret += "\n    metas: " + autil::StringUtil::toString(_info.metas);
    {
        autil::ScopedReadWriteLock lock(_tagLock, 'r');
        ret += "\n    tags: " + autil::StringUtil::toString(_info.tags);
    }
    ret += "\n";
}

} // namespace multi_call
