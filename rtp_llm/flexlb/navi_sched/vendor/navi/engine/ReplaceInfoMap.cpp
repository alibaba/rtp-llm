#include "navi/engine/ReplaceInfoMap.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"

namespace navi {

void ReplaceInfoMap::init(const std::vector<ResourceReplaceConfig> &resourceReplaceInfos) {
    RepeatedResourceReplaceInfoDef infoVecDef;
    for (const auto &info : resourceReplaceInfos) {
        auto infoDef = infoVecDef.Add();
        infoDef->set_from(info.from);
        infoDef->set_to(info.to);
    }
    init(infoVecDef);
}

void ReplaceInfoMap::init(const RepeatedResourceReplaceInfoDef &infoVecDef) {
    auto count = infoVecDef.size();
    std::vector<std::string> resourceList;
    NaviUnorderedSet<std::string> resourceSet;
    resourceList.reserve(count);
    resourceSet.reserve(count);
    for (int i = 0; i < count; i++) {
        const auto &replaceInfo = infoVecDef.Get(i);
        const auto &from = replaceInfo.from();
        if (resourceSet.end() != resourceSet.find(from)) {
            continue;
        }
        resourceSet.insert(from);
        resourceList.push_back(from);
    }
    auto replaceList = resourceList;
    for (int i = 0; i < count; i++) {
        const auto &replaceInfo = infoVecDef.Get(i);
        const auto &from = replaceInfo.from();
        const auto &to = replaceInfo.to();
        for (int j = 0; j < replaceList.size(); j++) {
            auto &tmp = replaceList[j];
            if (tmp == from) {
                tmp = to;
            }
        }
    }
    _replaceFrom2To.reserve(resourceList.size());
    for (int i = 0; i < resourceList.size(); i++) {
        const auto &from = resourceList[i];
        const auto &to = replaceList[i];
        if (from != to) {
            _replaceFrom2To[from] = to;
        }
    }
    NAVI_KERNEL_LOG(SCHEDULE1, "resource replace map: [%s]", autil::StringUtil::toString(_replaceFrom2To).c_str());
}

void ReplaceInfoMap::removeDup(const ReplaceInfoMap &configMap) {
    for (auto it = _replaceFrom2To.begin(); _replaceFrom2To.end() != it;) {
        const auto &configTo = configMap.getReplaceRTo(it->first);
        if (configTo == it->second) {
            _replaceFrom2To.erase(it++);
        } else {
            ++it;
        }
    }
}

const std::string &ReplaceInfoMap::getReplaceRTo(const std::string &from) const {
    if (_replaceFrom2To.empty()) {
        return from;
    }
    auto it = _replaceFrom2To.find(from);
    if (_replaceFrom2To.end() == it) {
        return from;
    }
    return it->second;
}

void ReplaceInfoMap::replace(const std::vector<std::pair<std::string, bool>> *dependVec, ResourceMap &resourceMap) const {
    if (!dependVec) {
        return;
    }
    if (_replaceFrom2To.empty()) {
        return;
    }
    ResourceMap newMap;
    for (const auto &pair : *dependVec) {
        const auto &name = pair.first;
        const auto &replaceTo = getReplaceRTo(name);
        auto resource = resourceMap.getResource(replaceTo);
        newMap.addResource(name, resource ? resource->shared_from_this() : nullptr);
    }
    resourceMap.setResource(newMap);
}

}

