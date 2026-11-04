#pragma once

#include "navi/common.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

class ResourceReplaceConfig;
class ResourceMap;

typedef google::protobuf::RepeatedPtrField<ResourceReplaceInfoDef> RepeatedResourceReplaceInfoDef;
class ReplaceInfoMap {
public:
    ReplaceInfoMap() {
    }
    void init(const RepeatedResourceReplaceInfoDef &infoVecDef);
    void init(const std::vector<ResourceReplaceConfig> &resourceReplaceInfos);
    void removeDup(const ReplaceInfoMap &configMap);
    const std::string &getReplaceRTo(const std::string &from) const;
    const NaviUnorderedMap<std::string, std::string> &getReplaceFrom2To() const {
        return _replaceFrom2To;
    }
    void replace(const std::vector<std::pair<std::string, bool>> *dependMap, ResourceMap &resourceMap) const;
private:
    NaviUnorderedMap<std::string, std::string> _replaceFrom2To;
};

NAVI_TYPEDEF_PTR(ReplaceInfoMap);

}

