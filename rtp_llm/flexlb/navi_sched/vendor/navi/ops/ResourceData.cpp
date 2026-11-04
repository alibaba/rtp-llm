#include "navi/ops/ResourceData.h"
#include "navi/util/CommonUtil.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

ResourceDataType::ResourceDataType()
    : Type(NAVI_TYPE_PARAM, RESOURCE_DATA_TYPE_ID)
{
}

REGISTER_BUILDIN_TYPE(ResourceDataType);

bool ResourceErrorInfo::latterThan(const ResourceErrorInfo &other) const {
    if (beginTimeUs > other.beginTimeUs) {
        return true;
    } else if (beginTimeUs == other.beginTimeUs) {
        if (endTimeUs > other.endTimeUs) {
            return true;
        } else if (unlikely(endTimeUs == other.endTimeUs)) {
            if (name > other.name) {
                return true;
            }
        }
    }
    return false;
}

ResourceData::ResourceData()
    : Data(RESOURCE_DATA_TYPE_ID)
    , _require(true)
    , _wait(false)
    , _stage(RS_UNKNOWN)
    , _ec(EC_NONE)
{
}
void ResourceData::initErrorInfo(int64_t beginTimeUs) {
    if (unlikely(_require && EC_NONE != _ec)) {
        auto errorInfo = std::make_shared<ResourceErrorInfo>();
        errorInfo->name = _name;
        errorInfo->object = NAVI_TLS_LOGGER ? NAVI_TLS_LOGGER->object : nullptr;
        errorInfo->beginTimeUs = beginTimeUs;
        errorInfo->endTimeUs = autil::TimeUtility::currentTime();
        errorInfo->ec = _ec;
        _errorInfo = errorInfo;
    }
}

bool ResourceData::validate(bool inputRequire, bool errorLog) const {
    if (_resource) {
        return true;
    }
    auto realRequire = _require || inputRequire;
    if (!realRequire) {
        return true;
    }
    if (_wait) {
        return true;
    }
    if (EC_NEED_REPLACE == _ec) {
        return true;
    }
    if (RS_EXTERNAL == _stage) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                      WARN,
                                      "lack resource [%s] stage [%s] ec [%s]",
                                      _name.c_str(),
                                      ResourceStage_Name(_stage).c_str(),
                                      CommonUtil::getErrorString(_ec));
        return false;
    }
    if (RS_RUN_GRAPH_EXTERNAL == _stage) {
        return true;
    }
    if (_stage < RS_RUN_GRAPH_EXTERNAL) {
        validateDepend(errorLog);
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                      WARN,
                                      "lack resource [%s] stage [%s] ec [%s]",
                                      _name.c_str(),
                                      ResourceStage_Name(_stage).c_str(),
                                      CommonUtil::getErrorString(_ec));
        return false;
    }
    if (RS_GRAPH <= _stage && _stage <= RS_KERNEL) {
        return validateDepend(errorLog);
    }
    if (EC_LACK_RESOURCE_GRAPH == _ec ||
        EC_LACK_RESOURCE_SUB_GRAPH == _ec ||
        EC_LACK_RESOURCE_SCOPE == _ec ||
        EC_LACK_RESOURCE_KERNEL == _ec)
    {
        return validateDepend(errorLog);
    }
    if (!(EC_LACK_RESOURCE_EXTERNAL == _ec || EC_BIND_NAMED_DATA == _ec)) {
        validateDepend(errorLog);
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                      WARN,
                                      "lack resource [%s] stage [%s] ec [%s]",
                                      _name.c_str(),
                                      ResourceStage_Name(_stage).c_str(),
                                      CommonUtil::getErrorString(_ec));
        return false;
    }
    return true;
}

bool ResourceData::validateDepend(bool errorLog) const {
    if (!_dependVec) {
        NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog, WARN, "null depend map");
        return false;
    }
    for (const auto &pair : *_dependVec) {
        const auto &dependName = pair.first;
        auto require = pair.second;
        auto it = _dependDatas.find(dependName);
        if (_dependDatas.end() == it) {
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                          WARN,
                                          "lack depend resource [%s], depend by [%s] require [1] stage [%s] ec [%s]",
                                          dependName.c_str(),
                                          _name.c_str(),
                                          ResourceStage_Name(_stage).c_str(),
                                          CommonUtil::getErrorString(_ec));
            return false;
        }
        if (!require) {
            continue;
        }
        if (!it->second->validate(true, errorLog)) {
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                          WARN,
                                          "lack depend resource [%s], depend by [%s] require [1] stage [%s] ec [%s]",
                                          dependName.c_str(),
                                          _name.c_str(),
                                          ResourceStage_Name(_stage).c_str(),
                                          CommonUtil::getErrorString(_ec));
            return false;
        }
    }
    return validateReplace(errorLog);
}

bool ResourceData::validateReplace(bool errorLog) const {
    if (!_replaceSet) {
        return true;
    }
    for (const auto &replaceName : *_replaceSet) {
        auto it = _dependDatas.find(replaceName);
        if (_dependDatas.end() == it) {
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                          WARN,
                                          "replacer [%s] not found in replace map, stage [%s] ec [%s]",
                                          replaceName.c_str(),
                                          ResourceStage_Name(_stage).c_str(),
                                          CommonUtil::getErrorString(_ec));
            return false;
        }
        if (!it->second->validate(true, errorLog)) {
            NAVI_KERNEL_FAIL_AS_ERROR_LOG(errorLog,
                                          WARN,
                                          "lack replacer resource [%s] to replace [%s], stage [%s] ec [%s]",
                                          replaceName.c_str(),
                                          _name.c_str(),
                                          ResourceStage_Name(_stage).c_str(),
                                          CommonUtil::getErrorString(_ec));
            return false;
        }
    }
    return true;
}

void ResourceData::addReplaceSummary(int32_t level, SummaryBuilder &builder) const {
    auto replaceTable = builder.newTable("Replace Table");
    if (!_replaceSet) {
        replaceTable->setDesc("Empty");
        return;
    }
    replaceTable->setDesc("registered replace info");
    auto nameCol = replaceTable->addColumn("name");
    size_t idx = 0;
    for (const auto &name : *_replaceSet) {
        nameCol->setData(idx, name);
        idx++;
    }
}

void ResourceData::addDependSummary(int32_t level, SummaryBuilder &builder) const {
    auto dependDataTable = builder.newTable("Depend Table");
    if (!_dependVec) {
        dependDataTable->setDesc("Empty");
        return;
    }
    dependDataTable->setDesc("depend resources");
    auto nameCol = dependDataTable->addColumn("name");
    auto requireCol = dependDataTable->addColumn("require");
    auto addrCol = dependDataTable->addColumn("addr");
    size_t idx = 0;
    for (const auto &pair : *_dependVec) {
        nameCol->setData(idx, pair.first);
        requireCol->setData(idx, pair.second);
        Resource *depend = nullptr;
        if (_resource) {
            depend = _resource->_dependMap.getResource(pair.first);
        }
        if (!depend) {
            addrCol->setData(idx, std::string("null"));
        } else {
            char buffer[128];
            autil::StringUtil::uint64ToHexStr((size_t)depend, buffer, sizeof(buffer));
            addrCol->setData(idx, std::string("0x") + buffer);
        }
        idx++;
    }
}

SummaryErrorCode ResourceData::getSummary(const std::string &query, SummaryBuilder &builder) const {
    auto level = autil::StringUtil::fromString<int32_t>(query);
    try {
        auto ec = SEC_OK;
        builder.setDesc("Resource: " + _name);
        auto resourceDataTable = builder.newTable("ResourceData Table");
        resourceDataTable->setDesc("resource create info");
        resourceDataTable->addColumn("name")->setData(0, _name);
        resourceDataTable->addColumn("stage")->setData(0, ResourceStage_Name(_stage));
        {
            char buffer[128];
            autil::StringUtil::uint64ToHexStr((size_t)_resource.get(), buffer, sizeof(buffer));
            resourceDataTable->addColumn("addr")->setData(0, std::string("0x") + buffer);
        }
        resourceDataTable->addColumn("require")->setData(0, _require);
        resourceDataTable->addColumn("wait")->setData(0, _wait);
        resourceDataTable->addColumn("ec")->setData(0, std::string(CommonUtil::getErrorString(_ec)));
        if (_resource) {
            ec = _resource->getSummary(query, builder);
        }
        if (level > 2) {
            addDependSummary(level, builder);
        }
        if (level > 3) {
            addReplaceSummary(level, builder);
        }
        return ec;
    } catch (const autil::legacy::ExceptionBase &e) {
        builder.setDesc(e.GetMessage());
        return SEC_FAILED;
    }
}

void ResourceData::mergeGigLocalSubResult(const multi_call::TopoSignatureSetPtr &left,
                                          multi_call::TopoSignatureSetPtr &right)
{
    if (!left) {
        return;
    }
    if (!right) {
        right = left;
    } else {
        if (*right == *left) {
            return;
        }
        right = std::make_shared<multi_call::TopoSignatureSet>(*right);
        for (auto sig : *left) {
            right->insert(sig);
        }
    }
}

}
