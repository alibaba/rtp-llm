#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/Resource.h"
#include "navi/engine/Type.h"

namespace navi {

class ResourceDataType : public Type
{
public:
    ResourceDataType();
private:
    ResourceDataType(const ResourceDataType &);
    ResourceDataType &operator=(const ResourceDataType &);
};

struct ResourceErrorInfo {
public:
    bool latterThan(const ResourceErrorInfo &other) const;
public:
    std::string name;
    void *object = nullptr;
    int64_t beginTimeUs = 0;
    int64_t endTimeUs = 0;
    ErrorCode ec = EC_NONE;
};

NAVI_TYPEDEF_PTR(ResourceErrorInfo);

class ResourceData : public Data
{
public:
    ResourceData();
private:
    ResourceData(const ResourceData &);
    ResourceData &operator=(const ResourceData &);
public:
    void initErrorInfo(int64_t beginTimeUs);
    bool validate(bool inputRequire, bool errorLog) const;
    bool validateReplace(bool errorLog) const;
    bool validateDepend(bool errorLog) const;
    SummaryErrorCode getSummary(const std::string &query, SummaryBuilder &builder) const override;
private:
    void addDependSummary(int32_t level, SummaryBuilder &builder) const;
    void addReplaceSummary(int32_t level, SummaryBuilder &builder) const;
public:
    static void mergeGigLocalSubResult(const multi_call::TopoSignatureSetPtr &left,
                                       multi_call::TopoSignatureSetPtr &right);
public:
    std::string _name;
    bool _require : 1;
    bool _wait : 1;
    ResourcePtr _resource;
    ResourceStage _stage;
    ErrorCode _ec;
    const std::vector<std::pair<std::string, bool>> *_dependVec = nullptr;
    std::map<std::string, std::shared_ptr<ResourceData>> _dependDatas;
    const NaviUnorderedSet<std::string> *_replaceSet = nullptr;
    ResourceErrorInfoPtr _errorInfo;
    multi_call::TopoSignatureSetPtr _gigMetaLocalSignatures;
};

NAVI_TYPEDEF_PTR(ResourceData);

}

