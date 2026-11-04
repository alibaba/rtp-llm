#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/Resource.h"
#include "navi/engine/Type.h"

namespace navi {

class ResourceNotifyDataType : public Type {
public:
    ResourceNotifyDataType();
private:
    ResourceNotifyDataType(const ResourceNotifyDataType &);
    ResourceNotifyDataType &operator=(const ResourceNotifyDataType &);
};

class ResourceNotifyData : public Data
{
public:
    ResourceNotifyData();
    ~ResourceNotifyData();
    ResourceNotifyData(const ResourceNotifyData &) = delete;
    ResourceNotifyData &operator=(const ResourceNotifyData &) = delete;
public:
    Resource *_resource = nullptr;
    ResourcePtr _result;
};

NAVI_TYPEDEF_PTR(ResourceNotifyData);

}

