#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/Resource.h"
#include "navi/engine/Type.h"

namespace navi {

class ResourceFlushDataType : public Type {
public:
    ResourceFlushDataType();
private:
    ResourceFlushDataType(const ResourceFlushDataType &);
    ResourceFlushDataType &operator=(const ResourceFlushDataType &);
public:
    static const std::string TYPE_ID;
};

class ResourceFlushData : public Data
{
public:
    ResourceFlushData();
    ~ResourceFlushData();
    ResourceFlushData(const ResourceFlushData &) = delete;
    ResourceFlushData &operator=(const ResourceFlushData &) = delete;
public:
    std::shared_ptr<AsyncPipe> _pipe;
};

NAVI_TYPEDEF_PTR(ResourceFlushData);

}

