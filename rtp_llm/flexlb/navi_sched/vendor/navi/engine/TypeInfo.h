#pragma once

#include "navi/common.h"

namespace navi {

class TypeInfo
{
public:
    TypeInfo(const std::string &typeId);
    ~TypeInfo();
private:
    TypeInfo(const TypeInfo &);
    TypeInfo &operator=(const TypeInfo &);
public:
    const std::string &getTypeId() const;
private:
    std::string _typeId;
};

}
