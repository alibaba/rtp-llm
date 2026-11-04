#pragma once

#include "autil/DataBuffer.h"

namespace navi {

class TypeContext
{
public:
    TypeContext(autil::DataBuffer &buffer);
    ~TypeContext();
private:
    TypeContext(const TypeContext &);
    TypeContext &operator=(const TypeContext &);
private:
    void setRawStr(const std::string &rawStr);
public:
    autil::DataBuffer &getDataBuffer();
    const std::string &getRawStr();
private:
    friend struct SerializeData;
private:
    autil::DataBuffer &_buffer;
    std::string _rawStr;
};

}
