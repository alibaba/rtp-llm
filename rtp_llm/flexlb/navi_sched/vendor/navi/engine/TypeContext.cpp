#include "navi/engine/TypeContext.h"

namespace navi {

TypeContext::TypeContext(autil::DataBuffer &buffer)
    : _buffer(buffer)
{
}

TypeContext::~TypeContext() {
}

void TypeContext::setRawStr(const std::string &rawStr) {
    _rawStr = rawStr;
}

autil::DataBuffer &TypeContext::getDataBuffer() {
    return _buffer;
}

const std::string &TypeContext::getRawStr() {
    return _rawStr;
}

}
