#include "navi/engine/TypeInfo.h"

namespace navi {

TypeInfo::TypeInfo(const std::string &typeId)
    : _typeId(typeId)
{
}

TypeInfo::~TypeInfo() {
}

const std::string &TypeInfo::getTypeId() const {
    return _typeId;
}

}

