#include "navi/engine/Type.h"
#include "navi/engine/StringHashTable.h"
#include "navi/proto/NaviSummary.pb.h"

namespace navi {

Type::Type(const char *sourceFile, int32_t line, const std::string &typeId)
    : Creator(sourceFile, line)
    , _typeId(typeId)
{
}

Type::~Type() {
}

TypeErrorCode Type::serialize(TypeContext &ctx, const DataPtr &data) const {
    return TEC_NOT_SUPPORT;
}

TypeErrorCode Type::deserialize(TypeContext &ctx, DataPtr &data) const {
    return TEC_NOT_SUPPORT;
}

void Type::fillProto(StringHashTable &stringHashTable, TypeRegistryDef &registryDef) const {
    Creator::fillProto(stringHashTable, *registryDef.mutable_registry());
    registryDef.set_type_id_hash(stringHashTable.getHash(_typeId));
}

}
