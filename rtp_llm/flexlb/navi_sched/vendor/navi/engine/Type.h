#pragma once

#include "navi/common.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/engine/Data.h"
#include "navi/engine/TypeContext.h"
#include "navi/resource/MemoryPoolR.h"

namespace navi {

class TypeRegistryDef;
class StringHashTable;

class Type : public Creator
{
public:
    Type(const char *sourceFile, int32_t line, const std::string &typeId);
    virtual ~Type();
private:
    Type(const Type &);
    Type& operator=(const Type &);
public:
    const std::string &getName() const override {
        return _typeId;
    }
    std::shared_ptr<autil::mem_pool::Pool> getPool() const {
        if (_memoryPoolR) {
            return _memoryPoolR->getPoolPtrWithQC();
        } else {
            return std::make_shared<autil::mem_pool::ChunkPool>(64 * 1024);
        }
    }
public:
    virtual TypeErrorCode serialize(TypeContext &ctx,
                                    const DataPtr &data) const;
    virtual TypeErrorCode deserialize(TypeContext &ctx, DataPtr &data) const;
private:
    void setMemoryPoolR(const MemoryPoolRPtr &memoryPoolR) {
        _memoryPoolR = memoryPoolR;
    }
    void fillProto(StringHashTable &stringHashTable, TypeRegistryDef &registryDef) const;
private:
    friend class CreatorManager;
private:
    std::string _typeId;
    MemoryPoolRPtr _memoryPoolR;
};

NAVI_TYPEDEF_PTR(Type);

#define NAVI_TYPE_PARAM __FILE__, __LINE__

#define REGISTER_TYPE(NAME) REGISTER_TYPE_HELPER(__COUNTER__, NAME)
#define REGISTER_TYPE_HELPER(ctr, NAME) REGISTER_TYPE_UNIQ(ctr, NAME, false)
#define REGISTER_TYPE_UNIQ(ctr, NAME, naviRegistry)                                                                    \
    __attribute__((constructor)) static void Register##NAME##Type() {                                                  \
        ::navi::CreatorRegistry::current(navi::RT_TYPE)->addCreatorFunc([]() -> ::navi::Creator * {                    \
            auto creator = new NAME();                                                                                 \
            creator->setNaviRegistry(naviRegistry);                                                                    \
            return creator;                                                                                            \
        });                                                                                                            \
    }
}
