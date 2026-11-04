#pragma once

#include "navi/common.h"
#include "navi/proto/KernelDef.pb.h"

namespace navi {

class ResourceDefBuilder
{
public:
    ResourceDefBuilder(ResourceDef *def, bool nameOnly = false);
    ~ResourceDefBuilder();
private:
    ResourceDefBuilder(const ResourceDefBuilder &);
    ResourceDefBuilder &operator=(const ResourceDefBuilder &);
public:
    ResourceDefBuilder &name(const std::string &name, ResourceStage stage);
    ResourceDefBuilder &dependOn(const std::string &name,
                                 bool require,
                                 const StaticResourceBindFunc &binder);
    ResourceDefBuilder &namedData(const std::string &name, bool require,
                                  const NamedDataBindFunc &binder);
    ResourceDefBuilder &dependByResource(const std::string &resource,
                                         const std::string &dynamicGroup);
    ResourceDefBuilder &dependByKernel(const std::string &kernel,
                                       const std::string &dynamicGroup);
    ResourceDefBuilder &enableBuildR(const std::vector<std::string> &candidates);
    ResourceDefBuilder &dynamicResourceGroup(
            const std::string &groupName,
            const DynamicResourceBindFunc &binder);
    ResourceDefBuilder &produce(const std::string &name);
    ResourceDefBuilder &enableReplaceR(const std::string &name);
    ResourceDefBuilder &buildMode(ResourceBuildMode mode);
    ResourceDefBuilder &setAsync(bool isAsyncR);
private:
    ResourceDef *def() const;
    const BindInfos &getBindInfos() const;
    friend class ResourceCreator;
private:
    ResourceDef *_def;
    BindInfos *_bindInfos = nullptr;
};

}
