#pragma once

#include "navi/common.h"
#include "navi/proto/KernelDef.pb.h"

namespace navi {

class KernelDef;

class KernelDefBuilder
{
public:
    KernelDefBuilder(KernelDef *def);
    ~KernelDefBuilder();
private:
    KernelDefBuilder(const KernelDefBuilder &);
    KernelDefBuilder& operator=(const KernelDefBuilder &);
public:
    KernelDefBuilder &name(const std::string &name);
    KernelDefBuilder &input(const std::string &name,
                            const std::string &typeId,
                            InputTypeDef inputType = IT_REQUIRE,
                            bool allowDisconnect = false);
    KernelDefBuilder &inputGroup(const std::string &name,
                                 const std::string &typeId,
                                 InputTypeDef channelInputType = IT_REQUIRE,
                                 InputTypeDef groupInputType = IT_REQUIRE,
                                 bool allowDisconnect = false);
    KernelDefBuilder &output(const std::string &name,
                             const std::string &typeId);
    KernelDefBuilder &outputGroup(const std::string &name,
                                  const std::string &typeId);
    KernelDefBuilder &dependOn(const std::string &name, bool require, const StaticResourceBindFunc &binder);
    KernelDefBuilder &resourceConfigKey(const std::string &name, const std::string &configKey);
    KernelDefBuilder &selectR(const std::vector<std::string> &candidates,
                              const ResourceSelectFunc &selectorFunc,
                              bool require,
                              const StaticResourceBindFunc &binder);
    KernelDefBuilder &enableBuildR(const std::vector<std::string> &candidates);
    KernelDefBuilder &dynamicResourceGroup(
            const std::string &groupName,
            const DynamicResourceBindFunc &binder);
    KernelDefBuilder &namedData(const std::string &name, bool require,
                                const NamedDataBindFunc &binder);
    KernelDefBuilder &jsonAttrs(const std::string &attrs);
private:
    KernelDef *def() const;
    const BindInfos &getBindInfos() const;
    friend class KernelCreator;
private:
    KernelDef *_def;
    BindInfos _bindInfos;
};

}
