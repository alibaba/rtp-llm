#pragma once

#include "google/protobuf/map.h"
#include "navi/builder/KernelDefBuilder.h"
#include "navi/common.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/proto/KernelDef.pb.h"
#include <autil/HashAlgorithm.h>

namespace navi {

class Kernel;
class KernelRegistryDef;
class CreatorManager;

struct ConstStringHasher {
    inline size_t operator()(const autil::StringView &str) const {
        return autil::HashAlgorithm::hashString(str.data(), str.size(), 0);
    }
};

struct PortInfo
{
    PortInfo(PortIndex index_, const void *def_)
        : index(index_)
        , def(def_)
    {
    }
    PortInfo()
        : def(nullptr)
    {
    }
    bool isValid() const {
        return def && index.isValid();
    }
    bool isGroup() const {
        return index.isGroup();
    }
    PortIndex index;
    const void *def;
};

typedef NaviUnorderedMap<autil::StringView, PortInfo, ConstStringHasher> PortMap;
typedef ::google::protobuf::Map<std::string, std::string> AttrMap;
typedef std::function<Kernel *(void *addr)> KernelCreateFunc;
typedef NaviUnorderedMap<std::string, DynamicResourceGroupDef *>
    DynamicResourceInfoMap;

class ResourceCreator;

class KernelMemEstimator {
public:
    size_t add(size_t size, size_t align, size_t count) {
        align = std::max(8lu, align);
        if (0 == baseAddr) {
            baseAddr += align;
        }
        if (count > 1) {
            size = NAVI_ALIGN(size, align);
        }
        auto offset = NAVI_ALIGN(baseAddr, align);
        baseAddr = offset + size * count;
        return offset;
    }
    size_t getBaseAddr() const {
        return baseAddr;
    }
private:
    size_t baseAddr = 0;
};

class KernelCreator : public Creator
{
private:
    typedef std::function<bool(Kernel *other, int64_t &castOffset)> KernelCastFunc;
public:
    KernelCreator(const char *sourceFile,
                  int32_t line,
                  bool naviRegistry,
                  const KernelDefBuilder &builder,
                  const KernelCreateFunc &func,
                  const KernelCastFunc &castFunc);
    KernelCreator();
    ~KernelCreator();
private:
    KernelCreator(const KernelCreator &);
    KernelCreator& operator=(const KernelCreator &);
public:
    bool init(bool isBuildin, Module *module) override;
public:
    CreatorPtr clone() const;
    void setBindInfos(const KernelDefBuilder &builder);
    const std::string &getName() const override {
        return _def->kernel_name();
    }
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    // todo return error msg
    virtual Kernel *create(void *addr) const {
        return _func(addr);
    }
    int64_t getCastOffsetValue() const {
        return _castOffset;
    }
    int64_t &getCastOffset() {
        return _castOffset;
    }
    uint32_t getNewSize() const {
        return _newSize;
    }
    uint32_t getAlignSize() const {
        return _alignSize;
    }
    size_t collectMemInfo(KernelMemEstimator &estimator) const {
        return estimator.add(_newSize, _alignSize, 1);
    }
    std::string getClassName() const override;
    bool isInlineConfigAndInit() const {
        return _inlineConfigAndInit;
    }
    size_t inputSize() const;
    size_t outputSize() const;
    size_t controlOutputSize() const;
    size_t inputGroupSize() const;
    size_t outputGroupSize() const;
    PortInfo getInputPortInfo(const autil::StringView &port, IndexType index) const;
    PortInfo getInputPortInfo(const std::string &name) const;
    const InputDef *getInputDef(IndexType index) const;
    PortInfo getOutputPortInfo(const std::string &name) const;
    std::string getInputPortName(PortIndex portIndex) const;
    std::string getOutputPortName(PortIndex portIndex) const;
    const std::vector<std::pair<std::string, bool>> &getDependResources() const {
        return _dependResourceVec;
    }
    const std::vector<std::pair<std::string, bool>> &getDependResourcesWithoutSelect() const {
        return _dependResourcesWithoutSelect;
    }
    const std::vector<std::pair<std::string, bool>> &getSelectDependResourceVec() const {
        return _selectDependResourceVec;
    }
    const std::set<std::string> &getEnableBuildResourceSet() const {
        return _enableBuildResourceSet;
    }
    bool addDependBy(const std::string &resource, const DependByDef *def);
    bool getPortTypeStr(const std::string &portName, IoType ioType,
                        std::string &typeStr) const;
    const KernelDef *def() const;
    const BindInfos &getBindInfos() const;
    const DynamicResourceInfoMap &getDynamicGroupInfoMap() const;
    autil::legacy::RapidDocument *getJsonAttrs() const {
        return _jsonAttrsDocument;
    }
    const NaviUnorderedMap<std::string, std::string> &getConfigKeyMap() const {
        return _configKeyMap;
    }
    bool postInit(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool initCastOffset(CreatorManager *manager);
    bool initSelectors(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool initEnableBuilds(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool initConfigKey(const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    void fillProto(StringHashTable &stringHashTable, KernelRegistryDef &registryDef) const;
private:
    // for clone
    KernelCreator(const char *sourceFile,
                  int32_t line,
                  bool naviRegistry,
                  KernelDef *def,
                  const KernelCreateFunc &func,
                  const KernelCastFunc &castFunc,
                  const BindInfos &bindInfos);

private:
    bool validateDef() const;
    bool initInputIndex();
    bool initOutputIndex();
    bool initInputGroupIndex();
    bool initOutputGroupIndex();
    bool initDependResources();
    bool initDynamicGroup();
    bool initAttrs();
    bool addToDependResourceMap(const DependResourceDef &resourceDef);
    bool addResourceDependRecur(const std::string &name,
                                bool require,
                                const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap);
    bool collectCommonGraphDepend(const std::string &candidate,
                                  const NaviUnorderedMap<std::string, ResourceCreator *> &resourceCreatorMap,
                                  NaviUnorderedSet<std::string> &refCountSet);
protected:
    KernelDef *_def;
    uint32_t _newSize = 0;
    uint32_t _alignSize = 0;
    bool _inlineConfigAndInit = false;
    OutputDef _finishPortDef;
    KernelCreateFunc _func;
    KernelCastFunc _castFunc;
    int64_t _castOffset = NAVI_INIT_CAST_OFFSET;
    // indexes for search
    PortMap _inputMap;
    PortMap _outputMap;
    PortMap _controlOutputMap;
    PortMap _inputGroupMap;
    PortMap _outputGroupMap;
    std::map<std::string, bool> _dependResources;
    std::vector<std::pair<std::string, bool>> _dependResourceVec;
    std::vector<std::pair<std::string, bool>> _dependResourcesWithoutSelect;
    std::vector<std::pair<std::string, bool>> _selectDependResourceVec;
    DynamicResourceInfoMap _dynamicGroupMap;
    std::set<std::string> _enableBuildResourceSet;
    NaviUnorderedMap<std::string, std::string> _configKeyMap;
    autil::legacy::RapidDocument *_jsonAttrsDocument = nullptr;
private:
    BindInfos _bindInfos;
};

NAVI_TYPEDEF_PTR(KernelCreator);

}
