#pragma once

#include "navi/engine/Kernel.h"
#include "navi/ops/ResourceData.h"
#include "navi/engine/MultiLayerResourceMap.h"

namespace navi {

class ReplaceInfoMap;

class ResourceCreateKernel : public Kernel
{
public:
    ResourceCreateKernel();
    ~ResourceCreateKernel();
    ResourceCreateKernel(const ResourceCreateKernel &) = delete;
    ResourceCreateKernel &operator=(const ResourceCreateKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
private:
    bool createResource(KernelComputeContext &ctx, ResourceDataPtr &output);
    bool collectInput(KernelComputeContext &ctx, bool &hasWait);
    bool addData(const DataPtr &data, bool eof, bool &hasWait, bool &allEof);
    ResourceDataPtr createOutput(bool hasWaitInput) const;
    bool canReuse() const;
    void doCreateResource(KernelComputeContext &ctx, ResourceDataPtr &output);
    void collectProduceResult(const ResourceDataPtr &output, ResourceDataPtr &produceResult);
    bool needProduce() const {
        return !_produceName.empty();
    }
    bool validateReplace() const;
private:
    std::string _resourceName;
    std::string _produceName;
    std::string _requireNode;
    int64_t _requireNodeIndex;
    bool _init : 1;
    bool _require : 1;
    bool _produceRequire : 1;
    bool _inputEof : 1;
    bool _outputEof : 1;
    bool _produceEof : 1;
    IndexType _resourceGroupIndex;
    ResourceStage _stage;
    ResourceMap _resourceMap;
    MultiLayerResourceMap _multiLayerResourceMap;
    const std::vector<std::pair<std::string, bool>> *_dependVec = nullptr;
    std::map<std::string, std::shared_ptr<ResourceData>> _depends;
    const NaviUnorderedSet<std::string> *_replaceSet = nullptr;
    const ReplaceInfoMap *_replaceInfoMap = nullptr;
    AsyncPipePtr _flushPipe;
    AsyncPipePtr _producePipe;
    ResourceDataPtr _output;
    ResourceDataPtr _produceResult;
    AsyncPipePtr _asyncPipe;
    multi_call::TopoSignatureSetPtr _gigMetaLocalSignatures;
};

}
