#include "navi/ops/ResourceSaveKernel.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/ops/ResourceData.h"
#include "navi/ops/ResourceKernelDef.h"

namespace navi {

ResourceSaveKernel::ResourceSaveKernel() {
}

ResourceSaveKernel::~ResourceSaveKernel() {
}

void ResourceSaveKernel::def(KernelDefBuilder &builder) const {
    builder.name(RESOURCE_SAVE_KERNEL)
        .inputGroup(RESOURCE_SAVE_KERNEL_INPUT, RESOURCE_DATA_TYPE_ID)
        .output(RESOURCE_SAVE_KERNEL_OUTPUT, "");
}

bool ResourceSaveKernel::config(navi::KernelConfigContext &ctx) {
    NAVI_JSONIZE(ctx, "ignore_produce_error", _ignoreProduceError, _ignoreProduceError);
    return true;
}

ErrorCode ResourceSaveKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

ErrorCode ResourceSaveKernel::compute(KernelComputeContext &ctx) {
    GroupDatas datas;
    if (!ctx.getGroupInput(0, datas)) {
        NAVI_KERNEL_LOG(ERROR, "get input failed");
    }
    multi_call::TopoSignatureSetPtr gigMetaLocalSignatures;
    bool hasWait = false;
    auto size = datas.size();
    bool hasError = false;
    bool abort = false;
    ResourceMap produceResourceMap;
    ResourceErrorInfoPtr errorInfo;
    for (int i = 0; i < size; i++) {
        const auto &streamData = datas[i];
        if (!streamData.hasValue) {
            continue;
        }
        auto resourceData = std::dynamic_pointer_cast<ResourceData>(streamData.data);
        if (!resourceData) {
            NAVI_KERNEL_LOG(ERROR, "create resource failed, unknown error");
            return EC_ABORT;
        }
        const auto &resource = resourceData->_resource;
        auto stage = resourceData->_stage;
        NAVI_KERNEL_LOG(DEBUG,
                        "save input resource [%s], stage[%s], eof [%d]",
                        resourceData->_name.c_str(),
                        ResourceStage_Name(resourceData->_stage).c_str(),
                        streamData.eof);
        if (stage >= RS_RUN_GRAPH_EXTERNAL) {
            PortIndex index(i, 0);
            NAVI_KERNEL_LOG(DEBUG,
                            "set input eof: [%s], stage[%s]",
                            resourceData->_name.c_str(),
                            ResourceStage_Name(resourceData->_stage).c_str());
            ctx.setInputEof(index);
            continue;
        }
        ResourceData::mergeGigLocalSubResult(resourceData->_gigMetaLocalSignatures, gigMetaLocalSignatures);
        if (resource) {
            NAVI_KERNEL_LOG(DEBUG, "create resource success [%s] [%p]", resource->getName().c_str(), resource.get());
            if (streamData.eof) {
                _resourceMap.overrideResource(resource);
            } else {
                produceResourceMap.overrideResource(resource);
            }
            continue;
        }
        if (!resourceData->_require) {
            NAVI_KERNEL_LOG(DEBUG, "not required resource ignored [%s]", resourceData->_name.c_str());
            continue;
        }
        if (resourceData->_wait) {
            NAVI_KERNEL_LOG(INFO, "need wait for resource creation [%s]", resourceData->_name.c_str());
            hasWait = true;
            continue;
        }
        bool ignoreError = (!streamData.eof && _ignoreProduceError && _saveSuccessCount > 0);
        if (resourceData->validate(false, !ignoreError)) {
            NAVI_KERNEL_LOG(DEBUG, "resource ignored [%s]", resourceData->_name.c_str());
            continue;
        }
        if (ignoreError) {
            NAVI_KERNEL_LOG(INFO, "create resource failed [%s]", resourceData->_name.c_str());
            hasError = true;
        } else {
            if (resourceData->_errorInfo) {
                if (!errorInfo || errorInfo->latterThan(*resourceData->_errorInfo)) {
                    errorInfo = resourceData->_errorInfo;
                }
            }
            NAVI_KERNEL_LOG(ERROR, "create resource failed [%s]", resourceData->_name.c_str());
            abort = true;
        }
    }
    if (abort) {
        _resourceErrorInfoCollectorR->updateErrorInfo(this, errorInfo);
        return EC_ABORT;
    }
    if (hasWait) {
        NAVI_KERNEL_LOG(INFO, "save skipped, need wait");
    } else {
        if (!hasError) {
            produceResourceMap.addResource(_resourceMap);
            if (!ctx._subGraph->saveResource(produceResourceMap)) {
                NAVI_KERNEL_LOG(ERROR, "save resource failed");
                return EC_ABORT;
            }
            _saveSuccessCount++;
        } else {
            NAVI_KERNEL_LOG(INFO, "some resource failed, save resource ignored");
        }
    }
    auto data = std::make_shared<ResourceData>();
    data->_require = false;
    data->_ec = (hasError || hasWait) ? EC_ABORT : EC_NONE;
    data->_errorInfo = errorInfo;
    data->_gigMetaLocalSignatures = gigMetaLocalSignatures;
    if (!ctx.setOutput(0, data, datas.eof())) {
        return EC_ABORT;
    }
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(ResourceSaveKernel);

}

