#include "navi/ops/ResourcePublishKernel.h"
#include "navi/ops/ResourceKernelDef.h"
#include "navi/ops/ResourceData.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

ResourcePublishKernel::ResourcePublishKernel() {
}

ResourcePublishKernel::~ResourcePublishKernel() {
}

void ResourcePublishKernel::def(KernelDefBuilder &builder) const {
    builder.name(RESOURCE_PUBLISH_KERNEL)
        .inputGroup(RESOURCE_PUBLISH_KERNEL_INPUT, RESOURCE_DATA_TYPE_ID)
        .output(RESOURCE_PUBLISH_KERNEL_OUTPUT, "");
}

bool ResourcePublishKernel::config(KernelConfigContext &ctx) {
    return true;
}

ErrorCode ResourcePublishKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

ErrorCode ResourcePublishKernel::compute(KernelComputeContext &ctx) {
    GroupPeekScope scope(ctx, 0);
    if (!scope.ok()) {
        NAVI_KERNEL_LOG(ERROR, "get input failed");
        return EC_ABORT;
    }
    multi_call::TopoSignatureSetPtr gigMetaLocalSignatures;
    const auto datas = scope.datas();
    for (const auto &streamData : datas) {
        if (streamData.eof) {
            continue;
        }
        auto resourceData = std::dynamic_pointer_cast<ResourceData>(streamData.data);
        if (!resourceData) {
            NAVI_KERNEL_LOG(ERROR, "create resource failed, unknown error");
            return EC_ABORT;
        }
        if (EC_NONE != resourceData->_ec) {
            NAVI_KERNEL_LOG(INFO, "save resource has error, publish ignored");
            return EC_NONE;
        }
        ResourceData::mergeGigLocalSubResult(resourceData->_gigMetaLocalSignatures, gigMetaLocalSignatures);
    }
    if (!ctx._subGraph->publishResource()) {
        NAVI_KERNEL_LOG(ERROR, "publish resource failed");
        return EC_ABORT;
    }
    auto data = std::make_shared<ResourceData>();
    data->_require = false;
    data->_gigMetaLocalSignatures = gigMetaLocalSignatures;
    ctx.setOutput(0, data, datas.eof());
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(ResourcePublishKernel);

}
