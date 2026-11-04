#include "navi/ops/ResourceCreateKernel.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Node.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/ops/ResourceKernelDef.h"
#include "navi/ops/ResourceNotifyData.h"
#include "navi/ops/ResourceFlushData.h"
#include "navi/resource/GigMetaR.h"

namespace navi {

ResourceCreateKernel::ResourceCreateKernel()
    : _requireNodeIndex(0)
    , _init(false)
    , _require(false)
    , _produceRequire(false)
    , _inputEof(false)
    , _outputEof(false)
    , _produceEof(false)
    , _resourceGroupIndex(INVALID_INDEX)
    , _stage(RS_UNKNOWN)
{
}

ResourceCreateKernel::~ResourceCreateKernel() {
}

void ResourceCreateKernel::def(KernelDefBuilder &builder) const {
    builder.name(RESOURCE_CREATE_KERNEL)
        .output(RESOURCE_CREATE_OUTPUT, RESOURCE_DATA_TYPE_ID)
        .output(RESOURCE_PRODUCE_OUTPUT, RESOURCE_DATA_TYPE_ID);
}

bool ResourceCreateKernel::config(KernelConfigContext &ctx) {
    ctx.Jsonize(RESOURCE_ATTR_NAME, _resourceName);
    ctx.Jsonize(RESOURCE_ATTR_REQUIRE_NODE, _requireNode, _requireNode);
    ctx.Jsonize(RESOURCE_ATTR_REQUIRE_NODE_INDEX, _requireNodeIndex, _requireNodeIndex);
    int64_t init = 0;
    int64_t require = 0;
    int64_t produceRequire = 0;
    ctx.Jsonize(RESOURCE_ATTR_INIT, init, init);
    ctx.Jsonize(RESOURCE_ATTR_REQUIRE, require, require);
    ctx.Jsonize(RESOURCE_ATTR_PRODUCE_NAME, _produceName, _produceName);
    ctx.Jsonize(RESOURCE_ATTR_PRODUCE_REQUIRE, produceRequire, produceRequire);
    _init = init;
    _require = (0 != require);
    _produceRequire = (0 != produceRequire);
    return true;
}

ErrorCode ResourceCreateKernel::init(KernelInitContext &ctx) {
    _resourceGroupIndex = ctx._node->getResourceInputGroupIndex();
    _multiLayerResourceMap.addResourceMap(&_resourceMap);
    auto localSubGraph = ctx._subGraph;
    _stage = localSubGraph->getResourceStage(_resourceName);
    localSubGraph->collectResourceMap(ctx._node->getScope(), _multiLayerResourceMap);
    _dependVec = localSubGraph->getResourceDependResourceVec(_resourceName);
    _replaceSet = localSubGraph->getResourceReplacerSet(_resourceName);
    _replaceInfoMap = ctx._node->getReplaceInfoMap();
    return EC_NONE;
}

bool ResourceCreateKernel::collectInput(KernelComputeContext &ctx, bool &hasWait) {
    if (INVALID_INDEX == _resourceGroupIndex) {
        _inputEof = true;
        return true;
    }
    GroupDatas datas;
    if (!ctx.doGetGroupInput(_resourceGroupIndex, EGT_GET, datas)) {
        NAVI_KERNEL_LOG(ERROR, "collect input resource failed");
        return false;
    }
    bool allEof = true;
    for (const auto &streamData : datas) {
        if (!addData(streamData.data, streamData.eof, hasWait, allEof)) {
            return false;
        }
    }
    if (_replaceInfoMap) {
        _replaceInfoMap->replace(_dependVec, _resourceMap);
    }
    _inputEof = allEof;
    return true;
}

bool ResourceCreateKernel::addData(const DataPtr &data, bool eof, bool &hasWait, bool &allEof) {
    if (!data) {
        if (eof) {
            return true;
        } else {
            NAVI_KERNEL_LOG(ERROR, "null data not eof");
            return false;
        }
    }
    auto resourceData = std::dynamic_pointer_cast<ResourceData>(data);
    if (!resourceData) {
        auto resourceFlushData = dynamic_cast<ResourceFlushData *>(data.get());
        if (!resourceFlushData) {
            NAVI_KERNEL_LOG(ERROR, "wrong data type: %s", data->getTypeId().c_str());
            return false;
        }
        _flushPipe = resourceFlushData->_pipe;
        return true;
    }
    if (resourceData->_wait) {
        hasWait = true;
    }
    NAVI_KERNEL_LOG(SCHEDULE1,
                    "resource [%s], input ResourceData [%s] resource [%p] eof [%d]",
                    _resourceName.c_str(),
                    resourceData->_name.c_str(),
                    resourceData->_resource.get(),
                    eof);
    if (!eof) {
        allEof = false;
    }
    _resourceMap.overrideResource(resourceData->_name, resourceData->_resource);
    if (_init) {
        _depends[resourceData->_name] = resourceData;
        ResourceData::mergeGigLocalSubResult(resourceData->_gigMetaLocalSignatures, _gigMetaLocalSignatures);
    }
    return true;
}

bool ResourceCreateKernel::createResource(KernelComputeContext &ctx, ResourceDataPtr &output) {
    bool hasWaitInput = false;
    if (!collectInput(ctx, hasWaitInput)) {
        NAVI_KERNEL_LOG(ERROR, "collect input resource failed");
        return false;
    }
    if (_producePipe && !_flushPipe) {
        NAVI_KERNEL_LOG(ERROR, "create resource failed, null flush pipe");
        return false;
    }
    output = createOutput(hasWaitInput);
    output->_dependVec = _dependVec;
    output->_replaceSet = _replaceSet;
    if (_init) {
        output->_dependDatas = _depends;
        auto replaceOk = output->validateReplace(false);
        if (!replaceOk) {
            output->_ec = EC_LACK_REPLACE_RESOURCE;
            NAVI_KERNEL_LOG(WARN, "create resource failed, lack replacer");
            return true;
        }
    }
    if (canReuse()) {
        output->_resource = _output->_resource;
        output->_ec = _output->_ec;
        return true;
    }
    doCreateResource(ctx, output);
    return true;
}

ResourceDataPtr ResourceCreateKernel::createOutput(bool hasWaitInput) const {
    auto output = std::make_shared<ResourceData>();
    output->_name = _resourceName;
    output->_require = _require;
    output->_wait = hasWaitInput;
    output->_stage = _stage;
    if (_init) {
        output->_gigMetaLocalSignatures = _gigMetaLocalSignatures;
    }
    return output;
}

bool ResourceCreateKernel::canReuse() const {
    if (!_output) {
        return false;
    }
    const auto &resource = _output->_resource;
    if (!resource) {
        return false;
    }
    const auto &dependMap = resource->getDepends().getMap();
    for (const auto &pair : dependMap) {
        const auto &name = pair.first;
        const auto &depend = pair.second;
        if (_multiLayerResourceMap.getResource(name) != depend.get()) {
            return false;
        }
    }
    return true;
}

void ResourceCreateKernel::doCreateResource(KernelComputeContext &ctx, ResourceDataPtr &output) {
    ProduceInfo info;
    info._produce = _producePipe;
    info._flush = _flushPipe;
    ResourcePtr resource;
    Node *requireNode = ctx._node;
    if (RS_KERNEL == _stage && !_init) {
        requireNode = ctx._subGraph->getNode(_requireNode, _requireNodeIndex);
        if (!requireNode) {
            NAVI_KERNEL_LOG(ERROR,
                            "create resource failed, get require kernel node "
                            "[%s] index [%ld] failed",
                            _requireNode.c_str(),
                            _requireNodeIndex);
            return;
        }
    }
    ErrorCode ec = EC_NONE;
    if (RS_KERNEL != _stage || !_init) {
        bool checkReuse = _init;
        auto failAsError = (_init && (_produceRequire || _require) && _stage < RS_RUN_GRAPH_EXTERNAL);
        ec = ctx._subGraph->createResource(
            _resourceName, _multiLayerResourceMap, info, checkReuse, requireNode, failAsError, resource);
    }
    output->_resource = std::move(resource);
    output->_ec = ec;
}

void ResourceCreateKernel::collectProduceResult(const ResourceDataPtr &output, ResourceDataPtr &produceResult) {
    if (!_producePipe) {
        return;
    }
    produceResult = std::make_shared<ResourceData>();
    produceResult->_name = _produceName;
    const auto &resource = output->_resource;
    ResourcePtr pipeData;
    while (true) {
        DataPtr data;
        bool eof = false;
        if (!_producePipe->getData(data, eof)) {
            break;
        }
        auto notifyData = dynamic_cast<ResourceNotifyData *>(data.get());
        if (!notifyData || (notifyData->_resource != resource.get())) {
            continue;
        }
        const auto &result = notifyData->_result;
        if (result && (_produceName != result->getName())) {
            NAVI_KERNEL_LOG(ERROR,
                            "wrong produced resource, expect [%s], got [%s]",
                            _produceName.c_str(),
                            result->getName().c_str());
            continue;
        }
        pipeData = result;
    }
    produceResult->_require = _produceRequire;
    if (pipeData) {
        produceResult->_resource = pipeData;
    } else if (_produceResult) {
        produceResult->_resource = _produceResult->_resource;
    }
    if (!resource) {
        NAVI_KERNEL_LOG(
            WARN, "can't produce resource [%s], lack producer [%s]", _produceName.c_str(), _resourceName.c_str());
    }
    produceResult->_stage = _stage;
    if (_init) {
        produceResult->_gigMetaLocalSignatures = _gigMetaLocalSignatures;
        auto gigMeta = dynamic_cast<GigMetaR *>(produceResult->_resource.get());
        if (gigMeta) {
            ResourceData::mergeGigLocalSubResult(gigMeta->getGigMetaLocalSignatures(),
                                                 produceResult->_gigMetaLocalSignatures);
        }
    }
    if (resource && !produceResult->_resource) {
        produceResult->_wait = true;
    } else {
        produceResult->_wait = false;
    }
}

ErrorCode ResourceCreateKernel::compute(KernelComputeContext &ctx) {
    auto beginTimeUs = autil::TimeUtility::currentTime();
    if (needProduce() && !_producePipe) {
        _producePipe = ctx.createAsyncPipe(ActivateStrategy::AS_INDEPENDENT);
        if (!_producePipe) {
            NAVI_KERNEL_LOG(ERROR, "create produce resource pipe failed");
            return EC_ABORT;
        }
        NAVI_KERNEL_LOG(INFO, "resource produce [%s]", _produceName.c_str());
    }
    ResourceDataPtr output;
    if (!createResource(ctx, output)) {
        NAVI_KERNEL_LOG(ERROR, "try create resource failed");
        return EC_ABORT;
    }
    if (!output) {
        NAVI_KERNEL_LOG(SCHEDULE1, "resource [%s] not created", _resourceName.c_str());
        return EC_NONE;
    }
    const auto &asyncR = output->_resource;
    if (asyncR && !asyncR->isReady()) {
        if (!_asyncPipe) {
            _asyncPipe = ctx.createAsyncPipe(ActivateStrategy::AS_INDEPENDENT);
            if (!_asyncPipe) {
                NAVI_KERNEL_LOG(ERROR, "create async resource pipe failed");
                return EC_ABORT;
            }
        }
        if (asyncR->setAsyncPipe(_asyncPipe)) {
            _output = output;
            ctx.setIgnoreDeadlock();
            NAVI_KERNEL_LOG(SCHEDULE1, "async resource [%s] waiting", _resourceName.c_str());
            return EC_NONE;
        }
    }
    if (!_outputEof) {
        _outputEof = _inputEof;
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "finish create resource [%s], resource data [%p] resource [%p] eof [%d] ec [%s]",
                        _resourceName.c_str(),
                        output.get(),
                        output ? output->_resource.get() : nullptr,
                        _outputEof,
                        CommonUtil::getErrorString(output ? output->_ec : EC_NONE));
        if (output) {
            output->initErrorInfo(beginTimeUs);
        }
        if (!ctx.setOutput(0, std::dynamic_pointer_cast<Data>(output), _outputEof)) {
            NAVI_KERNEL_LOG(ERROR, "set output failed");
            return EC_ABORT;
        }
        _output = output;
    }
    ResourceDataPtr produceResult;
    collectProduceResult(output, produceResult);
    if (!_produceEof && _producePipe) {
        auto produceEof = _inputEof && (!needProduce());
        NAVI_KERNEL_LOG(INFO,
                        "pipe [%p], resource produce success, ResourceData[%p], resource[%p], eof [%d]",
                        _producePipe.get(),
                        produceResult.get(),
                        produceResult ? produceResult->_resource.get() : nullptr,
                        produceEof);
        if (produceResult) {
            produceResult->initErrorInfo(beginTimeUs);
        }
        if (!ctx.setOutput(1, std::dynamic_pointer_cast<Data>(produceResult), produceEof)) {
            NAVI_KERNEL_LOG(ERROR, "set produce output failed");
            return EC_ABORT;
        }
        _produceEof = produceEof;
        _produceResult = produceResult;
    }
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(ResourceCreateKernel);

}
