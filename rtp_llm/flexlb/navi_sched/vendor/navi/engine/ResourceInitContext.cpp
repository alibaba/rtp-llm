#include "navi/log/NaviLogger.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/ResourceInitContext.h"
#include "navi/engine/Node.h"
#include "navi/engine/ResourceManager.h"
#include "navi/ops/ResourceNotifyData.h"
#include "navi/ops/ResourceData.h"

namespace navi {

bool ProduceNotifier::notify(const std::shared_ptr<Resource> &produceResult) {
    if (!_info._produce || !_info._flush) {
        NAVI_KERNEL_LOG(ERROR, "notify failed, null pipe");
        return false;
    }
    if (produceResult) {
        // init def
        produceResult->initName();
    }
    auto notifyData = std::make_shared<ResourceNotifyData>();
    notifyData->_resource = _resource;
    notifyData->_result = produceResult;
    auto ec = _info._produce->setDataOverride(std::dynamic_pointer_cast<Data>(notifyData), true);
    if (ec != EC_NONE) {
        NAVI_KERNEL_LOG(ERROR, "set produce data failed");
        return false;
    }
    auto flushData = std::make_shared<ResourceData>();
    flushData->_name = _name;
    flushData->_resource = produceResult;
    ec = _info._flush->setData(flushData);
    return (ec == EC_NONE);
}

ResourceInitContext::ResourceInitContext() : _partCount(1), _partId(0) {}

ResourceInitContext::ResourceInitContext(
        const ResourceManager *resourceManager,
        const ResourceCreator *creator,
        const std::string &configPath,
        const std::string &bizName,
        NaviPartId partCount,
        NaviPartId partId,
        const ResourceMap *resourceMap,
        const ProduceNotifierPtr &notifier,
        const NaviConfigContext &configContext,
        const SubNamedDataMap *namedDataMap,
        Node *requireNode)
    : _resourceManager(resourceManager)
    , _creator(creator)
    , _configPath(&configPath)
    , _bizName(&bizName)
    , _partCount(partCount)
    , _partId(partId)
    , _resourceMap(resourceMap)
    , _notifier(notifier)
    , _configContext(configContext)
    , _namedDataMap(namedDataMap)
    , _requireNode(requireNode)
{
}

ResourceInitContext::~ResourceInitContext() {
}

const std::string &ResourceInitContext::getConfigPath() const {
    return *_configPath;
}

ResourcePtr
ResourceInitContext::buildR(const std::string &name, KernelConfigContext ctx, ResourceMap &inputResourceMap) const {
    if (!_requireNode) {
        NAVI_KERNEL_LOG(ERROR, "only RS_KERNEL resource can buildR");
        return nullptr;
    }
    const auto &enableSet = _creator->getEnableBuildResourceSet();
    if (enableSet.end() == enableSet.find(name)) {
        NAVI_KERNEL_LOG(ERROR, "can't build resource [%s], not enabled", name.c_str());
        return nullptr;
    }
    MultiLayerResourceMap multiLayerResourceMap;
    _requireNode->getGraph()->collectResourceMap(_requireNode->getScope(), multiLayerResourceMap);
    multiLayerResourceMap.addResourceMap(_resourceMap);
    auto resource = _resourceManager->buildResourceRecur(name,
                                                         RS_KERNEL,
                                                         &ctx,
                                                         _namedDataMap,
                                                         multiLayerResourceMap,
                                                         _requireNode->getReplaceInfoMap(),
                                                         _requireNode,
                                                         false,
                                                         inputResourceMap);
    if (resource) {
        return resource->shared_from_this();
    } else {
        return nullptr;
    }
}

AsyncPipePtr ResourceInitContext::createRequireKernelAsyncPipe(ActivateStrategy activateStrategy) const {
    if (RS_KERNEL != _creator->getStage() || !_requireNode) {
        NAVI_KERNEL_LOG(DEBUG, "only RS_KERNEL resource can create async pipe");
        return nullptr;
    }
    return _requireNode->createAsyncPipe(activateStrategy);
}

}
