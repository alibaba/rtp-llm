#pragma once

#include "navi/engine/ResourceMap.h"
#include "navi/engine/ResourceConfigContext.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/CollectFlags.h"

namespace navi {

class Node;
class ResourceManager;
struct NamedData;
typedef NaviUnorderedMap<std::string, NamedData> SubNamedDataMap;
class ResourceCreator;
class AsyncPipe;
NAVI_TYPEDEF_PTR(AsyncPipe);

class ProduceInfo {
public:
    std::shared_ptr<AsyncPipe> _produce;
    std::shared_ptr<AsyncPipe> _flush;
};

class ProduceNotifier {
public:
    ProduceNotifier() {
    }
    ~ProduceNotifier() {
    }
public:
    bool notify(const std::shared_ptr<Resource> &produceResult);
private:
    friend class ResourceManager;
private:
    std::string _name;
    std::string _produce;
    Resource *_resource;
    ProduceInfo _info;
};

NAVI_TYPEDEF_PTR(ProduceNotifier);

class ResourceInitContext
{
public:
    ResourceInitContext();
    ResourceInitContext(const ResourceManager *resourceManager,
                        const ResourceCreator *creator,
                        const std::string &configPath,
                        const std::string &bizName,
                        NaviPartId partCount,
                        NaviPartId partId,
                        const ResourceMap *resourceMap,
                        const ProduceNotifierPtr &notifier,
                        const NaviConfigContext &configContext,
                        const SubNamedDataMap *namedDataMap,
                        Node *requireNode);
    ~ResourceInitContext();
private:
    ResourceInitContext(const ResourceInitContext &);
public:
    const std::string &getBizName() const {
        static const std::string unknown = "unknown";
        return _bizName ? *_bizName : unknown;
    }
    NaviPartId getPartCount() const {
        return _partCount;
    }
    NaviPartId getPartId() const {
        return _partId;
    }
    const std::string &getConfigPath() const;
    const ProduceNotifierPtr &getProduceNotifier() const {
        return _notifier;
    }
    ResourceConfigContext getConfigContext() const {
        return _configContext;
    }
    std::shared_ptr<Resource> buildR(const std::string &name, KernelConfigContext ctx, ResourceMap &inputResourceMap) const;
    AsyncPipePtr createRequireKernelAsyncPipe(ActivateStrategy activateStrategy = AS_REQUIRED) const;
private:
    const ResourceManager *_resourceManager = nullptr;
    const ResourceCreator *_creator = nullptr;
    const std::string *_configPath = nullptr;
    const std::string *_bizName = nullptr;
    NaviPartId _partCount = INVALID_NAVI_PART_COUNT;
    NaviPartId _partId = INVALID_NAVI_PART_ID;
    const ResourceMap *_resourceMap = nullptr;
    ProduceNotifierPtr _notifier;
    NaviConfigContext _configContext;
    const SubNamedDataMap *_namedDataMap = nullptr;
    Node *_requireNode = nullptr;
};

}
