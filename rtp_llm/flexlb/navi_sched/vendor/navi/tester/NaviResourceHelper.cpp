#include "navi/tester/NaviResourceHelper.h"

#include "autil/TimeUtility.h"
#include "autil/legacy/fast_jsonizable.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/MetricsReporterR.h"
#include "navi/config/NaviConfig.h"
#include "navi/config/NaviRegistryConfigMap.h"
#include "navi/engine/Resource.h"
#include "navi/resource/MemoryPoolR.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "navi/tester/NaviResDummyKernel.h"

using namespace autil::legacy;
using namespace autil;

namespace navi {

NaviResourceHelper::NaviResourceHelper()
    : _logLevel("warn")
{
    initBaseRes();
}

NaviResourceHelper::~NaviResourceHelper() {
    std::vector<std::vector<std::shared_ptr<Resource>>> resourceByStage;
    resourceByStage.resize(RS_UNKNOWN);
    auto resources = _resourceMap.getMap();
    for (auto &[_, res] : resources) {
        navi::ResourceDef resourceDef;
        ResourceDefBuilder builder(&resourceDef);
        res->def(builder);
        resourceByStage[resourceDef.stage()].emplace_back(res);
    }
    resources.clear();
    _resourceMap.clear();
    _testers.clear();
    // destruct by stage descend order
    for (int i = resourceByStage.size() - 1; i >= 0; --i) {
        resourceByStage[i].clear();
    }
}

bool NaviResourceHelper::snapshotConfig(const std::string &name, const std::string &configStr) {
    NaviRegistryConfig config;
    if (!config.loadConfig(name, configStr)) {
        NAVI_KERNEL_LOG(ERROR, "set config for resource[%s] failed", name.c_str());
        return false;
    }
    _snapshotConfigMap.setConfig(config);
    return true;
}

bool NaviResourceHelper::config(const std::string &name, const std::string &configStr) {
    NaviRegistryConfig config;
    if (!config.loadConfig(name, configStr)) {
        NAVI_KERNEL_LOG(ERROR, "set config for resource[%s] failed", name.c_str());
        return false;
    }
    _configMap.setConfig(config);
    return true;
}

bool NaviResourceHelper::config(const std::string &configStr) {
    RapidDocument document;
    auto ret = NaviConfig::parseToDocument(configStr, document);
    if (!ret) {
        NAVI_KERNEL_LOG(ERROR, "parse resource config failed, config [%s]", configStr.c_str());
        return false;
    }
    std::vector<NaviRegistryConfig> configVec;
    try {
        FromRapidValue(configVec, document);
    } catch (const std::exception &e) {
        NAVI_KERNEL_LOG(ERROR, "jsonize navi registry config vec has exception [%s]", e.what());
        return false;
    }
    for (auto &config : configVec) {
        _configMap.setConfig(std::move(config));
    }
    _document.emplace_back(std::move(document)); // hold document
    return true;
}

bool NaviResourceHelper::loadConfig(const std::string &configFile) { return false; }

void NaviResourceHelper::kernelConfig(const std::string &configStr) { _kernelConfig = configStr; }

void NaviResourceHelper::namedData(const std::string &name, std::shared_ptr<Data> data) {
    // TODO: check data name duplication??
    _namedDatas[name] = std::move(data);
}

std::shared_ptr<Resource> NaviResourceHelper::getOrCreateResPtr(const std::string &name) {
    auto res = _resourceMap.getResource(name);
    if (res) {
        return res ? res->shared_from_this() : nullptr;
    }
    return createResPtr(name);
}

Resource *NaviResourceHelper::getOrCreateRes(const std::string &name) { return getOrCreateResPtr(name).get(); }

void NaviResourceHelper::initBaseRes() {
    {
        auto resource = std::make_shared<MemoryPoolR>();
        resource->Init({});
        _resourceMap.addResource(resource);
    }
    {
        kmonitor::MetricsReporterPtr reporter(new kmonitor::MetricsReporter("naviResHelper", {}));
        _resourceMap.addResource(std::make_shared<kmonitor::MetricsReporterR>(reporter));
    }
}

std::shared_ptr<Resource> NaviResourceHelper::createResPtr(const std::string &name) {
    std::string kernelName = registerResKernel(name);
    KernelTesterBuilder builder(TM_RESOURCE_TEST);
    builder.logLevel(_logLevel);
    builder.kernel(kernelName);
    builder.kernelConfig(_kernelConfig);
    builder.input("default_input");
    for (auto &pair : _namedDatas) {
        builder.namedData(pair.first, pair.second);
    }
    for (const auto &module : _modules) {
        builder.module(module);
    }
    for (const auto &pair : _snapshotConfigMap.getConfigMap()) {
        builder.snapshotResourceConfig(pair.second.name, FastToJsonString(pair.second.config));
    }
    for (const auto &pair : _configMap.getConfigMap()) {
        builder.resourceConfig(pair.second.name, FastToJsonString(pair.second.config));
    }
    builder.resource(_resourceMap);
    auto tester = builder.build();
    if (!tester) {
        NAVI_KERNEL_LOG(ERROR, "build kernel tester failed");
        return {};
    }
    if (tester->hasError()) {
        NAVI_KERNEL_LOG(ERROR, "kernel tester has error[%s]", tester->getErrorMessage().c_str());
        return {};
    }
    _testers.push_back(tester);
    NAVI_KERNEL_LOG(DEBUG, "build kernel tester end");
    auto ret = _resourceMap.getResource(name);
    if (ret) {
        return ret->shared_from_this();
    } else {
        return nullptr;
    }
}

void NaviResourceHelper::collectDependRes(const std::shared_ptr<Resource> &resource) {
    if (!resource) {
        return;
    }
    auto &name = resource->getName();
    NAVI_KERNEL_LOG(DEBUG, "collect resource[%p,%s]", resource.get(), name.c_str());
    if (_resourceMap.hasResource(name)) {
        return;
    }
    for (const auto &pair : resource->_dependMap.getMap()) {
        collectDependRes(pair.second);
    }
    if (!_resourceMap.addResource(resource)) {
        assert(false);
    }
}

std::string NaviResourceHelper::registerResKernel(const std::string &name) {
    char tmpKernelName[512];
    snprintf(tmpKernelName,
             512,
             "resDummyKernelR:%s_S:%pT:%ld",
             name.c_str(),
             this,
             TimeUtility::monotonicTimeUs()); // TODO check name duplication
    std::string kernelName = tmpKernelName;

    auto *registry = ::navi::CreatorRegistry::buildin(navi::RT_KERNEL);
    registry->setInited(false);
    auto ret = registry->addCreatorFunc([kernelName, name, this]() -> ::navi::Creator * {
        auto def = new navi::KernelDef();
        navi::KernelDefBuilder builder(def);
        NaviResDummyKernel thisKernel;
        thisKernel.def(builder);
        builder.name(kernelName)
            .dependOn(name,
                      true,
                      [this](void *obj,
                             void *castObj,
                             navi::Resource *depend,
                             int64_t *objOffset,
                             int64_t *dependOffset) {
                          if (unlikely(objOffset && dependOffset)) {
                              *objOffset = NAVI_INVALID_CAST_OFFSET;
                              *dependOffset = NAVI_INVALID_CAST_OFFSET;
                              return false;
                          }
                          NAVI_KERNEL_LOG(DEBUG,
                                          "depend resource[%p] with kernel[%p]",
                                          depend,
                                          obj);
                          if (!depend) {
                              return true;
                          }
                          this->collectDependRes(depend->shared_from_this());
                          return true;
                      })
            .input("default_input", "");
        thisKernel.__navi__declare_depend(builder);
        auto newInfo = def->mutable_new_info();
        newInfo->set_new_size(sizeof(NaviResDummyKernel));
        newInfo->set_align_size(alignof(NaviResDummyKernel));
        return new ::navi::KernelCreator(
            __FILE__,
            __LINE__,
            false,
            builder,
            [](void *addr) {
                auto ptr = new (addr) NaviResDummyKernel();
                return ptr;
            },
            [](navi::Kernel *other, int64_t &castOffset) {
                NaviResDummyKernel *castValue = dynamic_cast<NaviResDummyKernel *>(other);
                if (castValue) {
                    castOffset = (((char *)castValue) - ((char *)other));
                    return true;
                } else {
                    return false;
                }
            });
    });
    registry->setInited(true);
    NAVI_KERNEL_LOG(DEBUG, "register kernel[%s]", kernelName.c_str());
    if (!ret) {
        assert(false);
    }
    return kernelName;
}

bool NaviResourceHelper::addExternalRes(std::shared_ptr<Resource> resource) {
    return _resourceMap.addResource(std::move(resource));
}

void NaviResourceHelper::setModules(const std::vector<std::string> &modules) {
    _modules = modules;
}

void NaviResourceHelper::logLevel(const std::string &level) {
    _logLevel = level;
}

}
