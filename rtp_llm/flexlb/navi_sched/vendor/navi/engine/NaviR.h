#pragma once

#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"
#include <multi_call/rpc/GigRpcServer.h>

namespace kmonitor {
class MetricsReporter;
}

namespace navi {

class Navi;
class NaviConfig;
class NaviHostInfo;
class ModuleManager;
class NaviSymbolTable;
class MemoryPoolR;
class EvTimer;

class NaviR : public navi::ExternalResource
{
public:
    NaviR();
    ~NaviR();
    NaviR(const NaviR &) = delete;
    NaviR &operator=(const NaviR &) = delete;
public:
    ErrorCode Init() const;
    ErrorCode update() const;
private:
    NaviLoggerPtr createLogger() const;
public:
    static const std::string RESOURCE_ID;
private:
    RESOURCE_DEPEND_DECLARE();
private:
    friend class Navi;
private:
    std::shared_ptr<multi_call::GigRpcServer> _gigRpcServerPtr;
    multi_call::GigRpcServer *_gigRpcServer = nullptr;
    std::string _name;
    std::string _initConfigPath;
    TestMode _testMode = TM_NONE;
    bool _waitNoQuery = true;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    std::shared_ptr<MemoryPoolR> _memoryPoolR;
    std::shared_ptr<NaviHostInfo> _hostInfo;
    std::shared_ptr<EvTimer> _evTimer;
    std::shared_ptr<NaviSymbolTable> _naviSymbolTable;

    std::shared_ptr<NaviConfig> _config;
    ResourceMap _rootResourceMap;
    std::shared_ptr<ModuleManager> _moduleManager;
    mutable std::shared_ptr<Navi> _navi;
};

NAVI_TYPEDEF_PTR(NaviR);

}
