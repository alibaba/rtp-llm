#include "navi/engine/NaviR.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/log/NaviLogManager.h"
#include "navi/log/ConsoleAppender.h"

namespace navi {

const std::string NaviR::RESOURCE_ID = "navi.buildin.navi_r";

NaviR::NaviR()
    : ExternalResource(RESOURCE_ID, RS_EXTERNAL)
{
}

NaviR::~NaviR() {
}

ErrorCode NaviR::Init() const {
    if (_navi) {
        return EC_NONE;
    }
    NAVI_KERNEL_LOG(INFO, "begin init, this [%p]", this);
    auto navi = std::make_shared<Navi>(_name);
    navi->setIsRealNavi();
    navi->setTestMode(_testMode);
    navi->setWaitNoQuery(_waitNoQuery);
    navi->_metricsReporter = _metricsReporter;
    navi->_memoryPoolR = _memoryPoolR;
    navi->_hostInfo = _hostInfo;
    navi->_evTimer = _evTimer;
    navi->_gigRpcServerPtr = _gigRpcServerPtr;
    _navi = navi;
    if (NAVI_TLS_LOGGER && NAVI_TLS_LOGGER->logger) {
        navi->_logger.logger = NAVI_TLS_LOGGER->logger;
    }
    bool ret = false;
    {
        NaviLoggerScope scope(navi->_logger);
        ret = navi->init(_initConfigPath, _gigRpcServer);
    }
    if (!ret) {
        return EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO, "init success, navi: [%p], this [%p]", _navi.get(), this);
    return EC_NONE;
}

ErrorCode NaviR::update() const {
    NAVI_KERNEL_LOG(INFO, "begin update, this [%p]", this);
    bool ret = false;
    {
        auto logger = createLogger();
        if (!logger) {
            return EC_ABORT;
        }
        NaviObjectLogger objectLogger(_navi.get(), "update", logger);
        NaviLoggerScope scope(objectLogger);
        ret = _navi->doUpdate("", _config, logger, _naviSymbolTable, _rootResourceMap, _moduleManager);
    }
    if (!ret) {
        NAVI_KERNEL_LOG(ERROR, "navi [%p] update failed", _navi.get());
        return EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO, "update success, snapshot: [%p], this [%p]", _navi->getSnapshot().get(), this);
    return EC_NONE;
}

NaviLoggerPtr NaviR::createLogger() const {
    auto logManager = std::make_shared<NaviLogManager>(_navi->getInstanceId());
    if (!logManager->init(_config->logConfig)) {
        std::string logConfigStr;
        if (!NaviConfig::prettyDumpToStr(_config->logConfig, logConfigStr)) {
            NAVI_KERNEL_LOG(ERROR, "init real navi log manager failed, dump navi config to str failed");
        } else {
            NAVI_KERNEL_LOG(ERROR, "init real navi log manager failed, config [%s]", logConfigStr.c_str());
        }
        return nullptr;
    }
    auto logger = logManager->createLogger(SessionId());
    if (NAVI_TLS_LOGGER && NAVI_TLS_LOGGER->logger) {
        auto traceAppender = NAVI_TLS_LOGGER->logger->getTraceAppender();
        if (traceAppender) {
            logger->setTraceAppender(traceAppender);
        }
        const auto &consoleAppender = NAVI_TLS_LOGGER->logger->getConsoleAppender();
        if (consoleAppender) {
            logger->addAppender(consoleAppender);
        }
    }
    return logger;
}

REGISTER_BUILDIN_RESOURCE(NaviR);

}
