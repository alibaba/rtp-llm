#include "navi/log/NaviLogManager.h"
#include "arpc/common/Exception.h"
#include "navi/log/Clock.h"
#include "navi/log/ConsoleAppender.h"
#include "navi/log/NaviLogger.h"
#include "autil/EnvUtil.h"
#include "navi/perf/NaviSymbolTable.h"
#include <execinfo.h>

namespace navi {

NaviLogManager::NaviLogManager(InstanceId instanceId) {
    _maxMessageLength = 1024;
    m_fileAppenders.reserve(8);
    m_bRunning = false;
    m_notified = false;
    _instanceId = instanceId;
    atomic_set(&_logCounter, 0);
}

NaviLogManager::~NaviLogManager()
{
    stop();
}

bool NaviLogManager::init(const LogConfig &config) {
    if (config.maxMessageLength > 0) {
        _maxMessageLength = config.maxMessageLength;
    } else {
        return false;
    }
    for (const auto &fileConfig : config.fileAppenders) {
        addFileAppender(fileConfig);
    }
    _needConsoleLog = autil::EnvUtil::getEnv("console_log", false);
    m_bRunning = true;
    m_worker.start(worker_routine, this, "NaviLog");
    notify();
    return true;
}

void NaviLogManager::addFileAppender(const FileLogConfig &config) {
    auto appender = new FileAppender(config.levelStr, this, config.fileName.c_str());
    appender->setAutoFlush(config.autoFlush);
    appender->setMaxSize(config.maxFileSize);
    appender->setDelayTime(config.translateDelayTime());
    appender->setCompress(config.compress);
    appender->setCacheLimit(config.cacheLimit);
    appender->setHistoryLogKeepCount(config.logKeepCount);
    appender->setAsyncFlush(config.asyncFlush);
    appender->setFlushThreshold(config.flushThreshold * 1024);
    appender->setFlushIntervalInMS(config.flushInterval);

    for (const auto &btFilter : config.btFilters) {
        appender->addBtFilter(btFilter);
    }
    auto patternLayout = std::make_shared<PatternLayout>();
    patternLayout->setLogPattern(config.logPattern);
    appender->setLayout(patternLayout);
    m_fileAppenders.push_back(appender);
}

NaviLoggerPtr NaviLogManager::createLogger(SessionId id) {
    auto logger = std::make_shared<NaviLogger>(_maxMessageLength, shared_from_this());
    if (!id.valid()) {
        id = getLoggerId();
    } else {
        id.currentInstance = _instanceId;
    }
    logger->init(id);
    if (_needConsoleLog) {
        logger->addAppender(std::make_shared<ConsoleAppender>("DEBUG"));
    }
    return logger;
}

int64_t NaviLogManager::flush(bool force) {
    ScopedLock guardLock(m_flushMutex);
    int64_t now = Clock::nowMS();
    int64_t nextFlushTime = now + s_iDefaultFlushIntervalInMS;
    for (size_t i = 0; i < m_fileAppenders.size(); ++i) {
        int64_t flushtime =
            m_fileAppenders[i]->asyncFlush(now, force);
        if (nextFlushTime > flushtime) {
            nextFlushTime = flushtime;
        }
    }
    return nextFlushTime;
}

void NaviLogManager::loop()
{
    while (true)
    {
        int64_t nexFlushTime = flush(false);
        wait(nexFlushTime);

        if (!m_bRunning) {
            return;
        }
    }

}

void NaviLogManager::stop()
{
    if (m_bRunning == true) {
        m_condition.lock();
        m_bRunning = false;
        m_condition.signal();
        m_condition.unlock();

        m_worker.stop();
    }

    flush(true);

    {
        for (auto appender : m_fileAppenders) {
            delete appender;
        }
        m_fileAppenders.clear();
    }
}

void NaviLogManager::notify()
{
    m_condition.lock();
    m_notified = true;
    m_condition.signal();
    m_condition.unlock();
}

void NaviLogManager::wait(uint64_t usec)
{
    m_condition.lock();
    if (!m_bRunning || m_notified) {
        m_notified = false;
        m_condition.unlock();
        return;
    }
    m_condition.wait(usec);
    m_notified = false;
    m_condition.unlock();
}

void NaviLogManager::worker_routine(void *pArg)
{
    ((NaviLogManager*) pArg)->loop();
}

void NaviLogManager::flushFile(FileAppender* appender)
{
    if (appender != NULL) {
        ScopedLock guardLock(m_flushMutex);
        appender->asyncFlush(Clock::nowMS(), true);
    }
}

SessionId NaviLogManager::getLoggerId() {
    SessionId id;
    id.instance = _instanceId;
    id.currentInstance = _instanceId;
    id.queryId = atomic_inc_return(&_logCounter);
    if (unlikely(0 == id.queryId)) {
        id.queryId = atomic_inc_return(&_logCounter);
    }
    return id;
}

InstanceId NaviLogManager::getInstanceId() const {
    return _instanceId;
}

void NaviLogManager::setNaviSymbolTable(const std::shared_ptr<NaviSymbolTable> &naviSymbolTable) {
    autil::ScopedWriteLock lock(_lock);
    _naviSymbolTable = naviSymbolTable;
}

std::shared_ptr<NaviSymbolTable> NaviLogManager::getNaviSymbolTable() const {
    autil::ScopedReadLock lock(_lock);
    return _naviSymbolTable;
}

std::string NaviLogManager::getCurrentBtString() const {
    auto naviSymbolTable = getNaviSymbolTable();
    if (naviSymbolTable) {
        return getBtStringNavi(naviSymbolTable);
    }
    arpc::common::Exception e;
    e.Init("", "", 0);
    return e.ToString();
}

std::string NaviLogManager::getBtStringNavi(const std::shared_ptr<NaviSymbolTable> &naviSymbolTable) const {
    constexpr int MAX_STACK_COUNT = 64;
    void *stackVec[MAX_STACK_COUNT];
    auto btCount = backtrace(stackVec, MAX_STACK_COUNT);
    std::string btStr;
    for (size_t i = 0; i < btCount; ++i) {
        auto ip = stackVec[i];
        btStr += naviSymbolTable->resolveAddr((size_t)ip) + "\n";
    }
    return btStr;
}

std::string NaviLogManager::getExceptionBt(const autil::legacy::ExceptionBase &e) const {
    auto naviSymbolTable = getNaviSymbolTable();
    std::string btStr;
    if (!naviSymbolTable) {
        return btStr;
    }
    size_t btCount = 0;
    auto stackVec = e.GetStacks(btCount);
    if (!stackVec || (0 == btCount)) {
        return btStr;
    }
    btStr += "\n";
    for (size_t i = 0; i < btCount; ++i) {
        auto ip = stackVec[i];
        btStr += naviSymbolTable->resolveAddr((size_t)ip) + "\n";
    }
    return btStr;
}

NaviLoggerPtr NaviLogManager::createDefaultLogger(InstanceId instanceId, const std::string &name) {
    auto logManager = std::make_shared<NaviLogManager>(instanceId);
    logManager->init(LogConfig::getDefaultLogConfig(name)); // default config;
    return logManager->createLogger();
}
}
