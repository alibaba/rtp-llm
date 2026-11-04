#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "autil/Lock.h"
#include "autil/Log.h"
#include "kmonitor_adapter/Monitor.h"

namespace kmonitor_adapter {

class MonitorFactory {
private:
    MonitorFactory();
    ~MonitorFactory();

public:
    Monitor *createMonitor(std::string serviceName);
    void close();

private:
    void init();

public:
    static MonitorFactory *getInstance();

private:
    std::map<std::string, Monitor *> _monitorMap;
    std::atomic_bool _valid{false};
    bool _owner = false;
    static autil::RecursiveThreadMutex _mutex;
};

} // namespace kmonitor_adapter
