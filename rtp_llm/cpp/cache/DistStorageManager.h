#pragma once

#include <chrono>
#include <future>

#include "autil/LockFreeThreadPool.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/DistStorage.h"

namespace rtp_llm {

// for multiple storage
class DistStorageManager {
public:
    DistStorageManager(const kmonitor::MetricsReporterPtr& metrics_reporter): metrics_reporter_(metrics_reporter) {};
    ~DistStorageManager();
    bool init(const DistStorageManagerInitParams& init_params);

public:
    bool lookup(const DistStorage::Item& key);
    bool get(DistStorage::Item& item);
    bool put(const DistStorage::Item& item);
    bool putIfNotExist(const DistStorage::Item& item);
    bool del(const DistStorage::Item& item);

private:
    enum class OpType {
        LOOKUP = 0,
        GET    = 1,
        PUT    = 2,
        DEL    = 3,
    };

    const std::shared_ptr<DistStorage> getStorage(const DistStorage::Item& item);
    bool        runWithTimeout(OpType op_type, const std::function<bool()>& func, int timeout_ms) const;
    std::string getOpTypeString(OpType op_type) const;

private:
    kmonitor::MetricsReporterPtr metrics_reporter_;

    DistStorageManagerInitParams init_params_;
    std::shared_ptr<DistStorage> storage_3fs_;
    std::shared_ptr<DistStorage> storage_local_mem_;

    std::unique_ptr<autil::LockFreeThreadPool> wait_task_thread_pool_;
    const size_t                               thread_num_{8};
    const size_t                               queue_size_{2000};
};

}  // namespace rtp_llm