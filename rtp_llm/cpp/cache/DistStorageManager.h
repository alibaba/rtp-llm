#pragma once

#include "rtp_llm/cpp/cache/DistStorage.h"

namespace rtp_llm {

// for multiple storage
class DistStorageManager {
public:
    DistStorageManager(const kmonitor::MetricsReporterPtr& metrics_reporter): metrics_reporter_(metrics_reporter){};
    bool init(const DistStorageManagerInitParams& init_params);

public:
    bool lookup(const DistStorage::Item& key);
    bool get(DistStorage::Item& item);
    bool put(const DistStorage::Item& item);
    bool del(const DistStorage::Item& item);

private:
    const std::shared_ptr<DistStorage>& getStorage(const DistStorage::Item& item);

private:
    kmonitor::MetricsReporterPtr metrics_reporter_;

    DistStorageManagerInitParams init_params_;
    std::shared_ptr<DistStorage> storage_3fs_;
    std::shared_ptr<DistStorage> storage_local_mem_;
};

}  // namespace rtp_llm