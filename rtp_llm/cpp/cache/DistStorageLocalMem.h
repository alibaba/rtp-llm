#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/DistStorage.h"

namespace rtp_llm {

class DistStorageLocalMem: public DistStorage {
public:
    DistStorageLocalMem(const kmonitor::MetricsReporterPtr& metrics_reporter);
    virtual ~DistStorageLocalMem() = default;

public:
    bool init(const DistStorageLocalMemInitParams& params);
    bool lookup(const DistStorage::Item& item) override;
    bool get(const DistStorage::Item& item) override;
    bool put(const DistStorage::Item& item) override;
    bool del(const DistStorage::Item& item) override;

private:
    kmonitor::MetricsReporterPtr metrics_reporter_;
};

}  // namespace rtp_llm