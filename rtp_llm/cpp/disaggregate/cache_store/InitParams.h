#pragma once

#include <stdint.h>
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

struct MessagerInitParams {
    uint32_t server_port         = 0;
    uint32_t io_thread_count     = 2;
    uint32_t worker_thread_count = 4;

    uint32_t rdma_server_port         = 0;
    uint32_t rdma_io_thread_count     = 1;
    uint32_t rdma_worker_thread_count = 2;

    int rdma_connect_timeout_ms{250};
    int rdma_qp_count_per_connection{2};
};

class CacheStoreInitParams {
public:
    uint32_t listen_port{15002};

    uint32_t thread_count{2};
    uint32_t queue_size{100};

    // memory util
    void*    stream{nullptr};
    bool     rdma_mode{true};
    uint32_t rdma_listen_port{0};

    int rdma_connect_timeout_ms{250};
    int rdma_qp_count_per_connection{2};

    uint32_t rdma_io_thread_count{4};
    uint32_t rdma_worker_thread_count{2};

    bool enable_metric{true};

    uint32_t messager_io_thread_count     = 4;
    uint32_t messager_worker_thread_count = 32;

    rtp_llm::DeviceBase*         device{nullptr};
    kmonitor::MetricsReporterPtr metrics_reporter;

    // for test
    std::shared_ptr<MemoryUtil> memory_util;
};

}  // namespace rtp_llm