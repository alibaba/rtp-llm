#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "kmonitor/client/core/MutableMetric.h"

namespace rtp_llm {

class CacheStoreMetricsReporter: public std::enable_shared_from_this<CacheStoreMetricsReporter> {
public:
    bool init();
    void stop();

    std::shared_ptr<CacheStoreClientStoreMetricsCollector> makeClientStoreMetricsCollector(uint32_t block_count);
    std::shared_ptr<CacheStoreServerStoreMetricsCollector> makeServerStoreMetricsCollector(uint32_t block_count);
    std::shared_ptr<CacheStoreClientLoadMetricsCollector>  makeClientLoadMetricsCollector(uint32_t block_count);
    std::shared_ptr<CacheStoreServerLoadMetricsCollector>
    makeServerLoadMetricsCollector(uint32_t block_count, uint32_t block_size, int64_t request_send_cost_us);

    void reportClientStore(CacheStoreClientStoreMetricsCollector* collector);
    void reportServerStore(CacheStoreServerStoreMetricsCollector* collector);
    void reportClientLoad(CacheStoreClientLoadMetricsCollector* collector);
    void reportServerLoad(CacheStoreServerLoadMetricsCollector* collector);

private:
    bool enable_{false};

    // client load
    std::unique_ptr<kmonitor::MutableMetric> client_load_qps_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_failed_qps_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_block_count_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_remote_load_block_count_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_total_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_local_load_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> client_load_remote_load_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> response_receive_cost_us_;

    // server load
    std::unique_ptr<kmonitor::MutableMetric> server_load_qps_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_failed_qps_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_block_count_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_block_size_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_total_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> request_send_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_write_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> server_load_connect_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> first_block_cost_us_;

    // client store
    std::unique_ptr<kmonitor::MutableMetric> client_store_qps_;
    std::unique_ptr<kmonitor::MutableMetric> client_store_failed_qps_;
    std::unique_ptr<kmonitor::MutableMetric> client_store_block_count_;
    std::unique_ptr<kmonitor::MutableMetric> client_store_total_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> client_store_local_store_cost_us_;
    std::unique_ptr<kmonitor::MutableMetric> client_store_remote_store_cost_us_;

    // server store
    std::unique_ptr<kmonitor::MutableMetric> server_store_qps_;
    std::unique_ptr<kmonitor::MutableMetric> server_store_failed_qps_;
    std::unique_ptr<kmonitor::MutableMetric> server_store_block_count_;
    std::unique_ptr<kmonitor::MutableMetric> server_store_total_cost_us_;
};

}  // namespace rtp_llm