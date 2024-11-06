#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitor.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

bool CacheStoreMetricsReporter::init() {
    auto kMonitor = kmonitor::KMonitorFactory::GetKMonitor("cache_store");
    if (kMonitor == nullptr) {
        FT_LOG_WARNING("arpc kmonitor client metric reporter init failed");
        return false;
    }
    kMonitor->SetServiceName("rtp_llm.cache_store");

#define REGISTER_METRIC(target, name, metricType, level)                                                               \
    do {                                                                                                               \
        std::string metricName = (name);                                                                               \
        target.reset(kMonitor->RegisterMetric(metricName, (metricType), (level)));                                     \
        if (nullptr == target) {                                                                                       \
            FT_LOG_ERROR("failed to register metric:[%s]", metricName.c_str());                                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

    // client load
    REGISTER_METRIC(client_load_qps_, "client.load.qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(client_load_failed_qps_, "client.load.failed_qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(client_load_block_count_, "client.load.block_count", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(client_load_total_cost_us_, "client.load.total_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(
        client_load_local_load_cost_us_, "client.load.local_load_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(
        client_load_remote_load_block_count_, "client.load.remote_load_block_count", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(
        client_load_remote_load_cost_us_, "client.load.remote_load_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(response_receive_cost_us_, "load.response_receive_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);

    // server load
    REGISTER_METRIC(server_load_qps_, "server.load.qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_failed_qps_, "server.load.failed_qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_block_count_, "server.load.block_count", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_block_size_, "server.load.block_size", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_total_cost_us_, "server.load.total_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(request_send_cost_us_, "load.request_send_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_connect_cost_us_, "server.load.connect_cost_us_", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(server_load_write_cost_us_, "server.load.write_cost_us_", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(first_block_cost_us_, "first_block_cost_us_", kmonitor::GAUGE, kmonitor::NORMAL);

    // client store
    REGISTER_METRIC(client_store_qps_, "client.store.qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(client_store_failed_qps_, "client.store.failed_qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(client_store_block_count_, "client.store.block_count", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(client_store_total_cost_us_, "client.store.total_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(
        client_store_local_store_cost_us_, "client.store.local_store_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(
        client_store_remote_store_cost_us_, "client.store.remote_store_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);

    // server store
    REGISTER_METRIC(server_store_qps_, "server.store.qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(server_store_failed_qps_, "server.store.failed_qps", kmonitor::QPS, kmonitor::NORMAL);
    REGISTER_METRIC(server_store_block_count_, "server.store.block_count", kmonitor::GAUGE, kmonitor::NORMAL);
    REGISTER_METRIC(server_store_total_cost_us_, "server.store.total_cost_us", kmonitor::GAUGE, kmonitor::NORMAL);

#undef REGISTER_METRIC

    enable_ = true;
    return true;
}

void CacheStoreMetricsReporter::stop() {
    enable_ = false;
}

std::shared_ptr<CacheStoreClientLoadMetricsCollector>
CacheStoreMetricsReporter::makeClientLoadMetricsCollector(uint32_t block_count) {
    return enable_ ? std::make_shared<CacheStoreClientLoadMetricsCollector>(shared_from_this(), block_count) : nullptr;
}

std::shared_ptr<CacheStoreServerLoadMetricsCollector> CacheStoreMetricsReporter::makeServerLoadMetricsCollector(
    uint32_t block_count, uint32_t block_size, int64_t request_send_cost_us) {
    return enable_ ? std::make_shared<CacheStoreServerLoadMetricsCollector>(
               shared_from_this(), block_count, block_size, request_send_cost_us) :
                     nullptr;
}

std::shared_ptr<CacheStoreClientStoreMetricsCollector>
CacheStoreMetricsReporter::makeClientStoreMetricsCollector(uint32_t block_count) {
    return enable_ ? std::make_shared<CacheStoreClientStoreMetricsCollector>(shared_from_this(), block_count) : nullptr;
}

std::shared_ptr<CacheStoreServerStoreMetricsCollector>
CacheStoreMetricsReporter::makeServerStoreMetricsCollector(uint32_t block_count) {
    return enable_ ? std::make_shared<CacheStoreServerStoreMetricsCollector>(shared_from_this(), block_count) : nullptr;
}

void CacheStoreMetricsReporter::reportClientStore(CacheStoreClientStoreMetricsCollector* collector) {
    if (!enable_ && collector) {
        return;
    }
    client_store_qps_->Report(1);
    if (!collector->success()) {
        client_store_failed_qps_->Report(1);
    }
    client_store_block_count_->Report(collector->blockCount());
    client_store_total_cost_us_->Report(collector->totalCostUs());
    client_store_local_store_cost_us_->Report(collector->localStoreCostUs());
    client_store_remote_store_cost_us_->Report(collector->remoteStoreCostUs());
}

void CacheStoreMetricsReporter::reportServerStore(CacheStoreServerStoreMetricsCollector* collector) {
    if (!enable_ && collector) {
        return;
    }
    server_store_qps_->Report(1);
    if (!collector->success()) {
        server_store_failed_qps_->Report(1);
    }
    server_store_block_count_->Report(collector->blockCount());
    server_store_total_cost_us_->Report(collector->totalCostUs());
}

void CacheStoreMetricsReporter::reportClientLoad(CacheStoreClientLoadMetricsCollector* collector) {
    if (!enable_ && collector) {
        return;
    }
    client_load_qps_->Report(1);
    if (!collector->success()) {
        client_load_failed_qps_->Report(1);
    }
    client_load_block_count_->Report(collector->blockCount());
    client_load_remote_load_block_count_->Report(collector->remoteLoadBlockCount());
    client_load_total_cost_us_->Report(collector->totalCostUs());
    client_load_local_load_cost_us_->Report(collector->localLoadCostUs());
    client_load_remote_load_cost_us_->Report(collector->remoteLoadCostUs());
    response_receive_cost_us_->Report(collector->responseReceiveCostUs());
    FT_ACCESS_LOG_DEBUG(
        "cache client load metrics: success: %d; load count: %d, remote load count: %d, total cost: %ldus, remote load cost %ldus",
        collector->success(),
        collector->blockCount(),
        collector->remoteLoadBlockCount(),
        collector->totalCostUs(),
        collector->remoteLoadCostUs());
}

void CacheStoreMetricsReporter::reportServerLoad(CacheStoreServerLoadMetricsCollector* collector) {
    if (!enable_ && collector) {
        return;
    }
    server_load_qps_->Report(1);
    if (!collector->success()) {
        server_load_failed_qps_->Report(1);
    }
    server_load_block_size_->Report(collector->blockSize());
    server_load_block_count_->Report(collector->blockCount());
    server_load_total_cost_us_->Report(collector->totalCostUs());
    request_send_cost_us_->Report(collector->requestSendCostUs());
    server_load_connect_cost_us_->Report(collector->connectCostUs());
    server_load_write_cost_us_->Report(collector->writeCostUs());
    first_block_cost_us_->Report(collector->firstBlockCostUs());
}

}  // namespace rtp_llm
