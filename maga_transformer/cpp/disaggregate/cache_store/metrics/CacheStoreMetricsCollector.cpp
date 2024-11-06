#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"

namespace rtp_llm {

int64_t subZeroOrAbove(int64_t lhs, int64_t rhs) {
    if (lhs <= 0 || rhs <= 0 || lhs <= rhs) {
        return 0;
    }
    return lhs - rhs;
}

CacheStoreClientStoreMetricsCollector::CacheStoreClientStoreMetricsCollector(
    const std::shared_ptr<CacheStoreMetricsReporter>& reporter, uint32_t block_count):
    reporter_(reporter), block_count_(block_count), start_time_us_(autil::TimeUtility::currentTimeInMicroSeconds()) {}

CacheStoreClientStoreMetricsCollector::~CacheStoreClientStoreMetricsCollector() {
    auto reporter = reporter_.lock();
    if (reporter) {
        reporter->reportClientStore(this);
    }
}

void CacheStoreClientStoreMetricsCollector::markStoreLocalBegin(
    const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector) {
    if (collector) {
        collector->store_local_begin_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void CacheStoreClientStoreMetricsCollector::markStoreLocalEnd(
    const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector) {
    if (collector) {
        collector->store_local_end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void CacheStoreClientStoreMetricsCollector::markStoreRequestBegin(
    const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector) {
    if (collector) {
        collector->store_request_begine_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void CacheStoreClientStoreMetricsCollector::markEnd(
    const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector, bool success) {
    if (collector) {
        collector->end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        collector->success_     = success;
    }
}

bool CacheStoreClientStoreMetricsCollector::success() const {
    return success_;
}

uint32_t CacheStoreClientStoreMetricsCollector::blockCount() const {
    return block_count_;
}

int64_t CacheStoreClientStoreMetricsCollector::totalCostUs() const {
    return subZeroOrAbove(end_time_us_, start_time_us_);
}

int64_t CacheStoreClientStoreMetricsCollector::localStoreCostUs() const {
    return subZeroOrAbove(store_local_end_time_us_, store_local_begin_time_us_);
}

int64_t CacheStoreClientStoreMetricsCollector::remoteStoreCostUs() const {
    return subZeroOrAbove(end_time_us_, store_local_end_time_us_);
}

CacheStoreServerStoreMetricsCollector::CacheStoreServerStoreMetricsCollector(
    const std::shared_ptr<CacheStoreMetricsReporter>& reporter, uint32_t block_count):
    reporter_(reporter), block_count_(block_count), start_time_us_(autil::TimeUtility::currentTimeInMicroSeconds()) {}

CacheStoreServerStoreMetricsCollector::~CacheStoreServerStoreMetricsCollector() {
    auto reporter = reporter_.lock();
    if (reporter) {
        reporter->reportServerStore(this);
    }
}

void CacheStoreServerStoreMetricsCollector::markEnd(
    const std::shared_ptr<CacheStoreServerStoreMetricsCollector>& collector, bool success) {
    if (collector) {
        collector->end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        collector->success_     = success;
    }
}
void CacheStoreServerLoadMetricsCollector::setConnectCost(
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, int64_t connect_cost_us) {
    if (collector) {
        collector->load_connect_cost_us_ = connect_cost_us;
    }
}

void CacheStoreServerLoadMetricsCollector::setStartWriteTime(
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector) {
    if (collector) {
        collector->start_write_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

bool CacheStoreServerStoreMetricsCollector::success() const {
    return success_;
}

uint32_t CacheStoreServerStoreMetricsCollector::blockCount() const {
    return block_count_;
}

int64_t CacheStoreServerStoreMetricsCollector::totalCostUs() const {
    return subZeroOrAbove(end_time_us_, start_time_us_);
}
int64_t CacheStoreServerLoadMetricsCollector::connectCostUs() const {
    return load_connect_cost_us_;
}
int64_t CacheStoreServerLoadMetricsCollector::writeCostUs() const {
    return subZeroOrAbove(end_time_us_, start_write_time_us_);
}

CacheStoreClientLoadMetricsCollector::CacheStoreClientLoadMetricsCollector(
    const std::shared_ptr<CacheStoreMetricsReporter>& reporter, uint32_t block_count):
    reporter_(reporter), block_count_(block_count), start_time_us_(autil::TimeUtility::currentTimeInMicroSeconds()) {}

CacheStoreClientLoadMetricsCollector::~CacheStoreClientLoadMetricsCollector() {
    auto reporter = reporter_.lock();
    if (reporter) {
        reporter->reportClientLoad(this);
    }
}

void CacheStoreClientLoadMetricsCollector::markLocalLoadBegin(
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    if (collector) {
        collector->local_load_begin_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void CacheStoreClientLoadMetricsCollector::markLocalLoadEnd(
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    if (collector) {
        collector->local_load_end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
    }
}

void CacheStoreClientLoadMetricsCollector::markLoadRequestBegin(
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector, uint32_t block_count) {
    if (collector) {
        collector->local_request_begin_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        collector->remote_load_block_count_     = block_count;
    }
}

void CacheStoreClientLoadMetricsCollector::markEnd(
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector, bool success) {
    if (collector) {
        collector->end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        collector->success_     = success;
    }
}

void CacheStoreClientLoadMetricsCollector::setResponseReceiveCost(
    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector, int64_t response_receive_cost_us) {
    if (collector) {
        collector->response_receive_cost_us_ = response_receive_cost_us;
    }
}

int64_t CacheStoreClientLoadMetricsCollector::responseReceiveCostUs() const {
    return response_receive_cost_us_;
}

bool CacheStoreClientLoadMetricsCollector::success() const {
    return success_;
}

uint32_t CacheStoreClientLoadMetricsCollector::blockCount() const {
    return block_count_;
}

uint32_t CacheStoreClientLoadMetricsCollector::remoteLoadBlockCount() const {
    return remote_load_block_count_;
}

int64_t CacheStoreClientLoadMetricsCollector::totalCostUs() const {
    return subZeroOrAbove(end_time_us_, start_time_us_);
}

int64_t CacheStoreClientLoadMetricsCollector::localLoadCostUs() const {
    return subZeroOrAbove(local_load_end_time_us_, local_load_begin_time_us_);
}

int64_t CacheStoreClientLoadMetricsCollector::remoteLoadCostUs() const {
    return subZeroOrAbove(end_time_us_, local_request_begin_time_us_);
}

CacheStoreServerLoadMetricsCollector::CacheStoreServerLoadMetricsCollector(
    const std::shared_ptr<CacheStoreMetricsReporter>& reporter,
    uint32_t                                          block_count,
    uint32_t                                          block_size,
    int64_t                                           request_send_cost_us):
    reporter_(reporter),
    block_count_(block_count),
    block_size_(block_size),
    start_time_us_(autil::TimeUtility::currentTimeInMicroSeconds()),
    request_send_cost_us_(request_send_cost_us) {}

CacheStoreServerLoadMetricsCollector::~CacheStoreServerLoadMetricsCollector() {
    auto reporter = reporter_.lock();
    if (reporter) {
        reporter->reportServerLoad(this);
    }
}

void CacheStoreServerLoadMetricsCollector::markEnd(
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, bool success) {
    if (collector) {
        collector->end_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();
        collector->success_     = success;
    }
}

void CacheStoreServerLoadMetricsCollector::setFirstBlockCostUs(
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, int64_t first_block_cost_us) {
    if (collector) {
        collector->first_block_cost_us_ = first_block_cost_us;
    }
}

bool CacheStoreServerLoadMetricsCollector::success() const {
    return success_;
}

uint32_t CacheStoreServerLoadMetricsCollector::blockCount() const {
    return block_count_;
}

uint32_t CacheStoreServerLoadMetricsCollector::blockSize() const {
    return block_size_;
}

int64_t CacheStoreServerLoadMetricsCollector::totalCostUs() const {
    return subZeroOrAbove(end_time_us_, start_time_us_);
}
int64_t CacheStoreServerLoadMetricsCollector::requestSendCostUs() const {
    return request_send_cost_us_;
}
int64_t CacheStoreServerLoadMetricsCollector::firstBlockCostUs() const {
    return first_block_cost_us_;
}

}  // namespace rtp_llm
