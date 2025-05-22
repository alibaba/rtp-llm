#pragma once

#include "autil/TimeUtility.h"
#include <memory>

namespace rtp_llm {

class CacheStoreMetricsReporter;

class CacheStoreClientStoreMetricsCollector {

public:
    CacheStoreClientStoreMetricsCollector(const std::shared_ptr<CacheStoreMetricsReporter>& reporter,
                                          uint32_t                                          block_count);
    ~CacheStoreClientStoreMetricsCollector();

public:
    static void markStoreLocalBegin(const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector);
    static void markStoreLocalEnd(const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector);
    static void markStoreRequestBegin(const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector);
    static void markEnd(const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector, bool success);

public:
    bool     success() const;
    uint32_t blockCount() const;
    int64_t  totalCostUs() const;        // store操作的总耗时
    int64_t  localStoreCostUs() const;   // store到本地的耗时
    int64_t  remoteStoreCostUs() const;  // store到对端的耗时, 包含对端的处理延迟

private:
    std::weak_ptr<CacheStoreMetricsReporter> reporter_;
    uint32_t                                 block_count_{0};
    int64_t                                  start_time_us_{0};
    int64_t                                  store_local_begin_time_us_{0};
    int64_t                                  store_local_end_time_us_{0};
    int64_t                                  store_request_begine_time_us_{0};
    int64_t                                  end_time_us_{0};
    bool                                     success_{false};
};

class CacheStoreServerStoreMetricsCollector {
public:
    CacheStoreServerStoreMetricsCollector(const std::shared_ptr<CacheStoreMetricsReporter>& reporter,
                                          uint32_t                                          block_count);
    ~CacheStoreServerStoreMetricsCollector();

public:
    static void markEnd(const std::shared_ptr<CacheStoreServerStoreMetricsCollector>& collector, bool success);

public:
    bool     success() const;
    uint32_t blockCount() const;
    int64_t  totalCostUs() const;

private:
    std::weak_ptr<CacheStoreMetricsReporter> reporter_;
    uint32_t                                 block_count_;
    int64_t                                  start_time_us_{0};
    int64_t                                  end_time_us_{0};
    bool                                     success_{false};
    int64_t                                  first_block_cost_us_{0};
};

class CacheStoreClientLoadMetricsCollector {

public:
    CacheStoreClientLoadMetricsCollector(const std::shared_ptr<CacheStoreMetricsReporter>& reporter,
                                         uint32_t                                          block_count);
    ~CacheStoreClientLoadMetricsCollector();

public:
    static void markLocalLoadBegin(const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector);
    static void markLocalLoadEnd(const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector);
    static void markLoadRequestBegin(const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                     uint32_t                                                     block_count);
    static void markEnd(const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector, bool success);
    static void setResponseReceiveCost(const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                       int64_t response_receive_cost_us);

public:
    bool     success() const;               // 是否成功
    uint32_t blockCount() const;            // 需要load的block数量
    uint32_t remoteLoadBlockCount() const;  // 从对端load的block数量
    int64_t  totalCostUs() const;           // load 操作的总耗时
    int64_t  localLoadCostUs() const;       // 从本地cache load的耗时
    int64_t  remoteLoadCostUs() const;      // 调用RPC的耗时
    int64_t  responseReceiveCostUs() const;

private:
    std::weak_ptr<CacheStoreMetricsReporter> reporter_;
    uint32_t                                 block_count_{0};
    int64_t                                  start_time_us_{0};
    uint32_t                                 remote_load_block_count_{0};
    bool                                     success_{false};
    int64_t                                  local_load_begin_time_us_{0};
    int64_t                                  local_load_end_time_us_{0};
    int64_t                                  local_request_begin_time_us_{0};
    int64_t                                  end_time_us_{0};
    int64_t                                  response_receive_cost_us_{0};
};

class CacheStoreServerLoadMetricsCollector {
public:
    CacheStoreServerLoadMetricsCollector(const std::shared_ptr<CacheStoreMetricsReporter>& reporter,
                                         uint32_t                                          block_count,
                                         uint32_t                                          block_size,
                                         int64_t                                           request_send_cost_us);
    ~CacheStoreServerLoadMetricsCollector();

public:
    static void markEnd(const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector, bool success);
    static void setConnectCost(const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                               int64_t                                                      connect_cost_us);
    static void setStartWriteTime(const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector);
    static void setFirstBlockCostUs(const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                    int64_t                                                      first_block_cost_us);

public:
    bool     success() const;
    uint32_t blockCount() const;
    uint32_t blockSize() const;
    int64_t  totalCostUs() const;
    int64_t  requestSendCostUs() const;
    int64_t  connectCostUs() const;
    int64_t  writeCostUs() const;
    int64_t  firstBlockCostUs() const;

private:
    std::weak_ptr<CacheStoreMetricsReporter> reporter_;
    bool                                     success_{false};
    uint32_t                                 block_count_{0};
    uint32_t                                 block_size_{0};
    int64_t                                  start_time_us_{0};
    int64_t                                  end_time_us_{0};
    int64_t                                  request_send_cost_us_{0};
    int64_t                                  load_connect_cost_us_{0};
    int64_t                                  start_write_time_us_{0};
    int64_t                                  first_block_cost_us_{0};
};

}  // namespace rtp_llm