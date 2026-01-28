#pragma once

#include <shared_mutex>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class TransferTask {
public:
    TransferTask(const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers, int64_t deadline_ms):
        layer_cache_buffers_(layer_cache_buffers), deadline_ms_(deadline_ms), start_time_us_(currentTimeUs()) {}
    ~TransferTask() = default;

public:
    void                              setCancelled();
    std::shared_ptr<LayerCacheBuffer> loadingLayerCacheBuffer(int layer_id, int partition_count, int partition_id);
    void                              notifyDone(int layer_id, bool success, int partition_count, int partition_id);

    bool success() const;
    bool cancelled() const;
    bool hasLoadingLayer() const;
    bool isTimeout() const;

    ErrorCode   errorCode() const;
    std::string errorMessage() const;

    int64_t firstLayerWaitTimeUs() const {
        return first_layer_wait_time_us_;
    }
    int64_t totalCostTimeUs() const {
        return total_cost_time_us_;
    }
    int64_t totalBlockCount() const;

    std::shared_ptr<LayerCacheBuffer> getLayerCacheBuffer(int layer_id) const;

private:
    bool isDoneNoLock(int layer_id, int partition_id) const;
    bool isLoadingNoLock(int layer_id, int partition_id) const;

private:
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    std::set<int>                                    done_layer_ids_;
    std::map<int, std::set<int>>                     done_layer_partition_ids_;
    std::map<int, std::set<int>>                     loading_layer_partition_ids_;
    bool                                             all_success_ = true;
    bool                                             cancelled_   = false;
    mutable std::shared_mutex                        mutex_;
    int64_t                                          deadline_ms_;
    int                                              partition_count_ = 0;

    // metric
    int64_t start_time_us_            = 0;
    int64_t first_layer_wait_time_us_ = 0;
    int64_t total_cost_time_us_       = 0;

    // error info
    ErrorCode   error_code_ = ErrorCode::NONE_ERROR;
    std::string error_msg_;
};

class TransferTaskStore {
public:
    TransferTaskStore();
    ~TransferTaskStore() = default;

    std::shared_ptr<TransferTask> addTask(const std::string&                                      unique_key,
                                          const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                          int64_t                                                 deadline_ms);

    std::shared_ptr<TransferTask> getTask(const std::string& unique_key) const;
    std::shared_ptr<TransferTask> stealTask(const std::string& unique_key);

    int64_t getTaskCount() const;

private:
    mutable std::shared_mutex mutex_;
    // [unique_key, TransferTask]
    std::map<std::string, std::shared_ptr<TransferTask>> task_map_;
};

}  // namespace rtp_llm