#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

bool TransferTask::success() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return all_success_ && done_layer_ids_.size() == layer_cache_buffers_.size();
}

bool TransferTask::cancelled() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cancelled_;
}

bool TransferTask::hasLoadingLayer() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return !loading_layer_partition_ids_.empty();
}

bool TransferTask::isTimeout() const {
    return currentTimeMs() >= deadline_ms_;
}

void TransferTask::setCancelled() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cancelled_  = true;
    error_code_ = ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
    error_msg_  = "TransferTask cancelled";
}

ErrorCode TransferTask::errorCode() const {
    if (isTimeout()) {
        return ErrorCode::P2P_CONNECTOR_WORKER_READ_TIMEOUT;
    }
    if (cancelled()) {
        return ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return error_code_;
}

std::string TransferTask::errorMessage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return error_msg_;
}

std::shared_ptr<LayerCacheBuffer> TransferTask::getLayerCacheBuffer(int layer_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                iter = layer_cache_buffers_.find(layer_id);
    if (iter == layer_cache_buffers_.end()) {
        return nullptr;
    }
    return iter->second;
}

void TransferTask::notifyDone(int layer_id, bool success, int partition_count, int partition_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!isLoadingNoLock(layer_id, partition_id)) {
        RTP_LLM_LOG_WARNING("TransferTask notifyDone failed: layer_id not loading, layer_id: %d, partition_id: %d",
                            layer_id,
                            partition_id);
        return;
    }
    if (!success) {
        all_success_ = false;
        error_code_  = ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED;
        error_msg_   = "TransferTask notifyDone failed: layer_id: " + std::to_string(layer_id)
                     + ", partition_id: " + std::to_string(partition_id);
        cancelled_ = true;
        RTP_LLM_LOG_WARNING("%s, task cancelled", error_msg_.c_str());
    }
    if (done_layer_partition_ids_.find(layer_id) == done_layer_partition_ids_.end()) {
        done_layer_partition_ids_[layer_id] = std::set<int>();
    }
    done_layer_partition_ids_[layer_id].insert(partition_id);
    loading_layer_partition_ids_[layer_id].erase(partition_id);
    if (loading_layer_partition_ids_[layer_id].empty()) {
        loading_layer_partition_ids_.erase(layer_id);
    }

    if (done_layer_partition_ids_[layer_id].size() == partition_count_) {
        done_layer_ids_.insert(layer_id);
    }

    if (done_layer_ids_.size() == layer_cache_buffers_.size()) {
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
    }
}

std::shared_ptr<LayerCacheBuffer>
TransferTask::loadingLayerCacheBuffer(int layer_id, int partition_count, int partition_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                iter = layer_cache_buffers_.find(layer_id);
    if (iter == layer_cache_buffers_.end()) {
        RTP_LLM_LOG_WARNING("TransferTask loadingLayerCacheBuffer failed: layer_id not found, layer_id: %d", layer_id);
        return nullptr;
    }

    if (partition_count_ == 0) {
        partition_count_ = partition_count;
    } else if (partition_count_ != partition_count) {
        RTP_LLM_LOG_WARNING(
            "TransferTask loadingLayerCacheBuffer failed: partition_count not match, partition_count: %d vs %d  , layer_id: %d",
            partition_count,
            partition_count_,
            layer_id);
        return nullptr;
    }

    if (isLoadingNoLock(layer_id, partition_id) || isDoneNoLock(layer_id, partition_id)) {
        RTP_LLM_LOG_WARNING(
            "TransferTask loadingLayerCacheBuffer failed: layer_id already loading or done, layer_id: %d", layer_id);
        return nullptr;
    }

    if (loading_layer_partition_ids_.find(layer_id) == loading_layer_partition_ids_.end()) {
        loading_layer_partition_ids_[layer_id] = std::set<int>();
    }
    loading_layer_partition_ids_[layer_id].insert(partition_id);

    if (first_layer_wait_time_us_ == 0) {
        first_layer_wait_time_us_ = currentTimeUs() - start_time_us_;
    }
    return iter->second;
}

bool TransferTask::isLoadingNoLock(int layer_id, int partition_id) const {
    return loading_layer_partition_ids_.find(layer_id) != loading_layer_partition_ids_.end()
           && loading_layer_partition_ids_.at(layer_id).find(partition_id)
                  != loading_layer_partition_ids_.at(layer_id).end();
}

bool TransferTask::isDoneNoLock(int layer_id, int partition_id) const {
    return done_layer_partition_ids_.find(layer_id) != done_layer_partition_ids_.end()
           && done_layer_partition_ids_.at(layer_id).find(partition_id) != done_layer_partition_ids_.at(layer_id).end();
}

int64_t TransferTask::totalBlockCount() const {
    int64_t                             total_block_count = 0;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& [layer_id, layer_cache_buffer] : layer_cache_buffers_) {
        total_block_count += layer_cache_buffer->blockIdMap().size();
    }
    return total_block_count;
}

// ==================== TransferTaskStore ====================

TransferTaskStore::TransferTaskStore() {}

std::shared_ptr<TransferTask>
TransferTaskStore::addTask(const std::string&                                      unique_key,
                           const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                           int64_t                                                 deadline_ms) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    task_map_[unique_key] = std::make_shared<TransferTask>(layer_cache_buffers, deadline_ms);
    return task_map_[unique_key];
}

std::shared_ptr<TransferTask> TransferTaskStore::getTask(const std::string& unique_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<TransferTask> TransferTaskStore::stealTask(const std::string& unique_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    auto task = it->second;
    task_map_.erase(it->first);
    return task;
}

int64_t TransferTaskStore::getTaskCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return task_map_.size();
}

}  // namespace rtp_llm