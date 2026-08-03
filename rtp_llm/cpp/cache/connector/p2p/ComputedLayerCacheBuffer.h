#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class ComputedLayerCacheBuffer {
public:
    ComputedLayerCacheBuffer(int64_t                                  request_id,
                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                             int64_t                                  deadline_ms);

    /// @brief 追加一层 cache buffer 并更新 deadline
    void addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    /// @brief 返回当前已存 buffer 数及指定 layer/tag 集合对应的缓冲区列表
    std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>>
    getBuffers(const std::set<std::string>& buffer_keys);

    /// @brief 阻塞等待层数变化，直到超过 last_layer_num 或 timeout_ms 超时
    void waitChange(int last_layer_num, int timeout_ms);

    int64_t deadlineMs() const {
        return deadline_ms_.load(std::memory_order_relaxed);
    }

private:
    int64_t                                          request_id_;
    std::map<std::string, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    std::atomic<int64_t>                             deadline_ms_;

    std::mutex              mutex_;
    std::condition_variable condition_variable_;
};

class ComputedLayerCacheBufferStore {
public:
    ComputedLayerCacheBufferStore();
    ~ComputedLayerCacheBufferStore();

public:
    /// @brief 按 request_id 获取或创建对应的 ComputedLayerCacheBuffer 并追加首层数据
    /// @return nullptr if request_id has been removed (late-arriving layers are rejected)
    std::shared_ptr<ComputedLayerCacheBuffer>
    addBuffer(int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    std::shared_ptr<ComputedLayerCacheBuffer> getBuffer(int64_t request_id) const;
    // Register the fixed late-layer acceptance horizon for a request. The
    // first value wins so callbacks from later layers cannot roll the horizon.
    std::optional<int64_t>
    registerRequestHorizon(int64_t request_id, int64_t horizon_ms, int64_t request_deadline_ms = 0);
    // StartLoad has consumed the whole-request resource. Promote the fixed
    // layer acceptance horizon to this physical transfer's deadline.
    std::optional<int64_t>
    activateRequestHorizon(int64_t request_id, int64_t horizon_ms, int64_t request_deadline_ms = 0);
    std::optional<int64_t> requestHorizon(int64_t request_id) const;
    void removeBuffer(int64_t request_id, int64_t request_deadline_ms = 0);
    void                   checkTimeout();
    int64_t                getBuffersCount() const;

private:
    struct RemovedRequestExpiry {
        int64_t expire_at_ms;
        int64_t request_id;
    };

    struct RemovedRequestExpiryCompare {
        bool operator()(const RemovedRequestExpiry& lhs, const RemovedRequestExpiry& rhs) const {
            return lhs.expire_at_ms > rhs.expire_at_ms;
        }
    };

    void markRemovedLocked(int64_t request_id, int64_t expire_at_ms);

    struct RequestHorizon {
        int64_t horizon_ms;
        int64_t request_deadline_ms;
    };

    // stores layer cache buffer already computed
    mutable std::mutex                                                     computed_buffers_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<ComputedLayerCacheBuffer>> computed_buffers_;
    std::unordered_map<int64_t, RequestHorizon>                           request_horizons_;

    // request_ids that have been explicitly removed; late addBuffer calls are rejected
    std::unordered_map<int64_t, int64_t> removed_request_ids_;  // request_id -> expire_at_ms
    std::priority_queue<RemovedRequestExpiry,
                        std::vector<RemovedRequestExpiry>,
                        RemovedRequestExpiryCompare>
        removed_request_expiry_queue_;
};

}  // namespace rtp_llm
