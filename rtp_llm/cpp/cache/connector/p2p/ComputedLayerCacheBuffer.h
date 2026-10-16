#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PNotification.h"
#include <atomic>
#include <functional>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtp_llm {

class ComputedLayerCacheBuffer {
public:
    ComputedLayerCacheBuffer(int64_t                                  request_id,
                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                             int64_t                                  deadline_ms);

    /// @brief 追加一层 cache buffer，deadline 只允许收紧
    void addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms);

    /// @brief 返回当前已存 buffer 数及指定 layer/tag 集合对应的缓冲区列表
    std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>>
    getBuffers(const std::set<std::string>& buffer_keys);

    /// @brief 阻塞等待层数变化，直到超过 last_layer_num 或 timeout_ms 超时
    void waitChange(int last_layer_num, int timeout_ms);

    void      setError(const ErrorInfo& error);
    ErrorInfo error() const;
    void      setErrorHandler(std::function<void(const ErrorInfo&)> handler);

    int64_t deadlineMs() const {
        return deadline_ms_.load(std::memory_order_relaxed);
    }

private:
    int64_t                                                  request_id_;
    std::map<std::string, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    std::atomic<int64_t>                             deadline_ms_;

    std::function<void(const ErrorInfo&)> error_handler_;
    ErrorInfo                             error_;
    mutable std::mutex                    mutex_;
    std::condition_variable condition_variable_;
};

class ComputedLayerCacheBufferStore {
public:
    ComputedLayerCacheBufferStore();
    ~ComputedLayerCacheBufferStore();

public:
    /// @brief 按 (request_id, request_deadline_ms) 获取或创建对应的 ComputedLayerCacheBuffer 并追加首层数据
    /// @return nullptr if the request key has been removed (late-arriving layers are rejected)
    std::shared_ptr<ComputedLayerCacheBuffer> addBuffer(int64_t                                  request_id,
                                                        const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                        int64_t                                  request_deadline_ms);

    std::shared_ptr<ComputedLayerCacheBuffer> getBuffer(int64_t request_id, int64_t request_deadline_ms) const;
    // Register the fixed late-layer acceptance horizon for a request. The
    // first value wins so callbacks from later layers cannot roll the horizon.
    std::optional<int64_t> registerRequestHorizon(int64_t request_id, int64_t horizon_ms, int64_t request_deadline_ms);
    // StartLoad has consumed the whole-request resource. Tighten the fixed
    // layer acceptance horizon to this physical transfer's deadline.
    std::optional<int64_t> activateRequestHorizon(int64_t request_id, int64_t horizon_ms, int64_t request_deadline_ms);
    std::optional<int64_t> requestHorizon(int64_t request_id, int64_t request_deadline_ms) const;
    void                   removeBuffer(int64_t request_id, int64_t request_deadline_ms);
    void                   checkTimeout();
    int64_t                getBuffersCount() const;
    int64_t                nextTimeoutMs() const;
    const std::shared_ptr<P2PNotification>& notification() const {
        return notification_;
    }

private:
    static constexpr int64_t kTombstoneRetentionMs = 60LL * 60 * 1000;

    // The original request deadline identifies the computation even after StartLoad
    // tightens the layer/transfer horizon.
    using RequestKey = std::pair<int64_t, int64_t>;
    struct RequestKeyHash {
        size_t operator()(const RequestKey& key) const {
            const auto id_hash = std::hash<int64_t>{}(key.first);
            return id_hash ^ (std::hash<int64_t>{}(key.second) + 0x9e3779b9 + (id_hash << 6) + (id_hash >> 2));
        }
    };

    struct RemovedRequestExpiry {
        int64_t expire_at_ms;
        RequestKey request_key;
    };

    struct RemovedRequestExpiryCompare {
        bool operator()(const RemovedRequestExpiry& lhs, const RemovedRequestExpiry& rhs) const {
            return lhs.expire_at_ms > rhs.expire_at_ms;
        }
    };

    void                                   markRemovedLocked(const RequestKey& request_key, int64_t now_ms);
    void checkTimeout(int64_t now_ms);
    const std::shared_ptr<P2PNotification> notification_{std::make_shared<P2PNotification>()};

    struct RequestHorizon {
        int64_t horizon_ms;
        int64_t request_deadline_ms;
    };

    // stores layer cache buffer already computed
    mutable std::mutex                                                     computed_buffers_mutex_;
    std::unordered_map<RequestKey, std::shared_ptr<ComputedLayerCacheBuffer>, RequestKeyHash> computed_buffers_;
    std::unordered_map<RequestKey, RequestHorizon, RequestKeyHash>                            request_horizons_;

    // Keep removed request keys for one hour; late layer and StartLoad calls are rejected.
    std::unordered_map<RequestKey, int64_t, RequestKeyHash> removed_requests_;  // request key -> expire_at_ms
    std::priority_queue<RemovedRequestExpiry,
                        std::vector<RemovedRequestExpiry>,
                        RemovedRequestExpiryCompare>
        removed_request_expiry_queue_;
};

}  // namespace rtp_llm
