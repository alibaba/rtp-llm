#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <set>
#include <unordered_map>
#include <mutex>

#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreClientLoadContext {
public:
    CacheStoreClientLoadContext(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                int64_t                                               context_id);
    ~CacheStoreClientLoadContext();

public:
    bool success() const;
    void cancel();
    void waitDone();

    void setFailed(ErrorCode ec, const std::string& error_info);

    bool                              isDone() const;
    std::shared_ptr<LayerCacheBuffer> getLayerCacheBuffer(int layer_id) const;

    int64_t contextId() const;

    void notifyLayerLoadDone(int layer_id);

private:
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    int64_t                                        context_id_;
    std::set<int>                                  done_layer_ids_;
};

class LoadContextStore {
public:
    LoadContextStore();
    ~LoadContextStore();

public:
    std::shared_ptr<CacheStoreClientLoadContext> getLoadContext(int64_t context_id) const;
    void    addLoadContext(const std::shared_ptr<CacheStoreClientLoadContext>& load_context);
    int64_t generateContextId();

private:
    std::atomic_int64_t                                                       context_id_generator_ = 0;
    mutable std::mutex                                                        load_context_map_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<CacheStoreClientLoadContext>> load_context_map_;
};

}  // namespace cache_store
}  // namespace rtp_llm