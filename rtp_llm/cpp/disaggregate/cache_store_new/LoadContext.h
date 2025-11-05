#pragma once

#include <vector>
#include <memory>
#include <atomic>

#include "rtp_llm/cpp/disaggregate/cache_store_new/LayerCacheBuffer.h"

namespace rtp_llm {

class LoadContext {
public:
    LoadContext(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers, int64_t context_id);
    ~LoadContext();

public:
    bool success() const;
    void cancel();
    void waitDone();

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
    std::shared_ptr<LoadContext> getLoadContext(int64_t context_id) const;
    void                         addLoadContext(const std::shared_ptr<LoadContext>& load_context);
    int64_t                      generateContextId();

private:
    std::atomic_int64_t                                       context_id_generator_ = 0;
    mutable std::mutex                                        load_context_map_mutex_;
    std::unordered_map<int64_t, std::shared_ptr<LoadContext>> load_context_map_;
};

}  // namespace rtp_llm