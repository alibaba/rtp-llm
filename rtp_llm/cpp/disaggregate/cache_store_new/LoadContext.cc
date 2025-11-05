#include "rtp_llm/cpp/disaggregate/cache_store_new/LoadContext.h"

namespace rtp_llm {

LoadContext::LoadContext(const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers, int64_t context_id):
    layer_cache_buffers_(layer_cache_buffers), context_id_(context_id) {}

LoadContext::~LoadContext() = default;

bool LoadContext::success() const {
    // TODO: implement
    return true;
}

void LoadContext::cancel() {
    // TODO: implement
}

void LoadContext::waitDone() {
    // TODO: implement
}

int64_t LoadContext::contextId() const {
    return context_id_;
}

void LoadContext::notifyLayerLoadDone(int layer_id) {
    done_layer_ids_.insert(layer_id);
}

bool LoadContext::isDone() const {
    return done_layer_ids_.size() == layer_cache_buffers_.size();
}

std::shared_ptr<LayerCacheBuffer> LoadContext::getLayerCacheBuffer(int layer_id) const {
    for (auto& layer_cache_buffer : layer_cache_buffers_) {
        if (layer_cache_buffer->layerId() == layer_id) {
            return layer_cache_buffer;
        }
    }
    return nullptr;
}

LoadContextStore::LoadContextStore() {}

LoadContextStore::~LoadContextStore() {}

std::shared_ptr<LoadContext> LoadContextStore::getLoadContext(int64_t context_id) const {
    std::lock_guard<std::mutex> lock(load_context_map_mutex_);
    return load_context_map_.at(context_id);
}

void LoadContextStore::addLoadContext(const std::shared_ptr<LoadContext>& load_context) {
    std::lock_guard<std::mutex> lock(load_context_map_mutex_);
    load_context_map_[load_context->contextId()] = load_context;
}

int64_t LoadContextStore::generateContextId() {
    static std::atomic<int64_t> context_id_generator(0);
    return context_id_generator.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace rtp_llm