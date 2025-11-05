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

}  // namespace rtp_llm