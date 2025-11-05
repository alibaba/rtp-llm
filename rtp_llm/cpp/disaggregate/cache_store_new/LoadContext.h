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

private:
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers_;
    int64_t                                        context_id_;
};

}  // namespace rtp_llm