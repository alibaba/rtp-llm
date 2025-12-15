#include "rtp_llm/cpp/cache_new/KVCacheConnectorCordinator.h"

namespace rtp_llm {

KVCacheConnectorCordinator::KVCacheConnectorCordinator() {}

KVCacheConnectorCordinator::~KVCacheConnectorCordinator() {}

std::shared_ptr<AsyncContext> KVCacheConnectorCordinator::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                                    const std::shared_ptr<Meta>&              meta) {

    // save resource for later use
    auto kv_cache_resource = allocator_->incRef(resource);

    std::vector<std::shared_ptr<AsyncContext>> contexts;
    for (const auto& connector : connectors_) {
        auto match_context = connector->asyncMatch(kv_cache_resource, meta);
        contexts.push_back(match_context);
    }
    auto fused_match_context = std::make_shared<FusedAsyncContext>(contexts);
    return std::make_shared<FusedAsyncReadContext>(fused_match_context);
}

std::shared_ptr<AsyncContext> KVCacheConnectorCordinator::asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                                     const std::shared_ptr<Meta>&              meta) {

    auto kv_cache_resource = allocator_->incRef(resource);
    auto write_contexts    = std::vector<std::shared_ptr<AsyncContext>>();
    for (const auto& connector : connectors_) {
        auto write_context = connector->asyncWrite(kv_cache_resource, meta);
        if (write_context) {
            write_contexts.push_back(write_context);
        }
    }
    auto fused_write_context = std::make_shared<FusedAsyncContext>(write_contexts);
    return std::make_shared<FusedAsyncWriteContext>(fused_write_context);
}

std::shared_ptr<AsyncContext> KVCacheConnectorCordinator::asyncWriteByLayer(
    int layer_id, const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    auto kv_cache_resource = allocator_->incRef(resource);
    auto write_context     = connector->asyncWriteByLayer(layer_id, kv_cache_resource, meta);
    if (write_context) {
        return write_context;
    }
    auto fused_write_context = std::make_shared<FusedAsyncContext>(contexts);
    return std::make_shared<FusedAsyncWriteContext>(fused_write_context);
}

void KVCacheConnectorCordinator::updateOnce() {
    std::lock_guard<std::mutex> lock(update_mutex_);
    for (auto it = fused_async_read_context_list_.begin(); it != fused_async_read_context_list_.end();) {
        auto fused_read_context = *it;
        if (fused_read_context->done()) {
            it = fused_async_read_context_list_.erase(it);
            continue;
        }
        if (fused_read_context->fused_match_context()->done() && fused_read_context->fused_read_context() == nullptr) {
            if (!fused_read_context->fused_match_context()->success()) {
                // match failed, cancel
                it = fused_async_read_context_list_.erase(it);
                continue;
            }
            // match success, start read
            int reuse_len = fused_read_context->resource()->reuseBlocksNum();
            for (auto& context : fused_read_context->fused_match_context()->contexts()) {
                if (!context) {
                    continue;
                }
                if (context->matchedBlockCount() > reuse_len) {
                    continue;
                }
                auto read_context =
                    context->asyncRead(fused_read_context->resource(),
                                       std::make_shared<Meta>(reuse_len, context->matchedBlockCount() - reuse_len));
                if (read_context) {
                    fused_read_context->fused_read_context()->add_context(read_context);
                }
            }
        }
        it++;
    }
    for (auto it = fused_async_write_context_list_.begin(); it != fused_async_write_context_list_.end();) {
        auto fused_write_context = *it;
        if (fused_write_context->done()) {
            it = fused_async_write_context_list_.erase(it);
            continue;
        }
        it++;
    }
}

}  // namespace rtp_llm