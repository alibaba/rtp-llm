#pragma once

#include <memory>

namespace rtp_llm {
namespace attention_accuracy_check {

template<typename Executor, typename CacheManager, typename Synchronize, typename EmptyCache>
void releaseResources(std::unique_ptr<Executor>&     executor,
                      std::shared_ptr<CacheManager>& cache_manager,
                      Synchronize                    synchronize,
                      EmptyCache                     empty_cache) {
    // NormalExecutor keeps a shared reference to the temporary KV pool.
    // Destroy it first so cache_manager.reset() releases the pool before the
    // allocator-wide cache cleanup runs.
    executor.reset();
    cache_manager.reset();
    synchronize();
    empty_cache();
}

}  // namespace attention_accuracy_check
}  // namespace rtp_llm
