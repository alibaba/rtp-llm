#pragma once

#include "block_tree_cache.h"

#include <memory>

namespace rtp_llm {
class StorageBackend;

namespace c_api {
std::shared_ptr<StorageBackend> makeHostStorage(btc_cache& cache, const btc_config& config);
}  // namespace c_api
}  // namespace rtp_llm
