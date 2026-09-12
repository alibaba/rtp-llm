#pragma once

#include <cstdint>
#include <memory>

#include "kvcm_client/kv_meta_object_client.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

namespace rtp_llm::detail {

// Internal dependency-injection seam for the opt-in KVCM implementation.  It
// keeps the native type out of the default client interface while allowing the
// adapter's batching, rollback, and exception boundaries to be tested without
// a live KVMeta service.
std::shared_ptr<MMKvcmClient>
createMMKvcmClientAdapter(std::unique_ptr<kv_cache_manager::KvMetaObjectClient> client,
                          std::uint64_t                                         max_object_bytes,
                          std::uint64_t                                         max_receipt_bytes);

}  // namespace rtp_llm::detail
