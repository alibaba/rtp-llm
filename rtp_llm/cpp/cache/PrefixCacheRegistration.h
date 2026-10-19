#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSetResource.h"

namespace rtp_llm {

struct PrefixCacheGroupSaveBinding {
    size_t                   group_set_id{0};
    GroupSetResource         resource;
    PrefixCacheSaveState     state{PrefixCacheSaveState::REGISTERED};
    PrefixCacheSaveOwnership ownership{PrefixCacheSaveOwnership::NONE};
    std::string              unfulfilled_reason;
    std::vector<size_t>       cached_path_indices;  // Indices into registration.cached_path_resources; no per-binding ref.
};

struct PrefixCacheRegistrationPoint {
    PrefixCachePlanPoint                     endpoint;
    std::vector<PrefixCacheGroupSaveBinding> group_bindings;
};

// Request-side intent, captured before matching or generation changes live keys.
// Group bindings start REGISTERED with empty resources. A later writer completion
// or an existing immutable checkpoint protected by CACHED_REF may satisfy them.
struct PrefixCacheRegistration {
    struct SourcePin {
        std::string  tag;
        size_t       slot{0};
        BlockIdxType block{NULL_BLOCK_IDX};
    };
    using CachedPathResource = CachedPrefixResources::PathResource;
    // One request reference per (tag, physical block), shared across all
    // registered points and historical paths. Never released by writer completion.
    std::vector<CachedPrefixResources::Pin> cached_source_pins;
    std::vector<CachedPathResource> cached_path_resources;
    // Dense SWA pages can leave the active runtime window before publication.
    // Pins preserve their allocation; these paths become ready only after forward completion.
    std::vector<CachedPrefixResources::Pin>   runtime_source_pins;
    std::vector<CachedPathResource>           runtime_path_resources;
    size_t                                    original_input_tokens{0};
    std::vector<int>                          token_ids;
    CacheKeysType                             base_keys;
    std::vector<PrefixCacheRegistrationPoint> points;
    bool                                      prepared{false};
    size_t                                    prepared_resume_endpoint{0};
    bool                                      content_completed{false};
    bool                                      required_continuation_ready{false};
    std::vector<SourcePin>                    source_pins;
    std::vector<TaggedBlockIdPair>            copy_mappings;
};

}  // namespace rtp_llm
