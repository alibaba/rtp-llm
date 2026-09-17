#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>

namespace rtp_llm {

enum class DSV41ReplayMode {
    FULL,
    BOUNDED_CHECKPOINT_V1
};

// Model-side identity payload: the execution-policy strings the generic cache
// config carries as an opaque model extension (see CacheConfig::dsv41_model_identity).
struct DSV41ModelIdentity {
    std::string model_revision;
    std::string replay_mode{"full"};
    uint32_t    tail_policy_version{1};
    uint32_t    replay_window{128};
};

struct DSV41CacheIdentity {
    std::string     model_revision;
    std::string     layout_fingerprint;
    DSV41ReplayMode replay_mode{DSV41ReplayMode::FULL};
    uint32_t        tail_policy_version{1};
    uint32_t        replay_window{128};
    uint32_t        physical_swa_entries{0};

    auto fields() const {
        return std::tie(
            model_revision, layout_fingerprint, replay_mode, tail_policy_version, replay_window, physical_swa_entries);
    }
    bool operator==(const DSV41CacheIdentity& rhs) const {
        return fields() == rhs.fields();
    }
    bool operator<(const DSV41CacheIdentity& rhs) const {
        return fields() < rhs.fields();
    }
    // Seeds the rolling cache-key hash. Behavior-sensitive: the seed value is
    // part of every live cache key and must not change.
    int64_t cacheKeySeed() const {
        return static_cast<int64_t>(std::hash<std::string>{}(
            model_revision + ':' + layout_fingerprint + ':' + std::to_string(static_cast<int>(replay_mode)) + ':'
            + std::to_string(tail_policy_version) + ':' + std::to_string(replay_window) + ':'
            + std::to_string(physical_swa_entries)));
    }
};

struct DSV41SwaRange {
    int64_t valid_start{0};
    int64_t valid_end{0};
    int64_t replay_floor{0};
    bool    operator==(const DSV41SwaRange& rhs) const {
        return std::tie(valid_start, valid_end, replay_floor)
               == std::tie(rhs.valid_start, rhs.valid_end, rhs.replay_floor);
    }
};

struct DSV41CheckpointMetadata {
    DSV41CacheIdentity            identity;
    int64_t                       materialized_end{0};
    int64_t                       encoder_materialized_end{0};
    int64_t                       decoder_checkpoint_end{0};
    int64_t                       aux_valid_start{0};
    int64_t                       aux_valid_end{0};
    std::array<int64_t, 4>        global_entries{};
    std::array<int64_t, 4>        index_entries{};
    std::array<DSV41SwaRange, 43> swa;
    std::array<int64_t, 3>        history_token_ids{};
    std::array<uint8_t, 3>        history_image_mask{};
    bool                          history_ready{false};
    bool                          draft_committed{false};
    bool                          pair_empty{false};

    bool operator==(const DSV41CheckpointMetadata& rhs) const {
        return identity == rhs.identity
               && std::tie(materialized_end,
                           encoder_materialized_end,
                           decoder_checkpoint_end,
                           aux_valid_start,
                           aux_valid_end,
                           global_entries,
                           index_entries,
                           swa,
                           history_token_ids,
                           history_image_mask,
                           history_ready,
                           draft_committed,
                           pair_empty)
                      == std::tie(rhs.materialized_end,
                                  rhs.encoder_materialized_end,
                                  rhs.decoder_checkpoint_end,
                                  rhs.aux_valid_start,
                                  rhs.aux_valid_end,
                                  rhs.global_entries,
                                  rhs.index_entries,
                                  rhs.swa,
                                  rhs.history_token_ids,
                                  rhs.history_image_mask,
                                  rhs.history_ready,
                                  rhs.draft_committed,
                                  rhs.pair_empty);
    }
};

}  // namespace rtp_llm
