#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace rtp_llm {

enum class DSV41ReplayMode {
    FULL,
    BOUNDED_CHECKPOINT_V1
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
    int64_t cacheKeySeed() const {
        validate();
        return static_cast<int64_t>(std::hash<std::string>{}(
            model_revision + ':' + layout_fingerprint + ':' + std::to_string(static_cast<int>(replay_mode)) + ':'
            + std::to_string(tail_policy_version) + ':' + std::to_string(replay_window) + ':'
            + std::to_string(physical_swa_entries)));
    }
    void validate() const {
        if (model_revision.empty() || layout_fingerprint.empty() || tail_policy_version != 1 || replay_window != 128
            || physical_swa_entries < replay_window
            || (replay_mode != DSV41ReplayMode::FULL && replay_mode != DSV41ReplayMode::BOUNDED_CHECKPOINT_V1)) {
            throw std::invalid_argument("invalid V4.1 checkpoint model/layout/mode identity");
        }
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

    void validate(size_t reuse_unit) const {
        identity.validate();
        if ((reuse_unit != 128 && reuse_unit != 256 && reuse_unit != 1024 && reuse_unit != 2048)
            || materialized_end <= 0 || materialized_end > 1048576 || materialized_end % reuse_unit != 0
            || encoder_materialized_end != materialized_end || decoder_checkpoint_end != materialized_end
            || !history_ready || !draft_committed || !pair_empty) {
            throw std::invalid_argument("V4.1 prefix requires a complete aligned target/draft checkpoint");
        }
        const int64_t needed_start = std::max<int64_t>(0, materialized_end - 128);
        if (aux_valid_start < 0 || aux_valid_start > needed_start || aux_valid_end != materialized_end) {
            throw std::invalid_argument("V4.1 checkpoint has incomplete draft aux rows");
        }
        for (size_t owner = 0; owner < 4; ++owner) {
            const auto entries = materialized_end / (owner == 3 ? 1 : 2);
            if (global_entries[owner] != entries || index_entries[owner] != entries) {
                throw std::invalid_argument("V4.1 checkpoint is missing owner global/index entries");
            }
        }
        for (size_t layer = 0; layer < swa.size(); ++layer) {
            const auto& range = swa[layer];
            if (range.valid_start < 0 || range.valid_start > needed_start || range.valid_end != materialized_end
                || range.valid_end - range.valid_start > identity.physical_swa_entries || range.replay_floor < 0
                || range.replay_floor > range.valid_start
                || ((identity.replay_mode == DSV41ReplayMode::FULL || layer <= 20) && range.replay_floor != 0)) {
                throw std::invalid_argument("V4.1 checkpoint has incomplete or inconsistent SWA ranges");
            }
        }
        for (auto mask : history_image_mask) {
            if (mask > 1) {
                throw std::invalid_argument("invalid V4.1 canonical history image mask");
            }
        }
    }
};

// Published by the model after producing the named rows and persistent state.
// Unlike a reusable checkpoint, a PD boundary may be odd or target-only.
struct DSV41ExecutionState {
    int64_t              request_id{-1};
    int64_t              materialized_end{0};
    int64_t              encoder_materialized_end{0};
    int64_t              decoder_checkpoint_end{0};
    uint32_t             draft_layers{0};
    int64_t              aux_valid_start{0};
    int64_t              aux_valid_end{0};
    std::vector<int64_t> global_entries;
    std::vector<int64_t> index_entries;
    std::vector<int64_t> swa_valid_start;
    std::vector<int64_t> swa_valid_end;
    std::vector<int64_t> swa_replay_floor;
    std::vector<int64_t> pair_positions;
    std::vector<uint8_t> pair_valid;
    std::vector<int64_t> history_token_ids;
    std::vector<uint8_t> history_image_mask;
    bool                 history_ready{false};
    bool                 draft_committed{false};

    void validate(const DSV41CacheIdentity& identity, uint32_t expected_draft_layers) const {
        identity.validate();
        if (request_id < 0 || materialized_end <= 0 || materialized_end > 1048576
            || encoder_materialized_end != materialized_end || decoder_checkpoint_end != materialized_end
            || (expected_draft_layers != 0 && expected_draft_layers != 3) || draft_layers != expected_draft_layers
            || global_entries.size() != 4 || index_entries.size() != 4 || pair_positions.size() != 3
            || pair_valid.size() != 3 || history_token_ids.size() != 3 || history_image_mask.size() != 3
            || !history_ready || swa_valid_start.size() != 40 + draft_layers
            || swa_valid_end.size() != swa_valid_start.size() || swa_replay_floor.size() != swa_valid_start.size()) {
            throw std::invalid_argument("V4.1 execution publication is incomplete or has an invalid boundary");
        }
        const int64_t needed_start = std::max<int64_t>(0, materialized_end - 128);
        if (draft_layers == 0 && (draft_committed || aux_valid_start != 0 || aux_valid_end != 0)) {
            throw std::invalid_argument("V4.1 target-only publication cannot certify draft/aux state");
        }
        if (draft_layers != 0
            && (!draft_committed || aux_valid_start < 0 || aux_valid_start > needed_start
                || aux_valid_end != materialized_end)) {
            throw std::invalid_argument("V4.1 execution publication is missing committed draft/aux state");
        }
        for (size_t owner = 0; owner < global_entries.size(); ++owner) {
            const int64_t entries = materialized_end / (owner == 3 ? 1 : 2);
            if (global_entries[owner] != entries || index_entries[owner] != entries) {
                throw std::invalid_argument("V4.1 execution publication is missing owner KV/index entries");
            }
        }
        for (size_t layer = 0; layer < swa_valid_start.size(); ++layer) {
            if (swa_valid_start[layer] < 0 || swa_valid_start[layer] > needed_start
                || swa_valid_end[layer] != materialized_end
                || swa_valid_end[layer] - swa_valid_start[layer] > identity.physical_swa_entries
                || swa_replay_floor[layer] < 0 || swa_replay_floor[layer] > swa_valid_start[layer]
                || ((identity.replay_mode == DSV41ReplayMode::FULL || layer <= 20) && swa_replay_floor[layer] != 0)) {
                throw std::invalid_argument("V4.1 execution publication has incomplete SWA ranges");
            }
        }
        for (size_t owner = 0; owner < pair_valid.size(); ++owner) {
            const bool pending = materialized_end % 2 != 0;
            if (pair_valid[owner] != static_cast<uint8_t>(pending)
                || pair_positions[owner] != (pending ? materialized_end - 1 : -1)) {
                throw std::invalid_argument("V4.1 execution publication has stale ratio2 pair state");
            }
        }
        for (size_t i = 0; i < history_token_ids.size(); ++i) {
            const bool missing = materialized_end + static_cast<int64_t>(i) < 3;
            if (history_image_mask[i] > 1 || (missing && (history_token_ids[i] != -1 || history_image_mask[i] != 0))
                || (!missing && history_token_ids[i] < 0)) {
                throw std::invalid_argument("V4.1 execution publication has invalid canonical history");
            }
        }
    }

    DSV41CheckpointMetadata checkpoint(const DSV41CacheIdentity& identity, size_t reuse_unit) const {
        validate(identity, 3);
        DSV41CheckpointMetadata metadata;
        metadata.identity                 = identity;
        metadata.materialized_end         = materialized_end;
        metadata.encoder_materialized_end = encoder_materialized_end;
        metadata.decoder_checkpoint_end   = decoder_checkpoint_end;
        metadata.aux_valid_start          = aux_valid_start;
        metadata.aux_valid_end            = aux_valid_end;
        std::copy(global_entries.begin(), global_entries.end(), metadata.global_entries.begin());
        std::copy(index_entries.begin(), index_entries.end(), metadata.index_entries.begin());
        for (size_t layer = 0; layer < metadata.swa.size(); ++layer)
            metadata.swa[layer] = {swa_valid_start[layer], swa_valid_end[layer], swa_replay_floor[layer]};
        std::copy(history_token_ids.begin(), history_token_ids.end(), metadata.history_token_ids.begin());
        std::copy(history_image_mask.begin(), history_image_mask.end(), metadata.history_image_mask.begin());
        metadata.history_ready   = history_ready;
        metadata.draft_committed = draft_committed;
        metadata.pair_empty      = materialized_end % 2 == 0;
        metadata.validate(reuse_unit);
        return metadata;
    }
};

struct DSV41ExecutionProgress {
    int64_t request_id{-1};
    int64_t encoder_materialized_end{0};
    int64_t decoder_checkpoint_end{0};
};

// The connector owns the immutable CPU snapshot. Keeping this handle protects
// N's complete state while the live GPU rings advance towards T.
class DSV41CheckpointSnapshot {
public:
    explicit DSV41CheckpointSnapshot(DSV41CheckpointMetadata metadata): metadata_(std::move(metadata)) {}
    virtual ~DSV41CheckpointSnapshot() = default;
    const DSV41CheckpointMetadata& metadata() const {
        return metadata_;
    }

private:
    const DSV41CheckpointMetadata metadata_;
};

class DSV41CacheState {
public:
    struct View {
        DSV41CacheIdentity                                    identity;
        int64_t                                               encoder_materialized_end{0};
        int64_t                                               decoder_checkpoint_end{0};
        int64_t                                               target_ready_end{0};
        int64_t                                               protected_prefix_end{0};
        int64_t                                               final_handoff_end{0};
        bool                                                  finished{false};
        bool                                                  cancelled{false};
        std::optional<DSV41CheckpointMetadata>                completed;
        std::optional<DSV41ExecutionState>                    execution;
        std::vector<std::shared_ptr<DSV41CheckpointSnapshot>> snapshots;
    };

    explicit DSV41CacheState(DSV41CacheIdentity identity) {
        identity.validate();
        state_.identity = std::move(identity);
    }
    View view() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }
    void advanceEncoder(int64_t end) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        advanceEncoderLocked(end);
    }
    void publishProgress(const DSV41ExecutionProgress& progress) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (state_.identity.replay_mode != DSV41ReplayMode::BOUNDED_CHECKPOINT_V1 || progress.request_id < 0
            || progress.decoder_checkpoint_end != state_.decoder_checkpoint_end) {
            throw std::invalid_argument("V4.1 encoder-only progress cannot complete decoder state");
        }
        advanceEncoderLocked(progress.encoder_materialized_end);
    }

private:
    void advanceEncoderLocked(int64_t end) {
        if (end < state_.encoder_materialized_end || end > 1048576
            || (state_.final_handoff_end > 0 && end > state_.final_handoff_end)) {
            throw std::invalid_argument("invalid V4.1 encoder progress");
        }
        if (state_.protected_prefix_end > 0 && end > state_.protected_prefix_end && !prefixProtected()) {
            throw std::logic_error("V4.1 must snapshot complete N before advancing to T");
        }
        state_.encoder_materialized_end = end;
        if (end > state_.decoder_checkpoint_end) {
            state_.target_ready_end = 0;
            state_.execution.reset();
        }
    }

public:
    void completeDecoder(const DSV41CheckpointMetadata& metadata, size_t reuse_unit) {
        metadata.validate(reuse_unit);
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (!(metadata.identity == state_.identity) || metadata.materialized_end != state_.encoder_materialized_end
            || metadata.materialized_end < state_.decoder_checkpoint_end) {
            throw std::invalid_argument("V4.1 decoder checkpoint does not match encoder progress or identity");
        }
        state_.decoder_checkpoint_end = metadata.materialized_end;
        state_.completed              = metadata;
        state_.target_ready_end = metadata.materialized_end;
    }
    void markTargetReady(int64_t end) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (state_.identity.replay_mode != DSV41ReplayMode::FULL || end < state_.encoder_materialized_end || end <= 0
            || end > 1048576
            || (state_.protected_prefix_end > 0 && end > state_.protected_prefix_end && !prefixProtected()))
            throw std::invalid_argument("V4.1 target completion has an invalid execution boundary");
        state_.encoder_materialized_end = end;
        state_.target_ready_end         = end;
    }
    void publishExecution(const DSV41ExecutionState& execution, uint32_t expected_draft_layers) {
        execution.validate(state_.identity, expected_draft_layers);
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (execution.materialized_end < state_.encoder_materialized_end
            || (state_.final_handoff_end > 0 && execution.materialized_end > state_.final_handoff_end)
            || (state_.protected_prefix_end > 0 && execution.materialized_end > state_.protected_prefix_end
                && !prefixProtected())) {
            throw std::invalid_argument("V4.1 execution publication is stale or bypasses protected N");
        }
        state_.encoder_materialized_end = execution.materialized_end;
        state_.decoder_checkpoint_end   = execution.materialized_end;
        state_.target_ready_end         = execution.materialized_end;
        state_.execution                = execution;
    }
    void protect(const std::shared_ptr<DSV41CheckpointSnapshot>& snapshot) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (!snapshot || !state_.completed || !(snapshot->metadata() == *state_.completed)
            || snapshot->metadata().materialized_end != state_.decoder_checkpoint_end
            || state_.encoder_materialized_end != state_.decoder_checkpoint_end) {
            throw std::invalid_argument("V4.1 cannot protect an incomplete or stale checkpoint");
        }
        state_.snapshots.push_back(snapshot);
    }
    void completeHandoff(int64_t end, bool target_swa_ready, bool draft_swa_ready, bool aux_ready) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (end != state_.encoder_materialized_end || end != state_.final_handoff_end || !target_swa_ready
            || !draft_swa_ready || !aux_ready) {
            throw std::invalid_argument("V4.1 handoff requires complete target/draft SWA and valid aux at T");
        }
        state_.decoder_checkpoint_end = end;
    }
    void finish(int64_t end) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (end != state_.encoder_materialized_end || end != state_.decoder_checkpoint_end
            || (state_.final_handoff_end > 0 && end != state_.final_handoff_end)
            || (state_.protected_prefix_end > 0 && state_.protected_prefix_end < end && !prefixProtected())) {
            throw std::logic_error("V4.1 cannot finish before both axes and protected N are complete");
        }
        state_.finished = true;
    }
    void restore(const DSV41CheckpointMetadata& metadata, size_t reuse_unit) {
        metadata.validate(reuse_unit);
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (!(metadata.identity == state_.identity) || state_.encoder_materialized_end > metadata.materialized_end
            || state_.encoder_materialized_end != state_.decoder_checkpoint_end) {
            throw std::invalid_argument("V4.1 restore requires a same-mode complete prior boundary");
        }
        state_.encoder_materialized_end = metadata.materialized_end;
        state_.decoder_checkpoint_end   = metadata.materialized_end;
        state_.completed                = metadata;
        state_.target_ready_end = metadata.materialized_end;
        state_.execution.reset();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.cancelled = true;
        state_.finished  = false;
        state_.snapshots.clear();
        state_.completed.reset();
        state_.execution.reset();
        state_.target_ready_end = 0;
    }

private:
    void active() const {
        if (state_.finished || state_.cancelled) {
            throw std::logic_error("V4.1 cache request is no longer active");
        }
    }
    bool prefixProtected() const {
        for (const auto& snapshot : state_.snapshots) {
            if (snapshot->metadata().materialized_end == state_.protected_prefix_end) {
                return true;
            }
        }
        return false;
    }
    mutable std::mutex mutex_;
    View               state_;
};

}  // namespace rtp_llm
