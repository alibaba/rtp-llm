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
        int64_t                                               protected_prefix_end{0};
        int64_t                                               final_handoff_end{0};
        bool                                                  finished{false};
        bool                                                  cancelled{false};
        bool                                                  published{false};
        std::optional<DSV41CheckpointMetadata>                completed;
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
    void requireProtectedPrefix(int64_t prefix_end, int64_t handoff_end) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (prefix_end < 0 || prefix_end < state_.encoder_materialized_end || prefix_end > handoff_end
            || handoff_end <= 0 || handoff_end > 1048576
            || (state_.final_handoff_end > 0
                && (state_.protected_prefix_end != prefix_end || state_.final_handoff_end != handoff_end))) {
            throw std::invalid_argument("V4.1 protected prefix must be selected before advancing past N");
        }
        state_.protected_prefix_end = prefix_end;
        state_.final_handoff_end    = handoff_end;
    }
    void advanceEncoder(int64_t end) {
        std::lock_guard<std::mutex> lock(mutex_);
        active();
        if (end < state_.encoder_materialized_end || end > 1048576
            || (state_.final_handoff_end > 0 && end > state_.final_handoff_end)) {
            throw std::invalid_argument("invalid V4.1 encoder progress");
        }
        if (state_.protected_prefix_end > 0 && end > state_.protected_prefix_end && !prefixProtected()) {
            throw std::logic_error("V4.1 must snapshot complete N before advancing to T");
        }
        state_.encoder_materialized_end = end;
    }
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
            || (state_.protected_prefix_end > 0 && !prefixProtected())) {
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
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.published) {
            throw std::logic_error("V4.1 cannot cancel a successfully published request");
        }
        state_.cancelled = true;
        state_.finished  = false;
        state_.snapshots.clear();
        state_.completed.reset();
    }
    bool publishSnapshots(const std::function<bool(const View&)>& publish) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!state_.finished || state_.cancelled)
            return false;
        if (state_.published)
            return true;
        if (!publish(state_))
            return false;
        state_.snapshots.clear();
        state_.published = true;
        return true;
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
