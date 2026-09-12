#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rtp_llm/cpp/cache/DSV41CacheState.h"

namespace rtp_llm {

struct DSV41GpuCheckpointData {
    DSV41CheckpointMetadata             metadata;
    size_t                              reuse_unit{0};
    std::vector<int64_t>                keys;
    std::array<std::vector<int32_t>, 6> blocks;

    void validateProducer(const DSV41CacheState::View& producer) const {
        validate();
        if (!producer.finished || producer.cancelled || !producer.completed || !(producer.identity == metadata.identity)
            || !(*producer.completed == metadata) || producer.encoder_materialized_end != metadata.materialized_end
            || producer.decoder_checkpoint_end != metadata.materialized_end) {
            throw std::invalid_argument("V4.1 GPU cache cannot publish incomplete or stale live checkpoint backing");
        }
    }

    void validate() const {
        metadata.validate(reuse_unit);
        const size_t count = metadata.materialized_end / reuse_unit;
        if (keys.size() != count || std::set<int64_t>(keys.begin(), keys.end()).size() != count) {
            throw std::invalid_argument("V4.1 GPU checkpoint requires its complete unique prefix key chain");
        }
        for (size_t group = 0; group < blocks.size(); ++group) {
            const auto& ids = blocks[group];
            if (ids.size() != (group < 4 ? count : 1)
                || std::any_of(ids.begin(), ids.end(), [](int32_t id) { return id <= 0; })
                || std::set<int32_t>(ids.begin(), ids.end()).size() != ids.size()) {
                throw std::invalid_argument("V4.1 GPU checkpoint is missing complete physical owner/ring backing");
            }
        }
    }
};

class DSV41GpuCheckpointCache {
public:
    struct Snapshot {
        const DSV41GpuCheckpointData data;

    private:
        friend class DSV41GpuCheckpointCache;
        Snapshot(DSV41GpuCheckpointData value, std::shared_ptr<void> backing):
            data(std::move(value)), backing_(std::move(backing)) {}
        const std::shared_ptr<void> backing_;
    };
    using Lease  = std::shared_ptr<const Snapshot>;
    using Retain = std::function<std::shared_ptr<void>(const DSV41GpuCheckpointData&)>;

    bool publish(const DSV41GpuCheckpointData& data, const Retain& retain, bool resident = false) {
        data.validate();
        if (!retain)
            throw std::invalid_argument("V4.1 GPU checkpoint requires an owning backing lease");
        std::lock_guard<std::mutex> lock(mutex_);
        const Key                   key{data.metadata.identity, data.keys.back()};
        const auto                  previous = entries_.find(key);
        if (previous != entries_.end()) {
            if (previous->second.snapshot->data.keys != data.keys
                || !(previous->second.snapshot->data.metadata == data.metadata)
                || previous->second.snapshot->data.reuse_unit != data.reuse_unit
                || previous->second.snapshot->data.blocks != data.blocks)
                return false;
            previous->second.resident |= resident;
            previous->second.access = ++access_;
            return true;
        }
        auto backing = retain(data);
        if (!backing)
            return false;
        auto snapshot = Lease(new Snapshot(data, std::move(backing)));
        entries_.emplace(key, Entry{std::move(snapshot), resident, ++access_});
        return true;
    }

    Lease match(const DSV41CacheIdentity& identity, const std::vector<int64_t>& keys, size_t reuse_unit, size_t limit) {
        identity.validate();
        std::lock_guard<std::mutex> lock(mutex_);
        limit     = std::min(limit, keys.size());
        auto best = entries_.end();
        // This component scans retained checkpoints. Production-scale lookup
        // indexing and checkpoint inventory limits still need qualification.
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const auto& data = it->second.snapshot->data;
            if (!(data.metadata.identity == identity) || data.reuse_unit != reuse_unit || data.keys.size() > limit
                || !std::equal(data.keys.begin(), data.keys.end(), keys.begin()))
                continue;
            if (best == entries_.end() || data.keys.size() > best->second.snapshot->data.keys.size())
                best = it;
        }
        if (best == entries_.end())
            return {};
        best->second.access = ++access_;
        // Taking the shared ownership while locked prevents eviction from
        // releasing any owner page between matching and acquiring its refs.
        return best->second.snapshot;
    }

    std::vector<Lease> takeOldestJointEvictable(int group = -1) {
        if (group < -1 || group >= 6)
            throw std::invalid_argument("invalid V4.1 GPU eviction group");
        std::lock_guard<std::mutex> lock(mutex_);
        struct Tree {
            bool     protected_tree{false};
            uint64_t access{0};
        };
        std::map<Key, Tree> trees;
        for (const auto& [key, entry] : entries_) {
            auto& tree = trees[{key.identity, entry.snapshot->data.keys.front()}];
            tree.protected_tree |= entry.resident || entry.snapshot.use_count() != 1;
            tree.access = std::max(tree.access, entry.access);
        }
        auto oldest = trees.end();
        for (auto it = trees.begin(); it != trees.end(); ++it) {
            if (!it->second.protected_tree && (oldest == trees.end() || it->second.access < oldest->second.access))
                oldest = it;
        }
        if (oldest == trees.end())
            return {};
        std::vector<Lease> removed;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->first.identity == oldest->first.identity
                && it->second.snapshot->data.keys.front() == oldest->first.key) {
                removed.push_back(it->second.snapshot);
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        return removed;
    }

    Lease leaseOldestForTransfer() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        oldest = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const auto& data           = it->second.snapshot->data;
            bool        protected_tree = false;
            for (const auto& [_, other] : entries_) {
                if (other.snapshot->data.metadata.identity == data.metadata.identity
                    && other.snapshot->data.keys.front() == data.keys.front())
                    protected_tree |= other.resident || other.snapshot.use_count() != 1;
            }
            if (!protected_tree && (oldest == entries_.end() || it->second.access < oldest->second.access))
                oldest = it;
        }
        // The entry stays published until the destination copy commits. A
        // failed or abandoned transfer only drops this lease and can retry.
        return oldest == entries_.end() ? Lease{} : oldest->second.snapshot;
    }

    void commitTransfer(const Lease& lease) {
        if (!lease)
            throw std::invalid_argument("V4.1 GPU transfer requires its matched backing");
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        found = entries_.find({lease->data.metadata.identity, lease->data.keys.back()});
        if (found == entries_.end() || found->second.snapshot != lease)
            return;
        // Residency can be promoted while the destination copy is in flight.
        for (const auto& [key, entry] : entries_) {
            if (entry.resident && key.identity == lease->data.metadata.identity
                && entry.snapshot->data.keys.front() == lease->data.keys.front())
                return;
        }
        entries_.erase(found);
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

private:
    struct Key {
        DSV41CacheIdentity identity;
        int64_t            key;
        bool               operator<(const Key& rhs) const {
            return std::tie(identity, key) < std::tie(rhs.identity, rhs.key);
        }
    };
    struct Entry {
        Lease    snapshot;
        bool     resident{false};
        uint64_t access{0};
    };
    mutable std::mutex   mutex_;
    std::map<Key, Entry> entries_;
    uint64_t             access_{0};
};

}  // namespace rtp_llm
