#pragma once

#include <algorithm>
#include <array>
#include <unordered_map>
#include <utility>
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"

namespace rtp_llm {

// Logical publication state, protected by the owning BlockTree mutex. It is
// updated even when transport admission fails so snapshots can reconcile drops.
class KVCacheEventState {
public:
    void clear() {
        for (auto& groups : groups_) {
            groups.clear();
        }
    }

    void update(int64_t                key,
                KVCacheEventMedium     medium,
                std::vector<size_t>    resident_groups,
                KVCacheEventPublisher& publisher) {
        auto&                     groups = groups_[static_cast<size_t>(medium)];
        const auto                it     = groups.find(key);
        const std::vector<size_t> empty;
        const auto&               previous = it == groups.end() ? empty : it->second;
        for (const auto group : previous) {
            if (!std::binary_search(resident_groups.begin(), resident_groups.end(), group)) {
                publish(publisher, KVCacheEventType::BLOCK_DELETE, key, group, medium);
            }
        }
        for (const auto group : resident_groups) {
            if (!std::binary_search(previous.begin(), previous.end(), group)) {
                publish(publisher, KVCacheEventType::BLOCK_ADD, key, group, medium);
            }
        }
        if (resident_groups.empty()) {
            if (it != groups.end()) {
                groups.erase(it);
            }
        } else {
            groups[key] = std::move(resident_groups);
        }
    }

    void erase(int64_t key, KVCacheEventPublisher& publisher) {
        for (const auto medium : kKVCacheEventMediums) {
            update(key, medium, {}, publisher);
        }
    }

    KVCacheSnapshot snapshot() const {
        KVCacheSnapshot result;
        result.version = version_;
        for (const auto medium : kKVCacheEventMediums) {
            const auto& groups = groups_[static_cast<size_t>(medium)];
            result.groups(medium).insert(groups.begin(), groups.end());
        }
        return result;
    }

private:
    void publish(
        KVCacheEventPublisher& publisher, KVCacheEventType type, int64_t key, size_t group, KVCacheEventMedium medium) {
        ++version_;
        KVCacheEvent event{type, key, group};
        event.medium = medium;
        (void)publisher.tryPublish(event);
    }

    std::array<std::unordered_map<int64_t, std::vector<size_t>>, 3> groups_;
    int64_t                                                         version_{-1};
};

}  // namespace rtp_llm
