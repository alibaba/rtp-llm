#include "rtp_llm/cpp/cache/events/KVCMLogicalMirror.h"

#include <algorithm>
#include <utility>

namespace rtp_llm::detail {

bool KVCMLogicalMirror::seed(const KVCacheSnapshot& snapshot) {
    if (snapshot.block_keys.size() > max_keys_) {
        return false;
    }
    decltype(keys_) replacement;
    replacement.reserve(snapshot.block_keys.size());
    replacement.insert(snapshot.block_keys.begin(), snapshot.block_keys.end());
    keys_.swap(replacement);
    return true;
}

bool KVCMLogicalMirror::apply(const std::vector<KVCacheEvent>& events) {
    for (const auto& event : events) {
        if (event.type == KVCacheEventType::BLOCK_ADD) {
            if (keys_.find(event.block_key) == keys_.end() && keys_.size() >= max_keys_) {
                return false;
            }
            keys_.insert(event.block_key);
        } else {
            keys_.erase(event.block_key);
        }
    }
    return true;
}

KVCacheSnapshot KVCMLogicalMirror::snapshot() const {
    KVCacheSnapshot result;
    result.block_keys.reserve(keys_.size());
    result.block_keys.insert(result.block_keys.end(), keys_.begin(), keys_.end());
    std::sort(result.block_keys.begin(), result.block_keys.end());
    return result;
}

void KVCMLogicalMirror::release() noexcept {
    decltype(keys_){}.swap(keys_);
}

std::vector<KVCacheEvent> KVCMLogicalMirror::nextMutationBatch(const std::vector<int64_t>& source_keys,
                                                               const std::vector<int64_t>& target_keys,
                                                               size_t&                     source_index,
                                                               size_t&                     target_index,
                                                               size_t                      max_batch_size) {
    std::vector<KVCacheEvent> mutations;
    mutations.reserve(max_batch_size);
    while (mutations.size() < max_batch_size
           && (source_index < source_keys.size() || target_index < target_keys.size())) {
        if (target_index == target_keys.size()
            || (source_index < source_keys.size() && source_keys[source_index] < target_keys[target_index])) {
            mutations.push_back({KVCacheEventType::BLOCK_DELETE, source_keys[source_index++], 0});
        } else if (source_index == source_keys.size() || target_keys[target_index] < source_keys[source_index]) {
            mutations.push_back({KVCacheEventType::BLOCK_ADD, target_keys[target_index++], 0});
        } else {
            ++source_index;
            ++target_index;
        }
    }
    return mutations;
}

}  // namespace rtp_llm::detail
