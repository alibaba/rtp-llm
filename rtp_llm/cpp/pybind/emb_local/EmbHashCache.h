#pragma once

#include "rtp_llm/cpp/pybind/EmbHashTensorBatch.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace rtp_llm::emb_local {

class HashCache {
public:
    using Hashes   = std::vector<at::Tensor>;
    using Snapshot = std::tuple<Hashes, std::string, bool>;

    explicit HashCache(int64_t capacity): capacity_(validCapacity(capacity)) {}

    void put(const std::string& key, const Hashes& hashes, const std::string& generation, bool approved) {
        putForGeneration(key, hashes, generation, approved, false, [](auto action) { action(); });
    }

    // withCurrent holds the embedding index lock only while running a short
    // hash-index mutation. Copying tensors and releasing retired storage happen
    // outside both locks. The allowed lock order is embedding -> hash.
    template<typename WithCurrent>
    void putForGeneration(const std::string& key,
                          const Hashes&      hashes,
                          const std::string& generation,
                          bool               approved,
                          bool               deduplicate,
                          WithCurrent        withCurrent) {
        if (key.empty() || !enabled()) {
            return;
        }
        auto    value = std::make_shared<Entry>(key, generation, charge(key, hashes, generation), approved);
        Entries retired;
        bool    started = false;
        try {
            withCurrent([&] {
                std::lock_guard lock(mutex_);
                if (capacity_ == 0) {
                    return;
                }
                const auto pending  = pending_.find(key);
                const auto resident = index_.find(key);
                if (deduplicate) {
                    auto existing = pending != pending_.end() ? pending->second :
                                    resident != index_.end()  ? *resident->second :
                                                                nullptr;
                    if (existing && existing->generation == generation) {
                        existing->approved |= approved;
                        return;
                    }
                }
                pending_.insert_or_assign(key, value);
                started            = true;
                size_t reclaimable = resident == index_.end() ? 0 : retire(resident, retired);
                if (value->bytes <= capacity_) {
                    while (!lru_.empty() && !fits(value->bytes, used() - reclaimable)) {
                        reclaimable += retire(index_.find(lru_.front()->key), retired);
                    }
                }
            });
            release(retired);
            if (!started) {
                return;
            }

            bool reserved = false;
            withCurrent([&] {
                std::lock_guard lock(mutex_);
                if (isPending(value) && fits(value->bytes, used())) {
                    building_ += value->bytes;
                    value->building = true;
                    reserved        = true;
                }
            });
            if (reserved) {
                value->batch = std::make_unique<EmbHashTensorBatch>(hashes);
                withCurrent([&] { publish(value); });
            }
        } catch (...) {
            release(retired);
            discard(value);
            throw;
        }
        discard(value);
    }

    std::optional<Snapshot> snapshot(const std::string& key, const std::optional<std::string>& generation, bool touch) {
        return capture(key, generation, touch, nullptr);
    }

    std::vector<std::optional<Snapshot>> metadata(const std::vector<std::string>& keys, size_t maxRows) {
        std::vector<std::optional<Snapshot>> values;
        values.reserve(keys.size());
        for (const auto& key : keys) {
            values.push_back(capture(key, {}, false, &maxRows));
        }
        return values;
    }

    bool contains(const std::string& key) const {
        std::lock_guard lock(mutex_);
        return index_.find(key) != index_.end();
    }

    bool approved(const std::string& key) const {
        std::lock_guard lock(mutex_);
        const auto      found = index_.find(key);
        return found != index_.end() && (*found->second)->approved;
    }

    bool markApproved(const std::string& key, const std::string& generation) {
        std::lock_guard lock(mutex_);
        const auto      found = index_.find(key);
        if (found == index_.end() || (*found->second)->generation != generation) {
            return false;
        }
        (*found->second)->approved = true;
        return true;
    }

    std::vector<std::string> keys(std::optional<size_t> limit) const {
        std::vector<std::string> keys;
        std::lock_guard          lock(mutex_);
        const auto               count = std::min(limit.value_or(lru_.size()), lru_.size());
        keys.resize(count);
        auto entry = lru_.rbegin();
        for (auto key = keys.rbegin(); key != keys.rend(); ++key, ++entry) {
            *key = (*entry)->key;
        }
        return keys;
    }

    void clear() {
        Entries retired;
        {
            std::lock_guard lock(mutex_);
            pending_.clear();
            while (!lru_.empty()) {
                retire(index_.find(lru_.front()->key), retired);
            }
        }
        release(retired);
    }

    void resize(int64_t capacity) {
        const auto next = validCapacity(capacity);
        Entries    retired;
        {
            std::lock_guard lock(mutex_);
            capacity_ = next;
            pending_.clear();
            size_t reclaimable = 0;
            while (!lru_.empty() && used() - reclaimable > capacity_) {
                reclaimable += retire(index_.find(lru_.front()->key), retired);
            }
        }
        release(retired);
    }

    bool enabled() const {
        std::lock_guard lock(mutex_);
        return capacity_ != 0;
    }

    std::array<size_t, 5> stats() const {
        std::lock_guard lock(mutex_);
        return {lru_.size(), resident_, building_, retired_, capacity_};
    }

private:
    struct Entry {
        Entry(const std::string& key, const std::string& generation, size_t bytes, bool approved):
            key(key), generation(generation), bytes(bytes), approved(approved) {}

        std::string                         key;
        std::string                         generation;
        size_t                              bytes;
        bool                                approved;
        bool                                building = false;
        bool                                retired  = false;
        size_t                              readers  = 0;
        std::unique_ptr<EmbHashTensorBatch> batch;
    };

    struct Reader {
        HashCache&                    cache;
        const std::shared_ptr<Entry>& value;

        ~Reader() {
            cache.finishRead(value);
        }
    };

    using Entries = std::list<std::shared_ptr<Entry>>;
    using Index   = std::unordered_map<std::string, Entries::iterator>;

    static size_t validCapacity(int64_t capacity) {
        TORCH_CHECK(capacity >= 0, "hash cache max_bytes must be non-negative");
        return static_cast<size_t>(capacity);
    }

    static size_t charge(const std::string& key, const Hashes& hashes, const std::string& generation) {
        // Charge the int32 data and estimate the native index, LRU node, keys
        // and TensorImpl allocations. Bound the sum before adding each tensor.
        constexpr auto maxBytes = std::numeric_limits<size_t>::max();
        TORCH_CHECK(generation.size() <= maxBytes - 256 && key.size() <= (maxBytes - 256 - generation.size()) / 2,
                    "EMB hash cache charge overflow");
        size_t bytes = 256 + 2 * key.size() + generation.size();
        for (const auto& hash : hashes) {
            constexpr size_t tensorOverhead = 256;
            TORCH_CHECK(bytes <= maxBytes - tensorOverhead
                            && static_cast<size_t>(hash.numel()) <= (maxBytes - bytes - tensorOverhead) / 4,
                        "EMB hash cache charge overflow");
            bytes += tensorOverhead + 4 * static_cast<size_t>(hash.numel());
        }
        return bytes;
    }

    size_t used() const {
        return resident_ + building_ + retired_;
    }

    bool fits(size_t bytes, size_t usedBytes) const {
        return usedBytes <= capacity_ && bytes <= capacity_ - usedBytes;
    }

    bool isPending(const std::shared_ptr<Entry>& value) const {
        const auto found = pending_.find(value->key);
        return found != pending_.end() && found->second == value;
    }

    void publish(const std::shared_ptr<Entry>& value) {
        std::lock_guard lock(mutex_);
        if (!isPending(value) || used() > capacity_) {
            return;
        }
        lru_.push_back(value);
        try {
            index_.emplace(value->key, std::prev(lru_.end()));
        } catch (...) {
            lru_.pop_back();
            throw;
        }
        pending_.erase(value->key);
        building_ -= value->bytes;
        resident_ += value->bytes;
        value->building = false;
    }

    void discard(const std::shared_ptr<Entry>& value) {
        // Only the publishing thread owns the uncommitted batch.
        if (value->building) {
            value->batch.reset();
        }
        std::lock_guard lock(mutex_);
        if (isPending(value)) {
            pending_.erase(value->key);
        }
        if (value->building) {
            building_ -= value->bytes;
            value->building = false;
        }
    }

    size_t retire(Index::iterator found, Entries& retired) {
        const auto  slot  = found->second;
        const auto& value = *slot;
        value->retired    = true;
        resident_ -= value->bytes;
        retired_ += value->bytes;
        const auto bytes = value->bytes;
        const bool ready = value->readers == 0;
        index_.erase(found);
        if (ready) {
            retired.splice(retired.end(), lru_, slot);
        } else {
            lru_.erase(slot);
        }
        return ready ? bytes : 0;
    }

    std::optional<Snapshot>
    capture(const std::string& key, const std::optional<std::string>& generation, bool touch, size_t* remainingRows) {
        std::shared_ptr<Entry> value;
        bool                   approved;
        {
            std::lock_guard lock(mutex_);
            const auto      found = index_.find(key);
            if (found == index_.end() || (generation && (*found->second)->generation != *generation)) {
                return {};
            }
            value = *found->second;
            if (touch) {
                lru_.splice(lru_.end(), lru_, found->second);
            }
            ++value->readers;
            approved = value->approved;
        }

        Reader reader{*this, value};
        if (remainingRows) {
            auto rows = value->batch->metadataRows(*remainingRows);
            if (!rows) {
                return {};
            }
            *remainingRows -= *rows;
        }
        return Snapshot(value->batch->cloneHashes(), value->generation, approved);
    }

    void finishRead(const std::shared_ptr<Entry>& value) {
        bool reclaim;
        {
            std::lock_guard lock(mutex_);
            reclaim = --value->readers == 0 && value->retired;
        }
        if (reclaim) {
            release(value);
        }
    }

    void release(const std::shared_ptr<Entry>& value) {
        value->batch.reset();
        std::lock_guard lock(mutex_);
        retired_ -= value->bytes;
    }

    void release(Entries& retired) {
        if (retired.empty()) {
            return;
        }
        size_t bytes = 0;
        for (const auto& value : retired) {
            value->batch.reset();
            bytes += value->bytes;
        }
        retired.clear();
        std::lock_guard lock(mutex_);
        retired_ -= bytes;
    }

    mutable std::mutex                                      mutex_;
    size_t                                                  capacity_;
    size_t                                                  resident_ = 0;
    size_t                                                  building_ = 0;
    size_t                                                  retired_  = 0;
    Entries                                                 lru_;
    Index                                                   index_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> pending_;
};

}  // namespace rtp_llm::emb_local
