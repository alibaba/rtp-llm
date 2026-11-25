#pragma once

#include "rtp_llm/cpp/pybind/emb_local/EmbLocalArena.h"
#include "rtp_llm/cpp/pybind/emb_local/EmbLocalTree.h"

#include <array>
#include <atomic>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm::emb_local {

enum class Tier {
    Cpu,
    Gpu
};

inline size_t tierIndex(Tier tier) {
    return static_cast<size_t>(tier);
}

using Blocks = std::array<std::shared_ptr<Arena::Block>, 2>;

struct Slice {
    int64_t              offset;
    int64_t              nbytes;
    at::ScalarType       type;
    std::vector<int64_t> sizes;
    at::Device           original;
    Tier                 placement;
};

struct Resident {
    uint64_t                    generation;
    Tier                        tier;
    Blocks                      storage;
    std::array<int64_t, 2>      bytes;
    int64_t                     tokens;
    std::vector<Slice>          slices;
    std::shared_ptr<const Tree> tree;
};

struct Audit {
    bool        decided         = false;
    bool        checked         = false;
    bool        verdict_present = false;
    bool        passed          = false;
    int64_t     code            = 0;
    std::string message;
    std::string origin;
};

// A request handle keeps its acquired generation alive across removal.
// The cell contains no Python reference. Its record can be replaced by
// promotion/demotion while readers retain a snapshot of the previous block.
class Cell {
public:
    explicit Cell(uint64_t generation): generation_(generation) {}
    uint64_t generation() const {
        return generation_;
    }
    std::shared_ptr<const Resident> record() const {
        return std::atomic_load_explicit(&record_, std::memory_order_acquire);
    }

    std::shared_ptr<const Resident> replace(std::shared_ptr<const Resident> next) {
        return std::atomic_exchange_explicit(&record_, std::move(next), std::memory_order_acq_rel);
    }

    Audit audit() const {
        std::lock_guard lock(audit_mutex_);
        return audit_;
    }

    void setVerdict(bool checked, bool present, bool passed, int64_t code, std::string message) {
        std::lock_guard lock(audit_mutex_);
        if (audit_.checked && !checked) {
            return;
        }
        audit_.decided         = true;
        audit_.checked         = checked;
        audit_.verdict_present = present;
        audit_.passed          = passed;
        audit_.code            = code;
        audit_.message         = std::move(message);
    }

    void setOrigin(std::string origin) {
        std::lock_guard lock(audit_mutex_);
        audit_.origin = std::move(origin);
    }

private:
    mutable std::mutex              audit_mutex_;
    Audit                           audit_;
    const uint64_t                  generation_;
    std::shared_ptr<const Resident> record_;
};

// Python waits on pending requests. The native index owns the generation
// against which publication, invalidation, and residency changes compare.
class Index {
public:
    struct Lookup {
        uint64_t                        generation;
        std::shared_ptr<const Resident> resident;
        std::shared_ptr<Cell>           cell;
    };

    struct Charge {
        std::array<int64_t, 2> bytes{};
        std::array<int64_t, 2> entries{};
        int64_t                tokens  = 0;
        int64_t                pending = 0;
    };

    Charge charge() const {
        std::lock_guard lock(mutex_);
        auto            result = charge_;
        result.pending         = entries_.size() - result.entries[0] - result.entries[1];
        return result;
    }

    Lookup claim(const std::string& key) {
        std::lock_guard lock(mutex_);
        const auto      found = entries_.find(key);
        if (found != entries_.end()) {
            auto& state = found->second;
            if (state.resident) {
                auto& lru = lruFor(state.resident->tier);
                lru.splice(lru.end(), lru, *state.lru);
            }
            return snapshot(state);
        }
        if (!accepting_) {
            return {};
        }
        const auto generation = ++generation_;
        if (generation == 0) {
            accepting_ = false;
            throw std::overflow_error("local EMB generation space exhausted");
        }
        auto [inserted, unique] = entries_.try_emplace(key, std::make_shared<Cell>(generation));
        return snapshot(inserted->second);
    }

    Lookup lookup(const std::string& key, bool consume) {
        std::lock_guard lock(mutex_);
        const auto      found = entries_.find(key);
        if (found == entries_.end()) {
            return {};
        }
        auto& state = found->second;
        if (consume && state.resident) {
            auto& lru = lruFor(state.resident->tier);
            lru.splice(lru.end(), lru, *state.lru);
        }
        return snapshot(state);
    }

    template<typename Action>
    bool withCurrent(const std::string& key, const Cell& cell, Action action) {
        std::lock_guard lock(mutex_);
        const auto      found = entries_.find(key);
        if (found == entries_.end() || found->second.cell.get() != &cell) {
            return false;
        }
        action();
        return true;
    }

    bool publish(const std::string&                     key,
                 const std::shared_ptr<Resident>&       proposed,
                 const std::shared_ptr<const Resident>& expected) {
        std::shared_ptr<const Resident> replaced;
        {
            std::lock_guard lock(mutex_);
            const auto      found = entries_.find(key);
            if (!accepting_ || found == entries_.end() || found->second.cell->generation() != proposed->generation
                || found->second.resident != expected) {
                return false;
            }
            auto& state    = found->second;
            auto& next     = lruFor(proposed->tier);
            auto  position = next.insert(next.end(), key);
            for (const auto& block : proposed->storage) {
                if (block && !block->publish()) {
                    next.erase(position);
                    return false;
                }
            }
            if (state.resident) {
                retire(*state.resident);
                account(*state.resident, -1);
                lruFor(state.resident->tier).erase(*state.lru);
                replaced = std::move(state.resident);
            }
            account(*proposed, 1);
            state.lru      = position;
            state.resident = proposed;
            state.cell->replace(proposed);
        }
        return true;
    }

    struct Removed {
        std::shared_ptr<const Resident> resident;
        bool                            removed = false;
        explicit                        operator bool() const {
            return removed;
        }
    };

    Removed remove(const std::string& key, uint64_t generation = 0, const Resident* expected = nullptr) {
        Removed result;
        {
            std::lock_guard lock(mutex_);
            const auto      found = entries_.find(key);
            if (found == entries_.end() || (generation && found->second.cell->generation() != generation)
                || (expected && found->second.resident.get() != expected)) {
                return {};
            }
            auto& state = found->second;
            if (state.resident) {
                lruFor(state.resident->tier).erase(*state.lru);
                retire(*state.resident);
                account(*state.resident, -1);
                result.resident = std::move(state.resident);
            }
            result.removed = true;
            entries_.erase(found);
        }
        return result;
    }

    std::optional<std::pair<std::string, std::shared_ptr<const Resident>>>
    oldest(Tier tier, const std::string* except, const std::string* also_except, bool requires_cpu_block) const {
        std::lock_guard lock(mutex_);
        for (const auto& key : lruFor(tier)) {
            const auto& record = entries_.at(key).resident;
            if ((!except || key != *except) && (!also_except || key != *also_except)
                && (!requires_cpu_block || record->bytes[tierIndex(Tier::Cpu)] > 0)) {
                return std::make_pair(key, record);
            }
        }
        return std::nullopt;
    }

    std::vector<std::pair<std::string, Tier>> residency(int64_t limit) const {
        std::lock_guard                           lock(mutex_);
        std::vector<std::pair<std::string, Tier>> snapshot;
        const size_t                              total = gpu_.size() + cpu_.size();
        snapshot.reserve(limit < 0 ? total : std::min(total, static_cast<size_t>(limit)));
        for (auto tier : {Tier::Gpu, Tier::Cpu}) {
            const auto& lru = lruFor(tier);
            for (auto it = lru.rbegin(); it != lru.rend(); ++it) {
                if (limit >= 0 && snapshot.size() == static_cast<size_t>(limit)) {
                    return snapshot;
                }
                snapshot.emplace_back(*it, tier);
            }
        }
        return snapshot;
    }

    std::pair<uint64_t, int64_t> clear(bool closing) {
        std::unordered_map<std::string, Entry> retired;
        std::pair<uint64_t, int64_t>           retired_through;
        {
            std::lock_guard lock(mutex_);
            accepting_ = accepting_ && !closing;
            for (const auto& [key, entry] : entries_) {
                if (entry.resident) {
                    retire(*entry.resident);
                }
            }
            retired_through = {generation_, charge_.entries[0] + charge_.entries[1]};
            charge_         = {};
            retired.swap(entries_);
            gpu_.clear();
            cpu_.clear();
        }
        return retired_through;
    }

private:
    void account(const Resident& object, int64_t sign) {
        charge_.tokens += object.tokens * sign;
        charge_.entries[tierIndex(object.tier)] += sign;
        for (size_t i = 0; i < charge_.bytes.size(); ++i) {
            charge_.bytes[i] += object.bytes[i] * sign;
        }
    }

    static void retire(const Resident& resident) noexcept {
        for (const auto& block : resident.storage) {
            if (block) {
                block->retire();
            }
        }
    }

    using Lru = std::list<std::string>;
    struct Entry {
        explicit Entry(std::shared_ptr<Cell> cell): cell(std::move(cell)) {}
        std::shared_ptr<Cell>           cell;
        std::shared_ptr<const Resident> resident;
        std::optional<Lru::iterator>    lru;
    };

    static Lookup snapshot(const Entry& state) {
        return {state.cell->generation(), state.resident, state.cell};
    }

    Lru& lruFor(Tier tier) {
        return tier == Tier::Gpu ? gpu_ : cpu_;
    }
    const Lru& lruFor(Tier tier) const {
        return tier == Tier::Gpu ? gpu_ : cpu_;
    }

    mutable std::mutex                     mutex_;
    std::unordered_map<std::string, Entry> entries_;
    Lru                                    gpu_;
    Lru                                    cpu_;
    // Keep monotonic identity across clear and re-claim.
    uint64_t generation_ = 0;
    bool     accepting_  = true;
    Charge   charge_;
};

}  // namespace rtp_llm::emb_local
