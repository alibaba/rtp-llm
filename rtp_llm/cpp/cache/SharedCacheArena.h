#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// One byte budget, with stable, overlapping typed address spaces. Slot id i
// in group g addresses i * stride[g]. Only allocated ranges may be accessed.
// Unlike a homogeneous parent-block pool, small groups can share the remaining
// bytes of a range; returning a range coalesces it with both neighbours.
class SharedCacheArena {
public:
    enum class Capacity {
        FREE,
        AVAILABLE,
        TOTAL
    };

    SharedCacheArena(size_t bytes, const std::vector<size_t>& payloads): bytes_(bytes) {
        for (auto payload : payloads) {
            if (payload == 0) {
                throw std::invalid_argument("shared cache payload must be positive");
            }
            strides_.push_back((payload + 255) / 256 * 256);
        }
        if (strides_.empty()) {
            throw std::invalid_argument("shared cache requires groups");
        }
        reserved_bytes_ = *std::max_element(strides_.begin(), strides_.end());
        if (bytes_ <= reserved_bytes_) {
            throw std::invalid_argument("shared cache budget cannot hold its null slots");
        }
        for (auto stride : strides_) {
            if (bytes_ / stride > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument("shared cache slot ids exceed int32");
            }
        }
        free_.emplace(reserved_bytes_, bytes_);
        available_ = free_;
    }

    size_t stride(size_t group) const {
        return strides_.at(group);
    }
    size_t bytes() const {
        return bytes_;
    }
    size_t usableBytes() const {
        return bytes_ - reserved_bytes_;
    }
    size_t freeBytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return rangeBytes(free_);
    }
    size_t availableBytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return rangeBytes(available_);
    }

    size_t slotCount(size_t group) const {
        return bytes_ / stride(group);
    }

    std::vector<int32_t> malloc(size_t group, size_t count) {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<int32_t>        ids;
        ids.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            size_t offset = take(free_, stride(group));
            if (offset == 0) {
                for (auto id : ids) {
                    releaseLocked(group, id);
                }
                return {};
            }
            ids.push_back(static_cast<int32_t>(offset / stride(group)));
            eraseRange(available_, offset, offset + stride(group));
            allocations_.emplace(offset, Allocation{group, false});
        }
        return ids;
    }

    void release(size_t group, int32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        releaseLocked(group, id);
    }

    void setReclaimable(size_t group, int32_t id, bool reclaimable) {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = allocations_.find(static_cast<size_t>(id) * stride(group));
        if (it == allocations_.end() || it->second.group != group) {
            throw std::logic_error("reference to unallocated shared cache slot");
        }
        if (it->second.reclaimable != reclaimable) {
            const size_t offset = it->first;
            if (reclaimable) {
                insert(available_, offset, offset + stride(group));
            } else {
                eraseRange(available_, offset, offset + stride(group));
            }
            it->second.reclaimable = reclaimable;
        }
    }

    size_t freeSlots(size_t group) const {
        std::lock_guard<std::mutex> lock(mu_);
        return countSlots(free_, stride(group));
    }

    size_t availableSlots(size_t group) const {
        std::lock_guard<std::mutex> lock(mu_);
        return countSlots(available_, stride(group));
    }

    size_t totalSlots(size_t group) const {
        return countSlots(Ranges{{reserved_bytes_, bytes_}}, stride(group));
    }

    // Admission simulates the entire request against one budget. Independent
    // per-group checks would promise the same free bytes more than once.
    bool canAllocate(const std::vector<size_t>& demand, Capacity capacity) const {
        std::lock_guard<std::mutex> lock(mu_);
        Ranges                      ranges = capacity == Capacity::TOTAL     ? Ranges{{reserved_bytes_, bytes_}} :
                                             capacity == Capacity::AVAILABLE ? available_ :
                                                                               free_;
        for (size_t group = 0; group < demand.size(); ++group) {
            for (size_t i = 0; i < demand[group]; ++i) {
                if (take(ranges, stride(group)) == 0) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    using Ranges = std::map<size_t, size_t>;  // start -> exclusive end
    struct Allocation {
        size_t group;
        bool   reclaimable;
    };

    static size_t rangeBytes(const Ranges& ranges) {
        size_t bytes = 0;
        for (const auto& range : ranges) {
            bytes += range.second - range.first;
        }
        return bytes;
    }

    static size_t aligned(size_t offset, size_t stride) {
        return (offset / stride + (offset % stride != 0)) * stride;
    }

    static size_t countSlots(const Ranges& ranges, size_t stride) {
        size_t slots = 0;
        for (const auto& range : ranges) {
            const size_t first = aligned(range.first, stride);
            if (first < range.second) {
                slots += (range.second - first) / stride;
            }
        }
        return slots;
    }

    static size_t take(Ranges& ranges, size_t stride) {
        for (auto it = ranges.begin(); it != ranges.end(); ++it) {
            const size_t start  = it->first;
            const size_t end    = it->second;
            const size_t offset = aligned(start, stride);
            if (offset >= end || stride > end - offset) {
                continue;
            }
            ranges.erase(it);
            if (start < offset) {
                ranges.emplace(start, offset);
            }
            if (offset + stride < end) {
                ranges.emplace(offset + stride, end);
            }
            return offset;
        }
        return 0;
    }

    static void insert(Ranges& ranges, size_t start, size_t end) {
        auto next = ranges.lower_bound(start);
        if (next != ranges.begin()) {
            auto previous = std::prev(next);
            if (previous->second == start) {
                start = previous->first;
                ranges.erase(previous);
            }
        }
        if (next != ranges.end() && end == next->first) {
            end = next->second;
            ranges.erase(next);
        }
        ranges.emplace(start, end);
    }

    static void eraseRange(Ranges& ranges, size_t start, size_t end) {
        auto next = ranges.upper_bound(start);
        if (next == ranges.begin()) {
            throw std::logic_error("shared cache range is occupied");
        }
        auto         it             = std::prev(next);
        const size_t previous_start = it->first;
        const size_t previous_end   = it->second;
        if (previous_end < end) {
            throw std::logic_error("shared cache range is occupied");
        }
        ranges.erase(it);
        if (previous_start < start) {
            ranges.emplace(previous_start, start);
        }
        if (end < previous_end) {
            ranges.emplace(end, previous_end);
        }
    }

    void releaseLocked(size_t group, int32_t id) {
        const size_t offset = static_cast<size_t>(id) * stride(group);
        auto         it     = allocations_.find(offset);
        if (it == allocations_.end() || it->second.group != group) {
            throw std::logic_error("double free or wrong shared cache group");
        }
        if (!it->second.reclaimable) {
            insert(available_, offset, offset + stride(group));
        }
        allocations_.erase(it);
        insert(free_, offset, offset + stride(group));
    }

    size_t                       bytes_;
    size_t                       reserved_bytes_;
    std::vector<size_t>          strides_;
    Ranges                       free_;
    Ranges                       available_;
    std::map<size_t, Allocation> allocations_;
    mutable std::mutex           mu_;
};

}  // namespace rtp_llm
