#pragma once

#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace rtp_llm::emb_local {

class Arena: public std::enable_shared_from_this<Arena> {
public:
    struct Stats {
        int64_t capacity;
        int64_t free;
        int64_t reserved;
        int64_t resident;
        int64_t retired;
        int64_t quarantined;
        int64_t largest_free;
    };

    class Block {
    public:
        Block(std::shared_ptr<Arena> arena, int64_t offset, int64_t bytes):
            arena_(std::move(arena)), offset_(offset), bytes_(bytes) {}

        ~Block() {
            if (claimed_) {
                arena_->release(*this);
            }
        }

        Block(const Block&)            = delete;
        Block& operator=(const Block&) = delete;

        at::Tensor bytes() const {
            return arena_->slab_.narrow(0, offset_, bytes_);
        }
        int64_t size() const {
            return bytes_;
        }
        int64_t offset() const {
            return offset_;
        }

        // May be called on a reserved output or a published input after a
        // CUDA submission whose completion cannot be proven.
        void quarantine() noexcept {
            arena_->quarantine(*this);
        }
        bool publish() noexcept {
            return arena_->publish(*this);
        }
        void retire() noexcept {
            arena_->retire(*this);
        }

    private:
        friend class Arena;
        enum class State {
            Reserved,
            Resident,
            Retired,
            Quarantined
        };
        std::shared_ptr<Arena> arena_;
        int64_t                offset_;
        int64_t                bytes_;
        State                  state_   = State::Reserved;
        bool                   claimed_ = false;
    };

    Arena(int64_t capacity, const at::Device& device, bool pinned): capacity_(capacity), free_bytes_(capacity) {
        TORCH_CHECK(capacity > 0, "local EMB arena requires a positive capacity");
        slab_ = at::empty({capacity}, at::TensorOptions().device(device).dtype(at::kByte).pinned_memory(pinned));
        free_.emplace(0, capacity);
    }

    [[nodiscard]] std::shared_ptr<Block> reserve(int64_t requested, int64_t alignment) {
        if (requested <= 0 || requested > capacity_) {
            return {};
        }
        requested = alignedSize(requested, alignment);
        std::unique_lock lock(mutex_);
        if (stopped_) {
            return {};
        }
        for (auto extent = free_.begin(); extent != free_.end(); ++extent) {
            const auto prefix = (alignment - extent->first % alignment) % alignment;
            if (extent->second - prefix < requested) {
                continue;
            }
            if (!quarantine_token_) {
                quarantine_token_ = std::make_unique<std::shared_ptr<Arena>>();
            }
            const auto offset = extent->first + prefix;
            const auto suffix = extent->second - prefix - requested;
            auto       block  = std::make_shared<Block>(shared_from_this(), offset, requested);
            if (prefix) {
                if (suffix) {
                    free_.emplace(offset + requested, suffix);
                }
                extent->second = prefix;
            } else if (suffix) {
                auto remainder = free_.extract(extent);
                remainder.key() += requested;
                remainder.mapped() = suffix;
                free_.insert(std::move(remainder));
            } else {
                free_.erase(extent);
            }
            free_bytes_ -= requested;
            reserved_ += requested;
            block->claimed_ = true;
            return block;
        }
        return {};
    }

    bool canReserveAfterEviction(int64_t requested, int64_t alignment) const {
        std::lock_guard lock(mutex_);
        return !stopped_ && requested <= capacity_ && alignedSize(requested, alignment) <= free_bytes_ + resident_;
    }

    bool publish(Block& block) noexcept {
        std::lock_guard lock(mutex_);
        if (stopped_ || block.state_ != Block::State::Reserved) {
            return false;
        }
        block.state_ = Block::State::Resident;
        reserved_ -= block.bytes_;
        resident_ += block.bytes_;
        return true;
    }

    void retire(Block& block) noexcept {
        std::lock_guard lock(mutex_);
        if (block.state_ == Block::State::Resident) {
            block.state_ = Block::State::Retired;
            resident_ -= block.bytes_;
            retired_ += block.bytes_;
        }
    }

    void stop() noexcept {
        std::lock_guard lock(mutex_);
        stopped_ = true;
    }

    Stats stats() const noexcept {
        std::lock_guard lock(mutex_);
        int64_t         largest = 0;
        for (const auto& [offset, bytes] : free_) {
            largest = std::max(largest, bytes);
        }
        return {capacity_, free_bytes_, reserved_, resident_, retired_, quarantined_, largest};
    }

private:
    int64_t alignedSize(int64_t requested, int64_t alignment) const {
        const auto padding = (alignment - requested % alignment) % alignment;
        return capacity_ - requested >= padding ? requested + padding : requested;
    }

    void quarantine(Block& block) noexcept {
        std::lock_guard lock(mutex_);
        if (block.state_ == Block::State::Quarantined) {
            return;
        }
        // The token is allocated before any reservation can reach CUDA.
        // Never allocate memory while handling an uncertain completion.
        if (!quarantine_owner_) {
            *quarantine_token_ = shared_from_this();
            quarantine_owner_  = quarantine_token_.release();
        }
        charge(block.state_) -= block.bytes_;
        block.state_ = Block::State::Quarantined;
        quarantined_ += block.bytes_;
    }

    void release(Block& block) noexcept {
        std::lock_guard lock(mutex_);
        if (block.state_ == Block::State::Quarantined) {
            return;
        }
        charge(block.state_) -= block.bytes_;
        auto       next = free_.lower_bound(block.offset_);
        const bool adjacent_left =
            next != free_.begin() && std::prev(next)->first + std::prev(next)->second == block.offset_;
        const bool adjacent_right = next != free_.end() && block.offset_ + block.bytes_ == next->first;
        if (adjacent_left) {
            auto left = std::prev(next);
            left->second += block.bytes_;
            if (adjacent_right) {
                left->second += next->second;
                free_.erase(next);
            }
        } else if (adjacent_right) {
            auto right  = free_.extract(next);
            right.key() = block.offset_;
            right.mapped() += block.bytes_;
            free_.insert(std::move(right));
        } else {
            try {
                free_.emplace_hint(next, block.offset_, block.bytes_);
            } catch (...) {
                // The interval remains unavailable and the cache can use
                // other extents. Do not destroy a healthy inference result.
                quarantined_ += block.bytes_;
                return;
            }
        }
        free_bytes_ += block.bytes_;
    }

    int64_t& charge(Block::State state) noexcept {
        switch (state) {
            case Block::State::Reserved:
                return reserved_;
            case Block::State::Resident:
                return resident_;
            case Block::State::Retired:
                return retired_;
            case Block::State::Quarantined:
                return quarantined_;
        }
        std::terminate();
    }

    const int64_t      capacity_;
    at::Tensor         slab_;
    mutable std::mutex mutex_;
    // One entry per disjoint free interval. Existing adjacent intervals
    // expand in place and do not allocate on the normal release path.
    std::map<int64_t, int64_t>              free_;
    int64_t                                 free_bytes_;
    int64_t                                 reserved_    = 0;
    int64_t                                 resident_    = 0;
    int64_t                                 retired_     = 0;
    int64_t                                 quarantined_ = 0;
    bool                                    stopped_     = false;
    std::unique_ptr<std::shared_ptr<Arena>> quarantine_token_;
    std::shared_ptr<Arena>*                 quarantine_owner_ = nullptr;
};

}  // namespace rtp_llm::emb_local
