#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <c10/core/Event.h>
#include <torch/extension.h>

namespace rtp_llm {

struct LinearReplayGpuHold {
    void recordLastUse(std::shared_ptr<torch::Event> event) {
        std::lock_guard<std::mutex> lock(mutex);
        last_use = std::move(event);
    }

    std::shared_ptr<torch::Event> lastUse() const {
        std::lock_guard<std::mutex> lock(mutex);
        return last_use;
    }

private:
    mutable std::mutex            mutex;
    std::shared_ptr<torch::Event> last_use;
};

struct LinearReplayLease: LinearReplayGpuHold {
    int32_t slot_id    = -1;
    int64_t generation = 0;
};

struct LinearReplayBlockHold: LinearReplayGpuHold {};

// Reuse is delayed by an event query; request completion never waits for the GPU.
class LinearReplayRetirementQueue {
public:
    ~LinearReplayRetirementQueue() {
        // Only engine teardown drains outstanding GPU ownership synchronously.
        for (auto& item : retired_) {
            if (item.event) {
                item.event->synchronize();
            }
            item.release();
        }
    }

    void retire(std::shared_ptr<torch::Event> event, std::function<void()> release) {
        std::lock_guard<std::mutex> lock(mutex_);
        retired_.push_back({std::move(event), std::move(release)});
        reapLocked();
    }

    void reap() {
        std::lock_guard<std::mutex> lock(mutex_);
        reapLocked();
    }

private:
    struct Retired {
        std::shared_ptr<torch::Event> event;
        std::function<void()>         release;
    };

    void reapLocked() {
        for (auto it = retired_.begin(); it != retired_.end();) {
            if (!it->event || it->event->query()) {
                it->release();
                it = retired_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::mutex           mutex_;
    std::vector<Retired> retired_;
};

class LinearReplaySlotPool: public std::enable_shared_from_this<LinearReplaySlotPool> {
public:
    explicit LinearReplaySlotPool(size_t slots): generations_(slots, 0), in_use_(slots, false) {}

    std::shared_ptr<LinearReplayLease> acquire() {
        retirement_.reap();
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t slot = 0; slot < in_use_.size(); ++slot) {
            if (in_use_[slot]) {
                continue;
            }
            in_use_[slot]     = true;
            auto* lease       = new LinearReplayLease;
            lease->slot_id    = static_cast<int32_t>(slot);
            lease->generation = ++generations_[slot];
            return std::shared_ptr<LinearReplayLease>(lease, [self = shared_from_this()](LinearReplayLease* value) {
                const auto slot  = value->slot_id;
                const auto event = value->lastUse();
                delete value;
                self->retirement_.retire(event, [pool = self.get(), slot] {
                    std::lock_guard<std::mutex> lock(pool->mutex_);
                    pool->in_use_[static_cast<size_t>(slot)] = false;
                });
            });
        }
        return nullptr;
    }

private:
    std::vector<int64_t>        generations_;
    std::vector<bool>           in_use_;
    std::mutex                  mutex_;
    LinearReplayRetirementQueue retirement_;
};

}  // namespace rtp_llm
