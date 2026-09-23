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
//
// A decode batch records the same completion event on many per-request holds. Keep
// those releases together so polling cost scales with the number of distinct GPU
// events rather than the number of requests. EventT is a template parameter only
// to make the batching policy testable without a CUDA device.
template<typename EventT>
class LinearReplayRetirementQueueT {
public:
    ~LinearReplayRetirementQueueT() {
        // Only engine teardown drains outstanding GPU ownership synchronously.
        for (auto& item : retired_) {
            if (item.event) {
                item.event->synchronize();
            }
            releaseAll(item);
        }
    }

    void retire(std::shared_ptr<EventT> event, std::function<void()> release) {
        if (!event) {
            release();
            return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& item : retired_) {
            if (item.event == event) {
                item.releases.push_back(std::move(release));
                return;
            }
        }

        if (event->query()) {
            lock.unlock();
            release();
            return;
        }
        retired_.push_back({std::move(event), {std::move(release)}});
    }

    void reap() {
        std::lock_guard<std::mutex> lock(mutex_);
        reapLocked();
    }

private:
    struct Retired {
        std::shared_ptr<EventT>            event;
        std::vector<std::function<void()>> releases;
    };

    static void releaseAll(Retired& item) {
        for (auto& release : item.releases) {
            release();
        }
    }

    void reapLocked() {
        for (auto it = retired_.begin(); it != retired_.end();) {
            if (!it->event || it->event->query()) {
                releaseAll(*it);
                it = retired_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::mutex           mutex_;
    std::vector<Retired> retired_;
};

using LinearReplayRetirementQueue = LinearReplayRetirementQueueT<torch::Event>;

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
