#include "rtp_llm/cpp/engine_base/sleep/SleepRoundFence.h"

namespace rtp_llm {

uint64_t SleepRoundFence::freeze() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  word = admitted_.fetch_or(kFrozen, std::memory_order_acq_rel);
    if (!(word & kFrozen)) {
        ++generation_;
        frozen_round_ = word & kRoundMask;
        target_.reset();
        quiesced_ = false;
    }
    return frozen_round_;
}

bool SleepRoundFence::setTarget(uint64_t round) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || !failure_.empty() || !(admitted_.load(std::memory_order_acquire) & kFrozen) || round < frozen_round_
        || round > kRoundMask || (target_ && *target_ != round)) {
        return false;
    }
    target_ = round;
    cv_.notify_all();
    return true;
}

SleepRoundFence::Permit SleepRoundFence::nextSlow() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        auto word = admitted_.load(std::memory_order_acquire);
        if (stopped_) {
            return {Action::STOP, generation_, word & kRoundMask};
        }
        const auto round = word & kRoundMask;
        if (!(word & kFrozen)) {
            if (round == kRoundMask) {
                failure_ = "sleep round counter exhausted";
                admitted_.fetch_or(kFrozen, std::memory_order_acq_rel);
                cv_.notify_all();
            } else if (admitted_.compare_exchange_strong(word, word + 1, std::memory_order_acq_rel)) {
                return {Action::RUN, generation_, round + 1};
            }
            continue;
        }
        if (failure_.empty() && target_) {
            if (round < *target_) {
                admitted_.fetch_add(1, std::memory_order_acq_rel);
                return {Action::RUN, generation_, round + 1};
            }
            if (!quiescing_ && !quiesced_) {
                quiescing_ = true;
                return {Action::QUIESCE, generation_, round};
            }
        }
        cv_.wait(lock);
    }
}

bool SleepRoundFence::wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto                   generation = generation_;
    if (!target_) {
        return false;
    }
    cv_.wait_for(lock, timeout, [this, generation] {
        return quiesced_ || stopped_ || !failure_.empty() || generation_ != generation;
    });
    return generation_ == generation && quiesced_ && !stopped_ && failure_.empty();
}

void SleepRoundFence::finishQuiesce(uint64_t generation, const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    // A device error must never be hidden by a concurrent stop or generation change.
    if (!error.empty()) {
        failure_ = error;
        admitted_.fetch_or(kFrozen, std::memory_order_acq_rel);
    }
    if (generation == generation_) {
        quiescing_ = false;
        quiesced_  = error.empty();
    }
    cv_.notify_all();
}

void SleepRoundFence::fail(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_ = error.empty() ? "executor failed during sleep quiesce" : error;
    admitted_.fetch_or(kFrozen, std::memory_order_acq_rel);
    cv_.notify_all();
}

bool SleepRoundFence::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || quiescing_ || !failure_.empty()) {
        return false;
    }
    ++generation_;
    target_.reset();
    quiesced_ = false;
    admitted_.fetch_and(kRoundMask, std::memory_order_release);
    cv_.notify_all();
    return true;
}

void SleepRoundFence::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    admitted_.fetch_or(kFrozen, std::memory_order_acq_rel);
    cv_.notify_all();
}

}  // namespace rtp_llm
