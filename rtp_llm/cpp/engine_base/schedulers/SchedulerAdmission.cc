#include "rtp_llm/cpp/engine_base/schedulers/SchedulerAdmission.h"

#include <stdexcept>

namespace rtp_llm {

SchedulerAdmission::SchedulerAdmission(): state_(std::make_shared<State>()) {}

SchedulerAdmission::Result SchedulerAdmission::admit(bool continuation) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!(continuation ? state_->continuations_open : state_->roots_open)) {
        return {};
    }
    if (state_->next_id == 0) {
        throw std::overflow_error("scheduler admission execution identity exhausted");
    }
    const auto id = state_->next_id++;
    // Construct the callback before insertion: allocation failure must never
    // leave an admitted record for which no completion notification exists.
    std::weak_ptr<State> weak = state_;
    Result               result{true, id, [weak, id]() {
                      if (auto state = weak.lock()) {
                          std::lock_guard<std::mutex> lock(state->mutex);
                          state->active.erase(id);
                      }
                  }};
    state_->active.emplace(id, AdmissionLease{});
    return result;
}

void SchedulerAdmission::closeRoots() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->roots_open = false;
}

void SchedulerAdmission::sealContinuations() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->roots_open         = false;
    state_->continuations_open = false;
}

void SchedulerAdmission::beginTermination() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->terminating = true;
    state_->roots_open  = false;
}

bool SchedulerAdmission::reopen() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->terminating) {
        return false;
    }
    state_->roots_open         = true;
    state_->continuations_open = true;
    return true;
}

bool SchedulerAdmission::rootsOpen() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->roots_open;
}

bool SchedulerAdmission::continuationsSealed() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return !state_->continuations_open;
}

bool SchedulerAdmission::terminating() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->terminating;
}

size_t SchedulerAdmission::activeCount() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->active.size();
}

}  // namespace rtp_llm
