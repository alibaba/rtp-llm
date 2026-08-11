#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

struct CompletionQueuePlan {
    size_t              queue_count = 0;
    std::vector<size_t> expected_completions;

    size_t queueIndexForWorker(size_t worker_index) const {
        if (queue_count == 0) {
            throw std::logic_error("cannot assign a worker without a completion queue");
        }
        return worker_index % queue_count;
    }
};

inline CompletionQueuePlan makeCompletionQueuePlan(size_t worker_count) {
    CompletionQueuePlan plan;
    if (worker_count == 0) {
        return plan;
    }

    plan.queue_count = (worker_count + 1) / 2;
    plan.expected_completions.assign(plan.queue_count, 0);
    for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        ++plan.expected_completions[plan.queueIndexForWorker(worker_index)];
    }
    return plan;
}

}  // namespace rtp_llm
