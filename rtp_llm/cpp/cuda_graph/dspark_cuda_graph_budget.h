#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// The budget counts draft positions; every request also contributes its anchor.
// Full width uses the dense backend and must never reuse packed metadata.
struct DSparkCudaGraphBudget {
    int  budget;
    bool packed;

    int tokensPerBatch() const {
        return budget + 1;
    }
};

inline std::vector<DSparkCudaGraphBudget> makeDSparkCudaGraphBudgets(int gamma, int configured_budget, bool adaptive) {
    if (gamma <= 0 || gamma == std::numeric_limits<int>::max() || configured_budget < -1 || configured_budget > gamma) {
        throw std::invalid_argument("invalid DSpARK CUDA graph budget");
    }
    std::vector<DSparkCudaGraphBudget> budgets;
    if (adaptive) {
        budgets.reserve(static_cast<size_t>(gamma) + 1);
        // Every rank captures in this order, independent of its online policy.
        for (int budget = 0; budget <= gamma; ++budget) {
            budgets.push_back({budget, budget < gamma});
        }
    } else {
        const int budget = configured_budget < 0 ? gamma : configured_budget;
        budgets.push_back({budget, budget < gamma});
    }
    return budgets;
}

inline std::optional<size_t> selectDSparkCudaGraphBudget(const std::vector<DSparkCudaGraphBudget>& budgets,
                                                         bool                                      packed,
                                                         int64_t                                   batch,
                                                         int64_t                                   tokens) {
    if (batch <= 0 || tokens <= 0 || tokens % batch != 0) {
        return std::nullopt;
    }
    const int64_t width = tokens / batch;
    for (size_t index = 0; index < budgets.size(); ++index) {
        if (budgets[index].packed == packed && budgets[index].tokensPerBatch() == width) {
            return index;
        }
    }
    return std::nullopt;
}

// Actual immutable capture geometry, queried before any proposal RNG draw.
struct DSparkCapturedCapacity {
    int64_t graph_batch;
    int64_t token_capacity;
    int     capture_budget;

    bool operator==(const DSparkCapturedCapacity& other) const {
        return graph_batch == other.graph_batch && token_capacity == other.token_capacity
               && capture_budget == other.capture_budget;
    }
};

struct DSparkCapacityPlan {
    DSparkCapturedCapacity graph;
    int64_t                real_batch;
    int64_t                query_rows;

    int64_t draftBudget() const {
        return query_rows - real_batch;
    }
};

inline std::optional<DSparkCapacityPlan>
makeDSparkCapacityPlan(const DSparkCapturedCapacity& graph, int64_t real_batch, int gamma) {
    if (gamma <= 0 || gamma == std::numeric_limits<int>::max() || graph.capture_budget < 0
        || graph.capture_budget >= gamma || real_batch <= 0 || graph.graph_batch < real_batch
        || graph.token_capacity > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    const int64_t width = static_cast<int64_t>(graph.capture_budget) + 1;
    if (graph.graph_batch > std::numeric_limits<int64_t>::max() / width
        || graph.token_capacity != graph.graph_batch * width
        || real_batch > std::numeric_limits<int64_t>::max() / (gamma + 1)) {
        return std::nullopt;
    }
    return DSparkCapacityPlan{graph, real_batch, std::min(graph.token_capacity, real_batch * (gamma + 1))};
}

// A unique fixed bank serves its frozen capacity plan, or the unchanged
// B*(k+1) geometry used by whole-idle-DP fake alignment. No Q/B bank inference.
inline bool isDSparkCapacityShape(const DSparkCapacityPlan& plan, int64_t rows) {
    return rows == plan.query_rows || rows == plan.real_batch * (plan.graph.capture_budget + 1);
}

inline int64_t dsparkDummyQueryLength(int64_t real_rows, int64_t capacity, int64_t ghost, int width) {
    return std::clamp<int64_t>(capacity - real_rows - ghost * width, 0, width);
}

}  // namespace rtp_llm
