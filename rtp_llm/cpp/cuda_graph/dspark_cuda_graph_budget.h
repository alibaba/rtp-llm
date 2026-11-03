#pragma once

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

}  // namespace rtp_llm
