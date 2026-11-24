#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rtp_llm {
namespace speculative {

// Host-only policy. The caller freezes choose() before the current proposal,
// then publishes completed confidence and CUDA-event costs for future rounds.
// Costs are complete rounds, indexed by the actual (batch, average budget).
// In particular, a graph measurement must not be reused for an eager candidate.
class DSparkAdaptiveBudget {
public:
    struct Decision {
        int     budget_per_request = 0;
        int64_t total_draft_budget = 0;
        double  expected_tokens    = 0.0;
        double  estimated_ms       = 0.0;  // Zero until this candidate has five samples.
        bool    exploring          = false;
    };

    DSparkAdaptiveBudget(int gamma, int fallback_budget): gamma_(gamma), fallback_budget_(fallback_budget) {
        if (gamma <= 0 || fallback_budget < 0 || fallback_budget > gamma) {
            throw std::invalid_argument("adaptive DSpARK requires gamma > 0 and fallback budget in [0, gamma]");
        }
    }

    Decision choose(const std::vector<int64_t>& active_ids) {
        validateIds(active_ids);
        pruneHistory(active_ids);
        const size_t batch_size = active_ids.size();
        if (batch_size == 0) {
            return {};
        }
        if (batch_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / gamma_) {
            throw std::overflow_error("adaptive DSpARK draft budget overflow");
        }

        // Historical conditional rates become prefix acceptance probabilities.
        // Unknown requests have drafts but no history, so use DeepInfer's 1.0
        // prior. No current proposal tensor can enter this API.
        std::vector<double> scores;
        scores.reserve(batch_size * gamma_);
        for (const auto id : active_ids) {
            const auto found  = history_.find(id);
            double     prefix = 1.0;
            for (int j = 0; j < gamma_; ++j) {
                const double rate = found == history_.end() ? 1.0 : found->second[j];
                prefix *= rate;
                scores.push_back(prefix);
            }
        }
        std::sort(scores.begin(), scores.end(), std::greater<double>());
        std::vector<double> expected(gamma_ + 1, static_cast<double>(batch_size));
        double              running = static_cast<double>(batch_size);
        for (size_t i = 0; i < scores.size(); ++i) {
            running += scores[i];
            if ((i + 1) % batch_size == 0) {
                expected[(i + 1) / batch_size] = running;
            }
        }

        auto& state = costState(batch_size);
        // Collect every candidate rather than extrapolating an unseen cost.
        // A rejected/unfinished observation leaves its candidate in warmup.
        for (int offset = 0; offset <= gamma_; ++offset) {
            const int budget = (state.cold_cursor + offset) % (gamma_ + 1);
            if (!state.costs[budget].median()) {
                state.cold_cursor = (budget + 1) % (gamma_ + 1);
                return decision(batch_size, budget, expected[budget], state, true);
            }
        }

        // Refresh unselected candidates as context length and load change.
        // This deterministic exploration also happens before proposal sampling.
        ++state.learned_rounds;
        if (state.learned_rounds % kProbeInterval == 0) {
            const int budget   = state.probe_cursor;
            state.probe_cursor = (budget + 1) % (gamma_ + 1);
            return decision(batch_size, budget, expected[budget], state, true);
        }

        int    best_budget     = 0;
        double best_throughput = -1.0;
        for (int budget = 0; budget <= gamma_; ++budget) {
            const double throughput = expected[budget] / *state.costs[budget].median();
            if (throughput > best_throughput) {
                best_throughput = throughput;
                best_budget     = budget;
            }
        }
        return decision(batch_size, best_budget, expected[best_budget], state, false);
    }

    // Ignore invalid timings rather than poisoning the median or admitting a
    // zero-cost candidate. Fake/warmup/seed-fallback rounds are filtered by caller.
    void observeCost(size_t batch_size, int budget_per_request, double elapsed_ms) {
        if (batch_size == 0 || budget_per_request < 0 || budget_per_request > gamma_ || !std::isfinite(elapsed_ms)
            || elapsed_ms <= 0.0) {
            return;
        }
        costState(batch_size).costs[budget_per_request].update(elapsed_ms);
    }

    void observeConfidence(const std::vector<int64_t>& ids, const std::vector<std::vector<float>>& conditional_rates) {
        validateIds(ids);
        if (ids.size() != conditional_rates.size()) {
            throw std::invalid_argument("adaptive DSpARK confidence must have one row per request");
        }
        for (const auto& rates : conditional_rates) {
            if (rates.size() != static_cast<size_t>(gamma_)) {
                throw std::invalid_argument("adaptive DSpARK confidence row must have gamma positions");
            }
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            auto& row = history_[ids[i]];
            row.resize(gamma_);
            for (int j = 0; j < gamma_; ++j) {
                const double rate = conditional_rates[i][j];
                row[j]            = std::isfinite(rate) && rate >= 0.0 && rate <= 1.0 ? rate : 0.0;
            }
        }
    }

    void pruneHistory(const std::vector<int64_t>& active_ids) {
        const std::unordered_set<int64_t> active(active_ids.begin(), active_ids.end());
        for (auto it = history_.begin(); it != history_.end();) {
            it = active.count(it->first) ? std::next(it) : history_.erase(it);
        }
    }

private:
    static constexpr size_t kWindowSize    = 9;
    static constexpr size_t kMinSamples    = 5;
    static constexpr size_t kMaxBatchSizes = 64;
    static constexpr size_t kProbeInterval = 32;

    struct CostWindow {
        std::array<double, kWindowSize> samples{};
        size_t                          size = 0;
        size_t                          next = 0;

        void update(double elapsed_ms) {
            samples[next] = elapsed_ms;
            next          = (next + 1) % kWindowSize;
            size          = std::min(size + 1, kWindowSize);
        }

        std::optional<double> median() const {
            if (size < kMinSamples) {
                return std::nullopt;
            }
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.begin() + size);
            const size_t middle = size / 2;
            return size % 2 == 0 ? sorted[middle - 1] * 0.5 + sorted[middle] * 0.5 : sorted[middle];
        }
    };

    struct BatchState {
        std::vector<CostWindow> costs;
        int                     cold_cursor;
        int                     probe_cursor;
        size_t                  learned_rounds = 0;
        uint64_t                last_use       = 0;

        BatchState(int gamma, int fallback): costs(gamma + 1), cold_cursor(fallback), probe_cursor(fallback) {}
    };

    static void validateIds(const std::vector<int64_t>& ids) {
        std::unordered_set<int64_t> unique;
        for (const auto id : ids) {
            if (!unique.insert(id).second) {
                throw std::invalid_argument("adaptive DSpARK requires unique active request identities");
            }
        }
    }

    BatchState& costState(size_t batch_size) {
        auto found = costs_.find(batch_size);
        if (found == costs_.end()) {
            if (costs_.size() == kMaxBatchSizes) {
                const auto oldest = std::min_element(costs_.begin(), costs_.end(), [](const auto& a, const auto& b) {
                    return a.second.last_use < b.second.last_use;
                });
                costs_.erase(oldest);
            }
            found = costs_.try_emplace(batch_size, gamma_, fallback_budget_).first;
        }
        found->second.last_use = ++last_use_;
        return found->second;
    }

    static Decision decision(size_t batch_size, int budget, double expected, const BatchState& state, bool exploring) {
        return {budget,
                static_cast<int64_t>(batch_size) * budget,
                expected,
                state.costs[budget].median().value_or(0.0),
                exploring};
    }

    int                                              gamma_;
    int                                              fallback_budget_;
    uint64_t                                         last_use_ = 0;
    std::unordered_map<size_t, BatchState>           costs_;
    std::unordered_map<int64_t, std::vector<double>> history_;
};

}  // namespace speculative
}  // namespace rtp_llm
