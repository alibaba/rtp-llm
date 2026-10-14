#include "rtp_llm/cpp/models/eplb/stats/ExpertStatsReportWindow.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace rtp_llm;

namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename F>
void rejects(F function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "Invalid input was not rejected");
}
}  // namespace

// CPU-only: test real ATen tensor lifetime without launching kernels or importing the engine.
int main() {
    try {
        at::set_num_threads(1);
        ExpertStatsReportWindow window;
        auto step = at::zeros({3, 4}, at::TensorOptions().dtype(at::kInt));
        auto* p = step.data_ptr<int32_t>();
        for (int i = 0; i < 4; ++i) {
            p[i] = 10;
        }
        p[4] = 20;
        p[5] = p[6] = 10;
        for (int i = 1; i <= 10; ++i) {
            check(window.add(step) == (i == 10), "Must report exactly on step ten");
        }
        auto snapshot = window.snapshot();
        check(snapshot.scalar_type() == at::kLong, "Window accumulation must be INT64");
        check(step[0][0].item<int>() == 10, "Report window mutated the producer tensor");
        RtpLLmEplbMetricsCollector collector;
        ExpertStatsReportWindow::fillCollector(snapshot, window.steps(), 1, collector);
        check(collector.gpu_loads[0] == std::vector<int64_t>({100, 100, 100, 100}), "Incorrect window totals");
        check(collector.gpu_loads[1] == std::vector<int64_t>({200, 100, 100, 0}), "Lost destination columns");
        check(collector.window_steps == 10 && collector.active_layer_count == 2, "Incorrect window metadata");
        check(collector.ib_layers == std::vector<double>({1, 2, 0}), "Incorrect layer imbalance");
        check(collector.ib_layer_mean == 1.5, "Empty layers must not dilute layer-mean imbalance");
        check(!collector.update_weights_qps, "Weight-update flag must default to false");
        rejects([&] { window.add(step); });
        window.reset();
        check(window.steps() == 0, "Reset did not clear the window counter");
        check(snapshot[0][0].item<int64_t>() == 100, "Reset mutated an owned snapshot");
        for (int i = 0; i < 10; ++i) {
            window.add(step);
        }
        check(window.snapshot().equal(snapshot), "Second window double-counted the first window");
        // A simulated second independent DP/CP source is added, not averaged.
        auto aggregated = window.snapshot();
        aggregated.add_(snapshot);
        ExpertStatsReportWindow::fillCollector(aggregated, 10, 1, collector);
        check(collector.gpu_loads[0][0] == 200, "Independent source counts must be summed");
        ExpertStatsReportWindow::fillCollector(aggregated, 10, 2, collector);
        check(collector.gpu_loads[0][0] == 100, "Explicit verified TP replication was not removed");
        check(collector.ib_layer_mean == 1.5, "TP normalization changed imbalance");
        aggregated[0][0].add_(1);
        rejects([&] { ExpertStatsReportWindow::fillCollector(aggregated, 10, 2, collector); });
        window.reset();
        step.zero_();
        for (int i = 0; i < 10; ++i) {
            window.add(step);
        }
        ExpertStatsReportWindow::fillCollector(window.snapshot(), 10, 1, collector);
        check(!collector.has_valid_load && collector.active_layer_count == 0 && collector.ib_layer_mean == 0,
              "Empty window retained old active-layer state");
        window.reset();
        window.add(step);
        window.reset();  // NONE / mode transition discards a partial window.
        check(window.steps() == 0, "Partial-window reset failed");
        rejects([&] { window.snapshot(); });
        rejects([&] { window.add(at::zeros({2, 4}, at::kInt)); });
        rejects([&] { window.add(at::zeros({3, 4}, at::kFloat)); });
        rejects([] { ExpertStatsReportWindow invalid(0); });
        rejects([&] { collector.setGpuLoads({{1, 2}, {3}}, 10); });
        rejects([&] { collector.setGpuLoads({{-1, 0}}, 10); });
        rejects([&] { collector.setGpuLoads({{1, 1}}, 0); });
        rejects([&] { collector.setGpuLoads({{1, 1}}, 10, 0); });
        collector.setGpuLoads({{400, 0, 0, 0}}, 10);
        check(collector.ib_layer_mean == 4, "Maximum imbalance should be EP size");
        const int64_t large = std::numeric_limits<int32_t>::max();
        ExpertStatsReportWindow large_window;
        auto large_step = at::full({1, 2}, large, at::kInt);
        for (int i = 0; i < 10; ++i) {
            large_window.add(large_step);
        }
        ExpertStatsReportWindow::fillCollector(large_window.snapshot(), 10, 1, collector);
        check(collector.gpu_loads[0][0] == large * 10, "INT32 input overflowed window accumulation");
        check(std::isfinite(collector.ib_layer_mean), "IB must remain finite");
        std::cout << "PASS: window cadence/reset, tensor isolation, 2D counts, IB, TP normalization, "
                     "DP/CP sum, empty windows, invalid inputs and INT64 accumulation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
