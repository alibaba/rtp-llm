#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"

namespace py = pybind11;

namespace rtp_llm {

struct AccuracySeq {
    size_t prefix_len;
    size_t chunk_len;
    size_t decode_steps;
};

struct AccuracyScenario {
    std::vector<AccuracySeq> seqs;
};

class AccuracyChecker {
public:
    AccuracyChecker(const ModelConfig&       model_config,
                    const RuntimeConfig&     runtime_config,
                    const ParallelismConfig& parallelism_config);

    static const std::vector<AccuracyScenario>& defaultScenarios();

    // Entry point for accuracy checks
    absl::Status runAll(Executor*                            executor,
                        ResourceContext&                     resource_context,
                        const std::vector<AccuracyScenario>& scenarios = defaultScenarios());

    // Sizes the temporary KV pool for peak block usage across all scenarios, plus a safety margin
    // The scenarios must match those passed to runAll()
    static size_t computeTemporaryBlockNum(const ModelConfig&                   model_config,
                                           const KVCacheConfig&                 kv_cache_config,
                                           const std::vector<AccuracyScenario>& scenarios = defaultScenarios());

private:
    torch::Tensor                     makeCheckPrompt(size_t len);
    std::shared_ptr<GenerateInput>    wrapAccuracyInput(torch::Tensor input_ids, bool need_release);
    absl::StatusOr<GenerateStreamPtr> buildAccuracyStream(torch::Tensor prompt, bool need_release);
    absl::StatusOr<GenerateStreamPtr> makeForkStream(
        torch::Tensor init_tokens, size_t history_len, bool is_decode, int cur_token, const GenerateStreamPtr& golden);
    absl::Status runScenario(const py::object& recorder, const AccuracyScenario& scenario);

    bool         worldAll(bool local_flag);
    absl::Status abortIfFailed(bool step_flag, const std::string& where);
    bool         runCheck(const AccuracyScenario& scenario, const std::string& scenario_base_name);
    absl::StatusOr<std::vector<std::string>> listCandidates(const py::object& recorder, const std::string& phase);
    absl::Status                             forward(const py::object&                     recorder,
                                                     const std::string&                    scenario_base_name,
                                                     const std::string&                    impl_name,
                                                     const std::string&                    phase,
                                                     const std::vector<GenerateStreamPtr>& active_streams,
                                                     const std::vector<size_t>&            active_seq_ids,
                                                     bool                                  build_streams_flag);

    const ModelConfig&       model_config_;
    const RuntimeConfig&     runtime_config_;
    const ParallelismConfig& parallelism_config_;

    // Borrowed only during runAll()
    Executor*        executor_         = nullptr;
    ResourceContext* resource_context_ = nullptr;
};

}  // namespace rtp_llm
