#pragma once

#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include <cstdint>

using namespace fastertransformer;

namespace rtp_llm {

class DeterministicExecutor: public ProposeExecutor {
public:
    explicit DeterministicExecutor(const EngineInitParams&                        score_model_engine_init_params,
                                   std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                                   ft::DeviceBase*                                device):
        ProposeExecutor(device), metrics_reporter_(propose_model_engine_init_params->metrics_reporter) {

        max_str_match_len_ =
            std::min(max_str_match_len_, (size_t)score_model_engine_init_params.gpt_init_parameter.max_seq_len_);
        min_str_match_len_ = autil::EnvUtil::getEnv("SP_MIN_STR_MATCH", 2);
        max_str_match_len_ = autil::EnvUtil::getEnv("SP_MAX_STR_MATCH", 1024);

        FT_LOG_INFO("DeterministicExecutor min str match size is %ld", min_str_match_len_);
        FT_LOG_INFO("DeterministicExecutor max str match size is %ld", max_str_match_len_);
    }

    ~DeterministicExecutor() {}

    absl::StatusOr<ProposeOutput> propose(const std::list<GenerateStreamPtr>& streams) override;

    absl::Status normalProcess(const std::list<GenerateStreamPtr>& streams) override {
        return absl::OkStatus();
    }

    void dynamicUpdateConfig(const ProposeDynamicConfig& config) override {
        return;
    }

    size_t reserveStep() const override {
        return max_str_match_len_;
    }

private:
    void ruleBasedTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

    void SpEditTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

    void PromptLookUpTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

    void postProcess(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

private:
    size_t                       min_str_match_len_ = 2;
    size_t                       max_str_match_len_ = 1024;
    kmonitor::MetricsReporterPtr metrics_reporter_  = nullptr;
};

}  // namespace rtp_llm