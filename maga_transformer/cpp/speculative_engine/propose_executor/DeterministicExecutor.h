#pragma once

#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include <cstddef>
#include <cstdint>

using namespace fastertransformer;

namespace rtp_llm {

class DeterministicExecutor: public ProposeExecutor {
public:
    explicit DeterministicExecutor(const EngineInitParams&                        score_model_engine_init_params,
                                   std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                                   ft::DeviceBase*                                device):
        ProposeExecutor(device) {

        propose_step_ = std::min(propose_model_engine_init_params->gen_num_per_circle,
                                 (size_t)score_model_engine_init_params.gpt_init_parameter.max_seq_len_);

        min_token_match_len_ = autil::EnvUtil::getEnv("SP_MIN_TOKEN_MATCH", 2);
        max_token_match_len_ = autil::EnvUtil::getEnv("SP_MAX_TOKEN_MATCH", 2);
        
        FT_LOG_INFO("DeterministicExecutor propose step is %ld", propose_step_);
        FT_LOG_INFO("DeterministicExecutor min token match size is %ld", min_token_match_len_);
        FT_LOG_INFO("DeterministicExecutor max token match size is %ld", max_token_match_len_);
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
        return propose_step_;
    }

private:
    void ruleBasedTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

    void SpEditTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output, bool use_sp_advice_prompt);

    void PromptLookUpTokenSelector(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output, bool use_sp_advice_prompt);

    void postProcess(const GenerateStreamPtr& stream, SpeculativeExecutorStreamOutputPtr& stream_output);

private:
    size_t min_token_match_len_ = 1;
    size_t max_token_match_len_ = 3;
    size_t propose_step_      = 5;
};

}  // namespace rtp_llm