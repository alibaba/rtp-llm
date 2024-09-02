#pragma once

#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"

namespace rtp_llm {

class PromptLookupExecutor: public ProposeExecutor {
public:
    explicit PromptLookupExecutor(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params,
                                  ft::DeviceBase*                                device):
        ProposeExecutor(device), metrics_reporter_(propose_model_engine_init_params->metrics_reporter) {}

    ~PromptLookupExecutor() {}

    absl::StatusOr<ProposeOutput> propose(const std::list<GenerateStreamPtr>& streams) override {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        // TODO(xyz): implement it
        return absl::OkStatus();
    }

    absl::Status process(const std::list<GenerateStreamPtr>& streams) override {
        return absl::OkStatus();
    }

    void dynamicUpdateConfig(const ProposeDynamicConfig& config) override {
        return;
    }

    size_t reserveStep() const override {
        return 0;
    }

private:
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm