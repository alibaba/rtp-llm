#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <string>

#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

struct ProposeModelEngineInitParams {
    ProposeModelEngineInitParams() {};

    // Constructor for vanilla propose model
    // Only model_config differs between propose and score models, other configs are shared
    ProposeModelEngineInitParams(size_t                           model_id,
                                 SpeculativeType                  sp_type,
                                 size_t                           gen_num_per_circle,
                                 const ModelConfig&               model_config,
                                 const EngineInitParams&          base_params,
                                 rtp_llm::Weights&&               gpt_weights,
                                 py::object                       py_model = py::none(),
                                 py::object                       py_eplb = py::none()):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(new EngineInitParams(
            model_id,
            model_config,
            base_params.parallelism_config,
            base_params.runtime_config,
            base_params.pd_sep_config,
            base_params.concurrency_config,
            base_params.fmha_config,
            base_params.kv_cache_config,
            base_params.profiling_debug_logging_config,
            base_params.hw_kernel_config,
            base_params.device_resource_config,
            base_params.moe_config,
            base_params.model_specific_config,
            base_params.sp_config,
            base_params.cache_store_config,
            base_params.misc_config,
            base_params.arpc_config,
            base_params.grpc_config,
            base_params.ffn_disaggregate_config,
            base_params.vit_config,
            std::move(gpt_weights),
            py_model,
            py_eplb)) {}

    // Consturctor for deterministic propose model
    ProposeModelEngineInitParams(SpeculativeType sp_type, size_t gen_num_per_circle):
        sp_type(sp_type), gen_num_per_circle(gen_num_per_circle) {}

    // Consturctor for mtp propose model
    ProposeModelEngineInitParams(SpeculativeType                                                 sp_type,
                                 size_t                                                          gen_num_per_circle,
                                 std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(nullptr),
        mtp_model_params_(std::move(mtp_model_params)) {
        validateRecurrentDraftParams();
    };

    bool draftModel() {
        return sp_type == SP_TYPE_VANILLA || sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE;
    }

    const EngineInitParams& getEngineInitParams() {
        if (sp_type == SP_TYPE_VANILLA) {
            return *vanilla_model_params;
        } else if (sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE) {
            validateRecurrentDraftParams();
            return *mtp_model_params_->front();
        } else {
            RTP_LLM_FAIL("error sp type[%d] do not have EngineInitParams", static_cast<int>(sp_type));
        }
    }

    const int genNumPerCircle() {
        return gen_num_per_circle;
    }

private:
    void validateRecurrentDraftParams() const {
        RTP_LLM_CHECK_WITH_INFO(gen_num_per_circle > 0,
                                "sampled recurrent proposal steps must be positive, got %zu",
                                gen_num_per_circle);
        RTP_LLM_CHECK_WITH_INFO(mtp_model_params_ != nullptr && mtp_model_params_->size() == 1
                                    && mtp_model_params_->front() != nullptr,
                                "sampled recurrent decoding requires exactly one draft model");
        const auto& draft_params = *mtp_model_params_->front();
        RTP_LLM_CHECK_WITH_INFO(draft_params.model_config_.num_layers > 0,
                                "sampled recurrent draft model must contain at least one layer, got %ld",
                                draft_params.model_config_.num_layers);
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(draft_params.model_config_.num_layers) == draft_params.gpt_weights.layers.size(),
            "sampled recurrent draft layer mismatch: config=%ld, weights=%zu",
            draft_params.model_config_.num_layers,
            draft_params.gpt_weights.layers.size());
    }

public:

    SpeculativeType                  sp_type;
    size_t                           gen_num_per_circle   = 0;
    std::unique_ptr<EngineInitParams> vanilla_model_params = nullptr;

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params_;
    py::object                                                      eagle_model;
    kmonitor::MetricsReporterPtr                                    metrics_reporter = nullptr;
};

}  // namespace rtp_llm
