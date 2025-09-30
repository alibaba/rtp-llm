#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <string>

#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

struct ProposeModelEngineInitParams {
    ProposeModelEngineInitParams() {};

    // Constructor for vanilla propose model
    ProposeModelEngineInitParams(size_t                           model_id,
                                 std::string                      sp_type,
                                 size_t                           gen_num_per_circle,
                                 const rtp_llm::GptInitParameter& gpt_init_parameter,
                                 rtp_llm::Weights&&               gpt_weights):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(new EngineInitParams(model_id, gpt_init_parameter, std::move(gpt_weights))) {}

    // Consturctor for deterministic propose model
    ProposeModelEngineInitParams(std::string sp_type, size_t gen_num_per_circle):
        sp_type(sp_type), gen_num_per_circle(gen_num_per_circle) {}

    // Consturctor for mtp propose model
    ProposeModelEngineInitParams(std::string                                                     sp_type,
                                 size_t                                                          gen_num_per_circle,
                                 std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(nullptr),
        mtp_model_params_(std::move(mtp_model_params)) {};

    bool draftModel() {
        return sp_type == "vanilla" || sp_type == "mtp" || sp_type == "eagle3" || sp_type == "eagle";
    }

    const rtp_llm::GptInitParameter& getGptInitParameter() {
        if (sp_type == "vanilla") {
            return vanilla_model_params->gpt_init_parameter;
        } else if (sp_type == "mtp" || sp_type == "eagle3" || sp_type == "eagle") {
            RTP_LLM_CHECK(!mtp_model_params_->empty());
            RTP_LLM_CHECK(mtp_model_params_->at(0) != nullptr);
            return mtp_model_params_->at(0)->gpt_init_parameter;
        } else {
            RTP_LLM_FAIL("error sp type[%s] do not have GptInitParameter", sp_type.c_str());
        }
    }

    const int genNumPerCircle() {
        return gen_num_per_circle;
    }

    std::string                       sp_type;
    size_t                            gen_num_per_circle   = 0;
    std::unique_ptr<EngineInitParams> vanilla_model_params = nullptr;

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params_;
    py::object                                                      eagle_model;
    kmonitor::MetricsReporterPtr                                    metrics_reporter = nullptr;
};

}  // namespace rtp_llm
