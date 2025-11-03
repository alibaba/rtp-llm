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
                                 const ModelConfig&               model_config,
                                 const MMModelConfig&             mm_model_config,
                                 const ParallelismConfig&         parallelism_config,
                                 const RuntimeConfig&              runtime_config,
                                 const EPLBConfig&                eplb_config,
                                 const PDSepConfig&               pd_sep_config,
                                 const ConcurrencyConfig&         concurrency_config,
                                 const FMHAConfig&                 fmha_config,
                                 const KVCacheConfig&              kv_cache_config,
                                 const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                                 const HWKernelConfig&             hw_kernel_config,
                                 const DeviceResourceConfig&       device_resource_config,
                                 const MoeConfig&                  moe_config,
                                 const ModelSpecificConfig&        model_specific_config,
                                 const SpeculativeExecutionConfig& sp_config,
                                 const CacheStoreConfig&           cache_store_config,
                                 const MiscellaneousConfig&         misc_config,
                                 const ArpcConfig&                 arpc_config,
                                 const FfnDisAggregateConfig&      ffn_disaggregate_config,
                                 rtp_llm::Weights&&               gpt_weights,
                                 py::object                       py_model = py::none()):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(new EngineInitParams(model_id, model_config, mm_model_config, parallelism_config, runtime_config, eplb_config, pd_sep_config, concurrency_config, fmha_config, kv_cache_config, profiling_debug_logging_config, hw_kernel_config, device_resource_config, moe_config, model_specific_config, sp_config, cache_store_config, misc_config, arpc_config, ffn_disaggregate_config, std::move(gpt_weights), py_model)) {}

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

    const EngineInitParams& getEngineInitParams() {
        if (sp_type == "vanilla") {
            return *vanilla_model_params;
        } else if (sp_type == "mtp" || sp_type == "eagle3" || sp_type == "eagle") {
            RTP_LLM_CHECK(!mtp_model_params_->empty());
            RTP_LLM_CHECK(mtp_model_params_->at(0) != nullptr);
            return *mtp_model_params_->at(0);
        } else {
            RTP_LLM_FAIL("error sp type[%s] do not have EngineInitParams", sp_type.c_str());
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
