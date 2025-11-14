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
    ProposeModelEngineInitParams(size_t                           model_id,
                                 SpeculativeType                  sp_type,
                                 size_t                           gen_num_per_circle,
                                 const ModelConfig&               model_config,
                                 const ParallelismConfig&         parallelism_config,
                                 const RuntimeConfig&              runtime_config,
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
                                 const VitConfig&                  vit_config,
                                 rtp_llm::Weights&&               gpt_weights,
                                 py::object                       py_model = py::none(),
                                 py::object                       py_eplb = py::none()):
        sp_type(sp_type),
        gen_num_per_circle(gen_num_per_circle),
        vanilla_model_params(new EngineInitParams(model_id, model_config, parallelism_config, runtime_config, pd_sep_config, concurrency_config, fmha_config, kv_cache_config, profiling_debug_logging_config, hw_kernel_config, device_resource_config, moe_config, model_specific_config, sp_config, cache_store_config, misc_config, arpc_config, ffn_disaggregate_config, vit_config, std::move(gpt_weights), py_model, py_eplb)) {}

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
        mtp_model_params_(std::move(mtp_model_params)) {};

    bool draftModel() {
        return sp_type == SP_TYPE_VANILLA || sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE;
    }

    const EngineInitParams& getEngineInitParams() {
        if (sp_type == SP_TYPE_VANILLA) {
            return *vanilla_model_params;
        } else if (sp_type == SP_TYPE_MTP || sp_type == SP_TYPE_EAGLE3 || sp_type == SP_TYPE_EAGLE) {
            RTP_LLM_CHECK(!mtp_model_params_->empty());
            RTP_LLM_CHECK(mtp_model_params_->at(0) != nullptr);
            return *mtp_model_params_->at(0);
        } else {
            RTP_LLM_FAIL("error sp type[%d] do not have EngineInitParams", static_cast<int>(sp_type));
        }
    }

    const int genNumPerCircle() {
        return gen_num_per_circle;
    }

    SpeculativeType                  sp_type;
    size_t                           gen_num_per_circle   = 0;
    std::unique_ptr<EngineInitParams> vanilla_model_params = nullptr;

    std::unique_ptr<std::vector<std::unique_ptr<EngineInitParams>>> mtp_model_params_;
    py::object                                                      eagle_model;
    kmonitor::MetricsReporterPtr                                    metrics_reporter = nullptr;
};

}  // namespace rtp_llm
