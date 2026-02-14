#pragma once
#include <cstddef>
#include <tuple>

#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "kmonitor/client/MetricsReporter.h"

namespace th = torch;

namespace rtp_llm {

using TensorMap          = std::unordered_map<std::string, th::Tensor>;
using TensorMaps         = std::vector<TensorMap>;
using ConstBufferPtrMap  = std::unordered_map<std::string, rtp_llm::ConstBufferPtr>;
using ConstBufferPtrMaps = std::vector<ConstBufferPtrMap>;

struct EngineInitParams {
    EngineInitParams() {};
    // This class is the only one that holds gpt_weights object globally.
    EngineInitParams(size_t                             model_id,
                     const ModelConfig&                 model_config,
                     const ParallelismConfig&           parallelism_config,
                     const RuntimeConfig&               runtime_config,
                     const PDSepConfig&                 pd_sep_config,
                     const ConcurrencyConfig&           concurrency_config,
                     const FMHAConfig&                  fmha_config,
                     const KVCacheConfig&               kv_cache_config,
                     const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                     const HWKernelConfig&              hw_kernel_config,
                     const DeviceResourceConfig&        device_resource_config,
                     const MoeConfig&                   moe_config,
                     const ModelSpecificConfig&         model_specific_config,
                     const SpeculativeExecutionConfig&  sp_config,
                     const CacheStoreConfig&            cache_store_config,
                     const MiscellaneousConfig&         misc_config,
                     const ArpcConfig&                  arpc_config,
                     const GrpcConfig&                  grpc_config,
                     const FfnDisAggregateConfig&       ffn_disaggregate_config,
                     const VitConfig&                   vit_config,
                     rtp_llm::Weights&&                 gpt_weights,
                     py::object                         py_model       = py::none(),
                     py::object                         weight_manager = py::none(),
                     py::object                         py_eplb        = py::none(),
                     py::object                         py_sp_model    = py::none()):
        model_id(model_id),
        model_config_(model_config),
        parallelism_config(parallelism_config),
        runtime_config(runtime_config),
        eplb_config(model_config.eplb_config),
        pd_sep_config(pd_sep_config),
        concurrency_config(concurrency_config),
        fmha_config(fmha_config),
        kv_cache_config(kv_cache_config),
        profiling_debug_logging_config(profiling_debug_logging_config),
        hw_kernel_config(hw_kernel_config),
        device_resource_config(device_resource_config),
        moe_config(moe_config),
        model_specific_config(model_specific_config),
        sp_config(sp_config),
        cache_store_config(cache_store_config),
        misc_config(misc_config),
        arpc_config(arpc_config),
        grpc_config(grpc_config),
        ffn_disaggregate_config(ffn_disaggregate_config),
        vit_config(vit_config),
        gpt_weights(std::move(gpt_weights)),
        py_model(py_model),
        py_eplb(py_eplb),
        py_sp_model(py_sp_model),
        weight_manager(weight_manager) {
        StaticConfig::user_ft_core_dump_on_exception = profiling_debug_logging_config.ft_core_dump_on_exception;
        StaticConfig::user_disable_pdl               = misc_config.disable_pdl;
        // default 1 minute and 1000
        ParallelInfo& global_parallel_info = ParallelInfo::globalParallelInfo();
        global_parallel_info.setTpSize(parallelism_config.tp_size);
        global_parallel_info.setPpSize(parallelism_config.pp_size);
        global_parallel_info.setEpSize(parallelism_config.ep_size);
        global_parallel_info.setDpSize(parallelism_config.dp_size);
        global_parallel_info.setWorldSize(parallelism_config.world_size);
        global_parallel_info.setWorldRank(parallelism_config.world_rank);
        global_parallel_info.setLocalWorldSize(parallelism_config.local_world_size);
        showDebugInfo();
    }

    size_t                       model_id;
    ModelConfig                  model_config_;
    ParallelismConfig            parallelism_config;
    NcclCommConfig               nccl_comm_config;  // initDevices uses this for NCCL ip/ports
    py::object                   server_config;     // Python ServerConfig; RPC/HTTP ports read from it
    RuntimeConfig                runtime_config;
    EPLBConfig                   eplb_config;
    PDSepConfig                  pd_sep_config;
    ConcurrencyConfig            concurrency_config;
    FMHAConfig                   fmha_config;
    KVCacheConfig                kv_cache_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    HWKernelConfig               hw_kernel_config;
    DeviceResourceConfig         device_resource_config;
    MoeConfig                    moe_config;
    ModelSpecificConfig          model_specific_config;
    SpeculativeExecutionConfig   sp_config;
    CacheStoreConfig             cache_store_config;
    MiscellaneousConfig          misc_config;
    ArpcConfig                   arpc_config;
    GrpcConfig                   grpc_config;
    FfnDisAggregateConfig        ffn_disaggregate_config;
    VitConfig                    vit_config;
    rtp_llm::Weights             gpt_weights;
    py::object                   py_model;
    py::object                   py_eplb;
    py::object                   py_sp_model;
    py::object                   weight_manager;
    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;

public:
    void showDebugInfo() const {
        // Show debug info for all configs
        RTP_LLM_LOG_INFO(
            "ModelConfig: max_seq_len=%ld, vocab_size=%ld", model_config_.max_seq_len, model_config_.vocab_size);
        RTP_LLM_LOG_INFO("ParallelismConfig: tp_size=%ld, ep_size=%ld, dp_size=%ld",
                         parallelism_config.tp_size,
                         parallelism_config.ep_size,
                         parallelism_config.dp_size);
        RTP_LLM_LOG_INFO("RuntimeConfig: %s", runtime_config.to_string().c_str());
    }
};

}  // namespace rtp_llm
