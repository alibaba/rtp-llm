#pragma once
#include <cstddef>
#include <tuple>

#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
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
    EngineInitParams(size_t                           model_id,
                     const rtp_llm::GptInitParameter& gpt_init_parameter,
                     rtp_llm::Weights&&               gpt_weights,
                     py::object                       py_model       = py::none(),
                     py::object                       weight_manager = py::none()):
        model_id(model_id),
        gpt_init_parameter(gpt_init_parameter),
        gpt_weights(std::move(gpt_weights)),
        py_model(py_model),
        weight_manager(weight_manager) {
        StaticConfig::user_ft_core_dump_on_exception =
            gpt_init_parameter.profiling_debug_logging_config.ft_core_dump_on_exception;
        StaticConfig::user_disable_pdl = gpt_init_parameter.misc_config.disable_pdl;
        // default 1 minute and 1000
        ParallelInfo& global_parallel_info = ParallelInfo::globalParallelInfo();
        global_parallel_info.setTpSize(gpt_init_parameter.parallelism_distributed_config.tp_size);
        global_parallel_info.setPpSize(gpt_init_parameter.parallelism_distributed_config.pp_size);
        global_parallel_info.setEpSize(gpt_init_parameter.parallelism_distributed_config.ep_size);
        global_parallel_info.setDpSize(gpt_init_parameter.parallelism_distributed_config.dp_size);
        global_parallel_info.setWorldSize(gpt_init_parameter.parallelism_distributed_config.world_size);
        global_parallel_info.setWorldRank(gpt_init_parameter.parallelism_distributed_config.world_rank);
        global_parallel_info.setLocalWorldSize(gpt_init_parameter.parallelism_distributed_config.local_world_size);
        Logger::log_level_ = gpt_init_parameter.profiling_debug_logging_config.log_level;
        gpt_init_parameter.showDebugInfo();
    }

    size_t                       model_id;
    rtp_llm::GptInitParameter    gpt_init_parameter;
    rtp_llm::Weights             gpt_weights;
    py::object                   py_model;
    py::object                   weight_manager;
    kmonitor::MetricsReporterPtr metrics_reporter = nullptr;

public:
    void showGptInitParameter() {
        gpt_init_parameter.showDebugInfo();
    }
};

}  // namespace rtp_llm
