#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initExecCtx(params);
}

EngineBase::~EngineBase() {}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initExecCtx(const EngineInitParams& params) {
    const auto rank =
        params.parallelism_config.dp_rank * params.parallelism_config.tp_size + params.parallelism_config.tp_rank;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    exec_init_params_ = rtp_llm::initExecCtx(params.parallelism_config,
                                             params.model_config_,
                                             params.eplb_config,
                                             params.fmha_config,
                                             params.device_resource_config,
                                             params.moe_config,
                                             params.sp_config,
                                             params.misc_config,
                                             params.profiling_debug_logging_config,
                                             params.hw_kernel_config,
                                             params.concurrency_config,
                                             params.ffn_disaggregate_config,
                                             params.runtime_config,
                                             params.model_specific_config,
                                             params.nccl_comm_config);
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
