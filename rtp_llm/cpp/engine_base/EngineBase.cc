#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initRuntime(params);
}

EngineBase::~EngineBase() {}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initRuntime(const EngineInitParams& params) {
    sleep_controller_.setEnabled(params.runtime_config.enable_sleep_mode);
    // The controller owns the level->discard-weights mapping; just hand it the
    // startup sleep_mode_level (2 opened the weights VMM region without host
    // cpu_backup at load time).
    sleep_controller_.setConfiguredLevel(params.runtime_config.sleep_mode_level);
    const auto rank =
        params.parallelism_config.dp_rank * params.parallelism_config.tp_size + params.parallelism_config.tp_rank;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
    mla_ops_type_    = rtp_llm::initRuntime(device_id,
                                         params.profiling_debug_logging_config.trace_memory,
                                         params.device_resource_config.enable_comm_overlap,
                                         params.model_config_.mla_ops_type);
    warmupNoBlockCopy();
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
