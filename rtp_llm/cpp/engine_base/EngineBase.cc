#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initRuntime(params);
}

EngineBase::~EngineBase() {}

std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
EngineBase::enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initRuntime(const EngineInitParams& params) {
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
    // Temporary cudacore diagnostics: register the worker identity and audit the
    // coredump configuration the driver actually adopted. Read-only, never blocks
    // startup, never throws.
    setCudacoreProcessIdentity(rank, static_cast<int>(device_id));
    auditCudacoreAttributesAtStartup();
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
