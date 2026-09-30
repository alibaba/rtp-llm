#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/CpuQuiesceCoordinator.h"
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

void EngineBase::setQuiesceCoordinator(std::shared_ptr<CpuQuiesceCoordinator> coordinator) {
    quiesce_coordinator_ = std::move(coordinator);  // startup only, before RPC publication
}

absl::Status EngineBase::coordinatedQuiesce(const std::string& token, int64_t timeout_ms) {
    if (timeout_ms < 0) {
        return absl::InvalidArgumentError("negative execution quiesce timeout");
    }
    if (!requiresCoordinatedSleepQuiesce()) {
        return quiesce(timeout_ms);
    }
    if (!quiesce_coordinator_) {
        return absl::FailedPreconditionError("CPU execution coordinator is not initialized");
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 60000);
    // freeze() is idempotent; the earlier freeze phase has already sealed all
    // peers. No GIL/CUDA work is allowed on this control path.
    auto target = quiesce_coordinator_->targetRound(token, freezeSleepRounds(), deadline);
    if (!target.ok()) {
        return target.status();
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
        return absl::DeadlineExceededError("CPU coordination exhausted execution quiesce deadline");
    }
    return quiesce(remaining.count(), *target);
}

void EngineBase::requestTermination() {
    termination_requested_.store(true, std::memory_order_release);
    if (scheduler_) {
        scheduler_->admission()->beginTermination();
    }
}

std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
EngineBase::enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
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
