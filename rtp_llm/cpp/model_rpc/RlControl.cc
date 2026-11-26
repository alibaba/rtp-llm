#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/utils/DevicePin.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

grpc::Status LocalRpcServer::RlControl(grpc::ServerContext*      context,
                                       const RlControlRequestPB* request,
                                       RlControlResponsePB*      response) {
    if (!engine_ || !engine_->rlEnabled()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "RTP_LLM_RL_MODE is not enabled"};
    }
    std::lock_guard<std::mutex> lock(rl_control_mutex_);
    const auto&                 parallel = maga_init_params_.parallelism_config;

    const auto op = request->operation();
    // Explicitly unsupported contract stubs are side-effect free and never fail the replica.
    const bool sleep_control = op == RlControlRequestPB::SLEEP || op == RlControlRequestPB::WAKE_UP;
    if ((sleep_control && !maga_init_params_.kv_cache_config.enable_rl_vmm_sleep)
        || op == RlControlRequestPB::WRITE_BUCKET || op == RlControlRequestPB::RELEASE_KV_CACHE
        || op == RlControlRequestPB::RESTORE_KV_CACHE) {
        return {grpc::StatusCode::UNIMPLEMENTED,
                "RL operation is reserved in the protocol contract but not yet implemented on this build"};
    }

    const bool protocol3 = request->protocol_version() == 3;
    const bool multi_dp  = parallel.dp_size > 1 || parallel.ep_size > parallel.tp_size;
    if (multi_dp && !protocol3) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "multi-DP/cross-DP EP RL requires protocol 3"};
    }
    if (protocol3) {
        if (parallel.pp_size != 1 || parallel.world_size != parallel.tp_size * parallel.dp_size
            || parallel.world_rank < 0 || parallel.world_rank >= parallel.world_size
            || request->expected_world_rank() != parallel.world_rank
            || request->expected_world_size() != parallel.world_size
            || request->expected_tp_size() != parallel.tp_size || request->expected_dp_size() != parallel.dp_size
            || request->expected_ep_size() != parallel.ep_size) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "RL protocol-3 world rank or topology mismatch"};
        }
    } else if ((request->protocol_version() != 0 && request->protocol_version() != 2) || parallel.pp_size != 1
               || parallel.dp_size != 1 || parallel.world_size != parallel.tp_size) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "single-replica RL requires protocol 2"};
    }

    const auto version = engine_->rlModelVersion();
    if (op != RlControlRequestPB::STATUS && rl_phase_ == "failed") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "RL replica failed; restart every replica rank"};
    }
    if (op != RlControlRequestPB::STATUS && request->expected_version() != version) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "stale or missing expected model version"};
    }
    if ((op == RlControlRequestPB::WRITE_WEIGHT || op == RlControlRequestPB::COMMIT_UPDATE)
        && (rl_phase_ != "updating" || request->update_id() != rl_update_id_)) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "no matching active weight transaction"};
    }
    if ((op == RlControlRequestPB::BEGIN_UPDATE || op == RlControlRequestPB::RESUME) && rl_phase_ != "paused") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "operation requires a paused replica"};
    }
    if (op == RlControlRequestPB::PAUSE && rl_phase_ != "running" && rl_phase_ != "paused") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "cannot pause an active weight transaction"};
    }

    bool wake_weights = false;
    bool wake_kv = false;
    if (sleep_control && rl_phase_ != "running" && rl_phase_ != "paused") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "cannot change memory residency during a refit"};
    }
    if (op == RlControlRequestPB::SLEEP && (request->sleep_level() != 2 || request->memory_tags_size() != 0)) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "sleep requires level 2 and no memory tags"};
    }
    if (op == RlControlRequestPB::WAKE_UP) {
        if (rl_phase_ != "paused" || request->memory_tags_size() == 0) {
            return {grpc::StatusCode::FAILED_PRECONDITION, "wake requires a paused replica and at least one tag"};
        }
        for (const auto& tag : request->memory_tags()) {
            if (tag == "weights" && !wake_weights) wake_weights = true;
            else if (tag == "kv_cache" && !wake_kv) wake_kv = true;
            else return {grpc::StatusCode::INVALID_ARGUMENT, "unknown or repeated wake tag: " + tag};
        }
    }
    if ((op == RlControlRequestPB::BEGIN_UPDATE || op == RlControlRequestPB::WRITE_WEIGHT
         || op == RlControlRequestPB::COMMIT_UPDATE) && !engine_->rlWeightsMapped()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "weights must be mapped before refit"};
    }
    if (op == RlControlRequestPB::RESUME && (!engine_->rlWeightsReady() || !engine_->rlKVCacheReady())) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "resume requires refitted weights and mapped KV"};
    }
    uint64_t released_bytes = 0;
    int64_t timeout_ms = request->timeout_ms();
    if (op != RlControlRequestPB::STATUS) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   context->deadline() - std::chrono::system_clock::now())
                                   .count();
        timeout_ms = std::min(timeout_ms, static_cast<int64_t>(remaining));
        if (timeout_ms <= 0 || timeout_ms > 600000) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "control timeout must be in (0, 600000] ms"};
        }
    }
    if (op == RlControlRequestPB::BEGIN_UPDATE
        && (request->max_buffer_bytes() == 0 || request->global_step() < rl_global_step_)) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "invalid update buffer limit or global step"};
    }

    auto require_ok = [](const absl::Status& status) {
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
    };

    try {
        setCurrentThreadDevice(parallel.local_rank);
        switch (op) {
            case RlControlRequestPB::STATUS:
                break;
            case RlControlRequestPB::PAUSE:
                require_ok(engine_->rlPause(timeout_ms));
                rl_phase_ = "paused";
                break;
            case RlControlRequestPB::RESUME:
                require_ok(engine_->rlResume(timeout_ms));
                rl_phase_ = "running";
                break;
            case RlControlRequestPB::SLEEP: {
                if (rl_phase_ == "running") {
                    require_ok(engine_->rlPause(timeout_ms));
                    rl_phase_ = "paused";
                }
                require_ok(engine_->rlClearCache());
#if USING_CUDA
                RTP_LLM_CHECK(cudaDeviceSynchronize() == cudaSuccess);
#endif
                released_bytes = engine_->rlSetKVMapping(false);
                if (engine_->rlWeightsMapped()) {
                    py::gil_scoped_acquire gil;
                    released_bytes += weight_manager_.attr("rl_unmap_weights")().cast<uint64_t>();
                    engine_->rlSetWeightsMapped(false);
                }
                break;
            }
            case RlControlRequestPB::WAKE_UP: {
                if (wake_weights && !engine_->rlWeightsMapped()) {
                    py::gil_scoped_acquire gil;
                    weight_manager_.attr("rl_remap_weights")();
                    engine_->rlSetWeightsMapped(true);
                }
                if (wake_kv) engine_->rlSetKVMapping(true);
                break;
            }
            case RlControlRequestPB::BEGIN_UPDATE: {
                if (!engine_->rlPaused()) {
                    throw std::runtime_error("native engine has not acknowledged quiescence");
                }
                try {
                    py::gil_scoped_acquire gil;
                    if (request->expected_source_names_size() == 0) {
                        weight_manager_.attr("rl_begin")(request->max_buffer_bytes());
                    } else {
                        py::list expected_sources;
                        for (const auto& name : request->expected_source_names()) {
                            expected_sources.append(py::str(name));
                        }
                        weight_manager_.attr("rl_begin")(request->max_buffer_bytes(), expected_sources);
                    }
                } catch (const py::error_already_set& py_err) {
                    throw std::runtime_error(py_err.what());
                }
                rl_update_id_ = version.substr(0, version.rfind(':') + 1) + std::to_string(rl_version_sequence_ + 1);
                rl_pending_global_step_ = request->global_step();
                rl_received_weights_    = 0;
                rl_applied_weights_     = 0;
                rl_phase_               = "updating";
                break;
            }
            case RlControlRequestPB::WRITE_WEIGHT: {
                try {
                    py::gil_scoped_acquire gil;
                    weight_manager_.attr("rl_push")(request->weight_name(),
                                                    py::bytes(request->cuda_ipc_metadata()),
                                                    static_cast<double>(timeout_ms) / 1000.0);
                } catch (const py::error_already_set& py_err) {
                    throw std::runtime_error(py_err.what());
                }
                ++rl_received_weights_;
                break;
            }
            case RlControlRequestPB::COMMIT_UPDATE: {
                try {
                    py::gil_scoped_acquire gil;
                    auto                   counts =
                        weight_manager_.attr("rl_finish")(static_cast<double>(timeout_ms) / 1000.0).cast<py::dict>();
                    rl_applied_weights_ = counts["internal_tensors"].cast<uint64_t>();
                    if (counts["hf_sources"].cast<uint64_t>() != rl_received_weights_) {
                        throw std::runtime_error("HF source ACK count mismatch");
                    }
                } catch (const py::error_already_set& py_err) {
                    throw std::runtime_error(py_err.what());
                }
                // Invalidate all logical prefix cache entries before publishing the new policy version.
                require_ok(engine_->rlClearCache());
                require_ok(engine_->rlSetModelVersion(rl_update_id_, rl_pending_global_step_));
                ++rl_version_sequence_;
                rl_global_step_ = rl_pending_global_step_;
                rl_update_id_.clear();
                rl_phase_ = "paused";
                break;
            }
            default:
                return {grpc::StatusCode::INVALID_ARGUMENT, "unknown RL control operation"};
        }
    } catch (const std::exception& error) {
        // In-place updates cannot roll back safely; keep admission permanently closed.
        rl_phase_ = "failed";
        RTP_LLM_LOG_ERROR("RL control failed: %s", error.what());
        return {grpc::StatusCode::INTERNAL,
                std::string("RL control failed: ") + std::string(error.what()).substr(0, 2048)};
    }

    response->set_protocol_version(protocol3 ? 3 : 2);
    response->set_phase(rl_phase_);
    response->set_model_version(engine_->rlModelVersion());
    response->set_update_id(rl_update_id_);
    response->set_global_step(rl_global_step_);
    response->set_tp_rank(parallel.tp_rank);
    response->set_tp_size(parallel.tp_size);
    response->set_world_rank(parallel.world_rank);
    response->set_world_size(parallel.world_size);
    response->set_dp_rank(parallel.dp_rank);
    response->set_dp_size(parallel.dp_size);
    response->set_ep_rank(parallel.ep_rank);
    response->set_ep_size(parallel.ep_size);
    response->set_received_weights(rl_received_weights_);
    response->set_applied_weights(rl_applied_weights_);
    response->set_quiescent(engine_->rlPaused());
    response->set_kv_cache_ready(engine_->rlKVCacheReady());
    response->set_kv_cache_bytes(engine_->rlKVCacheBytes());
    response->set_weights_ready(rl_phase_ != "failed" && rl_phase_ != "updating" && engine_->rlWeightsReady());
    response->set_released_vmm_bytes(released_bytes);
#if USING_CUDA
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(parallel.local_rank);
    response->set_torch_allocated_bytes(stats.allocated_bytes[0].current);
    response->set_torch_reserved_bytes(stats.reserved_bytes[0].current);
#endif
    return grpc::Status::OK;
}

}  // namespace rtp_llm
