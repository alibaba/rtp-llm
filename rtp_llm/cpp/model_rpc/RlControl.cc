#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace rtp_llm {

grpc::Status LocalRpcServer::RlControl(grpc::ServerContext*      context,
                                       const RlControlRequestPB* request,
                                       RlControlResponsePB*      response) {
    if (!engine_ || !engine_->rlEnabled()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "RTP_LLM_RL_MODE is not enabled"};
    }
    // Exactly one state-changing operation per rank. No Python GIL is held
    // while waiting for the engine loop or for another control RPC.
    std::lock_guard<std::mutex> lock(rl_control_mutex_);
    const auto                  op      = request->operation();
    const auto                  version = engine_->rlModelVersion();
    if (op != RlControlRequestPB::STATUS && rl_phase_ == "failed") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "RL replica failed; restart all TP ranks"};
    }
    if (op != RlControlRequestPB::STATUS && request->expected_version() != version) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "stale or missing expected model version"};
    }
    if ((op == RlControlRequestPB::WRITE_WEIGHT || op == RlControlRequestPB::COMMIT_UPDATE)
        && (rl_phase_ != "updating" || request->update_id() != rl_update_id_)) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "no matching active weight transaction"};
    }
    if ((op == RlControlRequestPB::BEGIN_UPDATE || op == RlControlRequestPB::RESUME
         || op == RlControlRequestPB::RELEASE_KV_CACHE || op == RlControlRequestPB::RESTORE_KV_CACHE)
        && rl_phase_ != "paused") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "operation requires a paused replica"};
    }
    if (op == RlControlRequestPB::PAUSE && rl_phase_ != "running" && rl_phase_ != "paused") {
        return {grpc::StatusCode::FAILED_PRECONDITION, "cannot pause an active weight transaction"};
    }
    if (op == RlControlRequestPB::RELEASE_KV_CACHE && maga_init_params_.hw_kernel_config.enable_cuda_graph) {
        // Reject this configuration error before changing ownership or phase.
        // Captured decode graphs retain the original KV allocation.
        return {grpc::StatusCode::FAILED_PRECONDITION, "RL decode CUDA Graph requires resident KV storage"};
    }
    int64_t timeout_ms = request->timeout_ms();
    if (op != RlControlRequestPB::STATUS) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(context->deadline()
                                                                                     - std::chrono::system_clock::now())
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
        setCurrentThreadDevice(maga_init_params_.parallelism_config.local_rank);
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
            case RlControlRequestPB::RELEASE_KV_CACHE:
                require_ok(engine_->rlReleaseKVCache());
                break;
            case RlControlRequestPB::RESTORE_KV_CACHE:
                require_ok(engine_->rlRestoreKVCache());
                break;
            case RlControlRequestPB::BEGIN_UPDATE: {
                if (!engine_->rlPaused()) {
                    throw std::runtime_error("native engine has not acknowledged quiescence");
                }
                py::gil_scoped_acquire gil;
                weight_manager_.attr("rl_begin")(request->max_buffer_bytes());
                rl_update_id_ = version.substr(0, version.rfind(':') + 1) + std::to_string(rl_version_sequence_ + 1);
                rl_pending_global_step_ = request->global_step();
                rl_received_weights_    = 0;
                rl_applied_weights_     = 0;
                rl_phase_               = "updating";
                break;
            }
            case RlControlRequestPB::WRITE_WEIGHT: {
                py::gil_scoped_acquire gil;
                weight_manager_.attr("rl_push")(request->weight_name(),
                                                py::bytes(request->cuda_ipc_metadata()),
                                                static_cast<double>(timeout_ms) / 1000.0);
                ++rl_received_weights_;
                break;
            }
            case RlControlRequestPB::COMMIT_UPDATE: {
                {
                    py::gil_scoped_acquire gil;
                    auto                   counts =
                        weight_manager_.attr("rl_finish")(static_cast<double>(timeout_ms) / 1000.0).cast<py::dict>();
                    rl_applied_weights_ = counts["internal_tensors"].cast<uint64_t>();
                    if (counts["hf_sources"].cast<uint64_t>() != rl_received_weights_) {
                        throw std::runtime_error("HF source ACK count mismatch");
                    }
                }
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
        // In-place updates cannot roll back safely. Admission remains closed;
        // the coordinator must terminate the entire replica on this response.
        rl_phase_ = "failed";
        // Python coverage errors can list hundreds of parameters. Keep the
        // complete diagnostic in the server log, below gRPC's metadata limit
        // on the wire so the client receives the actual failure status.
        RTP_LLM_LOG_ERROR("RL control failed: %s", error.what());
        return {grpc::StatusCode::INTERNAL,
                std::string("RL control failed: ") + std::string(error.what()).substr(0, 2048)};
    }
    response->set_protocol_version(2);
    response->set_phase(rl_phase_);
    response->set_model_version(engine_->rlModelVersion());
    response->set_update_id(rl_update_id_);
    response->set_global_step(rl_global_step_);
    response->set_tp_rank(maga_init_params_.parallelism_config.tp_rank);
    response->set_tp_size(maga_init_params_.parallelism_config.tp_size);
    response->set_received_weights(rl_received_weights_);
    response->set_applied_weights(rl_applied_weights_);
    response->set_quiescent(engine_->rlPaused());
    response->set_kv_cache_ready(engine_->rlKVCacheReady());
    response->set_kv_cache_bytes(engine_->rlKVCacheBytes());
#if USING_CUDA
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(maga_init_params_.parallelism_config.local_rank);
    response->set_torch_allocated_bytes(stats.allocated_bytes[0].current);
    response->set_torch_reserved_bytes(stats.reserved_bytes[0].current);
#endif
    return grpc::Status::OK;
}

}  // namespace rtp_llm
