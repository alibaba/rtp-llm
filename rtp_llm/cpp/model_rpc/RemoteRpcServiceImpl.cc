#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

#include <algorithm>
#include <thread>
#include <c10/util/ScopeExit.h>

namespace rtp_llm {

grpc::Status RemoteRpcServiceImpl::init(const EngineInitParams&                                params,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                        py::object                                             mm_process_engine) {
    if (params.pd_sep_config.role_type == RoleType::PREFILL) {
        prefill_server_ = std::make_shared<PrefillRpcServer>();
        local_server_   = prefill_server_;
    } else if (params.pd_sep_config.role_type == RoleType::DECODE) {
        decode_server_ = std::make_shared<DecodeRpcServer>();
        local_server_  = decode_server_;
    } else {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "PD server requires Prefill or Decode role");
    }
    const auto status = local_server_->init(params, std::move(propose_params), mm_process_engine);
    if (!status.ok()) {
        return status;
    }
    requires_group_drain_ = params.parallelism_config.world_size > 1;
    const auto& peers     = params.runtime_config.shutdown_worker_grpc_addrs;
    if (peers.size() == static_cast<size_t>(params.parallelism_config.world_size)) {
        shutdown_coordinator_ = std::make_unique<ShutdownCoordinator>(
            peers, params.parallelism_config.world_rank, [this] { return drainState(); }, [this] { beginShutdown(); });
    } else if (requires_group_drain_) {
        RTP_LLM_LOG_WARNING("missing complete shutdown group; graceful shutdown will wait until its deadline");
    }
    return grpc::Status::OK;
}

bool RemoteRpcServiceImpl::tryAdmitRequest() {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    if (draining_) {
        return false;
    }
    ++admitted_requests_;
    return true;
}

void RemoteRpcServiceImpl::finishRequest() {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    --admitted_requests_;
}

grpc::Status RemoteRpcServiceImpl::GenerateStreamCall(grpc::ServerContext*                   context,
                                                       const GenerateInputPB*                 request,
                                                       grpc::ServerWriter<GenerateOutputsPB>* writer) {
    if (!tryAdmitRequest()) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "PD worker is draining");
    }
    auto admission = c10::make_scope_exit([this] { finishRequest(); });
    return LocalRpcServiceImpl::GenerateStreamCall(context, request, writer);
}

grpc::Status RemoteRpcServiceImpl::BatchGenerateCall(grpc::ServerContext*        context,
                                                      const BatchGenerateInputPB* request,
                                                      BatchGenerateOutputsPB*     response) {
    if (!tryAdmitRequest()) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "PD worker is draining");
    }
    auto admission = c10::make_scope_exit([this] { finishRequest(); });
    return LocalRpcServiceImpl::BatchGenerateCall(context, request, response);
}

size_t RemoteRpcServiceImpl::onflightRequestNum() {
    size_t admitted;
    {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        admitted = admitted_requests_;
    }
    // Local servers also retain work after an RPC hands off to an asynchronous
    // queue; the service token covers the interval before their counter rises.
    return admitted + (local_server_ ? local_server_->onflightRequestNum() : 0);
}

ShutdownCoordinator::State RemoteRpcServiceImpl::drainState() {
    ShutdownCoordinator::State state;
    {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        state = {draining_, admitted_requests_};
    }
    state.onflight += (local_server_ ? local_server_->onflightRequestNum() : 0) + external_onflight_();
    return state;
}

void RemoteRpcServiceImpl::beginShutdown() {
    {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        draining_ = true;
    }
    LocalRpcServiceImpl::beginShutdown();
    if (prefill_server_) {
        // Preserve main's batch queue and pending-KV attachment drain contract.
        prefill_server_->beginShutdown();
    }
}

bool RemoteRpcServiceImpl::waitForDrain(ShutdownCoordinator::Deadline deadline) {
    if (shutdown_coordinator_) {
        return shutdown_coordinator_->wait(deadline);
    }
    if (!requires_group_drain_) {
        return LocalRpcServiceImpl::waitForDrain(deadline);
    }
    // An idle local rank does not prove that its TP/EP peers have drained.
    // Keep the engine and shutdown RPC endpoint alive until the caller's budget expires.
    while (std::chrono::system_clock::now() < deadline) {
        std::this_thread::sleep_until(
            std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
    }
    return false;
}

grpc::Status RemoteRpcServiceImpl::ExecuteFunction(grpc::ServerContext*     context,
                                                    const FunctionRequestPB* request,
                                                    FunctionResponsePB*      response) {
    if (request && request->has_shutdown_request()) {
        if (!shutdown_coordinator_ || !response) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "shutdown group is unavailable");
        }
        return shutdown_coordinator_->handle(request->shutdown_request(), response->mutable_shutdown_response());
    }
    return LocalRpcServiceImpl::ExecuteFunction(context, request, response);
}

}  // namespace rtp_llm
