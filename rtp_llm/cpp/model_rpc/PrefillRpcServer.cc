#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/PrefillRunWaiter.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PropagatedClientContext.h"
#include "rtp_llm/cpp/model_rpc/RequestDeadlineBudget.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadBudget.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"
#include "rtp_llm/cpp/model_rpc/SamplerGeneratorState.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <c10/core/InferenceMode.h>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

namespace {

constexpr int kRemoteLoadQuiesceTimeoutMs = 5000;

bool quiesceRemoteLoad(const std::shared_ptr<RpcService::Stub>& stub,
                       const std::string&                       allocation_token,
                       int64_t                                  load_deadline_unix_ms) {
    if (stub == nullptr) {
        return false;
    }
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now()
                         + std::chrono::milliseconds(kRemoteLoadQuiesceTimeoutMs));
    RemoteLoadQuiesceRequestPB request;
    request.set_allocation_token(allocation_token);
    request.set_load_deadline_unix_ms(load_deadline_unix_ms);
    request.set_local_only(false);
    RemoteLoadQuiesceResponsePB response;
    const auto                  status = stub->QuiesceRemoteLoad(&context, request, &response);
    return status.ok() && response.quiesced()
           && response.error_info().error_code() == ErrorCodePB::NONE_ERROR;
}

}  // namespace

#define CLIENT_GRPC_RET_IF_ERROR(prefill_context, state, error_code_value)                                             \
    if (!(state)) {                                                                                                    \
        auto   new_error_code = error_code_value;                                                                      \
        string new_error_msg  = "decode addr is " + prefill_context.decode_addr + ", ";                                \
        new_error_msg += "execute time is " + std::to_string(prefill_context.executeTimeMs()) + "ms, ";                \
        new_error_msg += "request timeout is " + std::to_string(prefill_context.request_timeout_ms) + "ms, ";          \
        new_error_msg += "rpc connection pointer is "                                                                  \
                         + std::to_string((int64_t)prefill_context.grpc_connection.channel.get()) + ", ";              \
        if (prefill_context.getStream()) {                                                                             \
            auto first_token_rt_ms = prefill_context.getStream()->getTimeInfo().first_token_rt_us / 1000;              \
            if (first_token_rt_ms) {                                                                                   \
                new_error_msg += "stream first token rt is " + std::to_string(first_token_rt_ms) + "ms, ";             \
            }                                                                                                          \
            auto wait_time_ms = prefill_context.getStream()->getTimeInfo().wait_time_us / 1000;                        \
            if (wait_time_ms) {                                                                                        \
                new_error_msg += "stream wait time is " + std::to_string(wait_time_ms) + "ms, ";                       \
            }                                                                                                          \
        }                                                                                                              \
        if (prefill_context.cache_lease_ticket != nullptr) {                                                           \
            prefill_context.cache_lease_ticket.reset();                                                                \
        }                                                                                                              \
        if (prefill_context.remote_load_cache_started && prefill_context.client_context != nullptr) {                  \
            prefill_context.client_context->TryCancel();                                                               \
        }                                                                                                              \
        auto status = prefill_context.closeGrpcStream();                                                               \
        if (!status.ok()) {                                                                                            \
            const auto& error_msg = status.error_message();                                                            \
            if (error_msg.find("Connect Failed") != std::string::npos) {                                               \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("No route to host") != std::string::npos) {                                      \
                new_error_code = ErrorCode::CONNECT_FAILED;                                                            \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection reset by peer") != std::string::npos) {                              \
                new_error_code = ErrorCode::CONNECTION_RESET_BY_PEER;                                                  \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("Connection timed out") != std::string::npos) {                                  \
                new_error_code = ErrorCode::CONNECT_TIMEOUT;                                                           \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("keepalive watchdog timeout") != std::string::npos) {                            \
                new_error_code = ErrorCode::KEEP_ALIVE_TIMEOUT;                                                        \
                prefill_context.closeGrpcConnection();                                                                 \
            }                                                                                                          \
            const auto remote_error = transGrpcStatusToErrorInfo(status, new_error_code);                              \
            new_error_code         = remote_error.code();                                                              \
            new_error_msg += remote_error.ToString();                                                                  \
        } else {                                                                                                       \
            if (prefill_context.client_stream) {                                                                       \
                new_error_msg += "server disconnected with status::ok";                                                \
            }                                                                                                          \
        }                                                                                                              \
        if (prefill_context.getStream()) {                                                                             \
            prefill_context.getStream()->reportEvent(StreamEvents::Error, new_error_code, new_error_msg);              \
        }                                                                                                              \
        prefill_context.error_info   = ErrorInfo(new_error_code, new_error_msg);                                       \
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);     \
        return;                                                                                                        \
    }

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                    py::object                                             mm_process_engine) {
    RTP_LLM_CHECK_WITH_INFO(maga_init_params.pd_sep_config.role_type == RoleType::PREFILL,
                            "prefill's role_type must be PREFILL");
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        return ret;
    }
    return grpc::Status::OK;
}

bool PrefillRpcServer::drainRemoteLoads() {
    return drainRemoteLoads(std::chrono::milliseconds(kRemoteLoadQuiesceTimeoutMs));
}

bool PrefillRpcServer::drainRemoteLoads(std::chrono::milliseconds grace) {
    return remote_load_leases_.stop(grace);
}

void PrefillRpcServer::stop() {
    RTP_LLM_CHECK_WITH_INFO(drainRemoteLoads(), "remote load leases failed to quiesce before prefill stop");
    LocalRpcServer::stop();
}

ErrorInfo PrefillRpcServer::waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream,
                                                grpc::ServerContext*            server_context) {
    const auto steady_now      = PrefillRunSteadyClock::now();
    const auto system_now      = PrefillRunSystemClock::now();
    const auto server_deadline = server_context == nullptr ? PrefillRunSystemClock::time_point::max() :
                                                             server_context->deadline();
    const auto configured_timeout = std::chrono::milliseconds(
        std::max<int64_t>(0, maga_init_params_.pd_sep_config.prefill_max_wait_timeout_ms));
    const auto deadline =
        makePrefillRunDeadline(steady_now, configured_timeout, system_now, server_deadline);
    auto wait_result = waitForPrefillRun(
        [&stream]() { return stream->statusInfo().hasError(); },
        [&stream]() {
            const auto status = stream->getStatus();
            return status == StreamState::RUNNING || status == StreamState::FINISHED;
        },
        [server_context]() { return server_context != nullptr && server_context->IsCancelled(); },
        deadline.value);

    if (wait_result == PrefillRunWaitResult::StreamError) {
        return stream->statusInfo();
    }
    if (wait_result == PrefillRunWaitResult::Ready) {
        if (stream->statusInfo().hasError()) {
            return stream->statusInfo();
        }
        if (PrefillRunSteadyClock::now() >= deadline.value) {
            wait_result = PrefillRunWaitResult::DeadlineExceeded;
        } else if (server_context != nullptr && server_context->IsCancelled()) {
            wait_result = PrefillRunWaitResult::Cancelled;
        } else {
            return ErrorInfo::OkStatus();
        }
    }

    ErrorCode   error_code;
    std::string error_msg;
    if (wait_result == PrefillRunWaitResult::Cancelled) {
        error_code = ErrorCode::CANCELLED;
        error_msg  = "request cancelled while waiting for prefill stream to run";
    } else if (deadline.limited_by_server_context) {
        error_code = ErrorCode::GENERATE_TIMEOUT;
        error_msg  = "request deadline exceeded while waiting for prefill stream to run";
    } else {
        error_code = ErrorCode::WAIT_TO_RUN_TIMEOUT;
        error_msg  = "prefill stream wait deadline exceeded after "
                    + std::to_string(configured_timeout.count()) + " ms";
    }
    stream->reportError(error_code, error_msg);
    return stream->statusInfo();
}

void PrefillRpcServer::getRpcConnection(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    const auto deadline_budget =
        makeRequestDeadlineBudget(prefill_context.rpc_context.request->request_deadline_unix_ms(),
                                  prefill_context.rpc_context.request->generate_config().timeout_ms(),
                                  currentTimeUs(),
                                  prefill_context.server_context == nullptr ?
                                      0 :
                                      requestDeadlineUnixUs(prefill_context.server_context->deadline()));
    if (deadline_budget.expired) {
        prefill_context.error_info =
            ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "request deadline expired before prefill admission");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    auto input = QueryConverter::transQuery(prefill_context.rpc_context.request, &deadline_budget);
    input->generate_config->pd_separation = true;
    if (engine_->isMTPEagle()) {
        input->generate_config->force_disable_sp_run = false;
    } else {
        input->generate_config->force_disable_sp_run = true;
    }
    prefill_context.generate_input = input;

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection", prefill_context.request_id);

    auto&                       role_addrs = prefill_context.generate_input->generate_config->role_addrs;
    std::shared_ptr<const Host> host;

    // Check if request specifies host for DECODE role
    for (auto& role_addr : role_addrs) {
        if (role_addr.role == RoleType::DECODE) {
            host = std::make_shared<const Host>(role_addr.ip, role_addr.grpc_port, role_addr.http_port);
            break;
        }
    }

    // If no host specified in request, check if there's a master role
    char* remote_rpc_server_ip_env = std::getenv("REMOTE_RPC_SERVER_IP");
    bool  has_master_role          = (remote_rpc_server_ip_env != nullptr && strlen(remote_rpc_server_ip_env) > 0);

    // If no host specified in request and no master role, this is a direct prefill request
    // In this case, we still need to select decode machines as specified in the requirements
    if (!host && !has_master_role) {
        // For direct prefill requests without master role, we still need to select decode machines
        // The current logic will fail as expected since no host is available
        RTP_LLM_LOG_DEBUG(
            "request [%ld] no host specified in request and no master role, need to select decode machines",
            prefill_context.request_id);
    }

    if (!host || host->ip.empty()) {
        prefill_context.error_info =
            ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for decode cluster " + decode_cluster_name_ + " failed");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    auto decode_addr    = host->ip + ":" + std::to_string(host->rpc_port);
    auto connect_status = resource_.rpc_pool.getConnection(decode_addr);
    if (!connect_status.ok()) {
        prefill_context.error_info   = ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                               "get grpc connection for decode addr " + decode_addr + " failed");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    prefill_context.decode_addr     = decode_addr;
    prefill_context.grpc_connection = connect_status.value();

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection done", prefill_context.request_id);
}

void PrefillRpcServer::multimodalProcess(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    auto& input = prefill_context.generate_input;
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto result = mm_processor_->updateMultimodalFeatures(input);
        CLIENT_GRPC_RET_IF_ERROR(prefill_context, result.ok(), result.code());

        auto mutable_request = const_cast<GenerateInputPB*>(prefill_context.rpc_context.request);
        mutable_request->clear_token_ids();
        // TODO(xinfei.sxf) optimize copy
        auto* ids_ptr = input->input_ids.data_ptr<int32_t>();
        for (size_t i = 0; i < input->input_ids.numel(); i++) {
            mutable_request->add_token_ids(ids_ptr[i]);
        }
    }
}

void PrefillRpcServer::remoteAllocateResource(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote allocate resource", prefill_context.request_id);
    auto request_timeout_ms = prefill_context.request_timeout_ms;
    auto max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    auto final_timeout_ms   = max_rpc_timeout_ms > 0 ? max_rpc_timeout_ms : MAX_GRPC_TIMEOUT_MS;
    final_timeout_ms = request_timeout_ms > 0 ? std::min(request_timeout_ms, final_timeout_ms) : final_timeout_ms;

    const auto system_now = RemoteLoadSystemClock::now();
    const auto steady_now = RemoteLoadSteadyClock::now();
    const auto request_start_unix_ms = prefill_context.generate_input == nullptr ?
                                           prefill_context.request_begin_time_us / 1000 :
                                           prefill_context.generate_input->begin_time_us / 1000;
    const auto request_deadline_unix_ms = saturatingDeadlineUnixMs(request_start_unix_ms, final_timeout_ms);
    const auto parent_deadline = prefill_context.server_context == nullptr ?
                                     RemoteLoadSystemClock::time_point::max() :
                                     prefill_context.server_context->deadline();
    const auto request_budget =
        makeRemoteLoadBudget(request_deadline_unix_ms, parent_deadline, system_now, steady_now);
    if (request_budget.expired()) {
        prefill_context.error_info =
            ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "request deadline expired before remote allocation");
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    const auto stop_error = serverContextStopError(prefill_context.server_context, "before remote decode allocation");
    if (stop_error.hasError()) {
        prefill_context.error_info   = stop_error;
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, stop_error);
        return;
    }

    prefill_context.client_context =
        makePropagatedClientContext(prefill_context.server_context, request_budget.system_deadline);
    prefill_context.client_stream =
        std::move(prefill_context.grpc_connection.stub->RemoteGenerate(prefill_context.client_context.get()));
    auto&             client_stream = prefill_context.client_stream;
    GenerateRequestPB alloc_request;
    alloc_request.set_stage(RemoteStage::ALLOCATE);
    alloc_request.set_client_id(process_id_);
    alloc_request.set_request_id(prefill_context.request_id);
    // TODO(xinfei.sxf) reduce copy
    GenerateInputPB* new_request = new GenerateInputPB(*prefill_context.rpc_context.request);
    alloc_request.set_allocated_input(new_request);
    for (auto& addrs : prefill_context.prefill_worker_cache_store_addrs) {
        alloc_request.add_peer_addrs(addrs);
    }

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Write(alloc_request), ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED);
    GenerateOutputsPB allocate_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Read(&allocate_response), ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    CLIENT_GRPC_RET_IF_ERROR(prefill_context,
                             allocate_response.error_info().error_code() == ErrorCodePB::NONE_ERROR,
                             transRPCErrorCode(allocate_response.error_info().error_code()));
    CLIENT_GRPC_RET_IF_ERROR(prefill_context,
                             !allocate_response.allocation_token().empty(),
                             ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    auto load_deadline = remoteLoadAllocationDeadline(allocate_response.allocation_token());
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, load_deadline.ok(), ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    prefill_context.allocation_token      = allocate_response.allocation_token();
    prefill_context.load_deadline_unix_ms = *load_deadline;
    RTP_LLM_LOG_DEBUG("request [%ld] remote allocate resource done", prefill_context.request_id);
}

void PrefillRpcServer::enqueueRequest(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", prefill_context.request_id);
    auto stream = engine_->enqueue(prefill_context.generate_input);
    prefill_context.setStream(stream);
    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", prefill_context.request_id);
}

void PrefillRpcServer::remoteLoadCacheStart(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache", prefill_context.request_id);
    prefill_context.error_info =
        waitStreamBeforeRun(prefill_context.getStream(), prefill_context.server_context);
    if (prefill_context.error_info.hasError()) {
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    auto stream = prefill_context.getStream();
    AtomicGuard request_guard(loading_cache_requests_);
    const auto hold_result = stream->holdKVCacheForPDSep();
    if (hold_result == PdSepCacheHoldResult::AlreadyLocalTerminal) {
        return;
    }
    if (hold_result == PdSepCacheHoldResult::HoldFailed) {
        const auto hold_error =
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "failed to retain KV cache before remote load");
        const auto local_outcome = stream->resolveRemoteLoadFailure(hold_error.code(), hold_error.ToString());
        if (local_outcome == RemoteGenerateWaitResult::LocalDone) {
            prefill_context.local_generate_done = true;
            return;
        }
        prefill_context.error_info = stream->statusInfo();
        prefill_context.error_status =
            serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    auto source_lease = stream->takeKVCacheForPDSep();
    if (source_lease == nullptr) {
        const auto hold_error =
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "failed to take KV cache lease before remote load");
        const auto local_outcome = stream->resolveRemoteLoadFailure(hold_error.code(), hold_error.ToString());
        if (local_outcome == RemoteGenerateWaitResult::LocalDone) {
            prefill_context.local_generate_done = true;
            return;
        }
        prefill_context.error_info   = stream->statusInfo();
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }

    const auto stub                  = prefill_context.grpc_connection.stub;
    const auto allocation_token      = prefill_context.allocation_token;
    const auto load_deadline_unix_ms = prefill_context.load_deadline_unix_ms;
    auto       quiesce               = prefill_context.remote_load_quiesce;
    if (!quiesce) {
        quiesce = [stub, allocation_token, load_deadline_unix_ms]() {
            return quiesceRemoteLoad(stub, allocation_token, load_deadline_unix_ms);
        };
    }
    auto       ticket                = remote_load_leases_.reserve(
        allocation_token,
        std::move(source_lease),
        std::move(quiesce));
    if (!ticket.ok()) {
        const auto load_error =
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "failed to retain KV cache for remote load");
        const auto local_outcome = stream->resolveRemoteLoadFailure(load_error.code(), load_error.ToString());
        if (local_outcome == RemoteGenerateWaitResult::LocalDone) {
            prefill_context.local_generate_done = true;
            return;
        }
        prefill_context.error_info   = stream->statusInfo();
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    prefill_context.cache_lease_ticket = std::move(*ticket);

    GenerateRequestPB load_request;
    load_request.set_stage(RemoteStage::LOAD);
    load_request.set_client_id(process_id_);
    load_request.set_request_id(prefill_context.request_id);
    load_request.set_start_time(currentTimeUs());
    load_request.set_allocation_token(allocation_token);
    load_request.set_load_deadline_unix_ms(load_deadline_unix_ms);

    if (!prefill_context.cache_lease_ticket->markStarted()) {
        const auto load_error =
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "failed to start retained remote load");
        const auto local_outcome = stream->resolveRemoteLoadFailure(load_error.code(), load_error.ToString());
        prefill_context.cleanupRemoteLoadCache();
        if (local_outcome == RemoteGenerateWaitResult::LocalDone) {
            prefill_context.local_generate_done = true;
            return;
        }
        prefill_context.error_info   = stream->statusInfo();
        prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
    prefill_context.remote_load_cache_started = true;

    bool        write_ok = false;
    std::string write_error_message = "remote load request write failed";
    try {
        write_ok = prefill_context.client_stream->Write(load_request);
    } catch (const std::exception& e) {
        write_error_message += ": " + std::string(e.what());
    } catch (...) {
        write_error_message += ": unknown exception";
    }
    if (!write_ok) {
        const auto load_error = ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, write_error_message);
        const auto local_outcome = stream->resolveRemoteLoadFailure(load_error.code(), load_error.ToString());
        prefill_context.cleanupRemoteLoadCache();
        if (local_outcome == RemoteGenerateWaitResult::LocalDone) {
            prefill_context.local_generate_done = true;
            return;
        }
        prefill_context.error_info = stream->statusInfo();
        prefill_context.error_status =
            serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        return;
    }
}

void PrefillRpcServer::pollLocalOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll local output", prefill_context.request_id);
    auto remote_generate_result = RemoteGenerateWaitResult::Error;
    auto first_status           = pollStreamOutput(prefill_context.server_context,
                                         prefill_context.request_key,
                                         prefill_context.rpc_context.writer,
                                         prefill_context.getStream(),
                                         &remote_generate_result);
    if (!first_status.ok()) {
        if (prefill_context.remote_load_cache_started) {
            prefill_context.deferred_local_status = first_status;
        } else {
            prefill_context.error_status = first_status;
        }
        return;
    }
    RTP_LLM_LOG_DEBUG("request [%ld] poll local output end", prefill_context.request_id);

    if (remote_generate_result == RemoteGenerateWaitResult::LocalDone) {
        prefill_context.local_generate_done = true;
        if (!prefill_context.remote_load_cache_started) {
            prefill_context.finished = true;
        }
        return;
    }
    if (remote_generate_result == RemoteGenerateWaitResult::Error) {
        auto stream = prefill_context.getStream();
        auto status = serializeErrorMsg(prefill_context.request_key, stream->statusInfo());
        if (prefill_context.remote_load_cache_started) {
            prefill_context.deferred_local_status = status;
        } else {
            prefill_context.error_status = status;
        }
    }
}

void PrefillRpcServer::remoteLoadCacheEnd(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!prefill_context.remote_load_cache_started) {
        return;
    }
    auto stream = prefill_context.getStream();
    auto fail_remote_load = [&](const ErrorInfo& load_error) {
        if (!prefill_context.local_generate_done && prefill_context.deferred_local_status.ok()) {
            stream->resolveRemoteLoadFailure(load_error.code(), load_error.ToString());
        }
        prefill_context.cleanupRemoteLoadCache();
        if (prefill_context.local_generate_done) {
            prefill_context.finished = true;
        } else if (!prefill_context.deferred_local_status.ok()) {
            prefill_context.error_status = prefill_context.deferred_local_status;
        } else {
            prefill_context.error_info   = stream->statusInfo();
            prefill_context.error_status =
                serializeErrorMsg(prefill_context.request_key, prefill_context.error_info);
        }
    };

    GenerateOutputsPB load_response;
    bool              read_ok = false;
    std::string       read_error_message = "remote load response failed";
    try {
        read_ok = prefill_context.client_stream->Read(&load_response);
    } catch (const std::exception& e) {
        read_error_message += ": " + std::string(e.what());
    } catch (...) {
        read_error_message += ": unknown exception";
    }
    if (!read_ok) {
        fail_remote_load(ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, read_error_message));
        return;
    }

    const auto error_code = transRPCErrorCode(load_response.error_info().error_code());
    if (error_code != ErrorCode::NONE_ERROR) {
        const auto& response_message = load_response.error_info().error_message();
        fail_remote_load(ErrorInfo(error_code,
                                   response_message.empty() ? "remote load response failed" : response_message));
        return;
    }

    if (!load_response.remote_load_quiesced()) {
        fail_remote_load(
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "remote load completed without quiescing workers"));
        return;
    }
    if (prefill_context.cache_lease_ticket == nullptr || !prefill_context.cache_lease_ticket->complete()) {
        fail_remote_load(
            ErrorInfo(ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED, "failed to complete retained remote load"));
        return;
    }

    stream->releaseKVCacheForPDSep();
    prefill_context.remote_load_cache_started = false;
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache done", prefill_context.request_id);
    if (!prefill_context.deferred_local_status.ok()) {
        prefill_context.error_status = prefill_context.deferred_local_status;
        return;
    }
    if (prefill_context.local_generate_done) {
        prefill_context.finished = true;
    }
}

void PrefillRpcServer::remoteGenerate(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote generate", prefill_context.request_id);
    std::shared_ptr<GenerateStream> stream = prefill_context.getStream();
    RTP_LLM_LOG_DEBUG("remote generate stream[%ld]: %s", stream->streamId(), stream->debugString().c_str());
    vector<int> all_token   = stream->currentExecuteTokens();
    int         first_token = all_token[all_token.size() - 1];
    RTP_LLM_LOG_DEBUG("first token token id %d", first_token);
    GenerateRequestPB generate_request;
    generate_request.set_client_id(process_id_);
    generate_request.set_request_id(prefill_context.request_id);
    generate_request.set_first_generate_token_id(first_token);
    generate_request.set_sampler_generator_state_version(kCurrentSamplerGeneratorStateVersion);
    auto generator_state = captureSamplerGeneratorState(
        stream->generateConfig()->random_seed.has_value(), stream->getGenerator());
    if (!generator_state.ok()) {
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, generator_state.status().ToString());
        return;
    }
    if (!generator_state->empty()) {
        generate_request.set_sampler_generator_state(*generator_state);
    }
    auto context_position_ids = stream->getContextPositionIds();
    if (context_position_ids.defined()) {
        generate_request.mutable_position_ids()->CopyFrom(
            {context_position_ids.data_ptr<int32_t>(),
             context_position_ids.data_ptr<int32_t>() + context_position_ids.numel()});
    }
    if (engine_->isMTPEagle()) {
        RTP_LLM_CHECK_WITH_INFO(stream->getProposeToken().size() > 0,
                                "mtp remote generate propose token should not be empty");
    }
    generate_request.mutable_propose_token_ids()->CopyFrom(
        {stream->getProposeToken().begin(), stream->getProposeToken().end()});

    auto sp_output_buffer = stream->getSPOutputBuffer();

    if (sp_output_buffer) {
        auto all_probs_cpu =
            sp_output_buffer->all_probs.is_cuda() ? sp_output_buffer->all_probs.cpu() : sp_output_buffer->all_probs;
        torch::Tensor hidden_states_cpu;
        if (!sp_output_buffer->hidden_states.defined()) {
            // dummy hidden states, so datatype is not important
            hidden_states_cpu = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat16));
        } else {
            hidden_states_cpu = sp_output_buffer->hidden_states.is_cuda() ? sp_output_buffer->hidden_states.cpu() :
                                                                            sp_output_buffer->hidden_states;
        }
        QueryConverter::transTensorPB(generate_request.mutable_propose_probs(), all_probs_cpu);
        QueryConverter::transTensorPB(generate_request.mutable_propose_hidden(), hidden_states_cpu);
    }

    generate_request.set_stage(RemoteStage::GENERATE);

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(generate_request), ErrorCode::REMOTE_GENERATE_FAILED);
}

void PrefillRpcServer::pollRemoteOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll remote output", prefill_context.request_id);
    auto&             request_id = prefill_context.request_id;
    GenerateOutputsPB response;
    auto              prefill_total_reuse_len  = prefill_context.getStream()->initialReuseLength();
    auto              prefill_local_reuse_len  = prefill_context.getStream()->localReuseLength();
    auto              prefill_remote_reuse_len = prefill_context.getStream()->remoteReuseLength();
    auto              prefill_memory_reuse_len = prefill_context.getStream()->memoryReuseLength();

    auto first_token_rt_us = prefill_context.getStream()->getTimeInfo().first_token_rt_us;
    while (prefill_context.client_stream->Read(&response)) {
        const auto stop_error =
            serverContextStopError(prefill_context.server_context, "forwarding remote decode output");
        if (stop_error.hasError()) {
            RTP_LLM_LOG_WARNING("request [%ld] stopped by deadline or cancellation", request_id);
            prefill_context.error_info   = stop_error;
            prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, stop_error);
            return;
        }
        if (response.flatten_output().aux_info_size() == 0) {
            RTP_LLM_LOG_ERROR("request [%ld] generate output size is 0", request_id);
            break;
        }
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            response.mutable_flatten_output()->mutable_aux_info(i)->set_pd_sep(true);
        }
        int64_t cost_time_us = currentTimeUs() - prefill_context.request_begin_time_us;
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            auto decode_total_reuse_len  = response.flatten_output().aux_info(i).total_reuse_len();
            auto decode_local_reuse_len  = response.flatten_output().aux_info(i).local_reuse_len();
            auto decode_remote_reuse_len = response.flatten_output().aux_info(i).remote_reuse_len();
            auto decode_memory_reuse_len = response.flatten_output().aux_info(i).memory_reuse_len();

            response.mutable_flatten_output()->mutable_aux_info(i)->set_first_token_cost_time_us(first_token_rt_us);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_cost_time_us(cost_time_us);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_total_reuse_len(prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_local_reuse_len(prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_remote_reuse_len(prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_memory_reuse_len(prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_total_reuse_len(
                prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_local_reuse_len(
                prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_remote_reuse_len(
                prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_memory_reuse_len(
                prefill_memory_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_total_reuse_len(decode_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_local_reuse_len(decode_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_remote_reuse_len(
                decode_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_memory_reuse_len(
                decode_memory_reuse_len);
        }
        if (!prefill_context.rpc_context.writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%ld] write outputs pb failed", request_id);
            const auto write_stop_error =
                serverContextStopError(prefill_context.server_context, "writing remote decode output");
            if (write_stop_error.hasError()) {
                prefill_context.error_info   = write_stop_error;
                prefill_context.error_status = serializeErrorMsg(prefill_context.request_key, write_stop_error);
            } else {
                prefill_context.error_status =
                    grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
            }
            return;
        }
    }
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.closeGrpcStream().ok(), ErrorCode::REMOTE_GENERATE_FAILED);
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(multimodalProcess, prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                  const GenerateInputPB*                 request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start generate stream call", request->request_id());
    c10::InferenceMode inference_guard(true);
    auto pd_separation = request->generate_config().max_new_tokens() > 1 && request->generate_config().num_beams() <= 1
                         && request->generate_config().variable_num_beams().size() == 0
                         && request->generate_config().num_return_sequences() <= 1
                         && request->generate_config().can_use_pd_separation();
    if (!pd_separation) {
        return LocalRpcServer::GenerateStreamCall(server_context, request, writer);
    }

    AtomicGuardPtr request_guard = make_shared<AtomicGuard>(onflight_requests_);
    RPCContext     rpc_context{request, writer};
    auto           prefill_context         = PrefillGenerateContext(&this->resource(),
                                                  rpc_context,
                                                  request->generate_config().timeout_ms(),
                                                  server_context,
                                                  metrics_reporter_,
                                                  meta_);
    prefill_context.onflight_requests      = onflight_requests_;
    prefill_context.loading_cache_requests = loading_cache_requests_;

    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;

    try {
        EXECUTE_WITH_RETRY(
            prepareAllocateResource, prefill_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
        if (prefill_context.hasError()) {
            RTP_LLM_LOG_WARNING(
                "request [%ld] prepare allocate resource failed after retry [%d] times, cost time ms [%ld], "
                "max retry time [%ld], max retry timeout ms [%ld]",
                prefill_context.request_id,
                prefill_context.retry_times,
                prefill_context.retry_cost_time_ms,
                max_retry_times + 1,
                max_retry_timeout_ms);
            return prefill_context.error_status;
        }
        EXECUTE_STAGE_FUNC(enqueueRequest, prefill_context);
        EXECUTE_STAGE_FUNC(remoteLoadCacheStart, prefill_context);
        EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
        EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
        meta_->dequeue(prefill_context.request_id, prefill_context.getStream());
        EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
        EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
        prefill_context.stat_info.nextStage();
    } catch (const std::exception& e) {
        auto error_msg = "request [" + prefill_context.request_key + "] catch exception [" + e.what() + "]";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return prefill_context.error_status;
    } catch (...) {
        auto error_msg               = "request [" + prefill_context.request_key + "] catch unknown exception";
        prefill_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return prefill_context.error_status;
    }

    RTP_LLM_LOG_DEBUG("request [%ld] all done", prefill_context.request_id);

    return grpc::Status::OK;
}

grpc::Status
PrefillRpcServer::RemoteFinish(grpc::ServerContext* context, const RemoteFinishRequestPB* request, EmptyPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    auto request_id = request->request_id();
    resource_.cache_store->markRequestEnd(std::to_string(request_id));
    return grpc::Status::OK;
}

}  // namespace rtp_llm
