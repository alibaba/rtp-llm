#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/telemetry/PhaseSpanSynthesizer.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <limits.h>
#include <c10/core/InferenceMode.h>

using namespace std;
using namespace autil::legacy;

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

namespace {

bool envValueIsTrue(const char* value) {
    return value != nullptr
           && (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "on") == 0
               || strcasecmp(value, "yes") == 0);
}

bool prefillTraceLogEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PREFILL_TRACE_LOG_ENABLE");
        if (value == nullptr) {
            value = std::getenv("PREFILL_CACHE_DEBUG_LOG");
        }
        if (value == nullptr) {
            value = std::getenv("KV_CACHE_DEBUG_LOG");
        }
        return envValueIsTrue(value);
    }();
    return enabled;
}

const char* prefillStageName(PrefillStatInfo::ExecuteStage stage) {
    switch (stage) {
        case PrefillStatInfo::start:
            return "start";
        case PrefillStatInfo::getRpcConnection:
            return "getRpcConnection";
        case PrefillStatInfo::multimodalProcess:
            return "multimodalProcess";
        case PrefillStatInfo::remoteAllocateResource:
            return "remoteAllocateResource";
        case PrefillStatInfo::enqueueRequest:
            return "enqueueRequest";
        case PrefillStatInfo::remoteLoadCacheStart:
            return "remoteLoadCacheStart";
        case PrefillStatInfo::pollLocalOutput:
            return "pollLocalOutput";
        case PrefillStatInfo::remoteLoadCacheEnd:
            return "remoteLoadCacheEnd";
        case PrefillStatInfo::RemoteGenerate:
            return "RemoteGenerate";
        case PrefillStatInfo::pollRemoteOutput:
            return "pollRemoteOutput";
        case PrefillStatInfo::finish:
            return "finish";
        default:
            return "unknown";
    }
}

void logPrefillFailureTrace(const char* event, PrefillGenerateContext& prefill_context) {
    if (!prefillTraceLogEnabled()) {
        return;
    }
    RTP_LLM_LOG_WARNING("Prefill request trace: event=%s request_id=%ld request_key=%s stage=%s retry_times=%ld "
                        "retry_cost_time_ms=%ld execute_time_ms=%ld decode_addr=%s grpc_code=%d grpc_message=%s "
                        "error_code=%d error_message=%s",
                        event,
                        prefill_context.request_id,
                        prefill_context.request_key.c_str(),
                        prefillStageName(prefill_context.stat_info.stage),
                        prefill_context.retry_times,
                        prefill_context.retry_cost_time_ms,
                        prefill_context.executeTimeMs(),
                        prefill_context.decode_addr.c_str(),
                        static_cast<int>(prefill_context.error_status.error_code()),
                        prefill_context.error_status.error_message().c_str(),
                        static_cast<int>(prefill_context.error_info.code()),
                        prefill_context.error_info.ToString().c_str());
}

}  // namespace

PrefillRpcServer::~PrefillRpcServer() = default;

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
        auto status = prefill_context.closeGrpcStream(ErrorCodeToString(new_error_code));                              \
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
            } else if (error_msg.find("Deadline Exceeded") != std::string::npos) {                                     \
                new_error_code = ErrorCode::DEADLINE_EXCEEDED;                                                         \
                prefill_context.closeGrpcConnection();                                                                 \
            } else if (error_msg.find("keepalive watchdog timeout") != std::string::npos) {                            \
                new_error_code = ErrorCode::KEEP_ALIVE_TIMEOUT;                                                        \
                prefill_context.closeGrpcConnection();                                                                 \
            }                                                                                                          \
            new_error_msg += error_msg;                                                                                \
            if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {                                         \
                new_error_code = ErrorCode::DECODE_MALLOC_FAILED;                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            if (prefill_context.client_stream) {                                                                       \
                new_error_msg += "server disconnected with status::ok";                                                \
            }                                                                                                          \
        }                                                                                                              \
        if (prefill_context.getStream()) {                                                                             \
            prefill_context.getStream()->reportEvent(StreamEvents::Error, new_error_code, new_error_msg);              \
        }                                                                                                              \
        setContextError(prefill_context, ErrorInfo(new_error_code, new_error_msg));                                    \
        logPrefillFailureTrace("client_grpc_error", prefill_context);                                                  \
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

ErrorInfo PrefillRpcServer::waitStreamBeforeRun(std::shared_ptr<GenerateStream> stream) {
    static int max_wait_timeout_us = maga_init_params_.pd_sep_config.prefill_max_wait_timeout_ms * 1000;
    auto       begin_time_us       = currentTimeUs();
    while (!stream->hasError() && stream->getStatus() == StreamState::WAITING) {
        usleep(100);
        auto current_time_us = currentTimeUs();
        auto cost_time_us    = current_time_us - begin_time_us;
        if (cost_time_us > max_wait_timeout_us) {
            string new_error_msg = "wait to run timeout, timeout is " + std::to_string(max_wait_timeout_us) + " us";
            stream->reportEvent(StreamEvents::Error, ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
            return ErrorInfo(ErrorCode::WAIT_TO_RUN_TIMEOUT, new_error_msg);
        }
    }
    if (stream->hasError()) {
        return stream->statusInfo();
    }
    return ErrorInfo::OkStatus();
}

void PrefillRpcServer::setContextError(PrefillGenerateContext& prefill_context, const ErrorInfo& error_info) {
    prefill_context.error_info = error_info;
    prefill_context.error_status =
        serializeErrorMsg(prefill_context.request_key, prefill_context.request_info, error_info);
}

void PrefillRpcServer::setContextError(PrefillGenerateContext& prefill_context,
                                       const ErrorInfo&        error_info,
                                       const grpc::Status&     error_status) {
    prefill_context.error_info   = error_info;
    prefill_context.error_status = error_status;
}

void PrefillRpcServer::prepareGenerateInput(PrefillGenerateContext& prefill_context) {
    if (!prefill_context.generate_input) {
        RTP_LLM_CHECK_WITH_INFO(engine_ != nullptr, "prefill rpc server engine is not initialized");
        auto input                                   = QueryConverter::transQuery(prefill_context.rpc_context.request);
        input->generate_config->pd_separation        = true;
        input->generate_config->force_disable_sp_run = !engine_->isMTPEagle();
        prefill_context.generate_input               = std::move(input);
    }
    prefill_context.request_info = prefill_context.generate_input->request_info;
}

void PrefillRpcServer::getRpcConnection(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    prefill_context.trace_server_address.clear();
    prefill_context.trace_server_port = 0;
    RTP_LLM_LOG_DEBUG("request [%ld] trans query", prefill_context.request_id);
    prepareGenerateInput(prefill_context);

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
        setContextError(
            prefill_context,
            ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for decode cluster " + decode_cluster_name_ + " failed"));
        logPrefillFailureTrace("get_rpc_connection_no_decode_host", prefill_context);
        return;
    }
    auto decode_addr    = host->ip + ":" + std::to_string(host->rpc_port);
    auto connect_status = resource_.rpc_pool.getConnection(decode_addr);
    if (!connect_status.ok()) {
        setContextError(prefill_context,
                        ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                  "get grpc connection for decode addr " + decode_addr + " failed"));
        prefill_context.decode_addr = decode_addr;
        logPrefillFailureTrace("get_rpc_connection_failed", prefill_context);
        return;
    }
    prefill_context.decode_addr     = decode_addr;
    prefill_context.grpc_connection = connect_status.value();
    if (prefill_context.trace_span_guard && prefill_context.trace_span_guard->valid()) {
        try {
            prefill_context.trace_server_address = host->ip;
            prefill_context.trace_server_port    = host->rpc_port;
        } catch (...) {
            prefill_context.trace_server_address.clear();
            prefill_context.trace_server_port = 0;
        }
    }

    RTP_LLM_LOG_DEBUG("request [%ld] get rpc connection done", prefill_context.request_id);
}

void PrefillRpcServer::multimodalProcess(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (prefill_context.multimodalProcessed()) {
        return;
    }

    auto& input = prefill_context.generate_input;
    RTP_LLM_CHECK_WITH_INFO(input != nullptr, "multimodal processing requires a prepared generate input");
    if (!input->multimodal_inputs || input->multimodal_inputs->empty()) {
        prefill_context.markMultimodalProcessed(false);
        return;
    }

    prefill_context.markMultimodalAttemptStarted();
    if (mm_processor_ == nullptr) {
        const auto error =
            ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "multimodal inputs require a configured multimodal processor");
        RTP_LLM_LOG_WARNING("request [%ld] rejected: %s", prefill_context.request_id, error.ToString().c_str());
        prefill_context.setRetryable(false);
        setContextError(prefill_context, error);
        logPrefillFailureTrace("multimodal_process_failed", prefill_context);
        return;
    }

    auto result = mm_processor_->updateMultimodalFeatures(input);
    if (!result.ok()) {
        prefill_context.setRetryable(isRetryableMultimodalError(result.code()));
        setContextError(prefill_context, result);
        logPrefillFailureTrace("multimodal_process_failed", prefill_context);
        return;
    }
    prefill_context.markMultimodalProcessed(true);
}

GenerateRequestPB PrefillRpcServer::buildAllocateRequest(PrefillGenerateContext& prefill_context) {
    GenerateRequestPB alloc_request;
    alloc_request.set_stage(RemoteStage::ALLOCATE);
    alloc_request.set_client_id(process_id_);
    alloc_request.set_request_id(prefill_context.request_id);

    GenerateInputPB* new_request = alloc_request.mutable_input();
    new_request->CopyFrom(*prefill_context.rpc_context.request);
    new_request->clear_group_size();
    new_request->clear_group_id();
    new_request->mutable_generate_config()->clear_group_timeout();
    RTP_LLM_CHECK_WITH_INFO(!prefill_context.tokenIdsExpanded() || prefill_context.generate_input != nullptr,
                            "expanded token ids require a prepared generate input");
    if (prefill_context.tokenIdsExpanded()) {
        new_request->clear_token_ids();
        const auto& input   = prefill_context.generate_input;
        auto*       ids_ptr = input->input_ids.data_ptr<int32_t>();
        for (size_t i = 0; i < input->input_ids.numel(); ++i) {
            new_request->add_token_ids(ids_ptr[i]);
        }
    }
    for (const auto& address : prefill_context.prefill_worker_cache_store_addrs) {
        alloc_request.add_peer_addrs(address);
    }
    return alloc_request;
}

void PrefillRpcServer::remoteAllocateResource(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to remote allocate resource", prefill_context.request_id);
    auto client_context = std::make_shared<ClientContext>();
    // P->D CLIENT span: each retry rebuilds ClientContext and opens a NEW
    // physical RemoteGenerate bidi stream (stub->RemoteGenerate
    // below), so one CLIENT span per attempt matches the OTel "one span per
    // physical RPC" convention and keeps a clean 1:1 with the Decode SERVER
    // span each attempt spawns. A failed attempt's span is settled inside
    // CLIENT_GRPC_RET_IF_ERROR (transport error name, or the business
    // ErrorCode when transport Finish()==OK); the retry fallback below is only
    // for attempts that failed before any gRPC call was made. The
    // rtp_llm.retry_attempt is the zero-based number of retries already
    // performed, keeping the retry chain visible without marking the initial
    // attempt as a retry.
    if (prefill_context.trace_span_guard && prefill_context.trace_span_guard->valid()) {
        if (prefill_context.pd_client_span_guard) {
            prefill_context.pd_client_span_guard->setAttribute(telemetry::kAttrErrorType, "Retry");
            prefill_context.pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kError,
                                                         "Prefill-to-decode RPC attempt failed before a retry");
        }
        auto client_span =
            telemetry::startChildClientSpan("rtp_llm.remote_generate",
                                            prefill_context.trace_span_guard->sharedSpan(),
                                            prefill_context.trace_server_address,
                                            prefill_context.trace_server_port,
                                            prefill_context.request_id,
                                            telemetry::retryAttemptFromExecutionCount(prefill_context.retry_times),
                                            "RpcService/RemoteGenerate");
        if (client_span != nullptr) {
            prefill_context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(client_span);
            telemetry::injectSpanToClientContext(client_context.get(), client_span);
        }
    }
    auto    request_timeout_ms = prefill_context.request_timeout_ms;
    auto    max_rpc_timeout_ms = maga_init_params_.pd_sep_config.max_rpc_timeout_ms;
    int64_t final_timeout_ms   = request_timeout_ms > 0 ? request_timeout_ms : max_rpc_timeout_ms;
    if (final_timeout_ms > 0) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(final_timeout_ms);
        client_context->set_deadline(deadline);
    }
    std::atomic_store(&prefill_context.client_context, client_context);
    // Close the publish-before-cancel window: either requestPriorityPreempt()
    // observes this ClientContext, or this check observes its cancel latch.
    if (prefill_context.cancel_state->load(std::memory_order_seq_cst)) {
        client_context->TryCancel();
    }
    // final_timeout_ms <= 0: skip set_deadline; gRPC treats it as no deadline.
    prefill_context.client_stream =
        std::move(prefill_context.grpc_connection.stub->RemoteGenerate(client_context.get()));
    auto&             client_stream = prefill_context.client_stream;
    GenerateRequestPB alloc_request = buildAllocateRequest(prefill_context);

    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Write(alloc_request), ErrorCode::REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED);
    GenerateOutputsPB allocate_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, client_stream->Read(&allocate_response), ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED);
    if (prefillTraceLogEnabled() && allocate_response.has_error_info()
        && allocate_response.error_info().error_code() != 0) {
        RTP_LLM_LOG_WARNING("Prefill request trace: event=remote_allocate_response_error request_id=%ld "
                            "decode_addr=%s remote_error_code=%d remote_error_message=%s",
                            prefill_context.request_id,
                            prefill_context.decode_addr.c_str(),
                            static_cast<int>(allocate_response.error_info().error_code()),
                            allocate_response.error_info().error_message().c_str());
    }
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
    auto start_time_us = currentTimeUs();
    auto wait_result   = waitStreamBeforeRun(prefill_context.getStream());
    prefill_context.stat_info.remote_load_cache_wait_stream_rt_us += currentTimeUs() - start_time_us;
    if (wait_result.hasError()) {
        setContextError(prefill_context, wait_result);
        logPrefillFailureTrace("wait_stream_before_run_failed", prefill_context);
        return;
    }
    AtomicGuard       request_guard(loading_cache_requests_);
    GenerateRequestPB load_request;
    load_request.set_client_id(process_id_);
    load_request.set_request_id(prefill_context.request_id);
    load_request.set_start_time(currentTimeUs());
    start_time_us = currentTimeUs();
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Write(load_request), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    prefill_context.stat_info.remote_load_cache_write_request_rt_us += currentTimeUs() - start_time_us;
}

void PrefillRpcServer::pollLocalOutput(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%ld] start to poll local output", prefill_context.request_id);
    auto first_status = pollStreamOutput(prefill_context.server_context,
                                         prefill_context.request_key,
                                         prefill_context.rpc_context.writer,
                                         prefill_context.getStream());
    if (!first_status.ok()) {
        auto stream = prefill_context.getStream();
        if (stream && stream->hasError()) {
            setContextError(prefill_context, stream->statusInfo(), first_status);
        } else if (first_status.error_code() == grpc::StatusCode::CANCELLED) {
            setContextError(
                prefill_context, ErrorInfo(ErrorCode::CANCELLED, first_status.error_message()), first_status);
        } else {
            ErrorDetailsPB error_details;
            if (!first_status.error_details().empty() && error_details.ParseFromString(first_status.error_details())
                && error_details.error_code() != static_cast<int64_t>(ErrorCode::NONE_ERROR)) {
                const auto& error_message = error_details.error_message().empty() ? first_status.error_message() :
                                                                                    error_details.error_message();
                setContextError(prefill_context,
                                ErrorInfo(static_cast<ErrorCode>(error_details.error_code()), error_message),
                                first_status);
            } else {
                setContextError(
                    prefill_context, ErrorInfo(ErrorCode::UNKNOWN_ERROR, first_status.error_message()), first_status);
            }
        }
        logPrefillFailureTrace("poll_local_output_failed", prefill_context);
        return;
    }
    RTP_LLM_LOG_DEBUG("request [%ld] poll local output end", prefill_context.request_id);

    auto stream = prefill_context.getStream();
    if (stream->hasError()) {
        prefill_context.finished = true;
        auto error_info          = stream->statusInfo();
        setContextError(prefill_context, error_info);
        logPrefillFailureTrace("local_stream_failed", prefill_context);
    }
}

void PrefillRpcServer::remoteLoadCacheEnd(PrefillGenerateContext& prefill_context) {
    RTP_LLM_PROFILE_FUNCTION();
    GenerateOutputsPB load_response;
    CLIENT_GRPC_RET_IF_ERROR(
        prefill_context, prefill_context.client_stream->Read(&load_response), ErrorCode::REMOTE_LOAD_KV_CACHE_FAILED);
    auto error_code = transRPCErrorCode(load_response.error_info().error_code());

    // Decode has finished loading cache, now safe to release KV cache blocks.
    // This is called after cache store transfer is complete.
    if (prefill_context.generate_input->generate_config->pd_separation) {
        prefill_context.getStream()->releaseKVCacheForPDSep();
    }

    CLIENT_GRPC_RET_IF_ERROR(prefill_context, error_code == ErrorCode::NONE_ERROR, error_code);
    RTP_LLM_LOG_DEBUG("request [%ld] remote load cache done", prefill_context.request_id);

    prefill_context.dequeueStreamFromRuntimeMeta();
    if (!prefill_context.getStream()->hasEvent(StreamEvents::NeedRemoteGenerate)) {
        RTP_LLM_LOG_DEBUG("request [%ld] pd-sep prefill finished locally without remote generate, "
                          "skipping remote generate stages",
                          prefill_context.request_id);
        // Exit here to keep the remote load-cache completion and release ordering intact.
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
    auto context_position_ids = stream->getContextPositionIds();
    if (context_position_ids.defined()) {
        generate_request.mutable_position_ids()->CopyFrom(
            {context_position_ids.data_ptr<int32_t>(),
             context_position_ids.data_ptr<int32_t>() + context_position_ids.numel()});
    }
    if (engine_->isMTPEagle() && !engine_->isDSpark()) {
        RTP_LLM_CHECK_WITH_INFO(stream->getProposeToken().size() > 0,
                                "mtp remote generate propose token should not be empty");
    }
    generate_request.mutable_propose_token_ids()->CopyFrom(
        {stream->getProposeToken().begin(), stream->getProposeToken().end()});

    auto sp_output_buffer = stream->getSPOutputBuffer();

    if (sp_output_buffer && !engine_->isDSpark()) {
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
    auto              prefill_memory_reuse_len = prefill_context.getStream()->hostReuseLength();
    auto              prefill_disk_reuse_len   = prefill_context.getStream()->diskReuseLength();
    // Decode workers do not receive ViT features in PD mode, so preserve the
    // prefill-side media usage metadata when forwarding their responses.
    const auto multimodal_lengths =
        prefill_context.generate_input ? prefill_context.generate_input->multimodalLengths() : std::map<int, int>{};

    auto first_token_rt_us = prefill_context.getStream()->getTimeInfo().first_token_rt_us;
    while (prefill_context.client_stream->Read(&response)) {
        if (prefill_context.isRequestCancelled()) {
            RTP_LLM_LOG_WARNING("request [%ld] cancel by user", request_id);
            auto status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            setContextError(prefill_context, ErrorInfo(ErrorCode::CANCELLED, status.error_message()), status);
            return;
        }
        if (response.flatten_output().aux_info_size() == 0) {
            RTP_LLM_LOG_ERROR("request [%ld] generate output size is 0", request_id);
            break;
        }
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            response.mutable_flatten_output()->mutable_aux_info(i)->set_pd_sep(true);
        }
        mergeMultimodalLengths(response, multimodal_lengths);
        int64_t cost_time_us = currentTimeUs() - prefill_context.request_begin_time_us;
        for (size_t i = 0; i < response.flatten_output().aux_info_size(); i++) {
            auto decode_total_reuse_len  = response.flatten_output().aux_info(i).total_reuse_len();
            auto decode_local_reuse_len  = response.flatten_output().aux_info(i).local_reuse_len();
            auto decode_remote_reuse_len = response.flatten_output().aux_info(i).remote_reuse_len();
            auto decode_memory_reuse_len = response.flatten_output().aux_info(i).memory_reuse_len();
            auto decode_disk_reuse_len   = response.flatten_output().aux_info(i).disk_reuse_len();

            response.mutable_flatten_output()->mutable_aux_info(i)->set_first_token_cost_time_us(first_token_rt_us);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_cost_time_us(cost_time_us);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_total_reuse_len(prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_local_reuse_len(prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_remote_reuse_len(prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_memory_reuse_len(prefill_memory_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_disk_reuse_len(prefill_disk_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_total_reuse_len(
                prefill_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_local_reuse_len(
                prefill_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_remote_reuse_len(
                prefill_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_memory_reuse_len(
                prefill_memory_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_prefill_disk_reuse_len(
                prefill_disk_reuse_len);

            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_total_reuse_len(decode_total_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_local_reuse_len(decode_local_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_remote_reuse_len(
                decode_remote_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_memory_reuse_len(
                decode_memory_reuse_len);
            response.mutable_flatten_output()->mutable_aux_info(i)->set_decode_disk_reuse_len(decode_disk_reuse_len);
        }
        if (!prefill_context.rpc_context.writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%ld] write outputs pb failed", request_id);
            // Both the synchronous gRPC writer and ResponseBufferWriter return false when their downstream
            // consumer is closed or cancelled. Treat this as cancellation so closeGrpcStream() calls TryCancel().
            setContextError(prefill_context, ErrorInfo(ErrorCode::CANCELLED, "request write outputs pb failed"));
            return;
        }
    }
    auto status = prefill_context.closeGrpcStream();
    if (!status.ok() && status.error_code() != grpc::StatusCode::CANCELLED) {
        CLIENT_GRPC_RET_IF_ERROR(prefill_context, false, ErrorCode::REMOTE_GENERATE_FAILED);
    }
}

void PrefillRpcServer::mergeMultimodalLengths(GenerateOutputsPB&        response,
                                              const std::map<int, int>& multimodal_lengths) {
    if (multimodal_lengths.empty()) {
        return;
    }
    for (int i = 0; i < response.flatten_output().aux_info_size(); ++i) {
        auto* output_lengths = response.mutable_flatten_output()->mutable_aux_info(i)->mutable_multimodal_lengths();
        output_lengths->clear();
        for (const auto& [type, length] : multimodal_lengths) {
            (*output_lengths)[type] = length;
        }
    }
}

grpc::Status PrefillRpcServer::prepareAllocateResource(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(getRpcConnection, prefill_context);
    EXECUTE_STAGE_FUNC(multimodalProcess, prefill_context);
    EXECUTE_STAGE_FUNC(remoteAllocateResource, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::syncPrefix(PrefillGenerateContext& prefill_context) {
    auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;
    int  retry_interval_ms    = 1;

    EXECUTE_WITH_RETRY(
        prepareAllocateResource, prefill_context, max_retry_times, max_retry_timeout_ms, retry_interval_ms);
    if (prefill_context.hasError()) {
        logPrefillFailureTrace("prepare_allocate_failed", prefill_context);
        RTP_LLM_LOG_WARNING(
            "request [%ld] prepare allocate resource failed after retry [%ld] times, cost time ms [%ld], "
            "max retry time [%ld], max retry timeout ms [%ld], retryable [%d], error code [%d:%s]",
            prefill_context.request_id,
            prefill_context.retry_times,
            prefill_context.retry_cost_time_ms,
            max_retry_times + 1,
            max_retry_timeout_ms,
            prefill_context.shouldRetry(),
            static_cast<int>(prefill_context.error_info.code()),
            ErrorCodeToString(prefill_context.error_info.code()).c_str());
        return prefill_context.error_status;
    }
    EXECUTE_STAGE_FUNC(enqueueRequest, prefill_context);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::finishStream(PrefillGenerateContext& prefill_context) {
    EXECUTE_STAGE_FUNC(remoteLoadCacheStart, prefill_context);
    EXECUTE_STAGE_FUNC(pollLocalOutput, prefill_context);
    EXECUTE_STAGE_FUNC(remoteLoadCacheEnd, prefill_context);
    EXECUTE_STAGE_FUNC(remoteGenerate, prefill_context);
    EXECUTE_STAGE_FUNC(pollRemoteOutput, prefill_context);
    prefill_context.stat_info.nextStage();
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::preferPriorityPreemption(PrefillGenerateContext& prefill_context,
                                                        const grpc::Status&     fallback) {
    if (!prefill_context.isPriorityPreempted()) {
        return fallback;
    }
    return serializeErrorMsg(prefill_context.request_key,
                             prefill_context.request_info,
                             ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request"));
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
    if (prefillTraceLogEnabled()) {
        RTP_LLM_LOG_INFO(
            "Prefill request trace: event=recv request_id=%ld pd_separation=%d token_ids=%d "
            "max_new_tokens=%d num_beams=%d num_return_sequences=%d can_use_pd_separation=%d timeout_ms=%ld",
            request->request_id(),
            pd_separation,
            request->token_ids_size(),
            request->generate_config().max_new_tokens(),
            request->generate_config().num_beams(),
            request->generate_config().num_return_sequences(),
            request->generate_config().can_use_pd_separation(),
            static_cast<int64_t>(request->generate_config().timeout_ms()));
    }
    if (!pd_separation) {
        if (prefillTraceLogEnabled()) {
            RTP_LLM_LOG_INFO("Prefill request trace: event=bypass_local request_id=%ld token_ids=%d",
                             request->request_id(),
                             request->token_ids_size());
        }
        return LocalRpcServer::GenerateStreamCall(server_context, request, writer);
    }

    AtomicGuardPtr request_guard = make_shared<AtomicGuard>(onflight_requests_);
    RPCContext     rpc_context{request, writer};
    auto           prefill_context         = PrefillGenerateContext(&this->resource(),
                                                  rpc_context,
                                                  request->generate_config().timeout_ms(),
                                                  server_context,
                                                  metrics_reporter_,
                                                  meta_,
                                                  maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms);
    prefill_context.onflight_requests      = onflight_requests_;
    prefill_context.loading_cache_requests = loading_cache_requests_;

    // Prefill SERVER span is created only on the PD path, AFTER the fallback
    // check above, so Local/Prefill each own exactly one SERVER span. RAII
    // guard covers EXECUTE_STAGE_FUNC early returns and exceptions.
    if (telemetry::TelemetryRuntime::isActive()) {
        auto span = telemetry::startRpcServerSpan(
            "rtp_llm.prefill_generate_stream_call", server_context, true, "RpcService/GenerateStreamCall");
        prefill_context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &prefill_context.error_status);
        // Bailian Unitrace index key (string) + internal numeric field
        prefill_context.trace_span_guard->setAttribute(telemetry::kAttrRequestId,
                                                       std::to_string(prefill_context.request_id));
        prefill_context.trace_span_guard->setAttribute(telemetry::kAttrRtpLlmRequestId, prefill_context.request_id);
    }
    telemetry::PhaseSpanSynthesisScope phase_span_scope([&prefill_context](bool exception_unwinding) {
        if (!prefill_context.trace_span_guard || !prefill_context.trace_span_guard->valid()) {
            return;
        }
        auto& stream = prefill_context.getStream();
        if (!stream) {
            return;
        }
        const auto             time_info  = stream->getTimeInfo();
        const bool             request_ok = prefill_context.error_status.ok() && !exception_unwinding;
        telemetry::PhaseTiming phase_timing;
        phase_timing.begin_time_us           = time_info.begin_time_us;
        phase_timing.running_started         = time_info.running_started;
        phase_timing.running_started_time_us = time_info.running_started_time_us;
        phase_timing.first_token_committed   = time_info.first_token_committed;
        phase_timing.first_token_time_us     = time_info.first_token_time_us;
        phase_timing.generation_done         = time_info.generation_done;
        phase_timing.generation_done_time_us = time_info.generation_done_time_us;
        phase_timing.synthesis_end_time_us   = currentTimeUs();
        phase_timing.request_id              = prefill_context.request_id;
        phase_timing.error_type              = request_ok ?
                                                   nullptr :
                                                   (!prefill_context.error_status.ok() ?
                                                        telemetry::grpcStatusCodeName(prefill_context.error_status.error_code()) :
                                                        "Exception");
        telemetry::synthesizePhaseSpans(
            prefill_context.trace_span_guard->sharedSpan(), phase_timing, telemetry::PhaseRole::Prefill, request_ok);
        if (request_ok && time_info.generation_done) {
            telemetry::setUsageTokenAttributes(
                *prefill_context.trace_span_guard, (int64_t)stream->inputLength(), (int64_t)stream->outputTokenLen());
        }
    });

    try {
        auto status = syncPrefix(prefill_context);
        if (!status.ok()) {
            return status;
        }
        status = finishStream(prefill_context);
        if (!status.ok()) {
            return status;
        }
    } catch (const std::exception& e) {
        auto error_msg = "request [" + prefill_context.request_key + "] catch exception [" + e.what() + "]";
        setContextError(prefill_context, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_msg));
        logPrefillFailureTrace("catch_exception", prefill_context);
        return prefill_context.error_status;
    } catch (...) {
        auto error_msg = "request [" + prefill_context.request_key + "] catch unknown exception";
        setContextError(prefill_context, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_msg));
        logPrefillFailureTrace("catch_unknown_exception", prefill_context);
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

grpc::Status
PrefillRpcServer::Cancel(grpc::ServerContext* /*context*/, const CancelRequestPB* request, CancelResponsePB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    if (request == nullptr || request->request_id() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
    }
    const auto result = onCancelRequest(request->request_id());
    switch (result) {
        case PriorityCancelResult::ACCEPTED:
            response->set_status(CancelStatusPB::CANCEL_STATUS_ACCEPTED);
            RTP_LLM_LOG_DEBUG("request [%ld] priority-preemption cancel accepted", request->request_id());
            break;
        case PriorityCancelResult::TOMBSTONED:
            response->set_status(CancelStatusPB::CANCEL_STATUS_TOMBSTONED);
            RTP_LLM_LOG_DEBUG("request [%ld] priority-preemption cancel tombstoned", request->request_id());
            break;
        case PriorityCancelResult::NOT_FOUND:
            response->set_status(CancelStatusPB::CANCEL_STATUS_NOT_FOUND);
            RTP_LLM_LOG_DEBUG("request [%ld] priority-preemption cancel not found", request->request_id());
            break;
    }
    return grpc::Status::OK;
}

}  // namespace rtp_llm
