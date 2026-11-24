#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <thread>
#include <c10/core/InferenceMode.h>
#include "autil/Scope.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MMProcessorConfig.h"
#include "rtp_llm/cpp/model_rpc/BatchStreamOutputCollector.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/telemetry/PhaseSpanSynthesizer.h"

using namespace std;

namespace rtp_llm {

namespace {
constexpr int64_t kRpcOutputWaitTimeoutMs        = 500;
constexpr size_t  kAllocatorDumpReplayHistoryMax = 1024;

std::string batchTraceMetadata(const grpc::ServerContext* context, const char* key, size_t max_size) {
    if (context == nullptr) {
        return {};
    }
    const auto& metadata = context->client_metadata();
    const auto  it       = metadata.find(key);
    if (it == metadata.end() || it->second.size() > max_size) {
        return {};
    }
    return std::string(it->second.data(), it->second.size());
}

class BatchItemTraceCarrier: public opentelemetry::context::propagation::TextMapCarrier {
public:
    explicit BatchItemTraceCarrier(const TraceContextPB& context): context_(context) {}
    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
        if (key == "traceparent" && context_.traceparent().size() <= 256) {
            return context_.traceparent();
        }
        if (key == "tracestate" && context_.tracestate().size() <= 512) {
            return context_.tracestate();
        }
        return "";
    }
    void Set(opentelemetry::nostd::string_view, opentelemetry::nostd::string_view) noexcept override {}

private:
    const TraceContextPB& context_;
};

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>
startBatchItemSpan(const GenerateInputPB&                                              input,
                   const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& rpc_span) {
    if (!telemetry::TelemetryRuntime::isActive()) {
        return {};
    }
    try {
        opentelemetry::trace::StartSpanOptions options;
        options.kind = opentelemetry::trace::SpanKind::kInternal;
        if (rpc_span) {
            options.parent = rpc_span->GetContext();
        }
        if (input.request_info().has_trace_context()) {
            BatchItemTraceCarrier carrier(input.request_info().trace_context());
            auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
            opentelemetry::context::Context empty_context;
            auto                            parent = propagator->Extract(carrier, empty_context);
            if (opentelemetry::trace::GetSpan(parent)->GetContext().IsValid()) {
                options.parent = parent;
            }
        }
        return telemetry::TelemetryRuntime::tracer()->StartSpan("rtp_llm.batch_generate_request", options);
    } catch (...) {
        return {};
    }
}

void recordBatchItemCompletion(telemetry::GrpcStatusSpanGuard&        guard,
                               const std::shared_ptr<GenerateStream>& stream,
                               bool                                   request_ok,
                               const char*                            error_type) {
    if (!guard.valid()) {
        return;
    }
    const auto             time_info = stream->getTimeInfo();
    telemetry::PhaseTiming timing;
    timing.begin_time_us           = time_info.begin_time_us;
    timing.running_started         = time_info.running_started;
    timing.running_started_time_us = time_info.running_started_time_us;
    timing.first_token_committed   = time_info.first_token_committed;
    timing.first_token_time_us     = time_info.first_token_time_us;
    timing.generation_done         = time_info.generation_done;
    timing.generation_done_time_us = time_info.generation_done_time_us;
    timing.synthesis_end_time_us   = currentTimeUs();
    timing.request_id              = stream->streamId();
    timing.error_type              = error_type;
    if (request_ok && time_info.generation_done) {
        const auto sequence_count =
            stream->hasNumBeams() ? std::max(stream->numReturnSequences(), 1) : stream->currentBatchSize();
        telemetry::setUsageTokenAttributes(
            guard, (int64_t)stream->inputLength(), (int64_t)(stream->outputTokenLen() * sequence_count));
    }
    telemetry::synthesizePhaseSpans(guard.sharedSpan(), timing, telemetry::PhaseRole::Fusion, request_ok);
}

std::string endpointHost(std::string endpoint) {
    const auto scheme_separator = endpoint.find(':');
    if (endpoint.compare(0, scheme_separator, "ipv4") == 0 || endpoint.compare(0, scheme_separator, "ipv6") == 0) {
        endpoint.erase(0, scheme_separator + 1);
    }
    if (!endpoint.empty() && endpoint.front() == '[') {
        const auto closing_bracket = endpoint.find(']');
        return closing_bracket == std::string::npos ? std::string() : endpoint.substr(1, closing_bracket - 1);
    }
    const auto port_separator = endpoint.rfind(':');
    return port_separator == std::string::npos ? endpoint : endpoint.substr(0, port_separator);
}

bool constantTimeEquals(const std::string& lhs, const std::string& rhs) {
    size_t       difference = lhs.size() ^ rhs.size();
    const size_t length     = std::max(lhs.size(), rhs.size());
    for (size_t index = 0; index < length; ++index) {
        const unsigned char left  = index < lhs.size() ? static_cast<unsigned char>(lhs[index]) : 0;
        const unsigned char right = index < rhs.size() ? static_cast<unsigned char>(rhs[index]) : 0;
        difference |= left ^ right;
    }
    return difference == 0;
}

bool validAllocatorDumpId(const std::string& dump_id) {
    if (dump_id.empty() || dump_id.size() > 64) {
        return false;
    }
    return std::all_of(dump_id.begin(), dump_id.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'A' && character <= 'Z')
               || (character >= 'a' && character <= 'z') || character == '-';
    });
}

std::string formatRequestLogTag(const std::string& request_key, const RequestInfo& request_info) {
    std::string tag = "request [" + request_key + "]";
    if (!request_info.trace_id.empty()) {
        tag += ", trace id [" + request_info.trace_id + "]";
    }
    if (!request_info.request_id.empty()) {
        tag += ", source request id [" + request_info.request_id + "]";
    }
    if (!request_info.frontend_ip.empty()) {
        tag += ", frontend ip [" + request_info.frontend_ip + "]";
    }
    if (!request_info.dash_ip.empty()) {
        tag += ", dash ip [" + request_info.dash_ip + "]";
    }
    if (!request_info.source_role.empty()) {
        tag += ", source role [" + request_info.source_role + "]";
    }
    return tag;
}

class BatchStreamCleanupGuard {
public:
    explicit BatchStreamCleanupGuard(std::vector<GenerateStreamPtr>& streams): streams_(streams) {}

    ~BatchStreamCleanupGuard() {
        if (dismissed_ || streams_.empty()) {
            return;
        }

        size_t cancelled_streams = 0;
        for (size_t i = handled_stream_count_; i < streams_.size(); ++i) {
            const auto& stream = streams_[i];
            if (!stream) {
                continue;
            }
            try {
                if (stream->getStatus() == StreamState::FINISHED || stream->hasError()
                    || stream->hasEvent(StreamEvents::GenerateDone)
                    || stream->hasEvent(StreamEvents::PrefillHandoffReady)) {
                    continue;
                }
                stream->reportError(ErrorCode::CANCELLED, "batch generate call exited before all streams finished");
                ++cancelled_streams;
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("failed to cancel batch stream [%ld]: %s", stream->streamId(), e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("failed to cancel batch stream [%ld]: unknown exception", stream->streamId());
            }
        }
        RTP_LLM_LOG_WARNING("BatchGenerateCall cleanup completed, cancelled_streams=%zu", cancelled_streams);
    }

    void markHandled(size_t stream_index) noexcept {
        if (handled_stream_count_ <= stream_index) {
            handled_stream_count_ = stream_index + 1;
        }
    }

    void dismiss() noexcept {
        dismissed_ = true;
    }

private:
    std::vector<GenerateStreamPtr>& streams_;
    size_t                          handled_stream_count_{0};
    bool                            dismissed_{false};
};

}  // namespace

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                  py::object                                    mm_process_engine) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_                      = maga_init_params;
    weight_manager_                        = maga_init_params.weight_manager;
    metrics_reporter_                      = maga_init_params.metrics_reporter;
    torch_allocator_dump_enabled_          = false;
    torch_allocator_dump_auth_token_       = "";
    torch_allocator_dump_cooldown_seconds_ = 60.0;
    {
        std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
        torch_allocator_dump_in_progress_              = false;
        torch_allocator_dump_has_completed_            = false;
        torch_allocator_dump_active_started_by_public_ = false;
        torch_allocator_dump_active_internal_started_  = false;
        torch_allocator_dump_active_id_.clear();
        torch_allocator_dump_ids_.clear();
        torch_allocator_dump_id_order_.clear();
    }
    if (!maga_init_params_.server_config.is_none()) {
        const auto& server_config = maga_init_params_.server_config;
        if (py::hasattr(server_config, "enable_torch_allocator_dump")) {
            torch_allocator_dump_enabled_ = server_config.attr("enable_torch_allocator_dump").cast<bool>();
        }
        if (py::hasattr(server_config, "torch_allocator_dump_auth_token")) {
            torch_allocator_dump_auth_token_ =
                server_config.attr("torch_allocator_dump_auth_token").cast<std::string>();
        }
        if (py::hasattr(server_config, "torch_allocator_dump_cooldown_seconds")) {
            torch_allocator_dump_cooldown_seconds_ =
                server_config.attr("torch_allocator_dump_cooldown_seconds").cast<double>();
        }
    }
    if (!std::isfinite(torch_allocator_dump_cooldown_seconds_) || torch_allocator_dump_cooldown_seconds_ < 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "allocator dump cooldown must be finite and non-negative");
    }
    if (torch_allocator_dump_enabled_ && torch_allocator_dump_auth_token_.empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "allocator dump authentication token is required when enabled");
    }
    RTP_LLM_LOG_INFO("LocalRpcServer aux_string %s", maga_init_params_.misc_config.aux_string.c_str());
    propose_maga_init_params_ = propose_params.get();
    if (maga_init_params_.parallelism_config.tp_rank == 0
        && !maga_init_params_.runtime_config.worker_grpc_addrs.empty()) {
        tp_broadcaster_ = std::make_shared<BroadcastManager>(maga_init_params_.runtime_config.worker_grpc_addrs);
        if (!tp_broadcaster_->init()) {
            RTP_LLM_LOG_WARNING("failed to init TP broadcaster");
            tp_broadcaster_.reset();
        }
    }

    const auto mm_decision = resolveAndLogMMProcessorKind(maga_init_params.model_config_.mm_model_config.is_multimodal,
                                                          maga_init_params.vit_config.vit_separation,
                                                          !mm_process_engine.is_none(),
                                                          maga_init_params.pd_sep_config.role_type,
                                                          maga_init_params.parallelism_config.tp_rank,
                                                          maga_init_params.parallelism_config.pp_rank,
                                                          maga_init_params.model_config_.model_type,
                                                          "LocalRpcServer");
    const auto mm_kind     = mm_decision.kind;
    if (!mm_decision.ok()) {
        RTP_LLM_LOG_ERROR("%s", mm_decision.error.c_str());
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, mm_decision.error);
    }

    {
        pybind11::gil_scoped_release release;
        RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                "running engine init with gil held may cause program hang, please check");
        engine_.reset(new NormalEngine(maga_init_params, std::move(propose_params)));
    }
    if (mm_kind == MMProcessorKind::LOCAL) {
        mm_processor_.reset(new LocalMultimodalProcessor(mm_process_engine,
                                                         maga_init_params.model_config_.mm_model_config,
                                                         maga_init_params.model_config_.max_seq_len));
    } else if (mm_kind == MMProcessorKind::REMOTE) {
        mm_processor_.reset(new RemoteMultimodalProcessor(maga_init_params.model_config_.mm_model_config,
                                                          maga_init_params.model_config_.max_seq_len,
                                                          maga_init_params.vit_config.output_transport,
                                                          metrics_reporter_,
                                                          maga_init_params.parallelism_config.local_rank));
    }

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    return serializeErrorMsg(request_key, RequestInfo(), error_info);
}

grpc::Status
LocalRpcServer::serializeErrorMsg(const string& request_key, const RequestInfo& request_info, ErrorInfo error_info) {
    const auto error_msg   = error_info.ToString();
    auto       log_message = error_msg;
    const auto role        = maga_init_params_.pd_sep_config.role_type;
    if (role == RoleType::PREFILL || role == RoleType::DECODE) {
        log_message = "reporter_role=" + roleTypeToString(role) + " stage=rpc_response request_key=" + request_key
                      + " tp_rank=" + std::to_string(maga_init_params_.parallelism_config.tp_rank)
                      + (request_info.request_id.empty() ? "" : " source_request_id=" + request_info.request_id)
                      + (request_info.trace_id.empty() ? "" : " trace_id=" + request_info.trace_id) + ": " + error_msg;
    }
    const auto request_log_tag = formatRequestLogTag(request_key, request_info);
    RTP_LLM_LOG_WARNING("%s, error_code=%d error_name=%s error_message=[%s]",
                        request_log_tag.c_str(),
                        static_cast<int>(error_info.code()),
                        ErrorCodeToString(error_info.code()).c_str(),
                        log_message.c_str());
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        RTP_LLM_LOG_WARNING("%s error details serialize to string failed", request_log_tag.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*                         context,
                                              const string&                                request_key,
                                              WriterInterface*                             writer,
                                              std::shared_ptr<GenerateStream>&             stream,
                                              const std::function<FirstError::Snapshot()>& check_remote) {
    RTP_LLM_PROFILE_FUNCTION();
    // Observe local and remote failures before waiting for another token.
    // Decode replays the Prefill token into its queue even for non-streaming
    // requests. Collect these internal blocks into one RPC output while retaining
    // optional first-token tensors; streaming and other roles keep block semantics.
    const bool  pd_decode             = stream->queryPdSep() && stream->resourceContext().role_type == RoleType::DECODE;
    const bool  collect_pd_output     = pd_decode && !stream->isStreaming();
    const bool  measure_token_output  = pd_decode && stream->isStreaming() && metrics_reporter_;
    int64_t     first_token_write_us  = 0;
    bool        second_token_reported = false;
    const auto& config                = *stream->generateConfig();
    BatchStreamOutputCollector complete_output(config.aux_info ? config.logits_index : std::nullopt);
    while (true) {
        auto remote_error = check_remote ? check_remote() : FirstError::Snapshot{};
        auto first        = FirstError::earlier(stream->firstError(), remote_error);
        if (first.error.hasError()) {
            stream->reportError(first.error.code(), first.error.ToString());
            return serializeErrorMsg(request_key, stream->generateInput()->request_info, first.error);
        }
        const auto result = stream->nextOutputForRpc(kRpcOutputWaitTimeoutMs);
        if (isCancelled(context)) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!result.ok()) {
            if (result.status().code() == ErrorCode::OUTPUT_QUEUE_NO_UPDATE) {
                continue;
            }
            if (result.status().code() == ErrorCode::FINISHED) {
                if (collect_pd_output) {
                    return serializeErrorMsg(request_key,
                                             stream->generateInput()->request_info,
                                             ErrorInfo(ErrorCode::UNKNOWN_ERROR,
                                                       "non-streaming PD request finished without terminal output"));
                }
                break;
            }
            return serializeErrorMsg(request_key, stream->generateInput()->request_info, result.status());
        }
        auto       output          = result.value();
        const bool terminal_output = !output.generate_outputs.empty()
                                     && std::all_of(output.generate_outputs.begin(),
                                                    output.generate_outputs.end(),
                                                    [](const auto& item) { return item.finished; });
        if (collect_pd_output && !output.frontend_metric_only) {
            const auto status = complete_output.add(std::move(output));
            if (!status.ok()) {
                stream->reportError(status.code(), status.ToString());
                return serializeErrorMsg(request_key, stream->generateInput()->request_info, status);
            }
            if (!terminal_output) {
                continue;
            }
            output = complete_output.finish();
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;

        QueryConverter::transResponse(&outputs_pb,
                                      &output,
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        updateAuxInfo(outputs_pb, stream);
        // Packaging runs without the stream lock and can overlap cancellation.
        // Recheck before handing a completed terminal response to the transport.
        if (isCancelled(context)) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        int64_t output_len = 0;
        if (measure_token_output && !second_token_reported && !output.frontend_metric_only) {
            for (const auto& item : output.generate_outputs) {
                if (item.output_ids.defined() && item.output_ids.numel() > 0) {
                    output_len = std::max<int64_t>(output_len, item.aux_info.output_len);
                }
            }
        }
        // Measure the start of the write carrying the token, and only report accepted writes.
        const auto write_time_us = output_len > 0 ? currentTimeUs() : 0;
        if (!writer->Write(outputs_pb)) {
            stream->reportError(ErrorCode::CANCELLED, "write outputs pb failed");
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            // WriterInterface uses false to signal that its downstream consumer has closed or cancelled.
            return grpc::Status(grpc::StatusCode::CANCELLED, "request output consumer closed");
        }
        if (write_time_us > 0) {
            if (first_token_write_us == 0) {
                first_token_write_us = write_time_us;
            }
            if (output_len >= 2) {
                second_token_reported = true;
                P2PConnectorMetricsCollector collector;
                collector.decode_first_to_second_token_output_us = write_time_us - first_token_write_us;
                const auto handoff_ready_us                      = stream->p2pHandoffReadyTimeUs();
                if (handoff_ready_us > 0) {
                    collector.decode_handoff_ready_to_second_token_output_us = write_time_us - handoff_ready_us;
                }
                metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &collector);
            }
        }
        if (!output.frontend_metric_only && stream->hasEvent(StreamEvents::PrefillHandoffReady)) {
            if (stream->queryPdSep() && stream->resourceContext().role_type == RoleType::PREFILL) {
                while (!isCancelled(context) && stream->getStatus() != StreamState::FINISHED) {
                    if (stream->checkTimeoutAndHasError()) {
                        return serializeErrorMsg(request_key, stream->statusInfo());
                    }
                    if (stream->deadlineMs() <= currentTimeMs()) {
                        stream->reportError(ErrorCode::GENERATE_TIMEOUT,
                                            "Prefill handoff completion deadline exceeded");
                        return serializeErrorMsg(request_key, stream->statusInfo());
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (isCancelled(context)) {
                    stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
                    RTP_LLM_LOG_WARNING("request [%s] cancelled by user during Prefill completion wait",
                                        request_key.c_str());
                    return serializeErrorMsg(request_key, stream->statusInfo());
                }
            }
            break;
        }
        if ((collect_pd_output && !output.frontend_metric_only) || (check_remote && terminal_output)) {
            // StartLoad already delivered the handoff. The companion Prefill
            // RPC may still be draining its own output and cleanup.
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("request [%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::pollBatchStreamOutput(grpc::ServerContext*                                  context,
                                                   const std::vector<GenerateStreamPtr>&                 streams,
                                                   BatchGenerateOutputsPB*                               response,
                                                   const std::function<void(int64_t, const ErrorInfo&)>& fail_item) {
    std::vector<BatchStreamOutputCollector> collectors;
    collectors.reserve(streams.size());
    for (const auto& stream : streams) {
        collectors.emplace_back(stream && stream->generateConfig()->aux_info ? stream->generateConfig()->logits_index :
                                                                               std::nullopt);
    }
    std::vector<bool> settled(streams.size(), false);
    while (true) {
        if (context && context->IsCancelled()) {
            return grpcStatusFromErrorInfo({ErrorCode::CANCELLED, "BatchGenerateCall cancelled"});
        }
        bool all_finished = true;
        bool progress     = false;
        for (size_t i = 0; i < streams.size(); ++i) {
            auto stream = streams[i];
            if (!stream || settled[i])
                continue;

            const auto fail = [&](const ErrorInfo& error) {
                fail_item(stream->generateInput()->request_id, error);
                settled[i] = true;
            };
            if (stream->hasError()) {
                fail(stream->statusInfo());
                continue;
            }

            bool terminal_output = false;
            if (stream->hasOutput()) {
                auto output = stream->nextOutput();
                if (!output.ok() && output.status().code() != ErrorCode::FINISHED) {
                    fail(output.status());
                    continue;
                }
                if (output.ok()) {
                    const auto& rows = output.value().generate_outputs;
                    terminal_output  = !rows.empty() && std::all_of(rows.begin(), rows.end(), [](const auto& row) {
                        return row.finished;
                    });
                    auto status      = collectors[i].add(std::move(output.value()));
                    if (!status.ok()) {
                        fail(status);
                        continue;
                    }
                    progress = true;
                }
            }
            if (terminal_output
                || (stream->isFinished() && !stream->hasOutput() && !stream->hasPendingFrontendMetrics())) {
                if (!terminal_output) {
                    fail({ErrorCode::EXECUTION_EXCEPTION, "batch item completed without terminal output"});
                    continue;
                }
                auto  output = collectors[i].finish();
                auto* pb     = response->mutable_results(static_cast<int>(i))->mutable_final_output();
                QueryConverter::transResponse(pb,
                                              &output,
                                              stream->generateConfig()->aux_info,
                                              maga_init_params_.misc_config.aux_string,
                                              stream->specialTokens().eos_token_id);
                updateAuxInfo(*pb, stream);
                settled[i] = true;
            } else if (stream->deadlineMs() <= currentTimeMs()) {
                fail({ErrorCode::GENERATE_TIMEOUT, "batch item deadline exceeded"});
            } else {
                all_finished = false;
            }
        }
        if (all_finished)
            return grpc::Status::OK;
        if (!progress)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

ErrorInfo LocalRpcServer::prepareInput(const GenerateInputPB&                                              input_pb,
                                       std::shared_ptr<GenerateInput>&                                     output,
                                       const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                                       grpc::ServerContext* server_context) {
    output.reset();
    try {
        output                 = QueryConverter::transQuery(&input_pb);
        const auto& model_type = maga_init_params_.model_config_.model_type;
        if ((model_type == "qwen4_exp" || model_type == "qwen4_exp_mtp") && output->v41_inputs) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR,
                             "Qwen3.8-Flash-Next requires ordinary image/video media inputs");
        }
        if (mm_processor_ != nullptr && (output->multimodal_inputs || output->v41_inputs)) {
            RTP_LLM_PROFILE_SCOPE("rpc.mm_update_features");
            auto mm_res = updateMultimodalFeaturesWithTrace(*mm_processor_, output, parent_span, server_context);
            if (!mm_res.ok()) {
                return mm_res;
            }
        }
    } catch (const std::invalid_argument& e) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, e.what());
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, e.what());
    } catch (...) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "input preparation failed with an unknown exception");
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo LocalRpcServer::collectStreamOutput(grpc::ServerContext*                  context,
                                              std::shared_ptr<GenerateStream>&      stream,
                                              const std::shared_ptr<GenerateInput>& input,
                                              GenerateOutputs&                      last_outputs) {
    std::vector<std::pair<uint64_t, double>> frontend_sp_tpot_samples;
    while (true) {
        const auto output_result = stream->nextOutputForRpc(kRpcOutputWaitTimeoutMs);
        if (isCancelled(context)) {
            const ErrorInfo cancellation(ErrorCode::CANCELLED, "request cancelled by client");
            stream->reportError(cancellation.code(), cancellation.ToString());
            return cancellation;
        }
        if (!output_result.ok()) {
            if (output_result.status().code() == ErrorCode::OUTPUT_QUEUE_NO_UPDATE) {
                continue;
            }
            if (output_result.status().code() != ErrorCode::FINISHED) {
                return output_result.status();
            }
            break;
        }
        const auto& output = output_result.value();
        frontend_sp_tpot_samples.insert(frontend_sp_tpot_samples.end(),
                                        output.frontend_sp_tpot_samples.begin(),
                                        output.frontend_sp_tpot_samples.end());
        if (!output.frontend_metric_only) {
            last_outputs = output;
        }
    }
    last_outputs.frontend_sp_tpot_samples = std::move(frontend_sp_tpot_samples);
    return ErrorInfo::OkStatus();
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) {
    return generateLocalStream(context, request, writer, nullptr);
}

grpc::Status LocalRpcServer::generateLocalStream(grpc::ServerContext*   context,
                                                 const GenerateInputPB* request,
                                                 WriterInterface*       writer,
                                                 PDCancelRegistry*      cancel_registry) try {
    RTP_LLM_PROFILE_SCOPE("rpc.generate_stream_call");
    c10::InferenceMode inference_guard(true);
    AtomicGuard        request_guard(onflight_requests_);
    if (request == nullptr) {
        return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::INVALID_PARAMS, "generate request must not be null"));
    }
    auto request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    PDCancelRegistry::Handle                   cancel_handle;
    std::optional<PDCancelRegistry::CallGuard> cancel_guard;
    auto                                       generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_, meta_);
    generate_context.onflight_requests    = &onflight_requests_;
    const int         uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    // gRPC SERVER span doubles as the request span on the fusion path; guard
    // destruction covers CHECK_ERROR_STATUS early returns.
    if (telemetry::TelemetryRuntime::isActive()) {
        auto span = telemetry::startRpcServerSpan(
            "rtp_llm.generate_stream_call", context, false, "RpcService/GenerateStreamCall");
        generate_context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &generate_context.error_status);
        // `request_id` is the Bailian Unitrace index key (string, verified);
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrRequestId, std::to_string(request_id));
    }
    telemetry::PhaseSpanSynthesisScope phase_span_scope([&generate_context](bool exception_unwinding) {
        generate_context.synthesizeTracePhases(telemetry::PhaseRole::Fusion, exception_unwinding);
    });
    if (cancel_registry) {
        const auto timeout_ms = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(),
                                                      maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
        // A fallback has no P2P handoff key. Admission still fences a Cancel that arrived first.
        const auto admission = cancel_registry->admit(*request, "", "", currentTimeMs() + timeout_ms, cancel_handle);
        if (!admission.ok()) {
            generate_context.recordRpcFailure(admission);
            return generate_context.error_status;
        }
        cancel_guard.emplace(*cancel_registry, cancel_handle);
    }
    std::shared_ptr<GenerateInput> input;
    {
        auto mm_res = prepareInput(*request,
                                   input,
                                   generate_context.trace_span_guard ?
                                       generate_context.trace_span_guard->sharedSpan() :
                                       opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{},
                                   context);
        if (!mm_res.ok()) {
            generate_context.error_info = mm_res;
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, mm_res);
        }
    }
    if (input) {
        generate_context.request_info = input->request_info;
    }
    if (cancel_handle && cancel_handle->canceled.load()) {
        generate_context.recordRpcFailure(cancel_handle->cancel_reason);
    }
    CHECK_ERROR_STATUS(generate_context);
    if (writer == nullptr) {
        generate_context.error_info = ErrorInfo(ErrorCode::INVALID_PARAMS, "generate response writer must not be null");
        generate_context.error_status =
            serializeErrorMsg(generate_context.request_key, generate_context.request_info, generate_context.error_info);
        return generate_context.error_status;
    }
    if (!engine_) {
        throw std::runtime_error("LocalRpcServer engine is not initialized");
    }

    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    {
        RTP_LLM_PROFILE_SCOPE("rpc.enqueue_engine");
        if (cancel_handle) {
            auto stream = engine_->makeStream(input);
            generate_context.setStream(stream);
            cancel_registry->attach(cancel_handle, stream);
            if (stream->hasError()) {
                stream->finishWithoutGenerate();
            } else {
                engine_->enqueue(stream);
            }
        } else {
            generate_context.setStream(engine_->enqueue(input));
        }
    }
    if (generate_context.getStream()->hasError()) {
        generate_context.error_status =
            serializeErrorMsg(generate_context.request_key, generate_context.getStream()->statusInfo());
    }
    CHECK_ERROR_STATUS(generate_context);

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key, writer, generate_context.getStream());
    if (generate_context.hasError() && generate_context.getStream() && generate_context.getStream()->hasError()) {
        generate_context.error_info = generate_context.getStream()->statusInfo();
    }
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
} catch (const std::exception& e) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(request_key,
                             ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                       "GenerateStreamCall failed with exception: " + std::string(e.what())));
} catch (...) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(
        request_key, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "GenerateStreamCall failed with unknown exception"));
}

grpc::Status LocalRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                               const BatchGenerateInputPB* request,
                                               BatchGenerateOutputsPB*     response) {
    return generateLocalBatch(context, request, response, nullptr);
}

grpc::Status LocalRpcServer::generateLocalBatch(grpc::ServerContext*        context,
                                                const BatchGenerateInputPB* request,
                                                BatchGenerateOutputsPB*     response,
                                                PDCancelRegistry*           cancel_registry) {
    grpc::Status rpc_status = grpc::Status::OK;
    auto         rpc_span =
        telemetry::startRpcServerSpan("rtp_llm.batch_generate_call", context, false, "RpcService/BatchGenerateCall");
    telemetry::GrpcStatusSpanGuard rpc_guard(rpc_span, &rpc_status);
    const int                      trace_batch_size = request ? request->inputs_size() : 0;
    rpc_guard.setAttribute("rtp_llm.batch.size", (int64_t)trace_batch_size);
    std::string external_request_id;
    if (rpc_guard.valid()) {
        try {
            const auto batch_id = batchTraceMetadata(context, "x-rtp-llm-batch-request-id", 38);
            if (batch_id.size() == 38 && batch_id.compare(0, 6, "batch-") == 0
                && std::all_of(batch_id.begin() + 6, batch_id.end(), [](char c) {
                       return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                   })) {
                rpc_guard.setAttribute(telemetry::kAttrRequestId, batch_id);
            }
            external_request_id = batchTraceMetadata(context, "x-rtp-llm-external-request-id", 128);
            if (!external_request_id.empty()) {
                rpc_guard.setAttribute(telemetry::kAttrExternalRequestId, external_request_id);
            }
        } catch (...) {
            // Observation metadata must never change the batch result.
        }
    }
    std::vector<grpc::Status>                                    item_statuses;
    std::vector<std::unique_ptr<telemetry::GrpcStatusSpanGuard>> item_guards;
    if (telemetry::TelemetryRuntime::isActive()) {
        try {
            item_statuses.resize(trace_batch_size);
            item_guards.reserve(trace_batch_size);
            for (int i = 0; i < trace_batch_size; ++i) {
                const auto request_id = std::to_string(request->inputs(i).request_id());
                auto       guard =
                    std::make_unique<telemetry::GrpcStatusSpanGuard>(startBatchItemSpan(request->inputs(i), rpc_span),
                                                                     &item_statuses[i],
                                                                     telemetry::SpanStatusSemantics::Logical);
                guard->setAttribute(telemetry::kAttrRequestId, request_id);
                if (!external_request_id.empty()) {
                    guard->setAttribute(telemetry::kAttrExternalRequestId, external_request_id);
                }
                guard->setAttribute("rtp_llm.batch.index", (int64_t)i);
                guard->setAttribute("gen_ai.span.kind", "LLM");
                guard->setAttribute(telemetry::kAttrRtpLlmPdSep, false);
                item_guards.push_back(std::move(guard));
            }
        } catch (...) {
            // Keep any existing guards alive for business-status settlement.
        }
    }
    telemetry::GrpcStatusSpanGuard empty_item_guard({}, nullptr, telemetry::SpanStatusSemantics::Logical);
    auto                           item_guard = [&](int index) -> telemetry::GrpcStatusSpanGuard& {
        return static_cast<size_t>(index) < item_guards.size() ? *item_guards[index] : empty_item_guard;
    };
    auto mark_item_error = [&](int index, ErrorCode code, const char* description, const char* error_type) {
        auto& guard = item_guard(index);
        if (!guard.valid()) {
            return;
        }
        try {
            item_statuses[index] = grpc::Status(
                code == ErrorCode::CANCELLED ? grpc::StatusCode::CANCELLED : grpc::StatusCode::INTERNAL, description);
            guard.setLogicalErrorType(error_type);
            guard.setAttribute(telemetry::kAttrRtpLlmErrorCode, (int64_t)code);
        } catch (...) {
            // Observation must not replace a business error with a transport error.
        }
    };
    std::vector<GenerateStreamPtr>        streams;
    std::vector<PDCancelRegistry::Handle> cancel_handles;
    // Destroy after the stream cleanup guard, including exception/early-return paths.
    autil::ScopeGuard       cancel_completion([&] {
        if (!cancel_registry)
            return;
        for (size_t i = 0; i < cancel_handles.size(); ++i) {
            if (!cancel_handles[i])
                continue;
            if (i < streams.size() && streams[i])
                meta_->dequeue(cancel_handles[i]->identity.request_id, streams[i]);
            cancel_registry->finishLocal(cancel_handles[i]);
        }
    });
    BatchStreamCleanupGuard stream_cleanup_guard(streams);

    try {
        RTP_LLM_PROFILE_SCOPE("rpc.batch_generate_call");
        c10::InferenceMode inference_guard(true);
        AtomicGuard        request_guard(onflight_requests_);
        if (request == nullptr || response == nullptr) {
            return rpc_status = serializeErrorMsg(
                       "unknown",
                       ErrorInfo(ErrorCode::INVALID_PARAMS, "batch generate request and response must not be null"));
        }
        const int batch_size = request->inputs_size();
        RTP_LLM_LOG_INFO("receive batch generate request, batch_size=%d", batch_size);
        if (batch_size == 0) {
            return grpc::Status::OK;
        }
        response->clear_results();

        std::vector<std::shared_ptr<GenerateInput>> inputs;
        inputs.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
            std::shared_ptr<GenerateInput> input;
            auto err = prepareInput(request->inputs(i), input, item_guard(i).sharedSpan(), context);
            if (!err.ok()) {
                // Fill error results for all requests (0..batch_size-1) to maintain 1:1 mapping.
                const bool invalid_params = err.code() == ErrorCode::INVALID_PARAMS;
                for (int j = 0; j < batch_size; j++) {
                    mark_item_error(
                        j, err.code(), "Batch input preparation failed", j == i ? "BusinessError" : "BatchAborted");
                    auto* result = response->add_results();
                    auto* err_pb = result->mutable_error_info();
                    // Internal ErrorCode values and RPC enum values are independent wire contracts.
                    err_pb->set_error_code(transErrorCodeToRPC(err.code()));
                    if (j == i) {
                        err_pb->set_error_message(invalid_params ? err.ToString() :
                                                                   "input preparation failed: " + err.ToString());
                    } else if (invalid_params) {
                        err_pb->set_error_message("batch aborted due to invalid parameters at index "
                                                  + std::to_string(i));
                    } else {
                        err_pb->set_error_message("batch aborted due to input preparation failure at index "
                                                  + std::to_string(i));
                    }
                }
                return grpc::Status::OK;
            }
            inputs.push_back(input);
        }

        if (!engine_) {
            for (int i = 0; i < batch_size; ++i) {
                mark_item_error(i, ErrorCode::EXECUTION_EXCEPTION, "Batch engine is unavailable", "BusinessError");
                auto* error_info = response->add_results()->mutable_error_info();
                error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
                error_info->set_error_message("LocalRpcServer engine is not initialized");
            }
            return grpc::Status::OK;
        }

        // Admit each request before enqueue so an early Cancel is a fence, not
        // a successful no-op. Rejected/canceled items must not abort their siblings.
        cancel_handles.resize(batch_size);
        std::vector<ErrorInfo>                      admission_errors(batch_size, ErrorInfo::OkStatus());
        std::vector<std::shared_ptr<GenerateInput>> active_inputs;
        std::vector<size_t>                         active_indices;
        streams.resize(batch_size);
        std::vector<bool> enqueue_successes(batch_size, false);
        for (int i = 0; i < batch_size; ++i) {
            if (cancel_registry) {
                const auto timeout = normalizeRpcTimeoutMs(request->inputs(i).generate_config().timeout_ms(),
                                                           maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
                admission_errors[i] =
                    cancel_registry->admit(request->inputs(i), "", "", currentTimeMs() + timeout, cancel_handles[i]);
            }
            if (!admission_errors[i].ok()) {
                streams[i] = engine_->makeStream(inputs[i]);
                streams[i]->reportError(admission_errors[i].code(), admission_errors[i].ToString());
                streams[i]->finishWithoutGenerate();
            } else {
                active_inputs.push_back(inputs[i]);
                active_indices.push_back(i);
            }
        }
        if (!active_inputs.empty()) {
            auto enqueue_result = engine_->enqueueMultiple(active_inputs);
            if (enqueue_result.first.size() != active_inputs.size()
                || enqueue_result.second.size() != active_inputs.size()) {
                BatchStreamCleanupGuard invalid_enqueue_cleanup(enqueue_result.second);
                throw std::runtime_error("enqueueMultiple result size mismatch");
            }
            for (size_t j = 0; j < active_indices.size(); ++j) {
                const auto i         = active_indices[j];
                enqueue_successes[i] = enqueue_result.first[j];
                streams[i]           = enqueue_result.second[j];
                if (cancel_handles[i] && streams[i]) {
                    meta_->enqueue(cancel_handles[i]->identity, streams[i]);
                    cancel_registry->attach(cancel_handles[i], streams[i]);
                }
            }
        }

        // collectStreamOutput is currently SERIAL: streams[0] must finish before streams[1] is drained.
        // For batch decode this is bounded (all streams advance together), but TODO: parallelize for
        // mixed-length batches.
        for (int i = 0; i < (int)streams.size(); i++) {
            if (!streams[i]) {
                throw std::runtime_error("enqueueMultiple returned a null stream at index " + std::to_string(i));
            }
            auto* result = response->add_results();
            if (!enqueue_successes[i]) {
                const auto code =
                    streams[i]->hasError() ? streams[i]->statusInfo().code() : ErrorCode::EXECUTION_EXCEPTION;
                mark_item_error(i, code, "Batch enqueue was rejected", "BusinessError");
                auto* error_info = result->mutable_error_info();
                if (streams[i]->hasError()) {
                    const auto stream_error = streams[i]->statusInfo();
                    error_info->set_error_code(transErrorCodeToRPC(stream_error.code()));
                    error_info->set_error_message(stream_error.ToString());
                } else {
                    error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
                    error_info->set_error_message("enqueueMultiple rejected input at index " + std::to_string(i));
                }
                stream_cleanup_guard.markHandled(static_cast<size_t>(i));
                recordBatchItemCompletion(item_guard(i), streams[i], false, "BusinessError");
                continue;
            }

            GenerateOutputs last_outputs;
            auto            err = collectStreamOutput(context, streams[i], inputs[i], last_outputs);
            stream_cleanup_guard.markHandled(static_cast<size_t>(i));
            const char* error_type =
                err.ok() ? nullptr : (err.code() == ErrorCode::CANCELLED ? "Cancelled" : "BusinessError");
            if (!err.ok()) {
                mark_item_error(i, err.code(), "Batch request failed", error_type);
                auto* err_pb = result->mutable_error_info();
                err_pb->set_error_code(transErrorCodeToRPC(err.code()));
                err_pb->set_error_message(err.ToString());
            } else {
                auto* output_pb = result->mutable_final_output();
                QueryConverter::transResponse(output_pb,
                                              &last_outputs,
                                              inputs[i]->generate_config->aux_info,
                                              maga_init_params_.misc_config.aux_string,
                                              streams[i]->specialTokens().eos_token_id);
            }
            recordBatchItemCompletion(item_guard(i), streams[i], err.ok(), error_type);
        }

        stream_cleanup_guard.dismiss();
        RTP_LLM_LOG_INFO("batch generate done, batch_size=%d", batch_size);
        return grpc::Status::OK;
    } catch (const std::exception& e) {
        const std::string error_message = "BatchGenerateCall failed with exception: " + std::string(e.what());
        const int         batch_size    = request ? request->inputs_size() : 0;
        RTP_LLM_LOG_ERROR("%s, batch_size=%d", error_message.c_str(), batch_size);
        if (response == nullptr) {
            return rpc_status = serializeErrorMsg("unknown", ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_message));
        }
        response->clear_results();
        for (int i = 0; i < batch_size; ++i) {
            mark_item_error(i, ErrorCode::EXECUTION_EXCEPTION, "Batch handler failed", "BusinessError");
            auto* error_info = response->add_results()->mutable_error_info();
            error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
            error_info->set_error_message(error_message);
        }
        return grpc::Status::OK;
    } catch (...) {
        const std::string error_message = "BatchGenerateCall failed with unknown exception";
        const int         batch_size    = request ? request->inputs_size() : 0;
        RTP_LLM_LOG_ERROR("%s, batch_size=%d", error_message.c_str(), batch_size);
        if (response == nullptr) {
            return rpc_status = serializeErrorMsg("unknown", ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_message));
        }
        response->clear_results();
        for (int i = 0; i < batch_size; ++i) {
            mark_item_error(i, ErrorCode::EXECUTION_EXCEPTION, "Batch handler failed", "BusinessError");
            auto* error_info = response->add_results()->mutable_error_info();
            error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
            error_info->set_error_message(error_message);
        }
        return grpc::Status::OK;
    }
}

grpc::Status
LocalRpcServer::GetCacheStatus(grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s, request cache version: [%d]",
                      context->peer().c_str(),
                      request->latest_cache_version());
    KVCacheInfo cache_status = getCacheStatusInfo(request->latest_cache_version(), request->need_cache_keys());
    response->set_available_kv_cache(cache_status.available_kv_cache);
    response->set_total_kv_cache(cache_status.total_kv_cache);
    response->set_block_size(cache_status.block_size);
    response->set_version(cache_status.version);
    auto* cache_map = response->mutable_cache_keys();
    for (const auto& key : cache_status.cached_keys) {
        (*cache_map)[static_cast<int64_t>(key)] = true;
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetWorkerStatus(grpc::ServerContext*   context,
                                             const StatusVersionPB* request,
                                             WorkerStatusPB*        response) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t request_begin_time_us   = currentTimeUs();
    int64_t latest_finished_version = request->latest_finished_version();
    RTP_LLM_LOG_DEBUG(
        "receive workerStatus rpc request from client: %s, latest_finished_version: %ld, config role_type: %d",
        context->peer().c_str(),
        latest_finished_version,
        maga_init_params_.pd_sep_config.role_type);

    WorkerStatusInfo status_info              = getWorkerStatusInfo(latest_finished_version);
    int64_t          request_after_ws_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("getWorkerStatusInfo took %ld us", request_after_ws_time_us - request_begin_time_us);

    const auto& engine_schedule_info = status_info.engine_schedule_info;
    // Field 1 is consumed as an opaque string by legacy Masters, whose role
    // codes are "RoleType.PREFILL" etc. Keep that exact spelling while the
    // typed field 20 is rolled out.
    response->set_role("RoleType." + roleTypeToString(status_info.role));
    response->set_role_type(static_cast<RoleTypePB>(status_info.role));

    for (const auto& task : engine_schedule_info.running_task_info_list) {
        TaskInfoPB* task_info = response->add_running_task_info();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_is_waiting(task.phase != rtp_llm::TaskPhase::RUNNING);
        task_info->set_batch_id(task.batch_id);
        task_info->set_execution_time_ms(task.execution_time_ms);
        task_info->set_priority_preemption_progress(
            static_cast<PriorityPreemptionProgressPB>(task.priority_preemption_progress));
        if (task.error_code != 0) {
            task_info->mutable_error_info()->set_error_code(task.error_code);
            task_info->mutable_error_info()->set_error_message(task.error_message);
        }
    }

    for (const auto& task : engine_schedule_info.finished_task_info_list) {
        TaskInfoPB* task_info = response->add_finished_task_list();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_is_waiting(task.phase != rtp_llm::TaskPhase::RUNNING);
        task_info->set_batch_id(task.batch_id);
        task_info->set_execution_time_ms(task.execution_time_ms);
        task_info->set_priority_preemption_progress(
            static_cast<PriorityPreemptionProgressPB>(task.priority_preemption_progress));
        if (task.error_code != 0) {
            task_info->mutable_error_info()->set_error_code(task.error_code);
            task_info->mutable_error_info()->set_error_message(task.error_message);
        }
    }

    response->set_dp_size(status_info.dp_size);
    response->set_tp_size(status_info.tp_size);
    response->set_status_version(status_info.status_version);
    response->set_latest_finished_version(status_info.latest_finished_version);
    response->set_alive(status_info.alive);
    response->set_precision(status_info.precision);
    response->set_dp_rank(status_info.dp_rank);
    response->set_max_seq_len(maga_init_params_.model_config_.max_seq_len);
    const auto& runtime_config = maga_init_params_.runtime_config;
    const auto& fifo_config    = runtime_config.fifo_scheduler_config;
    auto        batch_tokens   = fifo_config.max_batch_tokens_size;
    const bool  uses_fifo =
        maga_init_params_.parallelism_config.pp_size <= 1 && !runtime_config.use_batch_decode_scheduler
        && !(maga_init_params_.pd_sep_config.role_type == RoleType::PDFUSION
             && parsePDFusionSchedulerMode(fifo_config.pdfusion_scheduler_mode) == PDFusionSchedulerMode::RATIO);
    if (uses_fifo && fifo_config.max_batch_kv_len > 0) {
        // Legacy planners have one compute cap and use a full-prompt rectangle.
        // Bounding it by both FIFO sums is conservative; physical KV stays separate
        // so an oversized head request can still run alone. Negative T becomes
        // an unsigned limit in FIFO; zero T permits only the singleton fast path.
        batch_tokens = batch_tokens < 0 ? fifo_config.max_batch_kv_len :
                                          std::max<int64_t>(1, std::min(batch_tokens, fifo_config.max_batch_kv_len));
    }
    response->set_max_batch_tokens_size(batch_tokens);
    auto kv_info = engine_->getCacheStatusInfo(-1, false);
    response->set_available_kv_cache(kv_info.available_kv_cache);
    response->set_total_kv_cache(kv_info.total_kv_cache);
    auto       group_mode    = KV_CACHE_GROUP_MODE_UNSPECIFIED;
    const auto cache_manager = engine_->getCacheManager();
    if (cache_manager && cache_manager->cacheConfig().groupNums() > 0) {
        for (const auto& group : cache_manager->cacheConfig().topology().groups()) {
            // Match the HBM publisher's eligibility, including its exact tag exclusions.
            if (!cacheGroupPublishesEvents(group.policy)
                || group.policy.memory_placement != CacheMemoryPlacement::DEVICE || group.tag == "csa_state"
                || group.tag == "swa_kv") {
                continue;
            }
            if (group.policy.group_type == CacheGroupType::LINEAR) {
                group_mode = KV_CACHE_GROUP_MODE_WITH_MAMBA;
                break;
            }
            group_mode = KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY;
        }
    }
    response->set_kv_cache_group_mode(group_mode);
    reportWorkerStatusTime(request_begin_time_us, request_after_ws_time_us);
    return grpc::Status::OK;
}

WorkerStatusInfo LocalRpcServer::getWorkerStatusInfo(int64_t latest_finished_version) {
    WorkerStatusInfo status_info;
    status_info.engine_schedule_info    = getEngineScheduleInfo(latest_finished_version);
    status_info.role                    = maga_init_params_.pd_sep_config.role_type;
    status_info.dp_size                 = maga_init_params_.parallelism_config.dp_size;
    status_info.tp_size                 = maga_init_params_.parallelism_config.tp_size;
    status_info.dp_rank                 = maga_init_params_.parallelism_config.dp_rank;
    status_info.status_version          = currentTimeUs();
    status_info.latest_finished_version = status_info.engine_schedule_info.latest_finished_version;
    status_info.alive                   = true;
    auto quant_method                   = maga_init_params_.model_config_.quant_algo.getQuantMethod();

    switch (quant_method) {
        case QuantMethod::WeightOnlyPerCol:
            status_info.precision = "WeightOnlyPerCol";
            break;
        case QuantMethod::GptQ:
            status_info.precision = "GptQ";
            break;
        case QuantMethod::Awq:
            status_info.precision = "Awq";
            break;
        case QuantMethod::SmoothQuant:
            status_info.precision = "SmoothQuant";
            break;
        case QuantMethod::OmniQuant:
            status_info.precision = "OmniQuant";
            break;
        case QuantMethod::PerTensorQuant:
            status_info.precision = "PerTensorQuant";
            break;
        case QuantMethod::FP8Quant:
            status_info.precision = "FP8Quant";
            break;
        case QuantMethod::FP8PTPC:
            status_info.precision = "FP8PTPC";
            break;
        case QuantMethod::W4A8INT4PTPC:
            status_info.precision = "W4A8INT4PTPC";
            break;
        case QuantMethod::W8A8INT8PTPC:
            status_info.precision = "W8A8INT8PTPC";
            break;
        case QuantMethod::None:
            status_info.precision = "FP16";
            break;
        default:
            RTP_LLM_LOG_ERROR("unknown quant method: %d", static_cast<int>(quant_method));
            status_info.precision = "UNKNOWN";
    }
    return status_info;
}

KVCacheInfo LocalRpcServer::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    int64_t     request_begin_time_us = currentTimeUs();
    const auto& cache_info            = engine_->getCacheStatusInfo(latest_version, need_cache_keys);
    reportCacheStatusTime(request_begin_time_us);
    return cache_info;
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finished_version) {
    EngineScheduleInfo info               = meta_->getEngineScheduleInfo(latest_finished_version);
    auto               last_schedule_time = engine_->getLastScheduleTime();
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
}

grpc::Status LocalRpcServer::UpdateSchedulerInfo(grpc::ServerContext*                context,
                                                 const UpdateSchedulerInfoRequestPB* request,
                                                 EmptyPB*                            response) {
    const std::string scheduler_info = request->scheduler_info();
    engine_->getScheduler().updateSchedulerInfo(scheduler_info);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) {
    std::string log_level_str = request->log_level();
    uint32_t    log_level     = alog::LOG_LEVEL_INFO;
    if (log_level_str == "INFO" || log_level_str == "info") {
        log_level = alog::LOG_LEVEL_INFO;
    } else if (log_level_str == "WARNING" || log_level_str == "warning") {
        log_level = alog::LOG_LEVEL_WARN;
    } else if (log_level_str == "DEBUG" || log_level_str == "debug") {
        log_level = alog::LOG_LEVEL_DEBUG;
    } else if (log_level_str == "TRACE" || log_level_str == "trace") {
        log_level = alog::LOG_LEVEL_TRACE1;
    } else {
        RTP_LLM_LOG_WARNING("set log level failed, unknown log level: %s", log_level_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid log_level format");
    }
    auto& logger = rtp_llm::Logger::getEngineLogger();
    logger.setBaseLevel(log_level);
    return grpc::Status::OK;
}

namespace {
int64_t pressureNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
grpc::Status drainRemoteCompletions(EngineBase*                           engine,
                                    grpc::ServerContext*                  context,
                                    std::chrono::system_clock::time_point deadline) {
    if (std::chrono::system_clock::now() >= deadline) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "remote completion drain deadline exceeded"};
    }
    const auto manager = engine ? engine->getCacheManager() : nullptr;
    const auto cache   = manager ? manager->blockTreeCache() : nullptr;
    if (cache) {
        auto drained   = std::make_shared<std::promise<void>>();
        auto completed = drained->get_future();
        cache->drainRemoteCompletions([drained] { drained->set_value(); });
        // No cache/stream/scheduler/execution mutex is held here. gRPC sync
        // handlers have replacement pollers; MAX_POLLERS is not a handler cap.
        // The shared promise remains valid if the diagnostic deadline expires.
        if (completed.wait_until(deadline) != std::future_status::ready
            || std::chrono::system_clock::now() >= deadline) {
            return {grpc::StatusCode::DEADLINE_EXCEEDED, "remote completion drain deadline exceeded"};
        }
    }
    if (context->IsCancelled()) {
        return {grpc::StatusCode::CANCELLED, "diagnostic request cancelled after drain"};
    }
    return grpc::Status::OK;
}
void profileStatusToProto(const ProfileStatus& value, ProfileStatusPB* pb) {
    pb->set_supported(value.supported);
    pb->set_state(value.state);
    pb->set_trace_name(value.trace_name);
    pb->set_stop_requested(value.stop_requested);
    pb->set_collection_complete(value.collection_complete);
    pb->set_export_complete(value.export_complete);
    pb->set_collection_error(value.collection_error);
    pb->set_export_error(value.export_error);
    pb->set_file_name(value.file_name);
    pb->set_capture_begin_ns(value.capture_begin_ns);
    pb->set_capture_end_ns(value.capture_end_ns);
}
PressureTierPB pressureTier(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return PRESSURE_DEVICE;
        case Tier::HOST:
            return PRESSURE_HOST;
        case Tier::DISK:
            return PRESSURE_DISK;
        case Tier::REMOTE:
            return PRESSURE_REMOTE;
        case Tier::NONE:
            return PRESSURE_NONE;
    }
    return PRESSURE_TIER_UNSPECIFIED;
}
PressureGroupTypePB pressureGroup(CacheGroupType type) {
    switch (type) {
        case CacheGroupType::FULL:
            return PRESSURE_FULL;
        case CacheGroupType::SWA:
            return PRESSURE_SWA;
        case CacheGroupType::LINEAR:
            return PRESSURE_LINEAR;
    }
    return PRESSURE_GROUP_UNSPECIFIED;
}
PressureOperationPB pressureOperation(CacheTransferOperation operation) {
    switch (operation) {
        case CacheTransferOperation::LOAD:
            return PRESSURE_LOAD;
        case CacheTransferOperation::EVICT:
            return PRESSURE_EVICT;
        case CacheTransferOperation::STORE:
            return PRESSURE_STORE;
    }
    return PRESSURE_OPERATION_UNSPECIFIED;
}
void pressureWatermark(const TierWatermark& value, PressureWatermarkPB* output) {
    output->set_low_ratio(value.low_ratio);
    output->set_high_ratio(value.high_ratio);
}
void pressureSnapshotToProto(const CachePressureSnapshot& snapshot, CachePressureRankPB* output) {
    if (snapshot.ownership) {
        const auto& ownership = *snapshot.ownership;
        auto*       pb        = output->mutable_ownership();
        pb->set_mode(static_cast<PressureOwnershipPB::Mode>(ownership.mode));
        pb->mutable_owner_world_rank()->set_value(ownership.owner_world_rank);
        for (int rank : ownership.participant_world_ranks) {
            pb->add_participant_world_ranks(rank);
        }
        pb->set_publication_scope(ownership.all_participant_publication ? PressureOwnershipPB::ALL_PARTICIPANTS :
                                                                          PressureOwnershipPB::LOCAL);
    }
    if (snapshot.remote_runtime) {
        const auto& runtime = *snapshot.remote_runtime;
        auto*       pb      = output->mutable_remote_runtime();
        pb->set_version(1);
        for (const auto& [name, value] : runtime.options)
            (*pb->mutable_options())[name] = value;
        pb->set_callback_endpoint(runtime.callback_endpoint);
        pb->set_server_epoch(runtime.session.server_epoch);
        pb->set_session_id(runtime.session.session_id);
        pb->set_generation(runtime.session.generation);
        pb->set_layout_hash(runtime.layout_hash);
        pb->set_rank(runtime.rank);
        pb->set_expected_ranks(runtime.expected_ranks);
#define LIMIT(name) (*pb->mutable_limits())[#name] = runtime.limits.name
        LIMIT(total_bytes);
        LIMIT(source_bytes);
        LIMIT(write_slot_bytes);
        LIMIT(read_staging_bytes);
        LIMIT(write_staging_bytes);
        LIMIT(write_slots);
        LIMIT(max_rpc_bytes);
        LIMIT(max_blocks_per_rpc);
        LIMIT(max_read_operations);
        LIMIT(max_ranks);
        LIMIT(max_groups_per_rank);
        LIMIT(max_sources_per_group);
        LIMIT(max_members_per_group);
        LIMIT(replay_window);
        LIMIT(max_delay_ms);
#undef LIMIT
#define CREDIT(name) (*pb->mutable_credits())[#name] = runtime.credits.name
        CREDIT(local_read_bytes);
        CREDIT(local_write_bytes);
        CREDIT(service_read_bytes);
        CREDIT(service_write_bytes);
        CREDIT(read_operations);
        CREDIT(write_operations);
        CREDIT(first_write_slot);
        CREDIT(write_slots);
        CREDIT(max_slice_bytes);
        CREDIT(max_batch_records);
        CREDIT(max_write_slice_bytes);
#undef CREDIT
        for (const auto& layout : runtime.layouts) {
            auto* group = pb->add_layouts();
#define LAYOUT(name) group->set_##name(layout.name)
            LAYOUT(group);
            LAYOUT(pool);
            LAYOUT(payload_bytes);
            LAYOUT(storage_bytes);
            LAYOUT(stride_bytes);
            LAYOUT(source_count);
            LAYOUT(alignment);
            LAYOUT(dtype);
            LAYOUT(crc_enabled);
            LAYOUT(crc_format_version);
            LAYOUT(crc_footer_offset);
            LAYOUT(tokens_per_block);
            LAYOUT(cp_world_size);
            LAYOUT(cp_rank);
            LAYOUT(attention_window);
#undef LAYOUT
        }
    }
    output->set_remote_read_payload_bytes(snapshot.remote_read_payload_bytes);
    output->set_remote_write_acknowledged_payload_bytes(snapshot.remote_write_acknowledged_payload_bytes);
    auto* remote_write = output->mutable_mock_remote_write();
    remote_write->set_submitted(snapshot.mock_remote_write.submitted);
    remote_write->set_completed(snapshot.mock_remote_write.completed);
    remote_write->set_failed(snapshot.mock_remote_write.failed);
    remote_write->set_rejected(snapshot.mock_remote_write.rejected);
    remote_write->set_pending_requests(snapshot.mock_remote_write.pending_requests);
    remote_write->set_pending_intents(snapshot.mock_remote_write.pending_intents);
    remote_write->set_retained_descriptors(snapshot.mock_remote_write.retained_descriptors);
    remote_write->set_faulted(snapshot.remote_mock_safety.faulted);
    remote_write->set_quarantined_blocks(snapshot.remote_mock_safety.quarantined_blocks);
    if (snapshot.mock_cache) {
        const auto& metrics = *snapshot.mock_cache;
        auto*       pb      = output->mutable_mock_cache();
        pb->set_terminal_outcome_version(1);
        pb->set_eligible_blocks(metrics.eligible_blocks);
        pb->set_planned_memory_blocks(metrics.planned_memory_blocks);
        pb->set_planned_disk_blocks(metrics.planned_disk_blocks);
        pb->set_successful_memory_blocks(metrics.successful_memory_blocks);
        pb->set_successful_disk_blocks(metrics.successful_disk_blocks);
        pb->set_planned_remote_blocks(metrics.planned_remote_blocks);
        pb->set_successful_remote_blocks(metrics.successful_remote_blocks);
        pb->set_failed_blocks(metrics.failed_blocks);
        pb->set_disk_write_submitted_records(snapshot.mock_disk_write.submitted_records);
        pb->set_disk_write_completed_records(snapshot.mock_disk_write.completed_records);
        pb->set_disk_write_failed_records(snapshot.mock_disk_write.failed_records);
    }
    if (snapshot.local_store) {
        auto* pb = output->mutable_local_store();
        pb->set_released_requests(snapshot.local_store->released_requests);
        pb->set_pending_requests(snapshot.local_store->pending_requests);
        pb->set_abandoned_requests(snapshot.local_store->abandoned_requests);
        pb->set_rejected_host_stores(snapshot.local_store->rejected_host_stores);
        pb->set_rejected_disk_stores(snapshot.local_store->rejected_disk_stores);
    }
    if (snapshot.local_mock_targets) {
        const auto& targets = *snapshot.local_mock_targets;
        auto*       pb      = output->mutable_local_mock_targets();
        pb->set_memory_enabled(targets.memory_enabled);
        pb->set_disk_enabled(targets.disk_enabled);
        pb->set_memory_ratio(targets.memory_ratio);
        pb->set_disk_ratio(targets.disk_ratio);
        pb->set_memory_ppm(targets.memory_ppm);
        pb->set_disk_ppm(targets.disk_ppm);
    }
    if (snapshot.mock_memory) {
        const auto& metrics = *snapshot.mock_memory;
        auto*       pb      = output->mutable_mock_memory();
        pb->set_terminal_outcome_version(PressureMockMemoryPB::TRANSFER_TERMINAL_V1);
        pb->set_eligible_blocks(metrics.eligible_blocks);
        pb->set_planned_host_blocks(metrics.planned_host_blocks);
        pb->set_successful_h2d_blocks(metrics.successful_h2d_blocks);
        pb->set_failed_h2d_blocks(metrics.failed_h2d_blocks);
        pb->set_d2h_submitted_blocks(metrics.d2h_submitted_blocks);
        pb->set_d2h_completed_blocks(metrics.d2h_completed_blocks);
        pb->set_d2h_failed_blocks(metrics.d2h_failed_blocks);
    }
    for (const auto& pool : snapshot.pools) {
        auto* pb = output->add_pools();
        pb->set_pool_id(pool.pool_id);
        pb->set_tier(pressureTier(pool.tier));
        pb->set_pool_name(pool.pool_name);
        pb->set_block_size_bytes(pool.block_size_bytes);
        pb->set_total_blocks(pool.total_blocks);
        pb->set_free_blocks(pool.free_blocks);
        pb->set_used_blocks(pool.used_blocks);
        pb->set_available_blocks(pool.available_blocks);
        pb->set_active_blocks(pool.active_blocks);
        pb->set_block_cache_ref_blocks(pool.block_cache_ref_blocks);
        pb->set_load_ref_blocks(pool.load_ref_blocks);
        pb->set_eviction_target_ref_blocks(pool.eviction_ref_blocks);
        pb->set_store_ref_blocks(pool.store_ref_blocks);
        pb->set_pending_release_blocks(pool.pending_release_blocks);
        pb->set_published_host_blocks(pool.published_host_blocks);
        pb->set_mock_source_ref_blocks(pool.mock_source_ref_blocks);
        pb->set_mock_source_ref_bytes(pool.mock_source_ref_bytes);
        pb->set_disk_read_bytes(pool.disk_read_bytes);
        pb->set_disk_write_bytes(pool.disk_write_bytes);
        if (pool.request_ref_blocks_available) {
            pb->mutable_request_ref_blocks()->set_value(pool.request_ref_blocks);
        }
    }
    for (const auto& group : snapshot.groups) {
        auto* pb = output->add_groups();
        pb->set_group_id(group.group_id);
        pb->set_group_type(pressureGroup(group.group_type));
        pb->set_block_tokens(group.block_tokens);
        pb->set_payload_bytes(group.payload_bytes);
        pb->set_storage_bytes(group.storage_bytes);
        pb->set_crc_enabled(group.crc_enabled);
        for (const auto& tag : group.group_tags) {
            pb->add_group_tags(tag);
        }
        for (const auto& id : group.device_pool_ids) {
            pb->add_device_pool_ids(id);
        }
        if (group.host_pool_id) {
            pb->mutable_host_pool_id()->set_value(*group.host_pool_id);
        }
        if (group.disk_pool_id) {
            pb->mutable_disk_pool_id()->set_value(*group.disk_pool_id);
        }
        pressureWatermark(group.device_watermark, pb->mutable_device_watermark());
        pressureWatermark(group.host_watermark, pb->mutable_host_watermark());
        pressureWatermark(group.disk_watermark, pb->mutable_disk_watermark());
        pb->set_watermark_trigger_count(group.watermark_trigger_count);
    }
    for (const auto& transfer : snapshot.transfers) {
        auto* pb = output->add_transfers();
        pb->set_group_id(transfer.group_id);
        pb->set_pool_id(transfer.pool_id);
        pb->set_pool_tier(pressureTier(transfer.pool_tier));
        pb->set_operation(pressureOperation(transfer.operation));
        pb->set_source_tier(pressureTier(transfer.source_tier));
        pb->set_target_tier(pressureTier(transfer.target_tier));
        pb->set_copied_blocks(transfer.copied_blocks);
        pb->set_copied_bytes(transfer.copied_bytes);
        pb->set_published_blocks(transfer.published_blocks);
        pb->set_published_bytes(transfer.published_bytes);
        pb->set_failed_operations(transfer.failed_operations);
        pb->set_physical_failed_operations(transfer.physical_failed_operations);
    }
}
}  // namespace

std::string LocalRpcServer::makePressureInstance() {
    return std::to_string(getpid()) + "-" + std::to_string(pressureNowNs()) + "-"
           + std::to_string(std::random_device{}());
}

namespace {
grpc::Status validatePressureControl(const PressureSessionControlRequestPB& request) {
    if (request.session_id().empty() || request.session_id().size() > 256 || request.phase().size() > 256
        || request.action() < PressureSessionControlRequestPB::ENABLE
        || request.action() > PressureSessionControlRequestPB::DISABLE
        || request.recorder_capacity() > PressureStepRecorder::kMaxCapacity
        || (request.action() != PressureSessionControlRequestPB::ENABLE && request.recorder_capacity())
        || (request.action() == PressureSessionControlRequestPB::MARK_PHASE && request.phase().empty())) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "invalid pressure session control"};
    }
    return grpc::Status::OK;
}
}  // namespace

grpc::Status LocalRpcServer::ControlPressureDiagnosticsInternal(grpc::ServerContext*                   context,
                                                                const PressureSessionControlRequestPB* request,
                                                                PressureSessionControlResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    auto status = validatePressureControl(*request);
    if (!status.ok()) {
        return status;
    }
    auto* rank = response->add_ranks();
    rank->set_world_rank(maga_init_params_.parallelism_config.world_rank);
    rank->set_service_instance(pressure_instance_);
    if (context->IsCancelled()) {
        return {grpc::StatusCode::CANCELLED, "pressure control cancelled"};
    }
    PressureSessionControl control{request->session_id(),
                                   static_cast<PressureControlAction>(request->action()),
                                   request->phase(),
                                   request->recorder_capacity()};
    const auto             deadline = resolvePressureControlDeadline(context->deadline());
    auto                   result   = engine_ ? engine_->controlPressureDiagnostics(control, deadline) :
                                                ErrorResult<int64_t>(kPressureDiagnosticsUnsupported, "pressure diagnostics unsupported");
    if (!result.ok()) {
        rank->set_error(result.status().ToString());
        auto code = grpc::StatusCode::FAILED_PRECONDITION;
        if (result.status().code() == kPressureDiagnosticsUnsupported) {
            code = grpc::StatusCode::UNIMPLEMENTED;
        } else if (result.status().code() == ErrorCode::DEADLINE_EXCEEDED) {
            code = grpc::StatusCode::DEADLINE_EXCEEDED;
        }
        return {code, result.status().ToString()};
    }
    rank->set_applied_monotonic_ns(result.value());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::ControlPressureDiagnostics(grpc::ServerContext*                   context,
                                                        const PressureSessionControlRequestPB* request,
                                                        PressureSessionControlResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    auto status = validatePressureControl(*request);
    if (!status.ok()) {
        return status;
    }
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "pressure control fan-out requires tp_rank 0"};
    }
    const int        base = topology.world_rank - topology.tp_rank;
    std::set<int>    seen;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i) {
            ranks.push_back(base + i);
        }
    }
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology target rank"};
        }
    }
    if (ranks.size() != static_cast<size_t>(topology.tp_size)) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "pressure recording controls require the entire TP group"};
    }
    struct Pending {
        grpc::ClientContext                                                                context;
        PressureSessionControlResponsePB                                                   response;
        grpc::Status                                                                       status;
        std::shared_ptr<RpcService::Stub>                                                  stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<PressureSessionControlResponsePB>> reader;
        PressureSessionControlRankPB*                                                      record;
        int                                                                                rank;
    };
    // Do not reuse channels attached to a completion queue drained by an earlier request.
    RPCPool                               pressure_rpc_pool;
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    const auto                            deadline     = resolvePressureControlDeadline(context->deadline());
    const auto&                           addresses    = maga_init_params_.runtime_config.worker_grpc_addrs;
    PressureSessionControlRankPB*         local_record = nullptr;
    // Launch peers before waiting for the local engine boundary. Every rank shares the deadline.
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        if (rank == topology.world_rank) {
            local_record = record;
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            continue;
        }
        auto connection = pressure_rpc_pool.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncControlPressureDiagnosticsInternal(&call->context, *request, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    grpc::Status local_status;
    if (local_record) {
        if (context->IsCancelled()) {
            local_status = {grpc::StatusCode::CANCELLED, "pressure control cancelled"};
        } else {
            PressureSessionControl control{request->session_id(),
                                           static_cast<PressureControlAction>(request->action()),
                                           request->phase(),
                                           request->recorder_capacity()};
            auto                   result = engine_ ?
                                                engine_->controlPressureDiagnostics(control, deadline) :
                                                ErrorResult<int64_t>(kPressureDiagnosticsUnsupported, "pressure diagnostics unsupported");
            local_record->set_service_instance(pressure_instance_);
            if (result.ok()) {
                local_record->set_applied_monotonic_ns(result.value());
            } else {
                local_status = {
                    result.status().code() == kPressureDiagnosticsUnsupported ? grpc::StatusCode::UNIMPLEMENTED :
                    result.status().code() == ErrorCode::DEADLINE_EXCEEDED    ? grpc::StatusCode::DEADLINE_EXCEEDED :
                                                                                grpc::StatusCode::FAILED_PRECONDITION,
                    result.status().ToString()};
            }
        }
        if (!local_status.ok()) {
            local_record->set_error(local_status.error_message());
        }
    }
    bool cancellation_sent = false;
    for (size_t i = 0; i < pending.size();) {
        void*      tag = nullptr;
        bool       ok  = false;
        const auto now = std::chrono::system_clock::now();
        if (now >= deadline || context->IsCancelled()) {
            if (!cancellation_sent) {
                for (auto& call : pending) {
                    call->context.TryCancel();
                }
                cancellation_sent = true;
            }
            queue.Next(&tag, &ok);
        } else {
            // A peer may enqueue after the root's first empty TP broadcast, or
            // root may not be a target at all. Keep the existing idle TP loop
            // progressing only while this bounded control fan-out is pending.
            if (engine_) {
                engine_->wakePressureControlProgress();
            }
            if (queue.AsyncNext(&tag, &ok, std::min(deadline, now + std::chrono::milliseconds(10)))
                != grpc::CompletionQueue::GOT_EVENT) {
                continue;
            }
        }
        ++i;
        auto* call = static_cast<Pending*>(tag);
        if (ok && call->status.ok() && call->response.schema_version() == 1 && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid rank response" : call->status.error_message());
        }
    }
    queue.Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (queue.Next(&tag, &ok)) {}
    return ranks.size() == 1 && local_record ? local_status : grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetCachePressureInternal(grpc::ServerContext*          context,
                                                      const CachePressureRequestPB* request,
                                                      CachePressureResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    response->set_coordinator_received_ns(pressureNowNs());
    if (request->max_step_records() > kMaxPressureStepRecords) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "max_step_records exceeds 1024"};
    }
    const int rank = maga_init_params_.parallelism_config.world_rank;
    response->add_requested_ranks(rank);
    auto* record = response->add_ranks();
    record->set_world_rank(rank);
    record->set_service_instance(pressure_instance_);
    record->set_sequence(pressure_sequence_.fetch_add(1) + 1);
    if (context->IsCancelled()) {
        return {grpc::StatusCode::CANCELLED, "diagnostic request cancelled"};
    }
    if (request->drain_remote_completions()) {
        const auto deadline = std::min(
            context->deadline(), std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
        const auto status = drainRemoteCompletions(engine_.get(), context, deadline);
        if (!status.ok()) {
            record->set_error(status.error_message());
            return status;
        }
    }
    const auto begin = pressureNowNs();
    record->set_sampled_monotonic_ns(begin);
    auto result = engine_ ? engine_->getCachePressure() : unsupportedCachePressure();
    profileStatusToProto(engine_ ? engine_->getProfileStatus() : ProfileStatus{}, record->mutable_profile());
    record->set_collect_duration_ns(pressureNowNs() - begin);
    response->set_coordinator_sent_ns(pressureNowNs());
    if (!result.ok()) {
        record->set_error(result.status().ToString());
        return {result.status().code() == kPressureDiagnosticsUnsupported ? grpc::StatusCode::UNIMPLEMENTED :
                                                                            grpc::StatusCode::INTERNAL,
                result.status().ToString()};
    }
    pressureSnapshotToProto(result.value(), record);
    const auto cursor = request->after_step_sequence().find(rank);
    auto       page =
        engine_->getPressureSteps(request->session_id(),
                                  cursor == request->after_step_sequence().end() ? 0 : cursor->second,
                                  request->max_step_records() ? request->max_step_records() : kMaxPressureStepRecords);
    record->set_step_history_available(page.available);
    record->set_next_step_sequence(page.next_sequence);
    record->set_step_records_lost(page.lost);
    auto add_step = [&](const PressureStepRecord& step) {
        auto* pb = record->add_steps();
        pb->set_session_id(step.session_id);
        pb->set_phase(step.phase);
        pb->set_step_id(step.step_id);
        pb->set_sequence(step.sequence);
        pb->set_wall_start_ns(step.forward_enter_ns);
        pb->set_wall_end_ns(step.forward_return_ns);
        pb->set_cpu_start_ns(step.forward_enter_cpu_ns);
        pb->set_cpu_end_ns(step.forward_return_cpu_ns);
        pb->set_context_count(step.context_batch);
        pb->set_decode_count(step.decode_batch);
        pb->set_executed_tokens(step.tokens);
        // NormalExecutor executes one token per decode batch row.
        pb->set_context_tokens(step.tokens >= step.decode_batch ? step.tokens - step.decode_batch : 0);
        pb->set_decode_tokens(step.decode_batch);
        for (const auto& id : step.request_ids) {
            pb->add_source_request_ids(id);
        }
        for (const auto id : step.stream_ids) {
            pb->add_request_ids(id);
        }
        pb->set_completed(step.complete);
        pb->set_preparation_begin_ns(step.begin_ns);
        pb->set_forward_enter_ns(step.forward_enter_ns);
        pb->set_forward_return_ns(step.forward_return_ns);
        pb->set_end_ns(step.end_ns);
    };
    for (const auto& step : page.records) {
        add_step(step);
    }
    if (page.available) {
        // Only the scheduling rank observes the participating request identities.
        if (!page.active || !page.active->stream_ids.empty()) {
            record->mutable_activity()->mutable_running_requests()->set_value(
                page.active ? page.active->stream_ids.size() : 0);
        }
        if (const auto waiting = engine_->getPressureWaitingRequests()) {
            record->mutable_activity()->mutable_waiting_requests()->set_value(*waiting);
        }
    }
    if (page.active) {
        add_step(*page.active);
    }
    record->set_collect_duration_ns(pressureNowNs() - begin);
    response->set_coordinator_sent_ns(pressureNowNs());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetCachePressure(grpc::ServerContext*          context,
                                              const CachePressureRequestPB* request,
                                              CachePressureResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    response->set_coordinator_received_ns(pressureNowNs());
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "cache pressure fan-out requires tp_rank 0"};
    }
    if (request->max_step_records() > kMaxPressureStepRecords) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "max_step_records exceeds 1024"};
    }
    const int        base = topology.world_rank - topology.tp_rank;
    std::set<int>    seen;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i) {
            ranks.push_back(base + i);
        }
    }
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology target rank"};
        }
        response->add_requested_ranks(rank);
    }
    const auto             deadline      = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    CachePressureRequestPB local_request = *request;
    if (request->drain_remote_completions()) {
        // The validated TP coordinator owns completion intents, even when the
        // caller requests only worker snapshots or lists workers before it.
        // Drain before any fanout; selection and response order stay unchanged.
        const auto status = drainRemoteCompletions(engine_.get(), context, deadline);
        if (!status.ok()) {
            return status;
        }
        local_request.set_drain_remote_completions(false);
    }
    struct Pending {
        grpc::ClientContext                                                       context;
        CachePressureResponsePB                                                   response;
        grpc::Status                                                              status;
        std::shared_ptr<RpcService::Stub>                                         stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<CachePressureResponsePB>> reader;
        CachePressureRankPB*                                                      record;
        int                                                                       rank;
    };
    // Keep channel ownership paired with this request's completion queue.
    RPCPool                               pressure_rpc_pool;
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    grpc::Status                          local_status;
    const auto&                           addresses = maga_init_params_.runtime_config.worker_grpc_addrs;
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        const auto send = pressureNowNs();
        record->set_coordinator_send_ns(send);
        if (rank == topology.world_rank) {
            CachePressureResponsePB local;
            // Preserve each selected rank's physical callback drain, including
            // the owner, without extending the original fanout deadline.
            if (request->drain_remote_completions()) {
                local_status = drainRemoteCompletions(engine_.get(), context, deadline);
            }
            if (local_status.ok()) {
                local_status = GetCachePressureInternal(context, &local_request, &local);
            }
            if (local.ranks_size() == 1) {
                *record = local.ranks(0);
            }
            if (!local_status.ok()) {
                record->set_error(local_status.error_message());
            }
            record->set_coordinator_send_ns(send);
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        auto connection = pressure_rpc_pool.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncGetCachePressureInternal(&call->context, *request, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        void* tag = nullptr;
        bool  ok  = false;
        queue.Next(&tag, &ok);
        auto*      call = static_cast<Pending*>(tag);
        const auto send = call->record->coordinator_send_ns();
        if (ok && call->status.ok() && call->response.schema_version() == 1 && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid rank response" : call->status.error_message());
        }
        call->record->set_coordinator_send_ns(send);
        call->record->set_coordinator_receive_ns(pressureNowNs());
    }
    queue.Shutdown();
    void* drained_tag = nullptr;
    bool  drained_ok  = false;
    while (queue.Next(&drained_tag, &drained_ok)) {}
    response->set_coordinator_sent_ns(pressureNowNs());
    if (request->drain_remote_completions() && std::chrono::system_clock::now() >= deadline) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "remote completion snapshot deadline exceeded"};
    }
    // Preserve peer failures in successful aggregates. A single unsupported local backend is explicit.
    if (ranks.size() == 1 && ranks.front() == topology.world_rank && !local_status.ok()) {
        return local_status;
    }
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile from %s start_step=%d num_steps=%d enable_all_rank=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps(),
                     int(request->enable_all_rank()));
    if (!request->enable_all_rank()) {
        engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
        return grpc::Status::OK;
    }
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "enable_all_rank start_profile must be sent to tp_rank 0");
    }
    if (!tp_broadcaster_) {
        if (maga_init_params_.parallelism_config.tp_size <= 1) {
            RTP_LLM_LOG_INFO("start_profile enable_all_rank with tp_size=1, fallback to local start");
            engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
            return grpc::Status::OK;
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for enable_all_rank start_profile");
    }

    std::vector<StartProfileInternalRequestPB> requests(tp_broadcaster_->workerNum());
    for (auto& internal_request : requests) {
        internal_request.set_trace_name(request->trace_name());
        internal_request.set_start_step(request->start_step());
        internal_request.set_num_steps(request->num_steps());
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const StartProfileInternalRequestPB&        internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncStartProfileInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result =
        tp_broadcaster_->broadcast<StartProfileInternalRequestPB, EmptyPB>(requests, /*timeout_ms=*/3000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast start_profile_internal to tp group");
    }
    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast start_profile_internal to tp group failed");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartProfileInternal(grpc::ServerContext*                 context,
                                                  const StartProfileInternalRequestPB* request,
                                                  EmptyPB*                             response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile_internal from %s start_step=%d num_steps=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps());
    engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StopProfileInternal(grpc::ServerContext*                context,
                                                 const StopProfileInternalRequestPB* request,
                                                 StopProfileResponsePB*              response) {
    return stopProfileLocal(
        context,
        request,
        response,
        std::min(context->deadline(),
                 std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs)));
}

grpc::Status LocalRpcServer::stopProfileLocal(grpc::ServerContext*                  context,
                                              const StopProfileInternalRequestPB*   request,
                                              StopProfileResponsePB*                response,
                                              std::chrono::system_clock::time_point deadline) {
    response->Clear();
    auto* rank = response->add_ranks();
    rank->set_world_rank(maga_init_params_.parallelism_config.world_rank);
    rank->set_service_instance(pressure_instance_);
    if (context->IsCancelled() || std::chrono::system_clock::now() >= deadline) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "stop request deadline exceeded before acceptance"};
    }
    if (!engine_ || !engine_->getProfileStatus().supported) {
        return {grpc::StatusCode::UNIMPLEMENTED, "profile cancellation unsupported"};
    }
    if (request->trace_name().empty() || request->trace_name().size() > 64) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "stop requires the configured trace identity (1..64 bytes)"};
    }
    rank->set_accepted(engine_->stopTimelineProfiling(request->trace_name()));
    if (!rank->accepted()) {
        profileStatusToProto(engine_->getProfileStatus(), rank->mutable_profile());
        return {grpc::StatusCode::FAILED_PRECONDITION, "profile trace identity mismatch"};
    }
    while (true) {
        const auto status = engine_->getProfileStatus();
        profileStatusToProto(status, rank->mutable_profile());
        // A duplicate cleanup of the last finished trace never cancels a newer trace.
        if (status.trace_name != request->trace_name() || status.collection_complete)
            break;
        if (context->IsCancelled() || std::chrono::system_clock::now() >= deadline) {
            rank->set_error("stop accepted; owner-thread collection deadline exceeded");
            break;
        }
        engine_->wakePressureControlProgress();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StopProfile(grpc::ServerContext*        context,
                                         const StopProfileRequestPB* request,
                                         StopProfileResponsePB*      response) {
    response->Clear();
    StopProfileInternalRequestPB internal;
    internal.set_trace_name(request->trace_name());
    if (!request->enable_all_rank()) {
        if (!request->target_ranks().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "target_ranks requires enable_all_rank"};
        }
        return StopProfileInternal(context, &internal, response);
    }
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "all-rank stop requires tp_rank 0"};
    }
    const int        base = topology.world_rank;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i)
            ranks.push_back(base + i);
    }
    std::set<int> seen;
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology stop target"};
        }
    }
    struct Pending {
        grpc::ClientContext                                                     context;
        StopProfileResponsePB                                                   response;
        grpc::Status                                                            status;
        std::shared_ptr<RpcService::Stub>                                       stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<StopProfileResponsePB>> reader;
        StopProfileRankPB*                                                      record;
        int                                                                     rank;
    };
    const auto deadline = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    // Keep channel ownership paired with this request's completion queue.
    RPCPool                               pressure_rpc_pool;
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    const auto&                           addresses = maga_init_params_.runtime_config.worker_grpc_addrs;
    StopProfileRankPB*                    local     = nullptr;
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        if (rank == topology.world_rank) {
            local = record;
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            continue;
        }
        auto connection = pressure_rpc_pool.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncStopProfileInternal(&call->context, internal, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    if (local) {
        // Give local handling the same bounded deadline as peers.
        StopProfileResponsePB local_response;
        const auto            status = stopProfileLocal(context, &internal, &local_response, deadline);
        if (local_response.ranks_size())
            *local = local_response.ranks(0);
        if (!status.ok())
            local->set_error(status.error_message());
    }
    bool cancelled = false;
    for (size_t completed = 0; completed < pending.size();) {
        void*      tag = nullptr;
        bool       ok  = false;
        const auto now = std::chrono::system_clock::now();
        if (now >= deadline || context->IsCancelled()) {
            if (!cancelled) {
                for (auto& call : pending)
                    call->context.TryCancel();
                cancelled = true;
            }
            queue.Next(&tag, &ok);
        } else {
            // Internal responses wait for owner-thread collection, not merely
            // acceptance. Delayed peer flags therefore still receive TP progress.
            if (engine_)
                engine_->wakePressureControlProgress();
            if (queue.AsyncNext(&tag, &ok, std::min(deadline, now + std::chrono::milliseconds(10)))
                != grpc::CompletionQueue::GOT_EVENT)
                continue;
        }
        ++completed;
        auto* call = static_cast<Pending*>(tag);
        if (ok && call->status.ok() && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid stop response" : call->status.error_message());
        }
    }
    queue.Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (queue.Next(&tag, &ok)) {}
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::authorizeTorchAllocatorDump(const TorchAllocatorDumpRequestPB& request) const {
    if (!torch_allocator_dump_enabled_) {
        return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "allocator dump unavailable");
    }
    if (torch_allocator_dump_auth_token_.empty()
        || !constantTimeEquals(request.auth_token(), torch_allocator_dump_auth_token_)) {
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "allocator dump authentication failed");
    }
    if (!validAllocatorDumpId(request.dump_id())) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid allocator dump id");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::authorizeTorchAllocatorDumpInternal(grpc::ServerContext*               context,
                                                                 const TorchAllocatorDumpRequestPB& request) const {
    const auto authorization = authorizeTorchAllocatorDump(request);
    if (!authorization.ok()) {
        return authorization;
    }
    const auto& parallelism_config = maga_init_params_.parallelism_config;
    if ((parallelism_config.tp_size <= 1 && parallelism_config.pp_size <= 1)
        || !isTorchAllocatorDumpInternalPeer(context)) {
        return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "allocator dump internal fanout denied");
    }
    return grpc::Status::OK;
}

bool LocalRpcServer::isTorchAllocatorDumpInternalPeer(grpc::ServerContext* context) const {
    if (context == nullptr) {
        return false;
    }
    const auto peer_host = endpointHost(context->peer());
    if (peer_host.empty()) {
        return false;
    }
    return std::any_of(maga_init_params_.runtime_config.worker_grpc_addrs.begin(),
                       maga_init_params_.runtime_config.worker_grpc_addrs.end(),
                       [&peer_host](const std::string& worker_addr) { return endpointHost(worker_addr) == peer_host; });
}

grpc::Status
LocalRpcServer::beginTorchAllocatorDump(const std::string& dump_id, bool internal_fanout, bool* owns_admission) {
    std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
    if (owns_admission != nullptr) {
        *owns_admission = false;
    }
    if (internal_fanout && torch_allocator_dump_in_progress_ && torch_allocator_dump_active_started_by_public_
        && !torch_allocator_dump_active_internal_started_ && torch_allocator_dump_active_id_ == dump_id) {
        torch_allocator_dump_active_internal_started_ = true;
        return grpc::Status::OK;
    }
    if (torch_allocator_dump_ids_.count(dump_id) != 0) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "allocator dump id has already been used");
    }
    if (torch_allocator_dump_in_progress_) {
        return grpc::Status(grpc::StatusCode::ABORTED, "allocator dump already in progress");
    }
    if (torch_allocator_dump_has_completed_) {
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - torch_allocator_dump_last_completed_at_)
                .count();
        if (elapsed < torch_allocator_dump_cooldown_seconds_) {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "allocator dump cooldown active");
        }
    }
    if (torch_allocator_dump_id_order_.size() == kAllocatorDumpReplayHistoryMax) {
        torch_allocator_dump_ids_.erase(torch_allocator_dump_id_order_.front());
        torch_allocator_dump_id_order_.pop_front();
    }
    torch_allocator_dump_ids_.insert(dump_id);
    torch_allocator_dump_id_order_.push_back(dump_id);
    torch_allocator_dump_in_progress_              = true;
    torch_allocator_dump_active_started_by_public_ = !internal_fanout;
    torch_allocator_dump_active_internal_started_  = false;
    torch_allocator_dump_active_id_                = dump_id;
    if (owns_admission != nullptr) {
        *owns_admission = true;
    }
    return grpc::Status::OK;
}

void LocalRpcServer::finishTorchAllocatorDump() {
    std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
    torch_allocator_dump_in_progress_              = false;
    torch_allocator_dump_has_completed_            = true;
    torch_allocator_dump_active_started_by_public_ = false;
    torch_allocator_dump_active_internal_started_  = false;
    torch_allocator_dump_active_id_.clear();
    torch_allocator_dump_last_completed_at_ = std::chrono::steady_clock::now();
}

grpc::Status LocalRpcServer::aggregateTorchAllocatorDumpResults(const std::string&                             dump_id,
                                                                const std::vector<TorchAllocatorDumpResultPB>& results,
                                                                TorchAllocatorDumpResponsePB* response) const {
    for (const auto& result : results) {
        if (result.dump_id() != dump_id) {
            response->Clear();
            RTP_LLM_LOG_WARNING("allocator dump response id mismatch expected=%s actual=%s world_rank=%ld",
                                dump_id.c_str(),
                                result.dump_id().c_str(),
                                result.world_rank());
            return grpc::Status(grpc::StatusCode::DATA_LOSS, "allocator dump response id mismatch");
        }
    }
    for (const auto& result : results) {
        response->add_results()->CopyFrom(result);
    }
    return grpc::Status::OK;
}

TorchAllocatorDumpResultPB LocalRpcServer::dumpTorchAllocatorOnCurrentProcess(const std::string& dump_id) {
    const auto&                parallelism_config = maga_init_params_.parallelism_config;
    TorchAllocatorDumpResultPB result;
    result.set_world_rank(parallelism_config.world_rank);
    result.set_dp_rank(parallelism_config.dp_rank);
    result.set_tp_rank(parallelism_config.tp_rank);
    result.set_local_rank(parallelism_config.local_rank);
    result.set_pid(getpid());
    result.set_dump_id(dump_id);

    auto output_path = dumpTorchCudaOomDiagnostics(parallelism_config.local_rank, dump_id);
    result.set_success(!output_path.empty());
    result.set_file_path(std::move(output_path));
    if (!result.success()) {
        result.set_error("allocator dump failed; see backend log");
    }
    return result;
}

grpc::Status LocalRpcServer::executeAdmittedTorchAllocatorDump(const TorchAllocatorDumpRequestPB& request,
                                                               TorchAllocatorDumpResponsePB*      response) {
    const auto& parallelism_config = maga_init_params_.parallelism_config;
    if (parallelism_config.tp_size <= 1 && parallelism_config.pp_size <= 1) {
        return aggregateTorchAllocatorDumpResults(
            request.dump_id(), {dumpTorchAllocatorOnCurrentProcess(request.dump_id())}, response);
    }
    if (!tp_broadcaster_) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for allocator dump");
    }

    std::vector<TorchAllocatorDumpRequestPB> requests(tp_broadcaster_->workerNum(), request);
    auto                                     rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const TorchAllocatorDumpRequestPB&          internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncDumpTorchAllocatorInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result = tp_broadcaster_->broadcast<TorchAllocatorDumpRequestPB, TorchAllocatorDumpResultPB>(
        requests, /*timeout_ms=*/60000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast allocator dump to tp group");
    }
    try {
        broadcast_result->waitDone();
    } catch (const std::exception& exception) {
        RTP_LLM_LOG_WARNING(
            "allocator dump broadcast failed dump_id=%s: %s", request.dump_id().c_str(), exception.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, "allocator dump broadcast failed");
    }
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "allocator dump broadcast to tp group failed");
    }
    return aggregateTorchAllocatorDumpResults(request.dump_id(), broadcast_result->responses(), response);
}

grpc::Status LocalRpcServer::DumpTorchAllocator(grpc::ServerContext*               context,
                                                const TorchAllocatorDumpRequestPB* request,
                                                TorchAllocatorDumpResponsePB*      response) {
    response->Clear();
    const auto authorization = authorizeTorchAllocatorDump(*request);
    if (!authorization.ok()) {
        RTP_LLM_LOG_WARNING("rejected dump_torch_allocator from %s", context->peer().c_str());
        return authorization;
    }
    RTP_LLM_LOG_INFO("dump_torch_allocator from %s dump_id=%s", context->peer().c_str(), request->dump_id().c_str());
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "dump_torch_allocator must be sent to tp_rank 0");
    }
    const auto admission = beginTorchAllocatorDump(request->dump_id());
    if (!admission.ok()) {
        return admission;
    }

    try {
        const auto result = executeAdmittedTorchAllocatorDump(*request, response);
        finishTorchAllocatorDump();
        return result;
    } catch (...) {
        finishTorchAllocatorDump();
        throw;
    }
}

grpc::Status LocalRpcServer::DumpTorchAllocatorInternal(grpc::ServerContext*               context,
                                                        const TorchAllocatorDumpRequestPB* request,
                                                        TorchAllocatorDumpResultPB*        response) {
    response->Clear();
    const auto authorization = authorizeTorchAllocatorDumpInternal(context, *request);
    if (!authorization.ok()) {
        RTP_LLM_LOG_WARNING("rejected dump_torch_allocator_internal from %s", context->peer().c_str());
        return authorization;
    }
    bool       owns_admission = false;
    const auto admission      = beginTorchAllocatorDump(request->dump_id(), /*internal_fanout=*/true, &owns_admission);
    if (!admission.ok()) {
        return admission;
    }
    RTP_LLM_LOG_INFO(
        "dump_torch_allocator_internal from %s dump_id=%s", context->peer().c_str(), request->dump_id().c_str());
    try {
        response->CopyFrom(dumpTorchAllocatorOnCurrentProcess(request->dump_id()));
        if (owns_admission) {
            finishTorchAllocatorDump();
        }
        return grpc::Status::OK;
    } catch (...) {
        if (owns_admission) {
            finishTorchAllocatorDump();
        }
        throw;
    }
}

grpc::Status
LocalRpcServer::CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    response->set_health("OK");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::UpdateEplbConfig(grpc::ServerContext*             context,
                                              const UpdateEplbConfigRequestPB* request,
                                              EmptyPB*                         response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    const string mode_str = request->mode();
    EPLBConfig   config;
    if (mode_str == "EPLB" || mode_str == "eplb") {
        config.eplb_mode = EplbMode::EPLB;
    } else if (mode_str == "STATS" || mode_str == "stats") {
        config.eplb_mode = EplbMode::STATS;
    } else if (mode_str == "NONE" || mode_str == "none") {
        config.eplb_mode = EplbMode::NONE;
    } else if (mode_str == "ALL" || mode_str == "all") {
        config.eplb_mode = EplbMode::ALL;
    } else {
        RTP_LLM_LOG_WARNING("set eplb mode failed, unknown mode : %s", mode_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid eplb mode");
    }
    config.eplb_update_time = request->update_time();
    engine_->updateEplbConfig(config);
    return grpc::Status::OK;
}

void LocalRpcServer::reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_ws_time_us) {
    RpcWorkerStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = request_after_ws_time_us - request_begin_time_us;
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcWorkerStatusMetrics, RpcWorkerStatusMetricsCollector>(nullptr, &collector);
    }
}

void LocalRpcServer::reportCacheStatusTime(int64_t request_begin_time_us) {
    RpcCacheStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = (currentTimeUs() - request_begin_time_us);
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcCacheStatusMetrics, RpcCacheStatusMetricsCollector>(nullptr, &collector);
    }
}

::grpc::Status LocalRpcServer::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    const auto failure_status = [&](grpc::StatusCode status_code, const std::string& message) {
        if (!request->has_p2p_request()) {
            return grpc::Status(status_code, message);
        }
        const auto& p2p_request = request->p2p_request();
        ErrorInfo   error(status_code == grpc::StatusCode::CANCELLED ?
                            ErrorCode::CANCELLED :
                            ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                        message);
        if (response->has_p2p_response() && response->p2p_response().error_code() != ErrorCodePB::NONE_ERROR) {
            error = ErrorInfo(transRPCErrorCode(response->p2p_response().error_code()),
                              response->p2p_response().error_message());
        }
        const std::string detail =
            "reporter_role=" + roleTypeToString(maga_init_params_.pd_sep_config.role_type)
            + " stage=ExecuteFunction tp_rank=" + std::to_string(maga_init_params_.parallelism_config.tp_rank)
            + " request_id=" + std::to_string(p2p_request.request_id()) + " unique_key=" + p2p_request.unique_key()
            + " type=" + std::to_string(p2p_request.type()) + ": " + error.ToString();
        RTP_LLM_LOG_WARNING("P2P ExecuteFunction rejected, request_id=%ld unique_key=%s type=%d peer=%s "
                            "grpc_code=%d error_code=%d error_name=%s error=%s",
                            p2p_request.request_id(),
                            p2p_request.unique_key().c_str(),
                            static_cast<int>(p2p_request.type()),
                            context->peer().c_str(),
                            static_cast<int>(status_code),
                            static_cast<int>(error.code()),
                            ErrorCodeToString(error.code()).c_str(),
                            detail.c_str());
        ErrorDetailsPB details;
        details.set_error_code(static_cast<int>(error.code()));
        details.set_error_message(detail);
        // Preserve the existing transport status and failure path; carry the worker cause in details.
        return grpc::Status(status_code, detail, details.SerializeAsString());
    };
    RTP_LLM_LOG_DEBUG("receive execute function request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("execute function failed, request is cancelled");
        return failure_status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("execute function failed, engine is null");
        return failure_status(grpc::StatusCode::INTERNAL, "engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("execute function failed, cache manager is null");
        return failure_status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }
    FunctionRequestPB bounded_request;
    if (request->has_remote_request()
        && (request->remote_request().op() == REMOTE_OPERATION_READ
            || request->remote_request().op() == REMOTE_OPERATION_WRITE)) {
        if (request->remote_request().timeout_ms() <= 0) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "remote task budget must be positive");
        }
        const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      context->deadline() - std::chrono::system_clock::now())
                                      .count();
        if (remaining_ms <= 0) {
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "remote task expired before execution");
        }
        bounded_request.CopyFrom(*request);
        bounded_request.mutable_remote_request()->set_timeout_ms(
            std::min<int64_t>(request->remote_request().timeout_ms(), remaining_ms));
        request = &bounded_request;
    }
    if (!cache_manager->executeFunction(*request, *response)) {
        std::string request_case;
        if (request->has_mem_request()) {
            request_case = "mem_request";
        } else if (request->has_remote_request()) {
            request_case = "remote_request(trace_id=" + request->remote_request().trace_id() + ")";
        } else if (request->has_p2p_request()) {
            request_case = "p2p_request";
        } else {
            request_case = "unknown";
        }
        RTP_LLM_LOG_WARNING(
            "execute function failed, peer: %s, request_case: %s", context->peer().c_str(), request_case.c_str());
        const std::string error_msg = "execute function failed, request_case: " + request_case;
        return failure_status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    engine_->pause();
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s,", context->peer().c_str());
    engine_->restart();
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("Receive update weights request from: %s", context->peer().c_str());
    try {
        if (request->name().empty() || request->desc().empty() || request->method().empty()) {
            throw std::runtime_error("Missing required field(s) in request");
        }
        {
            py::gil_scoped_acquire acquire;
            py::dict               req;
            req["name"]   = request->name();
            req["desc"]   = request->desc();
            req["method"] = request->method();
            weight_manager_.attr("update")(req);
        }
        return grpc::Status::OK;
    } catch (const py::error_already_set& e) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        std::string err_msg = value ? PyUnicode_AsUTF8(value) : "Unknown Python error";
        return {grpc::StatusCode::INTERNAL, "exception from python: " + err_msg};
    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL, "exception from C++: " + std::string(e.what())};
    }
}
}  // namespace rtp_llm
