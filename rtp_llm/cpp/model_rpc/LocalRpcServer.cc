#include <memory>
#include <chrono>
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache_new/types.h"

using namespace std;

namespace rtp_llm {

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  py::object                                    mm_process_engine,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_ = maga_init_params;
    metrics_reporter_ = maga_init_params.metrics_reporter;
    RTP_LLM_LOG_INFO("LocalRpcServer aux_string %s",
                     maga_init_params_.gpt_init_parameter.misc_config.aux_string.c_str());
    if (propose_params) {
        propose_maga_init_params_ = propose_params.get();
        if (!mm_process_engine.is_none()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "Multimodal processing is not supported for speculative engine");
        }
        pybind11::gil_scoped_release release;
        RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                "running engine init with gil held may cause program hang, please check");
        std::unique_ptr<SpeculativeEngine> sp_engine =
            std::make_unique<SpeculativeEngine>(maga_init_params, std::move(propose_params));
        auto status = sp_engine->init();
        if (!status.ok()) {
            return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
        }
        engine_ = std::move(sp_engine);
    } else {
        {
            pybind11::gil_scoped_release release;
            RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                    "running engine init with gil held may cause program hang, please check");
            engine_.reset(new NormalEngine(maga_init_params));
        }
        if (!mm_process_engine.is_none()) {
            auto vit_separation = maga_init_params.gpt_init_parameter.vit_separation_;
            if (vit_separation == 2) {
                mm_processor_.reset(
                    new RemoteMultimodalProcessor(mm_process_engine, maga_init_params.gpt_init_parameter));
            } else if (vit_separation == 0) {
                mm_processor_.reset(
                    new LocalMultimodalProcessor(mm_process_engine, maga_init_params.gpt_init_parameter));
            } else {
                return grpc::Status(grpc::StatusCode::INTERNAL, "invalid vit separation value in config");
            }
        }
    }

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    const auto& error_msg = error_info.ToString();
    RTP_LLM_LOG_WARNING("request [%s], error code [%s], error message [%s]",
                        request_key.c_str(),
                        ErrorCodeToString(error_info.code()).c_str(),
                        error_msg.c_str());
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        RTP_LLM_LOG_WARNING("request [%s] error details serialize to string failed", request_key.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*             context,
                                              const string&                    request_key,
                                              WriterInterface*                 writer,
                                              std::shared_ptr<GenerateStream>& stream) {
    while (!stream->finished() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, result.status());
            } else {
                break;
            }
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(
            &outputs_pb, &(result.value()), maga_init_params_.gpt_init_parameter.misc_config.aux_string);
        if (context->IsCancelled()) {
            stream->cancel();
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            stream->cancel();
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->needRemoteGenerate()) {
            break;
        }
        if (stream->queryPdSep()) {
            stream->waitForRemoteGenerate();
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("request [%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) {
    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_, meta_);
    auto input = QueryConverter::transQuery(request);

    // need to check client has buffer at first
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    input->lora_id  = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    generate_context.setStream(engine_->enqueue(input));

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key, writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

grpc::Status
LocalRpcServer::GetCacheStatus(grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) {
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
    int64_t request_begin_time_us   = currentTimeUs();
    int64_t latest_finished_version = request->latest_finished_version();
    RTP_LLM_LOG_DEBUG(
        "receive workerStatus rpc request from client: %s, latest_finished_version: %ld, config role_type: %d",
        context->peer().c_str(),
        latest_finished_version,
        maga_init_params_.gpt_init_parameter.role_type_);

    WorkerStatusInfo status_info              = getWorkerStatusInfo(latest_finished_version);
    int64_t          request_after_ws_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("getWorkerStatusInfo took %ld us", request_after_ws_time_us - request_begin_time_us);

    const auto& engine_schedule_info = status_info.engine_schedule_info;
    response->set_role(status_info.role);

    for (const auto& task : engine_schedule_info.running_task_info_list) {
        TaskInfoPB* task_info = response->add_running_task_info();
        task_info->set_request_id(task.request_id);
        task_info->set_inter_request_id(task.inter_request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_is_waiting(task.is_waiting);
    }

    for (const auto& task : engine_schedule_info.finished_task_info_list) {
        TaskInfoPB* task_info = response->add_finished_task_list();
        task_info->set_request_id(task.request_id);
        task_info->set_inter_request_id(task.inter_request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_is_waiting(task.is_waiting);
    }
    response->set_dp_size(status_info.dp_size);
    response->set_tp_size(status_info.tp_size);
    response->set_status_version(status_info.status_version);
    response->set_alive(status_info.alive);
    response->set_precision(status_info.precision);
    reportWorkerStatusTime(request_begin_time_us, request_after_ws_time_us);
    return grpc::Status::OK;
}

WorkerStatusInfo LocalRpcServer::getWorkerStatusInfo(int64_t latest_finished_version) {
    WorkerStatusInfo status_info;
    status_info.engine_schedule_info = getEngineScheduleInfo(latest_finished_version);
    switch (maga_init_params_.gpt_init_parameter.role_type_) {
        case RoleType::PDFUSION:
            status_info.role = "RoleType.PDFUSION";
            break;
        case RoleType::PREFILL:
            status_info.role = "RoleType.PREFILL";
            break;
        case RoleType::DECODE:
            status_info.role = "RoleType.DECODE";
            break;
        case RoleType::VIT:
            status_info.role = "RoleType.VIT";
            break;
        case RoleType::FRONTEND:
            status_info.role = "RoleType.FRONTEND";
            break;
        default:
            status_info.role = "RoleType.UNKNOWN";
    }
    status_info.dp_size        = maga_init_params_.gpt_init_parameter.dp_size_;
    status_info.tp_size        = maga_init_params_.gpt_init_parameter.tp_size_;
    status_info.dp_rank        = maga_init_params_.gpt_init_parameter.dp_rank_;
    status_info.status_version = currentTimeUs();
    status_info.alive          = true;
    auto quant_method          = maga_init_params_.gpt_init_parameter.quant_algo_.getQuantMethod();

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

void LocalRpcServer::addLora(const std::string&                        adapter_name,
                             const rtp_llm::lora::loraLayerWeightsMap& lora_a_weights,
                             const rtp_llm::lora::loraLayerWeightsMap& lora_b_weights) {
    engine_->addLora(adapter_name, lora_a_weights, lora_b_weights);
}
void LocalRpcServer::removeLora(const std::string& adapter_name) {
    engine_->removeLora(adapter_name);
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finished_version) {
    EngineScheduleInfo                        info = meta_->getEngineScheduleInfo(latest_finished_version);
    std::vector<EngineScheduleInfo::TaskInfo> running_task_info_list = engine_->getScheduler().runningTaskList();
    for (auto& task_info : info.running_task_info_list) {
        for (auto& running_task : running_task_info_list) {
            if (task_info.inter_request_id == running_task.inter_request_id) {
                task_info.is_waiting = false;
            }
        }
    }
    auto last_schedule_time = engine_->getLastScheduleTime();
    // in case last_schedule_delta is negative
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
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

::grpc::Status LocalRpcServer::CopyCache(::grpc::ServerContext*      context,
                                         const ::CopyCacheRequestPB* request,
                                         ::CopyCacheResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive broadcast tp request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("copy cache failed, request is cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_) {
        RTP_LLM_CHECK_WITH_INFO(false, "copy cache failed, engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_CHECK_WITH_INFO(false, "copy cache failed, cache manager is null");
    }

    if (!cache_manager->copyCache(*request, *response)) {
        RTP_LLM_LOG_WARNING("cache manager copy cache failed, request: [%s]", request->DebugString().c_str());
        const std::string error_msg = "cache manager copy cache failed, request: [" + request->DebugString() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

}  // namespace rtp_llm
