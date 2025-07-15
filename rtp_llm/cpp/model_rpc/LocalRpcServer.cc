#include <memory>
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/proto/model_rpc_service.pb.h"

using namespace std;

namespace rtp_llm {

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  py::object                                    mm_process_engine,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_ = maga_init_params;
    metrics_reporter_ = maga_init_params.metrics_reporter;

    if (propose_params) {
        propose_maga_init_params_ = propose_params.get();
        RTP_LLM_LOG_INFO("init speculative engine");
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
        RTP_LLM_LOG_INFO("init normal engine");
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
        QueryConverter::transResponse(&outputs_pb, &(result.value()));
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
        if (stream->queryPdSep() && stream->waitForRemoteGenerate()) {
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

LoadBalanceInfo LocalRpcServer::getLoadBalanceInfo(int64_t latest_version) {
    return engine_->getLoadBalanceInfo(latest_version);
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

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finised_version) {
    auto info               = meta_->getEngineScheduleInfo(latest_finised_version);
    auto last_schedule_time = engine_->getLastScheduleTime();
    // in case last_schedule_delta is negative
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
}

::grpc::Status LocalRpcServer::RemoteGetCache(::grpc::ServerContext*              context,
                                              const ::BroadcastGetCacheRequestPB* request,
                                              ::BroadcastGetCacheResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive get cache rpc request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());

    const int64_t request_id = request->request_id();
    if (!engine_) {
        RTP_LLM_LOG_WARNING("get cache failed, receive get cache rpc request but engine is null, request: %ld",
                            request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("get cache failed, receive get cache rpc request but cache manager is null, request: %ld",
                            request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }

    std::vector<int64_t> cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    std::vector<int32_t> block_ids(request->block_ids().begin(), request->block_ids().end());
    if (!cache_manager->getCacheFrom3FSForRank(cache_keys, block_ids, request_id)) {
        RTP_LLM_LOG_DEBUG("get cache failed, receive get cache rpc request but get cache failed, request: [%s]",
                          request->ShortDebugString().c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager get cache failed");
    }
    return grpc::Status::OK;
}

::grpc::Status LocalRpcServer::DistKvCache(::grpc::ServerContext*        context,
                                           const ::DistKvCacheRequestPB* request,
                                           ::DistKvCacheResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive dist kvcache request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());

    const int64_t request_id = request->request_id();
    const auto    op_code    = request->op();
    if (op_code == ::DistKvCacheOp::UNKNOWN) {
        RTP_LLM_LOG_WARNING("dist kvcache failed, op code is unknown, request: %ld", request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, "op code is unknown");
    }

    if (!engine_) {
        RTP_LLM_LOG_WARNING("dist kvcache failed, engine is null, request: %ld", request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("dist kvcache failed, cache manager is null, request: %ld, op: %s",
                            request_id,
                            ::DistKvCacheOp_Name(op_code).c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }

    std::vector<int64_t>               cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    std::vector<int32_t>               block_ids(request->block_ids().begin(), request->block_ids().end());
    std::map<std::string, std::string> extra_metas;
    for (const auto& meta : request->extra_metas()) {
        extra_metas[meta.key()] = meta.value();
    }

    bool result = false;
    if (op_code == ::DistKvCacheOp::GET) {
        result = cache_manager->getCacheForRank(cache_keys, block_ids, request_id, extra_metas);
    } else {
        result = cache_manager->putCacheForRank(cache_keys, block_ids, request_id, extra_metas);
    }

    if (!result) {
        RTP_LLM_LOG_WARNING("dist kvcache failed, %s cache failed, request: [%s]",
                            ::DistKvCacheOp_Name(op_code).c_str(),
                            request->ShortDebugString().c_str());
        const std::string error_msg = "cache manager " + ::DistKvCacheOp_Name(op_code) + " failed";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

}  // namespace rtp_llm
