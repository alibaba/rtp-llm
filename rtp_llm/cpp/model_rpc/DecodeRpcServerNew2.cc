#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "autil/StringUtil.h"
#include <cstring>

namespace rtp_llm {

grpc::Status DecodeRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                       py::object                                             mm_process_engine,
                                       std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    prefill_server_caller_ = std::make_shared<PrefillServerCaller>(
        process_id_, maga_init_params.gpt_init_parameter.decode_polling_call_prefill_ms_, engine_->isMTPEagle());
    p2p_connector_decode_ =
        std::make_shared<P2PConnectorDecode>(maga_init_params.gpt_init_parameter,
                                             engine_->getDevice(),
                                             engine_->resourceContext().cache_manager->getAllocator());
    if (!p2p_connector_decode_->init()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, p2p_connector_decode is null");
        // return grpc::Status::OK;
        return grpc::Status(grpc::StatusCode::INTERNAL, "p2p_connector_decode init failed");
    }

    auto callback = p2p_connector_decode_->makeCallback();
    if (!callback) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, make callback failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "make callback failed");
    }
    tp_broadcast_service_->registerCallback(callback);

    RTP_LLM_LOG_INFO("decode rpc server new2 init success");
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                     const GenerateInputPB*                 request,
                                                     grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    RTP_LLM_LOG_INFO("decode rpc server new2 GenerateStreamCall, request: %lld", request->request_id());

    int64_t start_time_us = currentTimeUs();
    auto    deadline_us   = start_time_us + request->generate_config().timeout_ms() * 1000;  // 转换为微秒

    RTP_LLM_LOG_INFO("decode rpc server new2 GenerateStreamCall, get prefill role addr");
    auto prefill_role_addr = getPrefillRoleAddr(request);
    if (prefill_role_addr.ip.empty()) {
        RTP_LLM_LOG_WARNING("request [%lld] get prefill role addr failed", request->request_id());
        return serializeErrorMsg(std::to_string(request->request_id()),
                                 ErrorInfo(ErrorCode::GET_HOST_FAILED, "get prefill role addr failed"));
    }

    // TODO: inter_request_id
    auto unique_key = autil::NetUtil::getBindIp() + "_" + std::to_string(request->request_id()) + "_"
                      + std::to_string(currentTimeUs());

    RTP_LLM_LOG_INFO("decode rpc server new2 GenerateStreamCall, init stream");
    auto stream = initStream(request);
    if (!stream) {
        RTP_LLM_LOG_WARNING("request [%lld] init stream failed", request->request_id());
        return serializeErrorMsg(std::to_string(request->request_id()),
                                 ErrorInfo(ErrorCode::UNKNOWN_ERROR, "init stream failed"));
    }

    if (stream->kvCache().batchSize() > 1) {
        RTP_LLM_LOG_WARNING("request [%lld] batch size > 1, not supported", request->request_id());
        return serializeErrorMsg(std::to_string(request->request_id()),
                                 ErrorInfo(ErrorCode::INVALID_PARAMS, "batch size > 1 not supported"));
    }

    RTP_LLM_LOG_INFO("decode rpc server new2 GenerateStreamCall, call prefill");
    auto prefill_rpc_context = prefill_server_caller_->callPrefill(
        request, prefill_role_addr.ip, prefill_role_addr.grpc_port, unique_key, deadline_us);
    if (!prefill_rpc_context) {
        RTP_LLM_LOG_WARNING("request [%lld] call prefill failed", request->request_id());
        return serializeErrorMsg(std::to_string(request->request_id()),
                                 ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, "call prefill failed"));
    }

    if (!loadKVCacheFromPrefill(stream, unique_key, prefill_role_addr.ip, prefill_role_addr.grpc_port)) {
        RTP_LLM_LOG_WARNING("request [%lld] load KV cache from prefill failed", request->request_id());
        return serializeErrorMsg(std::to_string(request->request_id()),
                                 ErrorInfo(ErrorCode::LOAD_KV_CACHE_FAILED, "load KV cache from prefill failed"));
    }

    auto prefill_status = prefill_rpc_context->waitPrefillDone(stream, server_context, response_writer);
    if (!prefill_status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] wait prefill done failed", request->request_id());
        return prefill_status;
    }

    RTP_LLM_LOG_INFO("decode rpc server new2 GenerateStreamCall, wait prefill done");

    // prefill generate finished, no need to local generate
    if (prefill_rpc_context->isGenerateFinished()) {
        return grpc::Status::OK;
    }

    auto error_info = localGenerate(prefill_rpc_context, stream, server_context, response_writer);
    if (!error_info.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] local generate failed, error code %d(%s)",
                            request->request_id(),
                            error_info.code(),
                            error_info.ToString().c_str());
        return serializeErrorMsg(std::to_string(request->request_id()), error_info);
    }
    return grpc::Status::OK;
}

RoleAddr DecodeRpcServerNew2::getPrefillRoleAddr(const GenerateInputPB* request) {
    auto role_addrs = QueryConverter::getRoleAddrs(&request->generate_config());
    for (auto& role_addr : role_addrs) {
        if (role_addr.role == RoleType::PREFILL) {
            return role_addr;
        }
    }
    return RoleAddr(RoleType::PREFILL, "", 0, 0);
}

std::shared_ptr<GenerateStream> DecodeRpcServerNew2::initStream(const GenerateInputPB* request) {
    auto generate_input = QueryConverter::transQuery(request);
    auto stream         = engine_->makeStream(generate_input);
    if (!stream) {
        RTP_LLM_LOG_WARNING("request [%lld] init stream failed", request->request_id());
        return nullptr;
    }
    RTP_LLM_LOG_INFO("request [%lld] init stream, init KV block", request->request_id());
    auto status = stream->initKVBlock(0);
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] init KV block failed", request->request_id());
        return nullptr;
    }
    return stream;
}

bool DecodeRpcServerNew2::loadKVCacheFromPrefill(const std::shared_ptr<GenerateStream>& stream,
                                                 const std::string&                     unique_key,
                                                 const std::string&                     prefill_ip,
                                                 uint32_t                               prefill_port) {
    // TODO: fix this
    auto& stream_resource     = const_cast<BatchKVCacheResource&>(stream->kvCache()).resource(0);
    auto  stream_resource_ptr = std::make_shared<KVCacheResourceV1>(stream_resource);

    auto p2p_meta = std::make_shared<P2PConnectorDecodeMeta>(
        stream->generateInput()->request_id, unique_key, prefill_ip, prefill_port, stream->getDeadlineMs());
    auto p2p_load_context = p2p_connector_decode_->asyncRead(stream_resource_ptr, p2p_meta);
    if (!p2p_load_context) {
        RTP_LLM_LOG_WARNING("request [%lld] async read cache from prefill failed", stream->generateInput()->request_id);
        return false;
    }

    // TODO: wait done with timeout
    p2p_load_context->waitDone();
    if (!p2p_load_context->success()) {
        RTP_LLM_LOG_WARNING("request [%lld] async read cache from prefill failed", stream->generateInput()->request_id);
        return false;
    }
    return true;
}

ErrorInfo DecodeRpcServerNew2::localGenerate(const std::shared_ptr<PrefillServerCallerContext>& prefill_rpc_context,
                                             std::shared_ptr<GenerateStream>&                   stream,
                                             grpc::ServerContext*                               server_context,
                                             grpc::ServerWriter<GenerateOutputsPB>*             response_writer) {
    stream->setIsContextStream(false);
    stream->step();

    auto new_tokens = engine_->getDevice()->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)stream->nextBatchSize(), (size_t)1}, rtp_llm::AllocationType::HOST},
        {});

    // 从 response 的 output_ids 中获取第一个 token id
    const auto& output_pb     = prefill_rpc_context->response.generate_outputs(0);
    const auto& output_ids_pb = output_pb.output_ids();

    int32_t first_token_id = 0;
    if (output_ids_pb.data_type() == TensorPB_DataType::TensorPB_DataType_INT32
        && output_ids_pb.int32_data().size() >= sizeof(int32_t)) {
        // 从 int32_data 中读取第一个 int32 值
        std::memcpy(&first_token_id, output_ids_pb.int32_data().data(), sizeof(int32_t));
    } else {
        RTP_LLM_LOG_WARNING("request [%lld] invalid output_ids format, cannot extract first token id",
                            stream->generateInput()->request_id);
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "invalid output_ids format");
    }

    auto data = new_tokens->data<int32_t>();
    *data     = first_token_id;
    stream->incLastOutputPos();
    stream->update({new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    if (propose_maga_init_params_) {
        stream->setReuseLength(stream->seqLength() - 1);
        stream->setFallbackPrefixLength(stream->reuseLength());
        stream->setSpEditRun(false);
    }
    stream->resetBeginTime(currentTimeUs());
    engine_->enqueue(stream);
    auto grpc_status =
        pollStreamOutput(server_context, std::to_string(stream->generateInput()->request_id), response_writer, stream);
    if (!grpc_status.ok()) {
        // 将 gRPC 状态码转换为 ErrorCode
        ErrorCode error_code = ErrorCode::UNKNOWN_ERROR;
        if (grpc_status.error_code() == grpc::StatusCode::CANCELLED) {
            error_code = ErrorCode::CANCELLED;
        } else if (grpc_status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
            error_code = ErrorCode::GENERATE_TIMEOUT;
        } else if (grpc_status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
            error_code = ErrorCode::MALLOC_FAILED;
        }
        return ErrorInfo(error_code, grpc_status.error_message());
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
