#include <memory>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeEngine.h"
#include "maga_transformer/cpp/model_rpc/LocalRpcServer.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

grpc::Status LocalRpcServer::init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params) {
    maga_init_params_ = maga_init_params;
    metrics_reporter_ = maga_init_params.metrics_reporter;
    if (propose_params) {
        FT_LOG_INFO("init speculative engine");
        if (!mm_process_engine.is_none()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Multimodal processing is not supported for speculative engine");
        }
        std::unique_ptr<SpeculativeEngine> sp_engine = std::make_unique<SpeculativeEngine>(maga_init_params, std::move(propose_params));
        auto status = sp_engine->init();
        if (!status.ok()) {
            return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
        }
        engine_ = std::move(sp_engine);
    } else {
        FT_LOG_INFO("init normal engine");
        engine_.reset(new NormalEngine(maga_init_params));
        if (!mm_process_engine.is_none()) {
            mm_processor_.reset(new MultimodalProcessor(mm_process_engine,
                maga_init_params.gpt_init_parameter.mm_sep_tokens_,
                maga_init_params.gpt_init_parameter.include_sep_tokens_,
                maga_init_params.gpt_init_parameter.max_seq_len_));
        }
    }

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    const auto& error_msg = error_info.ToString();
    FT_LOG_WARNING("request [%s], error code [%s], error message [%s]",
              request_key.c_str(), ErrorCodeToString(error_info.code()).c_str(), error_msg.c_str());
    auto grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        FT_LOG_WARNING("request [%s] error details serialize to string failed", request_key.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*              context,
                                              const string&                     request_key,
                                              WriterInterface*                  writer,
                                              std::shared_ptr<GenerateStream>&  stream) {
    while (!stream->finished() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, result.status());
            } else {
                break;
            }
        }
        FT_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(&outputs_pb, &(result.value()));
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_WARNING("request:[%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            stream->cancel();
            FT_LOG_WARNING("request:[%s] write outputs pb failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->needRemoteGenerate()) {
            break;
        }
    }
    FT_LOG_DEBUG("request:[%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                    context,
                                                const GenerateInputPB*                  request,
                                                grpc::ServerWriter<GenerateOutputsPB>*  writer) {
    AtomicGuard request_guard(onflight_requests_);
    auto request_id = request->request_id();
    FT_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context = GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_);
    auto input = QueryConverter::transQuery(request);

    // need to check client has buffer at first
    // todo: catch python exception, such as download timeout
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            generate_context.error_status = grpc::Status(grpc::StatusCode::CANCELLED, mm_res.ToString());
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    input->lora_id = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    FT_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    generate_context.stream = engine_->enqueue(input);
    FT_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status = pollStreamOutput(
            context, generate_context.request_key, writer, generate_context.stream);
    return generate_context.error_status;
}

LoadBalanceInfo LocalRpcServer::getLoadBalanceInfo() {
    return engine_->getLoadBalanceInfo();
}

void LocalRpcServer::addLora(const std::string& adapter_name,
                             const ft::lora::loraLayerWeightsMap& lora_a_weights,
                             const ft::lora::loraLayerWeightsMap& lora_b_weights)
{
    engine_->addLora(adapter_name, lora_a_weights, lora_b_weights);
}
void LocalRpcServer::removeLora(const std::string& adapter_name) {
    engine_->removeLora(adapter_name);
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

}  // namespace rtp_llm
