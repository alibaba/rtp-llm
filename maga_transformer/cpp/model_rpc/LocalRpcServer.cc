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

grpc::Status LocalRpcServer::serializeErrorMsg(int64_t request_id, ErrorCode error_code, const std::string& error_msg) {
    FT_LOG_WARNING("request id [%lu], error code [%s], error message [%s]",
            request_id, toString(error_code).c_str(), error_msg.c_str());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_code));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg, error_details_serialized);
    } else {
        FT_LOG_WARNING("request:[%ld] SerializeToString error", request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
}

// TODO(xinfei.sxf) use request key to replace request_id
grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*                   context,
                                              int64_t                                request_id,
                                              grpc::internal::WriterInterface<GenerateOutputsPB>* writer,
                                              std::shared_ptr<GenerateStream>&       stream) {
    while (!stream->finished() || stream->hasOutput()) {
        const auto output_status = stream->nextOutput();
        if (!output_status.ok()) {
            auto           status = output_status.status();
            auto           error_code = transErrorCode(status.code());
            return serializeErrorMsg(request_id, error_code, status.ToString());
        }
        FT_LOG_DEBUG("request:[%ld] generate next output success", request_id);
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(&outputs_pb, &(output_status.value()));
        if (context->IsCancelled()) {
            FT_LOG_WARNING("request:[%ld] cancel", request_id);
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            FT_LOG_WARNING("request:[%ld] write outputs pb failed", request_id);
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->needRemoteGenerate()) {
            break;
        }
    }
    FT_LOG_DEBUG("request:[%ld] local generate done", request_id);

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::generate_stream(grpc::ServerContext*                   context,
                                             const GenerateInputPB*                 request,
                                             grpc::ServerWriter<GenerateOutputsPB>* writer) {
    AtomicGuard request_guard(onflight_requests_);
    auto request_id = request->request_id();
    FT_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context = GenerateContext(request->request_id());
    auto input = QueryConverter::transQuery(request);

    // need to check client has buffer at first
    // todo: catch python exception, such as download timeout
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, mm_res.ToString());
        }
    }

    input->lora_id = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    FT_LOG_DEBUG("request:[%ld] trans to stream success", request_id);
    generate_context.stream = engine_->enqueue(input);
    FT_LOG_DEBUG("request:[%ld] enqueue success", request_id);

    return pollStreamOutput(context, request_id, writer, generate_context.stream);
}

void LocalRpcServer::reportMetrics(RPCMetricsCollector* collector) {
    if (metrics_reporter_) {
        metrics_reporter_->report<RPCMetrics, RPCMetricsCollector>(nullptr, collector);
    }
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
