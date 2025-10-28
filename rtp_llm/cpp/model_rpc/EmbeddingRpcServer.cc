#include "rtp_llm/cpp/model_rpc/EmbeddingRpcServer.h"

namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {

grpc::Status EmbeddingRpcServiceImpl::decode(grpc::ServerContext*                   context,
                                             const EmbeddingInputPB*                request,
                                             grpc::ServerWriter<EmbeddingOutputPB>* writer) {
    int64_t request_id = request->request_id();
    RTP_LLM_LOG_INFO("Received embedding request id: %d", request_id);

    std::vector<int32_t> token_ids(request->token_ids().begin(), request->token_ids().end());

    std::vector<int32_t> token_type_ids(request->token_type_ids().begin(), request->token_type_ids().end());

    std::vector<int32_t> input_lengths(request->input_lengths().begin(), request->input_lengths().end());

    for (int i = 0; i < token_ids.size(); ++i) {
        RTP_LLM_LOG_INFO("token_ids[%d]: %d", i, token_ids[i]);
    }
    for (int i = 0; i < token_type_ids.size(); ++i) {
        RTP_LLM_LOG_INFO("token_type_ids[%d]: %d", i, token_type_ids[i]);
    }

    std::vector<MultimodalInput> multimodal_inputs;

    for (const auto& pb_feature : request->multimodal_features()) {
        MultimodalInput feature(pb_feature.multimodal_url(), torch::empty(1), pb_feature.multimodal_type());
        multimodal_inputs.emplace_back(std::move(feature));
    }

    std::optional<MultimodalFeature> multimodal_features = std::nullopt;
    auto                             embedding_input =
        std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.ToString());
        }
    }
    std::shared_ptr<EmbeddingOutput> embedding_output = embedding_engine_->decode(embedding_input);
    py::gil_scoped_acquire           acquire;
    EmbeddingOutputPB                outputs_pb;
    if (embedding_output->output.isTensor) {
        outputs_pb.set_output_is_tensor(true);
        QueryConverter::transTensorPB(outputs_pb.mutable_output_t(), embedding_output->output.t.value());
        if (context->IsCancelled()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
    } else {
        for (const auto& tensor_map_data : embedding_output->output.map.value()) {
            auto* embedding_map_pb = outputs_pb.add_output_map();
            for (const auto& [key, tensor_data] : tensor_map_data) {
                TensorPB tensor_pb;
                QueryConverter::transTensorPB(&tensor_pb, tensor_data);
                (*embedding_map_pb->mutable_tensor_map())[key] = tensor_pb;
            }
        }
        if (context->IsCancelled()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
    }
    return grpc::Status::OK;
}

grpc::Status EmbeddingRpcServiceImpl::health(grpc::ServerContext*            context,
                                             const EmbeddingHealthRequestPB* request,
                                             EmptyPB*                        writer) {
    RTP_LLM_LOG_INFO("Received embedding health check request");
    return grpc::Status::OK;
}
}  // namespace rtp_llm