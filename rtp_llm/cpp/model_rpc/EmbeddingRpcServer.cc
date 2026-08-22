#include "rtp_llm/cpp/model_rpc/EmbeddingRpcServer.h"
#include "rtp_llm/cpp/model_rpc/MMRpcCodec.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {

grpc::Status EmbeddingRpcServiceImpl::embedding(grpc::ServerContext*    context,
                                                const EmbeddingInputPB* request,
                                                EmbeddingOutputPB*      response) {
    int64_t                          request_id = 0;
    std::vector<int32_t>             token_ids;
    std::vector<int32_t>             token_type_ids;
    std::vector<int32_t>             input_lengths;
    std::vector<MultimodalInput>     multimodal_inputs;
    std::shared_ptr<EmbeddingInput>  embedding_input;
    std::shared_ptr<EmbeddingOutput> embedding_output;

    try {
        request_id = request->request_id();
        RTP_LLM_LOG_DEBUG("Received embedding request id: %d", request_id);
        token_ids      = std::vector<int32_t>(request->token_ids().begin(), request->token_ids().end());
        token_type_ids = std::vector<int32_t>(request->token_type_ids().begin(), request->token_type_ids().end());
        input_lengths  = std::vector<int32_t>(request->input_lengths().begin(), request->input_lengths().end());

        for (const auto& pb_feature : request->multimodal_features()) {
            multimodal_inputs.emplace_back(MMRpcCodec::transMMInputElement(
                pb_feature, QueryConverter::transTensor(pb_feature.multimodal_tensor())));
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("[Request Parsing] Failed for request_id %ld: %s", request_id, e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, std::string("Request parsing error: ") + e.what());
    }

    // Processing: Multimodal, Decode, Post-process, and Output Assembly
    try {
        // Stage 2: Multimodal Processing
        std::optional<MultimodalFeature> multimodal_features = std::nullopt;
        embedding_input =
            std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);

        if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
            auto mm_res =
                mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs, request->vit_role_addr());
            if (!mm_res.ok()) {
                RTP_LLM_LOG_WARNING("[Multimodal] updateMultimodalFeatures failed for request_id %ld, "
                                    "error code [%s], error message [%s]",
                                    request_id,
                                    ErrorCodeToString(mm_res.code()).c_str(),
                                    mm_res.ToString().c_str());
                return makeGrpcErrorStatus(mm_res);
            }
        }

        // Stage 3: Embedding Decode
        EmbeddingProfileConfig profile_config;
        profile_config.gen_timeline       = request->gen_timeline();
        profile_config.profile_step       = request->profile_step();
        profile_config.profile_trace_name = request->profile_trace_name();
        embedding_output                  = embedding_engine_->decode(embedding_input, profile_config);

        // Stage 4: Post Processing
        if (need_post_process_) {
            py::gil_scoped_acquire acquire;
            py::object             py_batch_output = py::cast(*embedding_output);
            py::object             result          = pyHandler_.attr("post_process")("", py_batch_output);
            *embedding_output                      = py::cast<EmbeddingOutput>(result);
        }

        // Stage 5: Output Assembly
        if (embedding_output->output.isTensor) {
            response->set_output_is_tensor(true);
            QueryConverter::transTensorPB(response->mutable_output_t(), embedding_output->output.t.value());
        } else {
            response->set_output_is_tensor(false);
            for (const auto& tensor_map_data : embedding_output->output.map.value()) {
                auto* embedding_map_pb = response->add_output_map();
                for (const auto& [key, tensor_data] : tensor_map_data) {
                    TensorPB tensor_pb;
                    QueryConverter::transTensorPB(&tensor_pb, tensor_data);
                    (*embedding_map_pb->mutable_tensor_map())[key] = tensor_pb;
                }
            }
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("[Processing] Python exception for request_id %ld: %s", request_id, e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, std::string("Processing error: ") + e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("[Processing] Failed for request_id %ld: %s", request_id, e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, std::string("Processing error: ") + e.what());
    }

    return grpc::Status::OK;
}

grpc::Status EmbeddingRpcServiceImpl::health(grpc::ServerContext*            context,
                                             const EmbeddingHealthRequestPB* request,
                                             EmptyPB*                        writer) {
    RTP_LLM_LOG_DEBUG("Received embedding health check request");
    return grpc::Status::OK;
}
}  // namespace rtp_llm
