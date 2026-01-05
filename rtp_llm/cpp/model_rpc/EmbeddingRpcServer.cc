#include "rtp_llm/cpp/model_rpc/EmbeddingRpcServer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
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

    int64_t request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("Received embedding request id: %d", request_id);
    std::vector<int32_t> token_ids(request->token_ids().begin(), request->token_ids().end());
    std::vector<int32_t> token_type_ids(request->token_type_ids().begin(), request->token_type_ids().end());
    std::vector<int32_t> input_lengths(request->input_lengths().begin(), request->input_lengths().end());

    std::vector<MultimodalInput> multimodal_inputs;
    for (const auto& pb_feature : request->multimodal_features()) {
        MultimodalInput feature(
            pb_feature.multimodal_url(), std::vector<torch::Tensor>{}, pb_feature.multimodal_type());
        multimodal_inputs.emplace_back(std::move(feature));
    }

    std::optional<MultimodalFeature> multimodal_features = std::nullopt;

    std::shared_ptr<EmbeddingInput> embedding_input =
        std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, request_id, multimodal_features);
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.ToString());
        }
    }
    std::shared_ptr<EmbeddingOutput> embedding_output = embedding_engine_->decode(embedding_input);
    try {
        request_id = request->request_id();
        RTP_LLM_LOG_DEBUG("Received embedding request id: %d", request_id);
        token_ids      = std::vector<int32_t>(request->token_ids().begin(), request->token_ids().end());
        token_type_ids = std::vector<int32_t>(request->token_type_ids().begin(), request->token_type_ids().end());
        input_lengths  = std::vector<int32_t>(request->input_lengths().begin(), request->input_lengths().end());

        for (const auto& pb_feature : request->multimodal_features()) {
            MultimodalInput feature(pb_feature.multimodal_url(), torch::empty(1), pb_feature.multimodal_type());
            multimodal_inputs.emplace_back(std::move(feature));
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
            auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input, multimodal_inputs);
            if (!mm_res.ok()) {
                throw std::runtime_error(mm_res.ToString());
            }
        }

        // Stage 3: Embedding Decode
        embedding_output = embedding_engine_->decode(embedding_input);

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