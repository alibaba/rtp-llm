#include "rtp_llm/cpp/model_rpc/EmbeddingRpcServer.h"

namespace py = pybind11;
namespace th = torch;

namespace rtp_llm {

std::shared_ptr<EmbeddingOutput> EmbeddingRpcServiceImpl::ConvertEngineOutputs(py::object result) {
    auto embedding_output = std::make_shared<EmbeddingOutput>();

    try {
        py::object py_outputs = result.attr("outputs");

        if (py_outputs.is_none()) {
            embedding_output->setError("Outputs is None");
            return embedding_output;
        }

        try {
            torch::Tensor tensor = py::cast<torch::Tensor>(py_outputs);
            embedding_output->setTensorOutput(tensor);
            return embedding_output;
        } catch (const py::cast_error& e) {}

        try {
            py::list                                          py_list = py::cast<py::list>(py_outputs);
            std::vector<std::map<std::string, torch::Tensor>> cpp_map_list;

            for (auto item : py_list) {
                py::dict                             py_dict = py::cast<py::dict>(item);
                std::map<std::string, torch::Tensor> cpp_map;

                for (auto pair : py_dict) {
                    std::string   key   = py::cast<std::string>(pair.first);
                    torch::Tensor value = py::cast<torch::Tensor>(pair.second);
                    cpp_map[key]        = value;
                }

                cpp_map_list.push_back(std::move(cpp_map));
            }

            embedding_output->setMapOutput(cpp_map_list);
            return embedding_output;
        } catch (const py::cast_error& e) {
            embedding_output->setError("Unsupported output type");
        }

    } catch (const py::error_already_set& e) {
        embedding_output->setError(e.what());
        PyErr_Clear();
    } catch (const std::exception& e) {
        embedding_output->setError(e.what());
    }

    return embedding_output;
}

grpc::Status EmbeddingRpcServiceImpl::decode(grpc::ServerContext*    context,
                                             const EmbeddingInputPB* request,
                                             EmbeddingOutputPB*      response) {

    int64_t request_id = request->request_id();
    RTP_LLM_LOG_INFO("Received embedding request id: %d", request_id);
    std::vector<int32_t> token_ids(request->token_ids().begin(), request->token_ids().end());
    std::vector<int32_t> token_type_ids(request->token_type_ids().begin(), request->token_type_ids().end());
    std::vector<int32_t> input_lengths(request->input_lengths().begin(), request->input_lengths().end());

    std::vector<MultimodalInput> multimodal_inputs;
    for (const auto& pb_feature : request->multimodal_features()) {
        MultimodalInput feature(pb_feature.multimodal_url(), torch::empty(1), pb_feature.multimodal_type());
        multimodal_inputs.emplace_back(std::move(feature));
    }

    std::optional<MultimodalFeature> multimodal_features = std::nullopt;

    std::shared_ptr<EmbeddingInput> embedding_input = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_inputs, multimodal_features);
    if (mm_processor_ != nullptr && !multimodal_inputs.empty()) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(embedding_input);
        if (!mm_res.ok()) {
            throw std::runtime_error(mm_res.ToString());
        }
    }
    std::shared_ptr<EmbeddingOutput> embedding_output = embedding_engine_->decode(embedding_input);
    try {
        py::gil_scoped_acquire acquire;
        TypedOutput&           typed_output = embedding_output->output;
        py::object             py_batch_output;

        if (typed_output.isTensor) {
            at::Tensor tensor = *typed_output.t;
            py_batch_output   = py::cast(tensor);
        } else {
            auto&    map_output = *typed_output.map;
            py::list py_list;
            for (auto& map_item : map_output) {
                py::dict py_dict;
                for (auto& pair : map_item) {
                    py_dict[pair.first.c_str()] = py::cast(pair.second);
                }
                py_list.append(py_dict);
            }
            py_batch_output = py_list;
        }
        py::module_ engine_module = py::module_::import("rtp_llm.async_decoder_engine.embedding.interface");
        py::object  EngineOutputs = engine_module.attr("EngineOutputs");
        py::object  engine_outputs =
            EngineOutputs(py::arg("outputs") = py_batch_output, py::arg("input_length") = input_lengths[0]);
        py::object result = pyHandler_.attr("post_process")("", engine_outputs);
        embedding_output  = ConvertEngineOutputs(result);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("python Convert exception : %s ", e.what());
        return grpc::Status(grpc::StatusCode::ABORTED, e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("cpp exception: %s", e.what());
        return grpc::Status(grpc::StatusCode::ABORTED, e.what());
    }
    if (embedding_output->output.isTensor) {
        response->set_output_is_tensor(true);
        QueryConverter::transTensorPB(response->mutable_output_t(), embedding_output->output.t.value());
        if (context->IsCancelled()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
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
        if (context->IsCancelled()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
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