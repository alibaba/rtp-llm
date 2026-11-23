#pragma once

#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class LocalMultimodalProcessor: public MultimodalProcessor {
public:
    using MultimodalProcessor::MultimodalProcessor;

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port    = "",
                                                      int64_t                                     request_id = 0,
                                                      grpc::ServerContext* server_context = nullptr) {
        if (mm_inputs.size() == 0) {
            return MultimodalOutput();
        } else if (!mm_process_engine_.is_none()) {
            try {
                py::gil_scoped_acquire acquire;
                auto                   res = mm_process_engine_.attr("mm_embedding_inputs_cpp")(
                    py::cast(mm_inputs, py::return_value_policy::copy),
                    request_id,
                    dashScopeMetadata(server_context, "x-dashscope-uid"),
                    dashScopeMetadata(server_context, "x-dashscope-service"));
                MultimodalOutput mm_embedding_res;
                auto             stacked = py::getattr(res, "stacked_embeddings", py::none());
                if (!stacked.is_none()) {
                    mm_embedding_res.mm_features = convertPyObjectToTensor(stacked).unbind(0);
                } else {
                    for (const auto& value : convertPyObjectToVec(res.attr("embeddings"))) {
                        mm_embedding_res.mm_features.emplace_back(convertPyObjectToTensor(value));
                    }
                }
                if (py::hasattr(res, "feature_hashes") && !res.attr("feature_hashes").is_none()) {
                    std::vector<torch::Tensor> hashes;
                    for (auto& value : convertPyObjectToVec(res.attr("feature_hashes"))) {
                        hashes.emplace_back(convertPyObjectToTensor(value));
                    }
                    mm_embedding_res.mm_feature_hashes = std::move(hashes);
                }
                auto                       position_id_vec = res.attr("position_ids");
                std::vector<torch::Tensor> position_ids;
                if (!position_id_vec.is_none()) {
                    for (auto& position_id : convertPyObjectToVec(position_id_vec)) {
                        auto pos = convertPyObjectToTensor(position_id);
                        position_ids.emplace_back(pos);
                    }
                    mm_embedding_res.mm_position_ids = position_ids;
                }
                auto                       extra_input_vec = res.attr("extra_input");
                std::vector<torch::Tensor> extra_input;
                if (!extra_input_vec.is_none()) {
                    for (auto& extra_input_item : convertPyObjectToVec(extra_input_vec)) {
                        extra_input.emplace_back(convertPyObjectToTensor(extra_input_item));
                    }
                    mm_embedding_res.mm_extra_input = extra_input;
                }
                return mm_embedding_res;
            } catch (py::error_already_set& e) {
                std::string error_msg = e.what();
                try {
                    py::gil_scoped_acquire gil;
                    py::object             exc = py::reinterpret_borrow<py::object>(e.value());
                    if (exc && py::hasattr(exc, "exception_type")) {
                        const auto exception_type = exc.attr("exception_type").cast<int>();
                        const auto message = py::hasattr(exc, "message") ? exc.attr("message").cast<std::string>() :
                                                                           py::str(exc).cast<std::string>();
                        if (auto error_code = parseMultimodalErrorCode(exception_type)) {
                            return ErrorInfo(*error_code, message);
                        }
                    }
                } catch (...) {
                    // Fall through to the legacy error mapping.
                }
                if (auto error_info = parseMultimodalErrorMessage(error_msg)) {
                    return *error_info;
                }
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error_msg);
            }
        } else {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "no mm process engine!");
        }
    }
};

}  // namespace rtp_llm
