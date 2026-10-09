#pragma once

#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class LocalMultimodalProcessor: public MultimodalProcessor {
public:
    using MultimodalProcessor::MultimodalProcessor;

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "",
                                                      const std::string& rendered_prompt = "") override {
        if (mm_inputs.size() == 0) {
            return MultimodalOutput();
        } else if (!mm_process_engine_.is_none()) {
            std::vector<std::string>   urls;
            std::vector<int32_t>       types;
            std::vector<torch::Tensor> tensors;
            for (auto& mm_input : mm_inputs) {
                urls.push_back(mm_input.url);
                tensors.push_back(mm_input.tensor);
                types.push_back(mm_input.mm_type);
            }
            try {
                py::gil_scoped_acquire acquire;

                std::vector<py::list> mm_preprocess_configs;
                for (auto& mm_input : mm_inputs) {
                    py::list mm_preprocess_config;
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.width);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.height);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.min_pixels);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.max_pixels);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.fps);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.min_frames);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.max_frames);
                    py::list crop_positions;
                    for (const float& crop_position : mm_input.mm_preprocess_config.crop_positions) {
                        crop_positions.append(crop_position);
                    }
                    mm_preprocess_config.append(crop_positions);
                    mm_preprocess_config.append(mm_input.mm_preprocess_config.mm_timeout_ms);
                    mm_preprocess_configs.push_back(mm_preprocess_config);
                }

                auto res = mm_process_engine_.attr("mm_embedding_cpp")(
                    urls, types, tensors, mm_preprocess_configs, rendered_prompt);
                auto mm_embedding_vec = convertPyObjectToVec(res.attr("embeddings"));

                MultimodalOutput           mm_embedding_res;
                std::vector<torch::Tensor> mm_features;
                for (auto& emb : mm_embedding_vec) {
                    mm_features.emplace_back(convertPyObjectToTensor(emb));
                }
                mm_embedding_res.mm_features               = mm_features;
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
                if (py::hasattr(res, "expansion_metadata")) {
                    auto metadata_obj = res.attr("expansion_metadata");
                    if (!metadata_obj.is_none()) {
                        std::vector<MultimodalExpansionMetadata> metadata;
                        auto                                     metadata_list = metadata_obj.cast<py::list>();
                        for (const auto& item : metadata_list) {
                            MultimodalExpansionMetadata entry;
                            if (!item.is_none()) {
                                auto value     = py::reinterpret_borrow<py::dict>(item);
                                entry.is_video = value["kind"].cast<std::string>() == "video";
                                if (entry.is_video) {
                                    entry.fps           = value["fps"].cast<double>();
                                    entry.frame_indices = {value["frame_index"].cast<int32_t>()};
                                }
                                entry.soft_tokens_per_frame = value["soft_tokens"].cast<int32_t>();
                                entry.frame_number          = value["frame_number"].cast<int32_t>();
                                entry.frame_count           = value["frame_count"].cast<int32_t>();
                            }
                            metadata.emplace_back(std::move(entry));
                        }
                        if (!metadata.empty()) {
                            mm_embedding_res.mm_expansion_metadata = std::move(metadata);
                        }
                    }
                }
                if (py::hasattr(res, "expanded_token_ids")) {
                    auto expanded_ids = res.attr("expanded_token_ids");
                    if (!expanded_ids.is_none()) {
                        mm_embedding_res.expanded_token_ids =
                            torch::tensor(expanded_ids.cast<std::vector<int32_t>>(), torch::kInt32);
                    }
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
