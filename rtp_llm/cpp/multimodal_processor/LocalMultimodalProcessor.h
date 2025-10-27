#pragma once

#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

namespace rtp_llm {

class LocalMultimodalProcessor: public MultimodalProcessor {
public:
    using MultimodalProcessor::MultimodalProcessor;

private:
    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        if (mm_inputs.size() == 0) {
            return MultimodalOutput();
        } else if (!mm_process_engine_.is_none()) {
            std::vector<std::string>          urls;
            std::vector<int32_t>              types;
            std::vector<torch::Tensor>        tensors;
            std::vector<std::vector<int32_t>> mm_preprocess_configs;
            for (auto& mm_input : mm_inputs) {
                urls.push_back(mm_input.url);
                tensors.push_back(mm_input.tensor);
                types.push_back(mm_input.mm_type);
                mm_preprocess_configs.push_back({mm_input.mm_preprocess_config.width,
                                                 mm_input.mm_preprocess_config.height,
                                                 mm_input.mm_preprocess_config.min_pixels,
                                                 mm_input.mm_preprocess_config.max_pixels,
                                                 mm_input.mm_preprocess_config.fps,
                                                 mm_input.mm_preprocess_config.min_frames,
                                                 mm_input.mm_preprocess_config.max_frames,
                                                 mm_input.mm_preprocess_config.mm_timeout_ms});
            }
            try {
                py::gil_scoped_acquire acquire;
                auto res = mm_process_engine_.attr("mm_embedding_cpp")(urls, types, tensors, mm_preprocess_configs);
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
                return mm_embedding_res;
            } catch (py::error_already_set& e) {
                std::string error_msg = e.what();
                if (error_msg.find("download failed") != std::string::npos) {
                    return ErrorInfo(ErrorCode::MM_DOWNLOAD_FAILED, error_msg);
                }
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error_msg);
            }
        } else {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "no mm process engine!");
        }
    }
};

}  // namespace rtp_llm