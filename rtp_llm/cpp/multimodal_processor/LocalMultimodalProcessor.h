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
        }

        // Mock implementation: Skip Python callback completely
        MultimodalOutput           mm_embedding_res;
        std::vector<torch::Tensor> mm_features;

        for (const auto& mm_input : mm_inputs) {
            int64_t           batch_size = 1;
            torch::Device     device     = torch::kCPU;
            torch::ScalarType dtype      = torch::kFloat;

            if (!mm_input.tensors.empty()) {
                batch_size = mm_input.tensors[0].size(0);
                device     = mm_input.tensors[0].device();
                dtype      = torch::kBFloat16;
            }

            // Generate random embedding [batch_size, hidden_size]
            // Ensure hidden_size_ is accessed from gpt_init_parameter_
            auto options        = torch::TensorOptions().device(device).dtype(dtype);
            auto rand_embedding = torch::randn({batch_size, gpt_init_parameter_.hidden_size_}, options);
            mm_features.push_back(rand_embedding);
        }

        mm_embedding_res.mm_features = mm_features;
        return mm_embedding_res;

        /* Original Python Callback Logic - Commented Out
        } else if (!mm_process_engine_.is_none()) {
            std::vector<std::string>                urls;
            std::vector<int32_t>                    types;
            std::vector<std::vector<torch::Tensor>> tensors;
            std::vector<std::vector<int32_t>>       mm_preprocess_configs;
            for (auto& mm_input : mm_inputs) {
                urls.push_back(mm_input.url);
                tensors.push_back(mm_input.tensors);
                types.push_back(mm_input.mm_type);
                mm_preprocess_configs.push_back({mm_input.mm_preprocess_config.width,
                                                 mm_input.mm_preprocess_config.height,
                                                 mm_input.mm_preprocess_config.min_pixels,
                                                 mm_input.mm_preprocess_config.max_pixels,
                                                 mm_input.mm_preprocess_config.fps,
                                                 mm_input.mm_preprocess_config.min_frames,
                                                 mm_input.mm_preprocess_config.max_frames});
            }
            try {
                py::gil_scoped_acquire acquire;
                auto res              = mm_process_engine_.attr("submit")(urls, types, tensors, mm_preprocess_configs);
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
        */
    }
};

}  // namespace rtp_llm