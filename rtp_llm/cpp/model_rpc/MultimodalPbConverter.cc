#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"

#include <numeric>

#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"

namespace rtp_llm {

MultimodalInputsPB MultimodalPbConverter::inputsToPb(const std::vector<MultimodalInput>& mm_inputs) {
    MultimodalInputsPB mm_inputs_pb;
    for (const auto& mm_input : mm_inputs) {
        auto now_input = mm_inputs_pb.add_multimodal_inputs();
        now_input->set_multimodal_url(mm_input.url);
        now_input->set_multimodal_type(mm_input.mm_type);
        TensorPbConvert::torchToPb(now_input->mutable_multimodal_tensor(), mm_input.tensor);
        preprocessConfigToPb(now_input->mutable_mm_preprocess_config(), mm_input.mm_preprocess_config);
    }
    return mm_inputs_pb;
}

ErrorResult<MultimodalOutput> MultimodalPbConverter::inlineOutputFromPb(const MultimodalOutputPB& output_pb) {
    // Convert malformed remote data into an error instead of propagating torch exceptions.
    try {
        torch::Tensor mm_embedding = TensorPbConvert::pbToTorch(output_pb.multimodal_embedding()), mm_position_id;
        bool          contain_pos         = output_pb.has_multimodal_pos_id();
        bool          contain_extra_input = output_pb.multimodal_extra_input_size() > 0;
        if (contain_pos) {
            mm_position_id = TensorPbConvert::pbToTorch(output_pb.multimodal_pos_id());
        }
        // RDMA receipts have no inline embedding and must not reach this decoder.
        if (mm_embedding.dim() == 0) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                             "inline multimodal response carries no embedding tensor");
        }
        MultimodalOutput     mm_output;
        std::vector<int64_t> split_sizes;
        for (auto split_size : output_pb.split_size()) {
            split_sizes.push_back(split_size);
        }
        const int64_t split_total = std::accumulate(split_sizes.begin(), split_sizes.end(), int64_t{0});
        if (split_sizes.empty() || split_total != mm_embedding.size(0)) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                             "inline multimodal response is inconsistent: split_sizes sum="
                                 + std::to_string(split_total) + " does not match mm_embedding.size(0)="
                                 + std::to_string(mm_embedding.size(0)));
        }
        mm_output.mm_features = mm_embedding.split(split_sizes, 0);
        if (contain_pos) {
            if (mm_position_id.dim() == 0 || split_total != mm_position_id.size(0)) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                                 "inline multimodal response is inconsistent: split_sizes sum="
                                     + std::to_string(split_total) + " does not match the position ids");
            }
            mm_output.mm_position_ids = mm_position_id.split(split_sizes, 0);
        }

        if (contain_extra_input) {
            if (output_pb.multimodal_extra_input_size() != static_cast<int>(split_sizes.size())) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                                 "inline multimodal response is inconsistent: extra_input count="
                                     + std::to_string(output_pb.multimodal_extra_input_size())
                                     + " does not match image count=" + std::to_string(split_sizes.size()));
            }
            // Extra inputs remain one tensor per image.
            std::vector<torch::Tensor> extra_inputs;
            extra_inputs.reserve(output_pb.multimodal_extra_input_size());
            for (const auto& extra_input_pb : output_pb.multimodal_extra_input()) {
                extra_inputs.emplace_back(TensorPbConvert::pbToTorch(extra_input_pb));
            }
            mm_output.mm_extra_input = std::move(extra_inputs);
        }
        return mm_output;
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         std::string("inline multimodal response could not be decoded: ") + e.what());
    }
}

void MultimodalPbConverter::preprocessConfigToPb(MMPreprocessConfigPB* config_pb,
                                                 const MMPreprocessConfig& config) {
    config_pb->set_width(config.width);
    config_pb->set_height(config.height);
    config_pb->set_min_pixels(config.min_pixels);
    config_pb->set_max_pixels(config.max_pixels);
    config_pb->set_fps(config.fps);
    config_pb->set_min_frames(config.min_frames);
    config_pb->set_max_frames(config.max_frames);
    config_pb->set_mm_timeout_ms(config.mm_timeout_ms);
    for (const float& crop_position : config.crop_positions) {
        config_pb->add_crop_positions(crop_position);
    }
}

}  // namespace rtp_llm
