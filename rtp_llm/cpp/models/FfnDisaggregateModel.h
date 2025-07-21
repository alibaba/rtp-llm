#pragma once
#include "rtp_llm/cpp/models/GptModel.h"

namespace rtp_llm {

class FfnDisaggregateModel: public GptModel {
    struct BatchSplitInfo {
        std::vector<std::vector<size_t>> micro_batch_sizes_list;
        std::vector<size_t>              total_micro_batch_sizes;

        std::string debugInfo() const {
            std::stringstream ss;
            for (int i = 0; i < total_micro_batch_sizes.size(); i++) {
                ss << "micro_batch_size_" << i << ": " << total_micro_batch_sizes[i] << ", list_val:";
                for (int j = 0; j < micro_batch_sizes_list[i].size(); j++) {
                    ss << micro_batch_sizes_list[i][j] << ",";
                }
                ss << std::endl;
            }
            return ss.str();
        }
    };

public:
    FfnDisaggregateModel(const GptModelInitParams& params): GptModel(params) {
        micro_batch_size_ = device_->getDeviceProperties().enable_layer_micro_batch != MicroBatchType::NONE ? 2 : 1;
    }

    GptModelOutputs forward(const GptModelInputs& inputs) override;

protected:
    GptModelOutputs forwardFFnService(const GptModelInputs& inputs);
    GptModelOutputs forwardNormal(const GptModelInputs& inputs);
    void            sendBatchSplitInfo(const std::vector<LayerMicroBatchInputs>& batch_infos);
    void            sendInputBuffer(const std::vector<LayerMicroBatchInputs>& batch_infos);
    BufferPtr       preAttentionOperation(BufferPtr input, int layer_id);
    BatchSplitInfo  recvBatchSplitInfo();
    void recvInputBuffer(const BatchSplitInfo& batch_split_info, std::vector<BufferPtr>& input_batch_buffers);
    int  micro_batch_size_ = 1;
};

}  // namespace rtp_llm