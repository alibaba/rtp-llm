#pragma once

#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"

#include <memory>
namespace rtp_llm {

class EmbeddingExecutor {
public:
    explicit EmbeddingExecutor(const MagaInitParams&                                                   params,
                               ft::NcclParam                                                           tensor_para,
                               ft::NcclParam                                                           pipeline_para,
                               const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                               const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights,
                               const HandlerBase&                                                      handler);

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

private:
    std::unique_ptr<ParallelModelWrapper> model_wrapper_;
    const HandlerBase&                    handler_;
    ft::NcclParam                         tensor_para_;
    ft::NcclParam                         pipeline_para_;
    ft::DeviceBase*                       device_;
    ft::BufferPtr                         max_position_ids_buf_;
    ft::DataType                          data_type_;

    ModelRequest generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<GptModelInputs> gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    std::unique_ptr<GptModelOutputs> copyResultToCPU(const GptModelOutputs& gpu_outputs) const;
    absl::Status updateStreams(std::unique_ptr<GptModelOutputs>& gpu_outputs, const std::list<EmbeddingStreamPtr>& streams) const;
    absl::Status createAttentionMask(GptModelInputs& model_input) const;
    void calcTokenNum(const std::list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const;    
    void init_position_ids(int max_seq_len);
};
}  // namespace rtp_llm