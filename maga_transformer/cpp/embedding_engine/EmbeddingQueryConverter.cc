#include "maga_transformer/cpp/embedding_engine/EmbeddingQueryConverter.h"
#include "maga_transformer/cpp/core/torch_utils/BufferTorchUtils.h"
#include <numeric>
#include <torch/extension.h>

namespace th = torch;

namespace rtp_llm {

EmbeddingStreamPtr EmbeddingQueryConverter::convertEmbeddingInputs(
    const th::Tensor& token_ids,
    const th::Tensor& token_type_ids,
    const th::Tensor& input_lengths,
    int request_id,
    std::optional<MultimodalFeature> multimodal_features)
{
    auto token_buffer_ptr =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)token_ids.size(0)},
            token_ids.data_ptr());

    auto token_type_buffer_ptr =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)token_type_ids.size(0)},
            token_type_ids.data_ptr());

    int64_t total_length = std::accumulate((int32_t*)input_lengths.data_ptr(), (int32_t*)input_lengths.data_ptr() + input_lengths.size(0), 0);

    auto input_lengths_buffer_ptr =
        std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU,
            rtp_llm::DataType::TYPE_INT32,
            std::vector<size_t>{(size_t)input_lengths.size(0)},
            input_lengths.data_ptr());
    auto input =
        std::make_shared<EmbeddingInput>(token_buffer_ptr, token_type_buffer_ptr, input_lengths_buffer_ptr, total_length, request_id, multimodal_features);
    return std::make_shared<EmbeddingStream>(input);
}

} // namespace rtp_llm
