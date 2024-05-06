#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"

namespace rtp_llm {
class EmbeddingQueryConverter {
public:
    static EmbeddingStreamPtr convertEmbeddingInputs(
        const torch::Tensor& token_ids, 
        const torch::Tensor& token_type_ids, 
        const torch::Tensor& input_lengths, 
        int request_id);
    static th::Tensor convertEmbeddingOutputs(EmbeddingStreamPtr stream);
};

} // namespace rtp_llm