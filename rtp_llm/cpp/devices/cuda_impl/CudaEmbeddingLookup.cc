#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/embedding_kernels.h"

using namespace std;

namespace rtp_llm {

BufferPtr CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens           = params.combo_tokens;
    const auto& embedding_table  = params.embedding_table;
    const auto& mask             = params.text_tokens_mask;
    const auto& position_ids     = params.position_ids;
    const auto& postition_table  = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeEmbeddingLookup,
                                     embeddings->data(),
                                     embedding_table.data(),
                                     params.input_embedding_scalar,
                                     postition_table.has_value() ? postition_table.value().get().data() : nullptr,
                                     token_type_table.has_value() ? token_type_table.value().get().data() : nullptr,
                                     tokens.data<int>(),
                                     position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr,
                                     token_types.has_value() ? token_types.value().get().data<int>() : nullptr,
                                     mask.has_value() ? mask.value().get().data<int>() : nullptr,
                                     token_num,
                                     hidden_size,
                                     stream_);
    check_cuda_error();
    return embeddings;
}

}  // namespace rtp_llm
