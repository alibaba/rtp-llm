#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

using namespace std;

namespace fastertransformer {

template<typename T>
void invokeEmebeddingLookup(void*           from_tensor,
                            const void*     embedding_table,
                            const void*     pos_table,
                            const void*   input_ids,
                            const void*   input_pos,
                            const int    token_num,
                            const int    hidden_units,
                            cudaStream_t stream)
{
    invokeEmebeddingLookup(
        (T*)from_tensor,
        (const T*)embedding_table,
        (const T*)pos_table,
        (const int*)input_ids,
        (const int*)input_pos,
        token_num,
        hidden_units,
        stream
    );
}

BufferPtr CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;
    const auto& position_ids = params.position_ids;
    const auto& postition_table = params.position_table;

    const auto token_num = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type = embedding_table.type();

    auto embeddings = allocateBuffer({embedding_table.type(), {token_num, hidden_size}});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(embedding_table.type(), invokeEmebeddingLookup,
        embeddings->data(),
        embedding_table.data(),
        postition_table.has_value() ? postition_table.value().get().data() : nullptr,
        tokens.data(),
        position_ids.has_value() ? position_ids.value().get().data() : nullptr,
        token_num,
        hidden_size,
        stream_
    );

    return move(embeddings);
}

} // namespace fastertransformer
