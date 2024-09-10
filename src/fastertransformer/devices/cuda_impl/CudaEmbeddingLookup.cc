#include "src/fastertransformer/devices/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

using namespace std;

namespace fastertransformer {


BufferPtr CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;
    const auto& mask = params.text_tokens_mask;
    const auto& position_ids = params.position_ids;
    const auto& postition_table = params.position_table;
    const auto& token_types = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeEmebeddingLookup,
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
        stream_
    );

    return embeddings;
}

BufferPtr CudaDevice::multimodalEmbedding(const MultimodalEmbeddingParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.multimodal_locs, "no multimodal input location found");
    const auto& embeddings = params.word_embeddings;
    const auto& features = params.multimodal_features.value().get();
    const auto& multimodal_locs = params.multimodal_locs.value().get();
    const auto mm_num = features.size();

    RUNTIME_ASSERT_OP_ARG(
        embeddings->typeSize() == features[0]->typeSize(),
        "type size of embeddings and multimodal features should be equal.");

    for (int i = 0; i < mm_num; ++i) {
        auto& feature = features[i];
        auto loc = multimodal_locs.dataWithOffset<int32_t>(i);
        copy({embeddings->view(*loc, feature->shape()[0]), *feature});
    }

    return move(embeddings);
}

} // namespace fastertransformer
