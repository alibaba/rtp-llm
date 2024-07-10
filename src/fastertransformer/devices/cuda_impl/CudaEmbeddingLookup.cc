#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

using namespace std;

namespace fastertransformer {


BufferPtr CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;
    const auto& mask = params.text_token_masks;
    const auto& position_ids = params.position_ids;
    const auto& postition_table = params.position_table;
    const auto& token_types = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto& multimodal_features = params.multimodal_features;
    const auto& multimodal_locs = params.mm_features_locs;

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
        multimodal_features ? mask.data<int>() : nullptr,
        token_num,
        hidden_size,
        stream_
    );

    if (multimodal_features) {
        RUNTIME_ASSERT_OP_ARG(multimodal_locs, "no multimodal input location found");
        return move(multimodalEmbedding({
            embeddings, multimodal_features.value(), multimodal_locs.value().get()
        }));
    }

    return move(embeddings);
}

BufferPtr CudaDevice::multimodalEmbedding(const MultimodalEmbeddingParams& params) {
    const auto& embeddings = params.word_embeddings;
    const auto& features = params.multimodal_features;
    const auto& multimodal_locs = params.multimodal_locs;
    const auto mm_num = features.size();
    const auto hidden_size = embeddings->shape()[1];

    // not deal with bf16
    RUNTIME_ASSERT_OP_ARG(
        embeddings->typeSize() == features[0].element_size(),
        "type size of embeddings and multimodal features should be equal.");

    for (int i = 0; i < mm_num; ++i) {
        auto& feature = features[i];
        auto loc = multimodal_locs.dataWithOffset<int>(i);
        check_cuda_error(
            cudaMemcpy(
                embeddings->dataWithOffset((*loc) * hidden_size), 
                feature.data_ptr(), 
                feature.sizes()[0] * embeddings->typeSize() * hidden_size, 
                cudaMemcpyDeviceToDevice
            ));
    }

    return move(embeddings);
}

} // namespace fastertransformer
