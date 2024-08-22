#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"

namespace fastertransformer {

BufferPtr ArmCpuDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens          = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;

    const auto token_num      = tokens.size();
    const auto hidden_size    = embedding_table.shape()[1];
    const auto data_type      = embedding_table.type();
    const auto data_type_size = getTypeSize(data_type);

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}, AllocationType::HOST});

    int copy_size = hidden_size * data_type_size;

    // select the rows from embedding table
    for (int index = 0; index < token_num; index++) {
        int row_index  = tokens.data<int>()[index];
        int src_offset = row_index * copy_size;
        int dst_offset = index * copy_size;

        std::memcpy(static_cast<char*>(embeddings->data()) + dst_offset, static_cast<char*>(embedding_table.data()) + src_offset, copy_size);
    }

    return embeddings;
}

}  // namespace fastertransformer
