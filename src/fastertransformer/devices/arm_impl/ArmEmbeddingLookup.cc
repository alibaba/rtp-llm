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

    auto copy_size = hidden_size * data_type_size;

    // select the rows from embedding table
    for (int index = 0; index < token_num; index++) {
        auto row_index  = tokens.data<int>()[index];
        auto src_offset = row_index * copy_size;
        auto dst_offset = index * copy_size;

        std::memcpy(static_cast<char*>(embeddings->data()) + dst_offset, static_cast<char*>(embedding_table.data()) + src_offset, copy_size);
    }

     // for BERT
    const auto& position_ids = params.position_ids;
    const auto& position_table = params.position_table;
    const auto& token_types = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const int* input_pos = position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr;
    const int* input_type = token_types.has_value() ? token_types.value().get().data<int>() : nullptr;

    if (data_type == DataType::TYPE_FP16) {
        const __fp16* pos_table = position_table.has_value() ? (__fp16*)(position_table.value().get().data()) : nullptr;
        const __fp16* type_table = token_type_table.has_value() ? (__fp16*)(token_type_table.value().get().data()) : nullptr;

        // case: both pos_table and type_table have values
        if (pos_table != nullptr && type_table != nullptr) {
            for (int index = 0; index < token_num; index++) {
                for (int hid = 0; hid < hidden_size; hid++) {
                    __fp16 pos_embed = pos_table[input_pos[index] * hidden_size + hid];
                    __fp16 type_embed = type_table[input_type[index] * hidden_size + hid];
                    __fp16* embed_ptr = (__fp16*)(embeddings->data()) + index * hidden_size + hid;
                    *embed_ptr += pos_embed + type_embed;
                }
            }
        }
    }

    return embeddings;
}

}  // namespace fastertransformer
