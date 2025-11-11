#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"

#if BUILDING_ARM_ONLY
#include <arm_neon.h>  // for convert_embedding_fp16_to_float
#include <omp.h>
#endif

namespace rtp_llm {

#if BUILDING_ARM_ONLY
void convert_embedding_fp16_to_float(const __fp16* input, float* output, int length) {
    int d = 0;
    for (; d <= length - 32; d += 32) {
        // Load 32 fp16 values
        float16x8_t fp16_vec0 = vld1q_f16(&input[d]);
        float16x8_t fp16_vec1 = vld1q_f16(&input[d + 8]);
        float16x8_t fp16_vec2 = vld1q_f16(&input[d + 16]);
        float16x8_t fp16_vec3 = vld1q_f16(&input[d + 24]);

        // Convert to float32
        float32x4_t float_vec0_low  = vcvt_f32_f16(vget_low_f16(fp16_vec0));
        float32x4_t float_vec0_high = vcvt_f32_f16(vget_high_f16(fp16_vec0));
        float32x4_t float_vec1_low  = vcvt_f32_f16(vget_low_f16(fp16_vec1));
        float32x4_t float_vec1_high = vcvt_f32_f16(vget_high_f16(fp16_vec1));
        float32x4_t float_vec2_low  = vcvt_f32_f16(vget_low_f16(fp16_vec2));
        float32x4_t float_vec2_high = vcvt_f32_f16(vget_high_f16(fp16_vec2));
        float32x4_t float_vec3_low  = vcvt_f32_f16(vget_low_f16(fp16_vec3));
        float32x4_t float_vec3_high = vcvt_f32_f16(vget_high_f16(fp16_vec3));

        // Store results
        vst1q_f32(&output[d], float_vec0_low);
        vst1q_f32(&output[d + 4], float_vec0_high);
        vst1q_f32(&output[d + 8], float_vec1_low);
        vst1q_f32(&output[d + 12], float_vec1_high);
        vst1q_f32(&output[d + 16], float_vec2_low);
        vst1q_f32(&output[d + 20], float_vec2_high);
        vst1q_f32(&output[d + 24], float_vec3_low);
        vst1q_f32(&output[d + 28], float_vec3_high);
    }
    for (; d < length; ++d) {
        output[d] = static_cast<float>(input[d]);
    }
}
#endif

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

        std::memcpy(static_cast<char*>(embeddings->data()) + dst_offset,
                    static_cast<char*>(embedding_table.data()) + src_offset,
                    copy_size);
    }

    // for BERT
    const auto& position_ids     = params.position_ids;
    const auto& position_table   = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const int* input_pos  = position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr;
    const int* input_type = token_types.has_value() ? token_types.value().get().data<int>() : nullptr;

    if (data_type == DataType::TYPE_FP16) {
        const __fp16* pos_table = position_table.has_value() ? (__fp16*)(position_table.value().get().data()) : nullptr;
        const __fp16* type_table =
            token_type_table.has_value() ? (__fp16*)(token_type_table.value().get().data()) : nullptr;

        // case: both pos_table and type_table have values
        if (pos_table != nullptr && type_table != nullptr) {
            for (int index = 0; index < token_num; index++) {
                for (int hid = 0; hid < hidden_size; hid++) {
                    __fp16 pos_embed  = pos_table[input_pos[index] * hidden_size + hid];
                    __fp16 type_embed = type_table[input_type[index] * hidden_size + hid];
                    __fp16* embed_ptr = (__fp16*)(embeddings->data()) + index * hidden_size + hid;
                    *embed_ptr += pos_embed + type_embed;
                }
            }
        }
    }

#if BUILDING_ARM_ONLY
    [[maybe_unused]] int numThreads = omp_get_num_threads();
    if (embeddings->type() == DataType::TYPE_FP16 && position_table.has_value() && token_type_table.has_value()) {
        size_t embeddings_m = embeddings->shape()[0];
        size_t embeddings_n = embeddings->shape()[1];
        auto   embeddings_fp32 =
            allocateBuffer({DataType::TYPE_FP32, {embeddings_m, embeddings_n}, AllocationType::HOST});
        auto embeddings_fp16 =
            allocateBuffer({DataType::TYPE_FP16, {embeddings_m, embeddings_n}, AllocationType::HOST});  // for cache
#pragma omp parallel for num_threads(numThreads) if (numThreads >= 2)
        for (size_t i = 0; i < embeddings_m; i++) {
            convert_embedding_fp16_to_float((__fp16*)embeddings->data() + i * embeddings_n,
                                            (float*)embeddings_fp32->data() + i * embeddings_n,
                                            embeddings_n);
        }

        // update infomation
        embeddings_fp16 = embeddings;
        embeddings      = embeddings_fp32;
    }
#endif

    return embeddings;
}

}  // namespace rtp_llm
