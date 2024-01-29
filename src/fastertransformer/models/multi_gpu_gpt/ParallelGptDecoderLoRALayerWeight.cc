#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLoRALayerWeight.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/cuda/memory_utils.h"
#include "src/fastertransformer/models/W.h"
namespace fastertransformer {
template<typename T>
void ParallelGptDecoderLoRALayerWeight<T>::setLoRAWeight(
    const std::string& name, int lora_id, T* lora_a, T* lora_b, int lora_rank)
{
    if (name == W::attn_qkv_w) {
        q_weights.setLoRAWeight(lora_id, lora_a, lora_b, lora_rank);
    } else if (name == W::attn_o_w) {
        attention_output_weights.setLoRAWeight(lora_id, lora_a, lora_b, lora_rank);
    } else if (name == W::ffn_w1) {
        ffn_intermediate_weights.setLoRAWeight(lora_id, lora_a, lora_b, lora_rank);
    } else if (name == W::ffn_w2) {
        ffn_output_weights.setLoRAWeight(lora_id, lora_a, lora_b, lora_rank);
    } else if (name == W::ffn_w3) {
        ffn_intermediate_weights2.setLoRAWeight(lora_id, lora_a, lora_b, lora_rank);
    } else {
        FT_CHECK_WITH_INFO(false, "error lora weight name");
    }
}
template<typename T>
void ParallelGptDecoderLoRALayerWeight<T>::removeLoRA(const int lora_id)
{
    q_weights.removeLoRAWeight(lora_id);
    attention_output_weights.removeLoRAWeight(lora_id);
    ffn_intermediate_weights.removeLoRAWeight(lora_id);
    ffn_intermediate_weights2.removeLoRAWeight(lora_id);
    ffn_output_weights.removeLoRAWeight(lora_id);
}
template struct ParallelGptDecoderLoRALayerWeight<float>;
template struct ParallelGptDecoderLoRALayerWeight<half>;
template struct ParallelGptDecoderLoRALayerWeight<nv_bfloat16>;
}  // namespace fastertransformer