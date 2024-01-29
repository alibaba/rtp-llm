#pragma once
#include <string>
#include "src/fastertransformer/kernels/calibrate_quantize_weight_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/utils/LoRAWeight.h"
namespace fastertransformer {
template<typename T>
struct ParallelGptDecoderLoRALayerWeight{
public:
    LoRAWeight<T> q_weights;
    LoRAWeight<T> attention_output_weights;
    LoRAWeight<T> ffn_intermediate_weights;
    LoRAWeight<T> ffn_intermediate_weights2;
    LoRAWeight<T> ffn_output_weights;
    ParallelGptDecoderLoRALayerWeight() {}
    void setLoRAWeight(const std::string& name, int lora_id, T* lora_a, T* lora_b, int lora_rank);
    void removeLoRA(const int lora_id);
};
}