#pragma once

#include <torch/all.h>
#include <cuda_runtime.h>

namespace rtp_llm {

void applySamplerTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures, cudaStream_t stream);
torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits);

struct RejectionSamplingInputs {
    torch::Tensor draft_probs;
    torch::Tensor draft_ids;
    torch::Tensor uniforms;
    torch::Tensor target_probs;
    torch::Tensor target_ids;
    torch::Tensor output_ids;
    torch::Tensor accepted;
    torch::Tensor do_sample;
    int target_stride;
    int batch;
    int steps;
    int vocab;
    bool point_mass;
};

cudaError_t dispatchRejectionSampling(const RejectionSamplingInputs& inputs, cudaStream_t stream);

}  // namespace rtp_llm
