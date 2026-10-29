#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace rtp_llm {
struct MtpNextInputRow {
    const int32_t* token;
    const int32_t* committed_length;
};
// One packed output allocation; 128 descriptors per launch, no metadata H2D.
// Sources are owned by request batch states until submission completes.
bool tryMtpPrepareNextInputs(const std::vector<MtpNextInputRow>& rows,
                              const torch::TensorOptions& options,
                              torch::Tensor& packed,
                              cudaStream_t stream);

// Preserve the BF16 selected rows returned by ModelOutputs and prepare their
// exact FP32 widening for the MTP head in the same kernel.
// Single-token MTP decode attention inputs; context/target-verify keep their own preparation.
bool tryMtpDecodeAttentionMetadata(const torch::Tensor& lengths,
                                   torch::Tensor&       cu,
                                   torch::Tensor&       cu_kv,
                                   torch::Tensor&       decode_cu,
                                   torch::Tensor&       lengths_plus_one,
                                   cudaStream_t         stream);

bool tryMtpHeadInputs(const torch::Tensor& hidden,
                      const torch::Tensor& indices,
                      torch::Tensor&       selected_hidden,
                      torch::Tensor&       head_input,
                      cudaStream_t         stream);

bool tryMtpAcceptedTokenGather(const std::vector<torch::Tensor>& tokens,
                               const std::vector<torch::Tensor>& lengths,
                               torch::Tensor&                    output,
                               cudaStream_t                      stream);
}  // namespace rtp_llm
