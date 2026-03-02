#pragma once
#include <torch/extension.h>
#include "hip/hip_runtime.h"

void chain_speculative_sampling(at::Tensor draft_probs,
                                at::Tensor draft_token_ids,
                                at::Tensor uniform_samples,
                                at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num,
                                bool       deterministic,
                                int64_t    hip_stream);

template<typename DType, typename IdType>
hipError_t invokeRejectionSampling(DType*      draft_probs,
                                   IdType*     draft_token_ids,
                                   DType*      uniform_samples,
                                   DType*      target_probs,
                                   IdType*     target_token_ids,
                                   int         target_token_stride,
                                   IdType*     output_token_ids,
                                   IdType*     output_accepted_token_num,
                                   bool*       do_sample,
                                   int         batch_size,
                                   int         num_speculative_tokens,
                                   int         target_vocab_size,
                                   hipStream_t stream);
