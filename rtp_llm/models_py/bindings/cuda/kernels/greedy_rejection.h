#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// Fully writes output_ids [batch, proposals+1] and accepted [batch].
// target_ids has shape [batch*(proposals+1), target_stride]; its last column
// contains the already sampled target token. Inputs and outputs are contiguous
// int32 tensors. The caller must restrict proposals to [0, 31].
extern "C" cudaError_t rtpLlmGreedyRejection(const int32_t* draft_ids,
                                            const int32_t* target_ids,
                                            int32_t*       output_ids,
                                            int32_t*       accepted,
                                            int            batch,
                                            int            proposals,
                                            int            target_stride,
                                            cudaStream_t   stream);
