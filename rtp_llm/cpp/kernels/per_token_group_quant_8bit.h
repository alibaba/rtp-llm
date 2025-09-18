#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <cmath>

namespace rtp_llm {

void per_token_group_quant_8bit(torch::Tensor input,
                                torch::Tensor output_q,
                                torch::Tensor output_s,
                                int64_t       group_size,
                                double        eps,
                                double        min_8bit,
                                double        max_8bit,
                                bool          scale_ue8m0);

}