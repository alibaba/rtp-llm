#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/PerTokenGroupQuantFp4.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/per_token_group_quant_fp4.h"

namespace torch_ext {

void per_token_group_quant_fp4(at::Tensor& input,
                               at::Tensor& output_q,
                               at::Tensor& output_s,
                               int64_t     group_size,
                               double      eps,
                               bool        use_packed_ue8m0) {
    rtp_llm::per_token_group_quant_fp4(input, output_q, output_s, group_size, eps, use_packed_ue8m0);
}

}  // namespace torch_ext
