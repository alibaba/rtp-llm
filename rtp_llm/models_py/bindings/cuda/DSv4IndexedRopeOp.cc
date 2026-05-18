#include "rtp_llm/models_py/bindings/cuda/DSv4IndexedRopeOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_indexed_inv_rope_fp8_quant.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_indexed_rmsnorm_rope.h"

namespace torch_ext {

void dsv4_indexed_rmsnorm_rope(at::Tensor& input,
                               at::Tensor& weight,
                               at::Tensor& freqs_cis,
                               at::Tensor& position_ids,
                               at::Tensor& output,
                               int64_t     rope_head_dim,
                               double      eps,
                               bool        has_weight) {
    rtp_llm::dsv4_indexed_rmsnorm_rope(input, weight, freqs_cis, position_ids, output, rope_head_dim, eps, has_weight);
}

void dsv4_indexed_inv_rope_fp8_quant(at::Tensor& input,
                                     at::Tensor& freqs_cis,
                                     at::Tensor& position_ids,
                                     at::Tensor& output_q,
                                     at::Tensor& output_s,
                                     int64_t     n_groups,
                                     int64_t     heads_per_group,
                                     int64_t     nope_dim,
                                     int64_t     rope_head_dim,
                                     double      eps,
                                     double      fp8_max,
                                     int64_t     kernel_mode) {
    rtp_llm::dsv4_indexed_inv_rope_fp8_quant(
        input,
        freqs_cis,
        position_ids,
        output_q,
        output_s,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_head_dim,
        eps,
        fp8_max,
        kernel_mode);
}

}  // namespace torch_ext
