#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/fp8_group_gemm.h"
#include "rtp_llm/cpp/kernels/scaled_fp8_quant.h"
#include "rtp_llm/cpp/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("cutlass_moe_mm",
                  &cutlass_moe_mm,
                  py::arg("out_tensors"),
                  py::arg("a_tensors"),
                  py::arg("b_tensors"),
                  py::arg("a_scales"),
                  py::arg("b_scales"),
                  py::arg("expert_offsets"),
                  py::arg("problem_sizes"),
                  py::arg("a_strides"),
                  py::arg("b_strides"),
                  py::arg("c_strides"),
                  py::arg("per_act_token"),
                  py::arg("per_out_ch"));

    rtp_ops_m.def("get_cutlass_moe_mm_data",
                  &get_cutlass_moe_mm_data,
                  py::arg("topk_ids"),
                  py::arg("expert_offsets"),
                  py::arg("problem_sizes1"),
                  py::arg("problem_sizes2"),
                  py::arg("input_permutation"),
                  py::arg("output_permutation"),
                  py::arg("num_experts"),
                  py::arg("n"),
                  py::arg("k"),
                  py::arg("blockscale_offsets") = py::none());

    rtp_ops_m.def("get_cutlass_batched_moe_mm_data",
                  &get_cutlass_batched_moe_mm_data,
                  py::arg("expert_offsets"),
                  py::arg("problem_sizes1"),
                  py::arg("problem_sizes2"),
                  py::arg("expert_num_tokens"),
                  py::arg("num_local_experts"),
                  py::arg("padded_m"),
                  py::arg("n"),
                  py::arg("k"));

    rtp_ops_m.def("get_cutlass_moe_mm_without_permute_info",
                  &get_cutlass_moe_mm_without_permute_info,
                  py::arg("topk_ids"),
                  py::arg("problem_sizes1"),
                  py::arg("problem_sizes2"),
                  py::arg("num_experts"),
                  py::arg("n"),
                  py::arg("k"),
                  py::arg("blockscale_offsets") = py::none());

    rtp_ops_m.def("per_tensor_quant_fp8",
                  &per_tensor_quant_fp8,
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("is_static"));

    rtp_ops_m.def(
        "per_token_quant_fp8", &per_token_quant_fp8, py::arg("input"), py::arg("output_q"), py::arg("output_s"));

    rtp_ops_m.def("moe_pre_reorder",
                  &moe_pre_reorder,
                  "moe ep permute kernel",
                  py::arg("input"),
                  py::arg("topk_ids"),
                  py::arg("token_expert_indices"),
                  py::arg("expert_map") = py::none(),
                  py::arg("n_expert"),
                  py::arg("n_local_expert"),
                  py::arg("topk"),
                  py::arg("align_block_size") = py::none(),
                  py::arg("permuted_input"),
                  py::arg("expert_first_token_offset"),
                  py::arg("inv_permuted_idx"),
                  py::arg("permuted_idx"));

    rtp_ops_m.def("moe_post_reorder",
                  &moe_post_reorder,
                  "moe ep unpermute kernel",
                  py::arg("permuted_hidden_states"),
                  py::arg("topk_weights"),
                  py::arg("inv_permuted_idx"),
                  py::arg("expert_first_token_offset") = py::none(),
                  py::arg("topk"),
                  py::arg("hidden_states"));

    registerBaseCudaBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}  // namespace rtp_llm
