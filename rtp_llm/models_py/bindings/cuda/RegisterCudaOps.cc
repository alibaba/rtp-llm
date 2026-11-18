#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/common/CpuTpStatus.h"

#ifdef RTP_ENABLE_SM120_LSA
#include "rtp_llm/models_py/kernels/cuda/qwen38_moe/reduce_gate.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#endif

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

#ifdef RTP_ENABLE_SM120_LSA
namespace {
void checkLsa(int status) {
    if (status)
        throw std::runtime_error(qwen38_lsa_error());
}

void registerLsaBindings(py::module& ops) {
    auto m = ops.def_submodule("_qwen38_lsa");
    m.def("device", [](int ordinal) {
        char bus_id[32];
        checkLsa(qwen38_lsa_device(ordinal, bus_id, sizeof(bus_id)));
        return std::string(bus_id);
    });
    m.def("check_peer", [](int ordinal, const std::string& peer) {
        checkLsa(qwen38_lsa_peer(ordinal, peer.c_str()));
    });
    m.def("unique_id", []() {
        std::string uid(qwen38_lsa_uid_size(), '\0');
        checkLsa(qwen38_lsa_unique_id(uid.data()));
        return py::bytes(uid);
    });
    m.def("allocate", [](int device, int hidden) {
        void* handle = nullptr;
        py::gil_scoped_release release;
        checkLsa(qwen38_lsa_allocate(device, hidden, &handle));
        return reinterpret_cast<uintptr_t>(handle);
    });
    m.def("connect", [](uintptr_t handle, int rank, const py::bytes& value) {
        const std::string uid = value;
        if (uid.size() != static_cast<size_t>(qwen38_lsa_uid_size()))
            throw std::invalid_argument("Invalid LSA NCCL unique ID size");
        py::gil_scoped_release release;
        checkLsa(qwen38_lsa_connect(reinterpret_cast<void*>(handle), rank, uid.data()));
    });
    m.def("check_capability", [](uintptr_t handle) {
        py::gil_scoped_release release;
        checkLsa(qwen38_lsa_check_capability(reinterpret_cast<void*>(handle)));
    });
    m.def("activate", [](uintptr_t handle) {
        py::gil_scoped_release release;
        checkLsa(qwen38_lsa_activate(reinterpret_cast<void*>(handle)));
    });
    m.def("apply", [](uintptr_t handle,
                      uintptr_t partial,
                      uintptr_t shared,
                      uintptr_t gate,
                      uintptr_t output,
                      int rows,
                      uintptr_t stream) {
        checkLsa(qwen38_lsa_apply(reinterpret_cast<void*>(handle),
                                 reinterpret_cast<const void*>(partial),
                                 reinterpret_cast<const void*>(shared),
                                 reinterpret_cast<const void*>(gate),
                                 reinterpret_cast<void*>(output),
                                 rows,
                                 reinterpret_cast<void*>(stream)));
    });
    m.def("reference", [](uintptr_t handle, uintptr_t partial, uintptr_t output, int rows, uintptr_t stream) {
        checkLsa(qwen38_lsa_reference(reinterpret_cast<void*>(handle),
                                     reinterpret_cast<const void*>(partial),
                                     reinterpret_cast<void*>(output),
                                     rows,
                                     reinterpret_cast<void*>(stream)));
    });
    m.def("destroy", [](uintptr_t handle) {
        py::gil_scoped_release release;
        qwen38_lsa_destroy(reinterpret_cast<void*>(handle));
    });
}
}  // namespace
#endif

void registerPyModuleOps(py::module& rtp_ops_m) {
#ifdef RTP_ENABLE_SM120_LSA
    registerLsaBindings(rtp_ops_m);
#endif
    rtp_ops_m.def("cpu_status_size", &cpu_status_size, py::arg("world_size") = 2);
    rtp_ops_m.def(
        "cpu_status_initialize",
        [](uintptr_t pointer, size_t bytes, int world_size) {
            return cpu_status_initialize(reinterpret_cast<void*>(pointer), bytes, world_size);
        },
        py::arg("pointer"),
        py::arg("bytes"),
        py::arg("world_size") = 2);
    rtp_ops_m.def("cpu_status_break",
                  [](uintptr_t pointer) { return cpu_status_break(reinterpret_cast<void*>(pointer)); });
    rtp_ops_m.def(
        "cpu_status_reduce",
        [](uintptr_t pointer, int rank, uint64_t generation, int local_ok, int timeout_ms) {
            return cpu_status_reduce(reinterpret_cast<void*>(pointer), rank, generation, local_ok, timeout_ms);
        },
        py::call_guard<py::gil_scoped_release>());

    rtp_ops_m.def("cublas_gemm_bf16_bf16_fp32",
                  &torch_ext::cublas_gemm_bf16_bf16_fp32,
                  "cuBLAS BF16 x BF16 GEMM with FP32 accumulation and FP32 output",
                  py::arg("input"),
                  py::arg("weight"));

    rtp_ops_m.def("per_tensor_quant_fp8",
                  &per_tensor_quant_fp8,
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("is_static"));

    rtp_ops_m.def(
        "per_token_quant_fp8", &per_token_quant_fp8, py::arg("input"), py::arg("output_q"), py::arg("output_s"));

    // NVFP4 mm/quant live in rtp_kernel.nvfp4. Keep the grouped experts
    // quantizers here; they are not part of that port.
#if defined(ENABLE_FP4)
    rtp_ops_m.def("scaled_fp4_experts_quant",
                  &scaled_fp4_experts_quant_sm100a,
                  py::arg("output"),
                  py::arg("output_scale"),
                  py::arg("input"),
                  py::arg("input_global_scale"),
                  py::arg("input_offset_by_experts"),
                  py::arg("output_scale_offset_by_experts"));

    rtp_ops_m.def("silu_and_mul_scaled_fp4_experts_quant",
                  &silu_and_mul_scaled_fp4_experts_quant_sm100a,
                  py::arg("output"),
                  py::arg("output_scale"),
                  py::arg("input"),
                  py::arg("input_global_scale"),
                  py::arg("mask"),
                  py::arg("use_silu_and_mul"));
#endif

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
