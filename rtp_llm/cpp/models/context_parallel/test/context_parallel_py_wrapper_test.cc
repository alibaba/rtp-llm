#include "rtp_llm/cpp/models/context_parallel/ContextParallelUtils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;
using namespace rtp_llm;

namespace unittest {

// Wrapper for contextParallelLoadBalanceSplit that returns a tuple
std::tuple<bool, std::vector<int>, std::vector<int>>
contextParallelLoadBalanceSplitWrapper(const std::vector<int>& total_input_tokens,
                                       std::vector<int>        input_tokens,
                                       std::vector<int>        shuffle_indices,
                                       int                     cp_rank,
                                       int                     cp_size,
                                       int                     cp_chunk_size,
                                       int                     cp_padding_size) {

    input_tokens.resize(cp_chunk_size);
    shuffle_indices.resize(cp_chunk_size);

    bool result = rtp_llm::contextParallelLoadBalanceSplit(
        total_input_tokens, input_tokens, shuffle_indices, cp_rank, cp_size, cp_chunk_size, cp_padding_size);

    return std::make_tuple(result, input_tokens, shuffle_indices);
}

PYBIND11_MODULE(libth_context_parallel_py_wrapper_test, m) {
    m.def("context_parallel_load_balance_split",
          &contextParallelLoadBalanceSplitWrapper,
          py::arg("total_input_tokens"),
          py::arg("input_tokens"),
          py::arg("shuffle_indices"),
          py::arg("cp_rank"),
          py::arg("cp_size"),
          py::arg("cp_chunk_size"),
          py::arg("cp_padding_size"),
          "Distribute input tokens across context parallel ranks with load balancing");

    m.def("generate_qkv_restore_indices",
          &rtp_llm::generateQKVRestoreIndices,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("cp_size"),
          "Generate indices to restore original token order after parallel processing");

    m.def("generate_qkv_padding_mask",
          &rtp_llm::generateQKVPaddingMask,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("prefill_cp_padding_lengths"),
          py::arg("cp_size"),
          "Generate padding mask for QKV tensors in context parallel scenarios");
}

}  // namespace unittest
