#include "rtp_llm/cpp/models/context_parallel/RoundRobinProcessor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;
using namespace rtp_llm;

namespace unittest {

class RoundRobinProcessorTestWrapper : public RoundRobinProcessor {
public:
    using RoundRobinProcessor::plan;
    using RoundRobinProcessor::generateQKVRestoreIndices;
    using RoundRobinProcessor::generateQKVPaddingMask;
};

std::tuple<bool, std::vector<int>, std::vector<int>>
roundRobinProcessorPlanWrapper(const std::vector<int>& total_input_tokens,
                               std::vector<int>        input_tokens,
                               std::vector<int>        shuffle_indices,
                               int                     cp_rank,
                               int                     cp_size,
                               int                     cp_chunk_size,
                               int                     cp_padding_size) {
    input_tokens.resize(cp_chunk_size);
    shuffle_indices.resize(cp_chunk_size);

    RoundRobinProcessorTestWrapper processor;
    bool result = processor.plan(
        total_input_tokens, input_tokens, shuffle_indices, cp_rank, cp_size, cp_chunk_size, cp_padding_size);

    return std::make_tuple(result, input_tokens, shuffle_indices);
}

torch::Tensor roundRobinGenerateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    RoundRobinProcessorTestWrapper processor;
    return processor.generateQKVRestoreIndices(prefill_cp_chunk_lengths, cp_size);
}

torch::Tensor roundRobinGenerateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                               const torch::Tensor& prefill_cp_padding_lengths,
                                               int                  cp_size) {
    RoundRobinProcessorTestWrapper processor;
    return processor.generateQKVPaddingMask(prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size);
}

PYBIND11_MODULE(libth_round_robin_py_wrapper_test, m) {
    m.def("round_robin_plan",
          &roundRobinProcessorPlanWrapper,
          py::arg("total_input_tokens"),
          py::arg("input_tokens"),
          py::arg("shuffle_indices"),
          py::arg("cp_rank"),
          py::arg("cp_size"),
          py::arg("cp_chunk_size"),
          py::arg("cp_padding_size"),
          "Distribute input tokens across context parallel ranks with round-robin pattern");

    m.def("round_robin_generate_qkv_restore_indices",
          &roundRobinGenerateQKVRestoreIndices,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("cp_size"),
          "Generate indices to restore original token order after round-robin parallel processing");

    m.def("round_robin_generate_qkv_padding_mask",
          &roundRobinGenerateQKVPaddingMask,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("prefill_cp_padding_lengths"),
          py::arg("cp_size"),
          "Generate padding mask for QKV tensors in round-robin context parallel scenarios");
}

}  // namespace unittest
