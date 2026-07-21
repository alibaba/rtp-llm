#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;
using namespace rtp_llm;

namespace unittest {

// Test-only wrapper class to expose protected methods for unit testing
class ZigZagProcessorTestWrapper: public ZigZagProcessor {
public:
    ZigZagProcessorTestWrapper(): ZigZagProcessor(ParallelismConfig{}) {}
    explicit ZigZagProcessorTestWrapper(const ParallelismConfig& cfg): ZigZagProcessor(cfg) {}
    using ZigZagProcessor::plan;
    using ZigZagProcessor::generateQKVRestoreIndices;
    using ZigZagProcessor::generateQKVPaddingMask;
    using ZigZagProcessor::computeLocalLastHidden;
};

// Wrapper for ZigZagProcessor::plan that returns a tuple
std::tuple<bool, std::vector<int>, std::vector<int>>
zigzagProcessorPlanWrapper(const std::vector<int>& total_input_tokens,
                           std::vector<int>        input_tokens,
                           std::vector<int>        shuffle_indices,
                           int                     cp_rank,
                           int                     cp_size,
                           int                     cp_chunk_size,
                           int                     cp_padding_size) {
    input_tokens.resize(cp_chunk_size);
    shuffle_indices.resize(cp_chunk_size);

    ZigZagProcessorTestWrapper processor;
    bool                       result = processor.plan(
        total_input_tokens, input_tokens, shuffle_indices, cp_rank, cp_size, cp_chunk_size, cp_padding_size);

    return std::make_tuple(result, input_tokens, shuffle_indices);
}

// Wrapper for ZigZagProcessor::generateQKVRestoreIndices
torch::Tensor zigzagGenerateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    ZigZagProcessorTestWrapper processor;
    return processor.generateQKVRestoreIndices(prefill_cp_chunk_lengths, cp_size);
}

// Wrapper for ZigZagProcessor::generateQKVPaddingMask
torch::Tensor zigzagGenerateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                           const torch::Tensor& prefill_cp_padding_lengths,
                                           int                  cp_size) {
    ZigZagProcessorTestWrapper processor;
    return processor.generateQKVPaddingMask(prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size);
}

torch::Tensor zigzagComputeLocalLastHidden(const torch::Tensor& hidden_chunk,
                                           const torch::Tensor& restore_indice,
                                           const torch::Tensor& padding_mask,
                                           const torch::Tensor& lm_output_indexes,
                                           int                  cp_rank,
                                           int                  cp_size) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_rank = cp_rank;
    parallelism_config.tp_size = cp_size;
    ZigZagProcessorTestWrapper processor(parallelism_config);

    GptModelInputs inputs;
    inputs.lm_output_indexes = lm_output_indexes.contiguous().clone();

    torch_ext::PyContextParallelParams cp_params;
    cp_params.prefill_qkv_restore_indice = restore_indice.contiguous().clone();
    cp_params.prefill_qkv_padding_mask   = padding_mask.contiguous().clone();

    return processor.computeLocalLastHidden(hidden_chunk.contiguous().clone(), inputs, cp_params).cpu().clone();
}

PYBIND11_MODULE(libth_context_parallel_py_wrapper_test, m) {
    m.def("context_parallel_load_balance_split",
          &zigzagProcessorPlanWrapper,
          py::arg("total_input_tokens"),
          py::arg("input_tokens"),
          py::arg("shuffle_indices"),
          py::arg("cp_rank"),
          py::arg("cp_size"),
          py::arg("cp_chunk_size"),
          py::arg("cp_padding_size"),
          "Distribute input tokens across context parallel ranks with load balancing (legacy wrapper)");

    m.def("generate_qkv_restore_indices",
          &zigzagGenerateQKVRestoreIndices,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("cp_size"),
          "Generate indices to restore original token order after parallel processing (legacy wrapper)");

    m.def("generate_qkv_padding_mask",
          &zigzagGenerateQKVPaddingMask,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("prefill_cp_padding_lengths"),
          py::arg("cp_size"),
          "Generate padding mask for QKV tensors in context parallel scenarios (legacy wrapper)");

    m.def("compute_local_last_hidden",
          &zigzagComputeLocalLastHidden,
          py::arg("hidden_chunk"),
          py::arg("restore_indice"),
          py::arg("padding_mask"),
          py::arg("lm_output_indexes"),
          py::arg("cp_rank"),
          py::arg("cp_size"),
          "Compute this rank's contribution to the CP gather-last-hidden result");
}

}  // namespace unittest
