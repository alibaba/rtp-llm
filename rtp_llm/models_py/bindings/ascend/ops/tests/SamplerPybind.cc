/**
 * Minimal pybind11 wrapper that exposes execSampleGreedy to Python.
 *
 * This module is designed to be loaded AFTER Python `import torch_npu`
 * (which registers NPU dispatch).  Because the module is a separate .so
 * loaded via Python's import system (not a DT_NEEDED pre-load), there
 * is no "Two accelerators" conflict.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;
using namespace rtp_llm;

PYBIND11_MODULE(sampler_test_module, m) {
    m.doc() = "Ascend sampler test bindings – wraps execSampleGreedy";

    m.def(
        "exec_sample_greedy",
        [](torch::Tensor logits,
           torch::Tensor input_lengths,
           torch::Tensor sequence_lengths,
           torch::Tensor token_ids,
           int64_t step,
           torch::Tensor top_k,
           torch::Tensor top_p,
           torch::Tensor temperature,
           std::optional<torch::Tensor> repetition_penalty,
           std::optional<torch::Tensor> no_repeat_ngram_size,
           std::optional<torch::Tensor> cum_log_probs,
           std::optional<torch::Tensor> output_log_probs,
           std::optional<torch::Tensor> output_all_probs,
           std::optional<torch::Tensor> presence_penalty,
           std::optional<torch::Tensor> frequency_penalty,
           std::optional<torch::Tensor> do_sample,
           std::optional<std::vector<at::Generator>> generators) -> py::dict {
            GreedyParams params{
                logits,
                input_lengths,
                sequence_lengths,
                token_ids,
                static_cast<size_t>(step),
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
                cum_log_probs,
                output_log_probs,
                output_all_probs,
                presence_penalty,
                frequency_penalty,
                do_sample,
                generators.value_or(std::vector<at::Generator>{}),
            };

            auto output = execSampleGreedy(params);

            py::dict result;
            result["token_ids"] = token_ids;  // modified in-place
            if (output.success.defined()) {
                result["success"] = output.success;
            }
            if (cum_log_probs.has_value()) {
                result["cum_log_probs"] = cum_log_probs.value();
            }
            if (output_all_probs.has_value()) {
                result["output_all_probs"] = output_all_probs.value();
            }
            return result;
        },
        py::arg("logits"),
        py::arg("input_lengths"),
        py::arg("sequence_lengths"),
        py::arg("token_ids"),
        py::arg("step"),
        py::arg("top_k"),
        py::arg("top_p"),
        py::arg("temperature"),
        py::arg("repetition_penalty")  = py::none(),
        py::arg("no_repeat_ngram_size") = py::none(),
        py::arg("cum_log_probs")       = py::none(),
        py::arg("output_log_probs")    = py::none(),
        py::arg("output_all_probs")    = py::none(),
        py::arg("presence_penalty")    = py::none(),
        py::arg("frequency_penalty")   = py::none(),
        py::arg("do_sample")           = py::none(),
        py::arg("generators")          = py::none(),
        "Execute greedy/random sampling on Ascend NPU.\n\n"
        "Returns dict with keys: token_ids, success, cum_log_probs, output_all_probs.");
}
