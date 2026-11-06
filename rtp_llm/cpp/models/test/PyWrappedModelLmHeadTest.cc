#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm::test {

// Exercise the production output paths without constructing transformer layers.
py::dict runLmHead(py::object           py_model,
                   const torch::Tensor& kernel,
                   torch::Tensor        hidden,
                   const torch::Tensor& indexes,
                   bool                 has_context_request,
                   bool                 need_all_logits,
                   bool                 already_selected,
                   int                  tp_size,
                   size_t               output_vocab_size,
                   const py::object&    all_logits_row_indexes,
                   const py::object&    all_logits_row_labels,
                   int64_t              all_logits_top_k) {
    initRuntime(/*device_id=*/0, /*trace_memory=*/false, /*enable_comm_overlap=*/false, MlaOpsType::AUTO);
    Weights weights;
    weights.layers.resize(1);
    auto head      = std::make_shared<DenseWeights>();
    head->kernel   = kernel;
    weights.lm_head = std::move(head);

    GptModelDescription description;
    description.data_type                    = DataType::TYPE_BF16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = hidden.size(1);
    description.output_vocab_size           = output_vocab_size;
    description.output_vocab_padded_size    = kernel.size(0) * tp_size;

    GptModelInitParams params{weights, description, std::nullopt};
    params.parallelism_config.tp_size                = tp_size;
    params.hw_kernel_config.enable_cuda_graph        = false;
    params.profile_debug_logging_config.check_nan    = true;
    params.hidden_size                               = hidden.size(1);
    params.max_seq_len                               = 64;

    PyWrappedModel model(params, std::move(py_model));
    GptModelInputs inputs;
    inputs.need_all_logits = need_all_logits;
    if (!all_logits_row_indexes.is_none()) {
        inputs.all_logits_row_indexes = all_logits_row_indexes.cast<torch::Tensor>();
    }
    if (!all_logits_row_labels.is_none()) {
        inputs.all_logits_row_labels = all_logits_row_labels.cast<torch::Tensor>();
    }
    inputs.all_logits_top_k = all_logits_top_k;
    auto outputs = already_selected ? model.forwardPostLayersLastHidden(hidden, inputs) :
                                      model.forwardPostLayers(hidden,
                                                              has_context_request,
                                                              need_all_logits,
                                                              indexes,
                                                              /*enable_sp=*/false,
                                                              hidden.size(0),
                                                              inputs,
                                                              torch::Tensor(),
                                                              /*skip_final_layernorm=*/true);
    py::dict result;
    result["logits"]             = outputs.logits;
    result["hidden_states"]      = outputs.hidden_states;
    result["all_hidden_states"]  = outputs.all_hidden_states;
    result["all_logits"]         = outputs.all_logits.defined() ? py::cast(outputs.all_logits) : py::none();
    result["all_logits_lm_rows"] = outputs.all_logits_lm_rows;
    auto cast_or_none            = [](const torch::Tensor& t) -> py::object {
        if (!t.defined()) {
            return py::none();
        }
        return py::cast(t);
    };
    result["all_logits_lse"]           = cast_or_none(outputs.all_logits_lse);
    result["all_logits_target_logits"] = cast_or_none(outputs.all_logits_target_logits);
    result["all_logits_topk_values"]   = cast_or_none(outputs.all_logits_topk_values);
    result["all_logits_topk_ids"]      = cast_or_none(outputs.all_logits_topk_ids);
    return result;
}

}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_lm_head_test, m) {
    m.def("run_lm_head",
          &rtp_llm::test::runLmHead,
          py::arg("py_model"),
          py::arg("kernel"),
          py::arg("hidden"),
          py::arg("indexes"),
          py::arg("has_context_request")    = true,
          py::arg("need_all_logits")        = false,
          py::arg("already_selected")       = false,
          py::arg("tp_size")                = 1,
          py::arg("output_vocab_size")      = 0,
          py::arg("all_logits_row_indexes") = py::none(),
          py::arg("all_logits_row_labels")  = py::none(),
          py::arg("all_logits_top_k")       = int64_t{0});
}
