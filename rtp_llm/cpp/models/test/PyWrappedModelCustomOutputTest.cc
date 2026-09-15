#include <mutex>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm::test {

// Exercise the real post-layers implementation with tiny deterministic weights,
// without a checkpoint or attention kernels. The BUILD target enables private
// access just like the existing PyWrappedModel integration tests.
py::dict runCustomOutput(py::object                   py_model,
                         py::object                   handler,
                         torch::Tensor                hidden,
                         torch::Tensor                lm_indexes,
                         std::optional<torch::Tensor> custom_indexes,
                         int64_t                      decode_batch,
                         bool                         python_norm,
                         bool                         all_logits,
                         std::optional<torch::Tensor> pre_hidden) {
    static std::once_flag runtime_once;
    std::call_once(runtime_once, []() { initRuntime(0, false, false, MlaOpsType::AUTO); });
    const auto width = hidden.size(1);
    Weights    weights;
    auto       lm_head      = std::make_shared<DenseWeights>();
    lm_head->kernel         = torch::eye(width, hidden.options());
    weights.lm_head         = lm_head;
    auto norm               = std::make_shared<LayerNormWeights>();
    norm->gamma             = torch::ones({width}, hidden.options());
    weights.final_layernorm = norm;
    GptModelDescription description;
    description.data_type                    = DataType::TYPE_FP32;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.size_per_head = width;
    GptModelInitParams params{weights, description, std::nullopt};
    params.device_resource_config.enable_layer_micro_batch = python_norm ? 0 : 1;
    PyWrappedModel model(params, std::move(py_model));
    auto           processor = std::make_shared<PostLayersProcessor>();
    processor->setHandler(std::move(handler));
    model.setPostLayersProcessor(processor);

    GptModelInputs inputs;
    inputs.combo_tokens          = torch::zeros({hidden.size(0)}, torch::kInt32);
    inputs.input_lengths         = torch::ones({lm_indexes.size(0)}, torch::kInt32);
    inputs.sequence_lengths      = torch::ones({decode_batch}, torch::kInt32);
    inputs.lm_output_indexes     = lm_indexes;
    inputs.custom_output_indexes = custom_indexes.value_or(torch::Tensor());
    inputs.need_all_logits       = all_logits;
    auto outputs = model.callForwardPostLayers(hidden, inputs, python_norm, -1, pre_hidden.value_or(torch::Tensor()));
    py::dict result;
    result["custom_output"]       = outputs.custom_output;
    result["custom_output_error"] = outputs.custom_output_error;
    result["logits"]              = outputs.logits;
    result["hidden_states"]       = outputs.all_hidden_states;
    return result;
}

}  // namespace rtp_llm::test

PYBIND11_MODULE(libth_pywrapped_model_custom_output_test, m) {
    torch_ext::registerPyOpDefs(m);
    m.def("run_post_layers", &rtp_llm::test::runCustomOutput);
    py::class_<rtp_llm::PostLayersProcessor>(m, "PostLayersProcessor")
        .def(py::init<>())
        .def("set_handler", &rtp_llm::PostLayersProcessor::setHandler)
        .def("uses_pre_final_norm", &rtp_llm::PostLayersProcessor::usesPreFinalNorm);
}
