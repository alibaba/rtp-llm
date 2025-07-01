#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <torch/extension.h>


namespace torch_ext {

struct PyModelInitResources {
    torch::Tensor k_cache_base;
    torch::Tensor v_cache_base;
};

struct PyAttentionInputs {
    std::shared_ptr<void> prefill_flash_infer_attn;
    std::shared_ptr<void> decode_flash_infer_attn;
};

struct PyModelInputs {
    torch::Tensor input_ids;
    PyAttentionInputs attention_inputs;
};

struct PyModelOutputs {
    torch::Tensor hidden_states;
};

void registerPyOpDefs(pybind11::module &m);

}

