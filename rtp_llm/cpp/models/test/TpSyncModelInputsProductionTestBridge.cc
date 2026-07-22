#include "rtp_llm/cpp/models/ModelTypes.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace py = pybind11;

namespace rtp_llm {
namespace {

using TpSyncModelInputsFn = void (*)(GptModelInputs&, const ParallelismConfig&);

// This is the exported C++ ABI symbol from the shipped libth_transformer.so.
// Resolving it dynamically keeps this bridge from compiling or linking another
// copy of tpSyncModelInputs; the test therefore crosses the same production DSO
// boundary as the Python runtime.
constexpr char kTpSyncModelInputsSymbol[] =
    "_ZN7rtp_llm17tpSyncModelInputsERNS_14GptModelInputsERKNS_17ParallelismConfigE";

TpSyncModelInputsFn loadProductionTpSyncModelInputs(const std::string& transformer_path) {
    static std::mutex          mu;
    static void*               transformer_handle = nullptr;
    static TpSyncModelInputsFn tp_sync            = nullptr;
    static std::string         loaded_path;

    std::lock_guard<std::mutex> lock(mu);
    if (tp_sync != nullptr) {
        if (loaded_path != transformer_path) {
            throw std::runtime_error("tpSyncModelInputs production bridge was already loaded from a different path");
        }
        return tp_sync;
    }

    transformer_handle = ::dlopen(transformer_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (transformer_handle == nullptr) {
        throw std::runtime_error(std::string("failed to load production libth_transformer.so: ") + ::dlerror());
    }
    ::dlerror();
    void*       symbol = ::dlsym(transformer_handle, kTpSyncModelInputsSymbol);
    const char* error  = ::dlerror();
    if (symbol == nullptr || error != nullptr) {
        throw std::runtime_error(std::string("failed to resolve production tpSyncModelInputs: ")
                                 + (error == nullptr ? "symbol address is null" : error));
    }

    tp_sync     = reinterpret_cast<TpSyncModelInputsFn>(symbol);
    loaded_path = transformer_path;
    return tp_sync;
}

torch::Tensor int32Tensor(const std::vector<int32_t>& values) {
    return torch::tensor(values, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
}

torch::Tensor int64Tensor(const std::vector<int64_t>& values) {
    return torch::tensor(values, torch::TensorOptions(torch::kInt64).device(torch::kCPU));
}

torch::Tensor boolTensor(const std::vector<uint8_t>& values) {
    return torch::tensor(values, torch::TensorOptions(torch::kBool).device(torch::kCPU));
}

py::dict runProductionTpSyncModelInputs(const std::string& transformer_path,
                                        int                tp_rank,
                                        bool               include_gpu,
                                        bool               root_combo_tokens_on_gpu,
                                        bool               empty_combo_tokens) {
    ParallelismConfig config;
    config.tp_size          = 2;
    config.tp_rank          = tp_rank;
    config.world_size       = 2;
    config.world_rank       = tp_rank;
    config.local_world_size = 2;
    config.local_rank       = tp_rank;

    GptModelInputs inputs;
    inputs.pd_separation = include_gpu;
    if (tp_rank == 0) {
        inputs.combo_tokens          = int32Tensor({11, 12, 13});
        inputs.input_lengths         = int32Tensor({2, 1});
        inputs.sequence_lengths      = int32Tensor({5, 6});
        inputs.prefix_lengths        = int32Tensor({0, 2});
        inputs.request_id            = int64Tensor({101, 202});
        inputs.request_pd_separation = boolTensor({0, 1});
        inputs.lm_output_indexes     = int32Tensor({1, 2});
        inputs.lm_output_lengths     = int32Tensor({7, 8});
        inputs.combo_position_ids    = int32Tensor({0, 1, 2});
        inputs.text_tokens_mask      = int32Tensor({1, 1, 0});
        if (empty_combo_tokens) {
            inputs.combo_tokens = int32Tensor({});
        }
        if (root_combo_tokens_on_gpu) {
            inputs.combo_tokens = inputs.combo_tokens.to(torch::Device(torch::kCUDA, tp_rank));
        }
        if (include_gpu) {
            inputs.kv_cache_kernel_block_id = int32Tensor({1, 2, 3, 4}).reshape({1, 2, 2});
            inputs.kv_cache_block_id        = int32Tensor({10, 11, 12, 20, 21, 22}).reshape({1, 2, 3});
            inputs.kv_cache_group_types     = int32Tensor({1});
            inputs.kv_cache_update_mapping  = int32Tensor({0, 4, 5, 0, 6, 7}).reshape({2, 3});
            inputs.cache_keys               = int64Tensor({1001, 1002, 1003, 2001, 2002, 2003}).reshape({2, 3});
            inputs.mm_features_locs         = int32Tensor({0, 2});
            inputs.multimodal_features      = std::vector<torch::Tensor>{
                torch::tensor(std::vector<float>{10.5f, 11.5f, 12.5f, 13.5f}, torch::TensorOptions(torch::kFloat32))
                    .reshape({2, 2})
                    .to(torch::Device(torch::kCUDA, tp_rank)),
                torch::tensor(std::vector<float>{20.5f, 21.5f}, torch::TensorOptions(torch::kFloat32))
                    .reshape({1, 2})
                    .to(torch::Device(torch::kCUDA, tp_rank)),
            };
            inputs.mm_extra_input = std::vector<torch::Tensor>{
                torch::tensor(std::vector<float>{30.5f, 31.5f, 32.5f}, torch::TensorOptions(torch::kFloat32))
                    .to(torch::Device(torch::kCUDA, tp_rank)),
                torch::tensor(std::vector<float>{40.5f}, torch::TensorOptions(torch::kFloat32))
                    .to(torch::Device(torch::kCUDA, tp_rank)),
            };
            inputs.last_hidden_states = torch::tensor(std::vector<float>{1.25f, 2.5f, 3.75f, 4.5f, 5.25f, 6.5f},
                                                      torch::TensorOptions(torch::kFloat32).device(torch::kCPU))
                                            .reshape({3, 2})
                                            .to(torch::Device(torch::kCUDA, tp_rank));
        }
    }

    loadProductionTpSyncModelInputs(transformer_path)(inputs, config);

    py::dict result;
    result["combo_tokens"]          = inputs.combo_tokens;
    result["input_lengths"]         = inputs.input_lengths;
    result["sequence_lengths"]      = inputs.sequence_lengths;
    result["prefix_lengths"]        = inputs.prefix_lengths;
    result["request_id"]            = inputs.request_id;
    result["request_pd_separation"] = inputs.request_pd_separation;
    result["lm_output_indexes"]     = inputs.lm_output_indexes;
    result["lm_output_lengths"]     = inputs.lm_output_lengths;
    result["combo_position_ids"]    = inputs.combo_position_ids;
    result["text_tokens_mask"]      = inputs.text_tokens_mask;
    if (include_gpu) {
        result["kv_cache_kernel_block_id"] = inputs.kv_cache_kernel_block_id;
        result["kv_cache_block_id"]        = inputs.kv_cache_block_id;
        result["kv_cache_group_types"]     = inputs.kv_cache_group_types;
        result["kv_cache_update_mapping"]  = inputs.kv_cache_update_mapping;
        result["cache_keys"]               = inputs.cache_keys;
        result["mm_features_locs"]         = inputs.mm_features_locs;
        result["multimodal_feature_0"]     = inputs.multimodal_features.value()[0].cpu();
        result["multimodal_feature_1"]     = inputs.multimodal_features.value()[1].cpu();
        result["mm_extra_input_0"]         = inputs.mm_extra_input.value()[0].cpu();
        result["mm_extra_input_1"]         = inputs.mm_extra_input.value()[1].cpu();
        result["last_hidden_states"]       = inputs.last_hidden_states.cpu();
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(libtp_sync_model_inputs_production_test_bridge, m) {
    m.def("run_tp_sync_model_inputs",
          &runProductionTpSyncModelInputs,
          py::arg("transformer_path"),
          py::arg("tp_rank"),
          py::arg("include_gpu"),
          py::arg("root_combo_tokens_on_gpu") = false,
          py::arg("empty_combo_tokens")       = false,
          "Call the exported tpSyncModelInputs from the shipped libth_transformer.so.");
}

}  // namespace rtp_llm
