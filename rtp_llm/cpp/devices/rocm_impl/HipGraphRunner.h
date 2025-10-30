#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/devices/rocm_impl/HipGraphUtils.h"
#include <ATen/hip/HIPGraph.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
// Forward declaration to avoid circular dependency
// #include "rtp_llm/models_py/bindings/rocm/AiterOp.h"
#include "rtp_llm/cpp/devices/GraphBase.h"
namespace py = pybind11;
namespace rtp_llm {

class HipGraphRunner: public GraphBase {
public:
    HipGraphRunner(const DeviceInitParams& params,
                   py::object              py_instance,
                   int                     kv_cache_block_offset,
                   DeviceBase*             device,
                   bool                    is_prefill_hip_graph_mode = false);

    ~HipGraphRunner();

    void           capture();
    void           captureOneBatchSize(int bs);
    void           prepareInputs(PyModelInputs& inputs);
    bool           canRun(PyModelInputs& inputs);
    void           replay(int bs);
    void           initCapture() override;
    void           initKernelInternalMemory();
    int            getCurrentRealGraphBs();
    PyModelOutputs forward(PyModelInputs& inputs) override;
    py::object     normalForward(PyModelInputs& inputs);
    void           setPositionEncoding(torch::Tensor position_encoding) override {
        // TODO: 实现你的逻辑，或先留空
    }

    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override {
        // TODO: 实现你的逻辑，或先留空
    }

    void setInputEmbeddingScalar(float input_embedding_scalar) override {
        // TODO: 实现你的逻辑，或先留空
    }

    void setModelDataType(caffe2::TypeMeta data_type) override {
        // TODO: 实现你的逻辑，或先留空
    }

private:
    void             copySmallerIntoLarger(const torch::Tensor& source_tensor, torch::Tensor& target_tensor);
    std::vector<int> getBatchSizesToCapture(int concurrency_limit);
    bool             tryGetRealGraphBatchSize(PyModelInputs& inputs);
    void extractValidHiddenStates(PyModelOutputs& outputs, const PyModelInputs& inputs, int32_t total_valid_tokens);
    std::string sizesToString(const torch::Tensor& tensor);
    std::string getDTypeName(const torch::Tensor& tensor);

    py::object                             py_forward_method_;
    py::object                             py_fill_params_method_;
    bool                                   enable_hip_graph_{false};
    bool                                   is_prefill_hip_graph_mode_{false};
    int                                    concurrency_limit_{32};
    at::hip::HIPStream                     capture_stream_;
    bool                                   enable_hip_graph_debug_mode_{false};
    int                                    hidden_size_;
    size_t                                 max_bs_{1};
    int                                    num_tokens_per_bs_{1};
    int                                    max_num_token_{1};
    int                                    current_batch_size_{1};
    int                                    current_real_graph_bs_{1};
    int                                    max_seq_len_{0};
    int                                    seq_size_per_block_{0};
    int                                    kv_cache_block_offset_{0};
    int                                    seq_len_sum_{0};
    std::vector<int>                       capture_range_;
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;

public:
    DeviceBase* device_{nullptr};
};

}  // namespace rtp_llm