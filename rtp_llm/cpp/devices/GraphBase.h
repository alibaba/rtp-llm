#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <unordered_map>
#include <functional>
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
namespace rtp_llm {

using namespace torch_ext;

class GraphBase {
public:
    GraphBase(py::object py_instance);

    GraphBase(const DeviceInitParams& params,
              py::object              py_instance,
              int                     kv_cache_block_offset,
              DeviceBase*             device,
              bool                    is_prefill_mode);

    virtual ~GraphBase() = default;

    // 设备特定的虚函数
    virtual void                                              initCapture();
    virtual void                                              setPositionEncoding(torch::Tensor position_encoding);
    virtual void                                              setTokenTypeEmbedding(torch::Tensor token_type_embedding);
    virtual void                                              setInputEmbeddingScalar(float input_embedding_scalar);
    virtual void                                              setModelDataType(caffe2::TypeMeta data_type);
    virtual void                                              replay(int bs);
    virtual void                                              deviceSpecificSync();
    virtual std::unique_ptr<void, std::function<void(void*)>> createStreamLife(void* capture_stream);
    virtual void                                              setParamsPtr(int bs, const PyModelOutputs& outputs);
    virtual void*                                             getDeviceStream();

    // 主要接口方法
    virtual PyModelOutputs forward(PyModelInputs& inputs);
    virtual bool           canRun(PyModelInputs& inputs);
    virtual int            getCurrentRealGraphBs();
    virtual bool           tryGetRealGraphBatchSize(PyModelInputs& inputs);
    virtual void
    extractValidHiddenStates(PyModelOutputs& outputs, const PyModelInputs& inputs, int32_t total_valid_tokens);
    virtual py::object normalForward(PyModelInputs& inputs);
    virtual void       initKernelInternalMemory();
    virtual void       captureOneBatchSize(int bs);
    virtual void       prepareInputs(PyModelInputs& inputs);
    virtual void       capture();

protected:
    void preparePrefillInputs(PyModelInputs& inputs, PyModelInputs& py_model_inputs_);

public:
    py::object        py_forward_method_, py_fill_params_method_;
    bool              enable_graph_{false}, is_prefill_graph_mode_{false}, enable_graph_debug_mode_{false};
    int               concurrency_limit_{32}, hidden_size_, max_seq_len_{0}, seq_size_per_block_{0};
    int               kv_cache_block_offset_{0}, current_batch_size_{1}, current_real_graph_bs_{1}, seq_len_sum_{0};
    size_t            max_bs_{1};
    int               num_tokens_per_bs_{1}, max_num_token_{1};
    std::vector<int>  capture_range_;
    CaptureMemoryHold capture_mem_hold_;
    torch::Tensor     position_encoding_, token_type_embedding_;
    float             input_embedding_scalar_;
    caffe2::TypeMeta  model_data_type_;
    DeviceBase*       device_{nullptr};

public:
    py::object                             py_instance_;
    std::unordered_map<int, GraphInstance> graph_instances_;
};
}  // namespace rtp_llm
