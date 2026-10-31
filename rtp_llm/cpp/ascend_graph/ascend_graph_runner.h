// AscendGraphRunner: ACL Graph (c10_npu::NPUGraph) runner for decode phase.
//
// Ascend counterpart of CudaGraphRunner: decode-only, reuses the existing
// GraphBase/CaptureMemoryHold infrastructure.
//
// Key differences from CudaGraphRunner:
//   * Uses c10_npu::NPUGraph instead of at::cuda::CUDAGraph.
//   * Uses c10_npu::NPUStream instead of CUDAStream.
//   * Uses aclrtMemcpyAsync (via tensor copy_) instead of custom fused kernels.
//   * Prefill graph mode is not supported (ACL Graph is decode-only today).

#pragma once

#include <unordered_map>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h"
#include "rtp_llm/cpp/ascend_graph/ascend_graph_utils.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

class AscendGraphRunner: public GraphBase {
public:
    AscendGraphRunner(const GraphParams& graph_params, py::object py_instance);
    ~AscendGraphRunner() override;

    void           initCapture() override;
    PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state) override;
    bool           canRun(const PyModelInputs& inputs,
                         CudaGraphState&      state,
                         CudaGraphCheckMode   mode = CudaGraphCheckMode::FORWARD) override;

    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;

    // Newer GraphBase interface stubs: ACL graph manages its own capture
    // session lifecycle (replayAndSyncCheck handles event hygiene), so the
    // capture session is never left dirty, and attention inputs are prepared
    // inside the runner's own replay path.
    bool captureSessionMayBeDirty() const override { return false; }
    void prepareAttentionInputs(const PyModelInputs& inputs,
                                CudaGraphState&      state,
                                bool                 skip_forward_event_sync = false) override {}

private:
    // Capture / replay
    void captureDecode();
    void captureDecodeOneBatchSize(int bs);
    void captureOneGraphInstance(int key, const char* key_type);
    void replayAndSyncCheck(int key, const char* key_type);
    void replayGraph(int key);
    void replayDecode(int bs);

    // Input preparation
    void prepareInputs(const PyModelInputs& inputs, CudaGraphState& state);
    void prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int num_tokens);

    // Bucket / canRun helpers
    bool           tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state);
    std::vector<int> getDecodeBatchSizesToCapture();

    // Capture-time tensor allocation helpers (mirror CudaGraphRunner equivalents)
    void initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    void initKernelInternalMemory();
    ascend_graph::AscendGraphMemHold createMemHold(PyModelInputs& inputs, int tokens_count);

    // Python refs
    py::object py_forward_method_;
    py::object py_attn_pyobj_method_;

    // Config (mirrors CudaGraphRunner fields, drop prefill-related ones)
    bool     enable_graph_{false};
    bool     enable_graph_debug_mode_{false};
    bool     is_target_verify_{false};
    size_t   max_bs_{1};
    int      num_tokens_per_bs_{1};
    int      max_num_token_{1};
    int      max_seq_len_{0};
    int      seq_size_per_block_{0};
    int      kernel_seq_size_per_block_{0};
    int      hidden_size_{0};
    int      sp_steps_{0};

    c10::ScalarType model_data_type_;

    // Bucket: batch_size -> AscendGraphInstance
    std::vector<int>                                            capture_range_;
    std::vector<int>                                            decode_capture_batch_sizes_;
    std::unordered_map<int, ascend_graph::AscendGraphInstance>  graph_instances_;

    // Max-size shared capture mem hold (slices used per bucket)
    ascend_graph::AscendGraphMemHold capture_mem_hold_;

    // Capture stream + pool
    ascend_graph::GraphStream    capture_stream_;
    ascend_graph::GraphPoolHandle shared_graph_pool_{};

    // Forward completion event
    torch::Event forward_event_;

    // Bert embedding extras
    torch::Tensor position_encoding_;
    torch::Tensor token_type_embedding_;
    float         input_embedding_scalar_{0.0f};

    // TensorOptions
    at::TensorOptions options_npu_int32_;
    at::TensorOptions options_cpu_int32_;
    at::TensorOptions options_npu_float_;

    // Hybrid KV cache
    std::vector<int32_t> kv_cache_layer_to_group_;
    int32_t              kv_cache_group_num_{0};
    // Tag-based hybrid cache groups (mirrors engine-side AttentionInputsByTag)
    std::vector<std::string> group_tags_;
};

}  // namespace rtp_llm
