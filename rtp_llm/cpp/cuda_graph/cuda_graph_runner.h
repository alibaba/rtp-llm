#pragma once

#include <array>
#include <atomic>
#include <exception>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"

namespace py = pybind11;

namespace rtp_llm {

class DirtyCudaGraphCaptureError: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PrefillCudaGraphUnsupportedBackendError: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const GraphParams& graph_params,
                    py::object         py_instance,
                    const char*        forward_method_name = "forward"):
        GraphBase(std::move(py_instance)),
        enable_cuda_graph_(graph_params.enable_cuda_graph),
        is_prefill_cuda_graph_mode_(graph_params.is_prefill_cuda_graph_mode),
        is_target_verify_(graph_params.is_target_verify),
        role_(graph_params.role),
        capture_stream_(cuda_graph::graphGetStreamFromPool(true)),
        enable_cuda_graph_debug_mode_(graph_params.enable_cuda_graph_debug_mode),
        num_tokens_per_bs_(graph_params.num_tokens_per_bs),
        max_seq_len_(graph_params.max_seq_len),
        seq_size_per_block_(graph_params.tokens_per_block),
        kernel_seq_size_per_block_(graph_params.kernel_tokens_per_block),
        hidden_size_(graph_params.hidden_size),
        input_hidden_size_(graph_params.input_hidden_size),
        hc_mult_(static_cast<int>(graph_params.hc_mult)),
        sp_steps_(graph_params.sp_steps),
        prefill_capture_seq_lens_(graph_params.prefill_capture_seq_lens),
        decode_capture_batch_sizes_(graph_params.decode_capture_batch_sizes),
        position_encoding_(graph_params.position_encoding),
        token_type_embedding_(graph_params.token_type_embedding),
        input_embedding_scalar_(graph_params.input_embedding_scalar),
        model_data_type_(graph_params.model_data_type),
        kv_cache_group_tags_(graph_params.kv_cache_group_tags),
        position_id_len_factor_(graph_params.position_id_len_factor),
        prefill_cuda_graph_max_requests_(graph_params.prefill_cuda_graph_max_requests),
        prefill_cuda_graph_pad_token_id_(graph_params.prefill_cuda_graph_pad_token_id),
        prefill_scratch_kernel_block_ids_(graph_params.prefill_scratch_kernel_block_ids) {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("CudaGraphRunner constructor: Python instance is null or none.");
        }
        if (kernel_seq_size_per_block_ <= 0) {
            throw std::runtime_error("CudaGraphRunner constructor: kernel_tokens_per_block must be > 0.");
        }
        max_bs_ = graph_params.max_context_batch_size;
        if (role_ == CudaGraphRole::AUTO) {
            role_ = is_target_verify_ ? CudaGraphRole::TARGET_VERIFY :
                    is_prefill_cuda_graph_mode_ ?
                                        (num_tokens_per_bs_ == max_seq_len_ ? CudaGraphRole::EMBEDDING_PREFILL :
                                                                              CudaGraphRole::MTP_DRAFT_PREFILL) :
                                        CudaGraphRole::DECODE;
        }
        is_prefill_cuda_graph_mode_ = role_ == CudaGraphRole::EMBEDDING_PREFILL
                                      || role_ == CudaGraphRole::MTP_DRAFT_PREFILL
                                      || role_ == CudaGraphRole::GENERATIVE_PREFILL;
        is_target_verify_ = role_ == CudaGraphRole::TARGET_VERIFY;
        for (auto& counter : prefill_cuda_graph_fallback_log_counts_) {
            counter.store(0, std::memory_order_relaxed);
        }
        py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
        py_forward_method_    = py_instance_.attr(forward_method_name);
        options_cuda_int32_   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
        options_cpu_int32_    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        options_cuda_float_ = torch::TensorOptions().dtype(model_data_type_).device(torch::kCUDA).requires_grad(false);
        RTP_LLM_LOG_INFO("Initialize CudaGraphRunner with parameters below: \n \
            enable_cuda_graph_: %d, max_bs_: %d, enable_cuda_graph_debug_mode_: %d, max_seq_len_: %d, kernel_seq_size_per_block_: %d, \
            hidden_size_: %d, input_hidden_size_: %zu, num_tokens_per_bs_: %d, role_: %d, is_prefill_cuda_graph_mode_: %d, is_target_verify_: %d",
                         enable_cuda_graph_,
                         max_bs_,
                         enable_cuda_graph_debug_mode_,
                         max_seq_len_,
                         kernel_seq_size_per_block_,
                         hidden_size_,
                         input_hidden_size_,
                         num_tokens_per_bs_,
                         static_cast<int>(role_),
                         is_prefill_cuda_graph_mode_,
                         is_target_verify_);
    }

    ~CudaGraphRunner() {
        RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
        if (capture_session_may_be_dirty_.load(std::memory_order_acquire)) {
            RTP_LLM_LOG_ERROR("Skip CUDA graph runner drain because capture/session cleanup did not complete; "
                              "the caller must fail model initialization instead of continuing eager execution");
        } else {
            try {
                cuda_graph::GraphStreamGuard stream_guard(capture_stream_);
                cuda_graph::graphDeviceSynchronize();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("CUDA graph runner drain failed during teardown: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_WARNING("CUDA graph runner drain failed during teardown with unknown exception");
            }
        }
        py::gil_scoped_acquire gil;
        py_instance_.release();
        RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
    }
    void           captureDecode();
    void           capturePrefill();
    void           captureDecodeOneBatchSize(int bs);
    void           capturePrefillOneSeqLen(int seq_len);
    void           prepareInputs(const PyModelInputs& inputs, CudaGraphState& state);
    void           prepareInputData(const PyModelInputs& inputs, CudaGraphState& state);
    void           prepareAttentionInputs(const PyModelInputs& inputs, CudaGraphState& state) override;
    void           updateKVCacheKernelBlockId(const PyModelInputs& inputs, CudaGraphState& state) override;
    bool           canRun(const PyModelInputs& inputs,
                          CudaGraphState&      state,
                          CudaGraphCheckMode   mode = CudaGraphCheckMode::FORWARD) override;
    void           replayGraph(int key);
    void           replayDecode(int bs);
    void           replayPrefill(int seq_len);
    int            getCurrentRealGraphSize(const CudaGraphState& state) const;
    PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state) override;
    void           initCapture() override;

    bool captureSessionMayBeDirty() const override {
        return capture_session_may_be_dirty_.load(std::memory_order_acquire);
    }

    // Factory methods for test: take GraphParams so callers can reuse the same struct
    static CudaGraphRunner* createForPrefill(py::object py_instance, GraphParams params);
    static CudaGraphRunner* createForDecode(py::object py_instance, GraphParams params);

private:
    // Common capture logic for both prefill and decode
    void captureOneGraphInstance(int key, const char* key_type);
    // Common replay and sync check logic
    void replayAndSyncCheck(int key, const char* key_type);

    bool isEmbeddingStylePrefillCudaGraph() const {
        return role_ == CudaGraphRole::EMBEDDING_PREFILL;
    }
    bool isMtpDraftPrefillCudaGraph() const {
        return role_ == CudaGraphRole::MTP_DRAFT_PREFILL;
    }
    bool isGenerativePrefillCudaGraph() const {
        return role_ == CudaGraphRole::GENERATIVE_PREFILL;
    }
    bool usesFixedCapacityMtpDraftPrefillCudaGraph() const {
        // DSpARK propose/commit now run as construction-time-role decode graphs
        // (is_prefill_cuda_graph_mode_ == false), so only the HC-shaped MTP draft
        // prefill keeps the fixed-capacity Python path: slicing its output buffer
        // would mismatch the forward_decode [B * q_len, dim] result in
        // captureOneGraphInstance.
        return isMtpDraftPrefillCudaGraph() && hc_mult_ > 1;
    }
    // Common input preparation logic for capture
    void prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens);
    void prepareInputEmbeddings(const PyModelInputs& inputs, PyModelInputs& captured_inputs);
    // Common memory hold creation logic
    CaptureMemoryHold createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count);
    void              initKernelInternalMemory();
    py::object        prepareFmhaImpl(const PyModelInputs& inputs, bool is_cuda_graph);
    void              initPrefillScratchTensors();
    void              logCudaGraphPoolMemory(const char* phase);
    void              setPositionEncoding(torch::Tensor position_encoding) override;
    void              setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void              setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    std::vector<int> getDecodeBatchSizesToCapture();
    std::vector<int> getPrefillSequenceLengthsToCapture();
    /// Select graph key for decode; false if no captured graph can serve current_batch_size (e.g. lower_bound hit end).
    bool tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state, bool observe_fallback);
    /// Select graph key for prefill; false if capture_range_ empty or seq_len above max captured (lower_bound hit end).
    bool tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, CudaGraphState& state, bool observe_fallback);
    bool validateComboPositionIds(const PyModelInputs&  inputs,
                                  const CudaGraphState& state,
                                  const torch::Tensor&  captured_position_ids,
                                  size_t&               copy_numel) const;
    bool
    canReplaySelectedGraph(const PyModelInputs& inputs, const CudaGraphState& state, CudaGraphCheckMode mode) const;
    void                    initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs);
    void                    initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token);
    void                    initCaptureAttentionInputsPost();
    py::object              py_forward_method_;
    py::object              py_attn_pyobj_method_;
    bool                    enable_cuda_graph_{false};
    bool                    is_prefill_cuda_graph_mode_{false};
    bool                    is_target_verify_{false};
    CudaGraphRole           role_{CudaGraphRole::AUTO};
    cuda_graph::GraphStream capture_stream_;
    bool                    enable_cuda_graph_debug_mode_{false};
    size_t                  max_bs_{1};
    int                     num_tokens_per_bs_{1};
    int                     max_num_token_{1};
    int                     max_seq_len_{0};
    int                     seq_size_per_block_{0};
    int                     kernel_seq_size_per_block_{0};
    int                     hidden_size_{0};
    size_t                  input_hidden_size_{0};
    int                     hc_mult_{1};
    int                     sp_steps_{0};
    std::vector<int>        capture_range_;
    std::vector<int>        prefill_capture_seq_lens_;    // Pre-configured sequence lengths from Python
    std::vector<int>        decode_capture_batch_sizes_;  // Pre-configured batch sizes from Python
    // capture seqLen -> GraphInstance (prefill)
    // batch_size -> GraphInstance (decode)
    std::unordered_map<int, GraphInstance> graph_instances_;
    CaptureMemoryHold                      capture_mem_hold_;
    torch::Tensor                          position_encoding_;
    torch::Tensor                          token_type_embedding_;
    float                                  input_embedding_scalar_;
    c10::ScalarType                        model_data_type_;
    at::TensorOptions                      options_cuda_int32_;
    at::TensorOptions                      options_cpu_int32_;
    at::TensorOptions                      options_cuda_float_;
    cuda_graph::GraphPoolHandle            shared_graph_pool_{};

    std::vector<std::string>      kv_cache_group_tags_;
    int                           position_id_len_factor_ = 0;  // 0 = model has no combo_position_ids
    int                           prefill_cuda_graph_max_requests_{0};
    int                           prefill_cuda_graph_pad_token_id_{0};
    std::vector<std::vector<int>> prefill_scratch_kernel_block_ids_;
    std::vector<torch::Tensor>    prefill_scratch_kernel_block_ids_host_;
    std::vector<torch::Tensor>    prefill_scratch_kernel_block_ids_device_;
    torch::Tensor                 prefill_cuda_graph_padding_offset_host_;
    mutable std::atomic<uint64_t> combo_position_fallback_count_{0};
    static constexpr size_t       kPrefillCudaGraphStatusCount =
        static_cast<size_t>(PrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH) + 1;
    mutable std::array<std::atomic<uint64_t>, kPrefillCudaGraphStatusCount> prefill_cuda_graph_fallback_log_counts_;
    mutable std::atomic<uint64_t>                                           prefill_cuda_graph_replay_log_count_{0};

    // event to record forward done
    torch::Event forward_event_ = cuda_graph::makeGraphEvent();

    std::atomic<bool> prepared_attention_inputs_    = false;
    std::atomic<bool> capture_session_may_be_dirty_ = false;
    // True only while request overrides have been staged but not yet consumed
    // by a successful graph replay. This makes abandoned async preparation
    // recoverable without clearing metadata for the common request path.
    std::atomic<bool> input_embedding_metadata_dirty_ = false;
};

}  // namespace rtp_llm
