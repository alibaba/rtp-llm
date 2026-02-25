#include "rtp_llm/cpp/devices/rocm_impl/HipGraphRunner.h"

#include <cstring>
#include "rtp_llm/cpp/devices/rocm_impl/HipGraphUtils.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

namespace rtp_llm {

HipGraphRunner::HipGraphRunner(const DeviceInitParams& params,
                               py::object              py_instance,
                               c10::ScalarType         model_data_type,
                               int                     num_tokens_per_bs,
                               bool                    is_prefill_hip_graph_mode):
    GraphBase(std::move(py_instance)), nccl_capture_ctx_(std::make_shared<HipGraphNcclCaptureContext>()) {
    GraphBackendCallbacks callbacks;
    callbacks.event_device_type = c10::DeviceType::HIP;
    callbacks.debug_file_prefix = "hip_graph_tokens";

    callbacks.memcpy_async = [](const torch::Tensor& src, torch::Tensor& dst, size_t size) {
        if (!src.defined() || src.numel() <= 0) {
            return;
        }
        hipStream_t stream = at::hip::getCurrentHIPStream().stream();
        if (src.is_cuda() && dst.is_cuda()) {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToDevice, stream));
        } else if (!src.is_cuda() && !dst.is_cuda()) {
            std::memcpy(dst.data_ptr(), src.data_ptr(), size);
        } else if (src.is_cuda() && !dst.is_cuda()) {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyDeviceToHost, stream));
        } else {
            ROCM_CHECK(hipMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, hipMemcpyHostToDevice, stream));
        }
    };
    callbacks.device_synchronize         = []() { ROCM_CHECK(hipDeviceSynchronize()); };
    callbacks.record_forward_event       = [](torch::Event& event) { event.record(at::hip::getCurrentHIPStream()); };
    callbacks.synchronize_forward_stream = []() {
        auto stream = at::hip::getCurrentHIPStream();
        ROCM_CHECK(hipStreamSynchronize(stream.stream()));
    };

    auto capture_stream           = at::hip::getStreamFromPool(true);
    callbacks.with_capture_stream = [capture_stream](const std::function<void()>& fn) {
        HipGraphStreamLife stream_life(capture_stream);
        fn();
    };

    callbacks.should_skip_decode_capture = [](py::object py_instance, bool is_prefill_mode) {
        if (is_prefill_mode) {
            return false;
        }
        py::gil_scoped_acquire gil;
        bool                   has_kv_cache = true;
        if (py::hasattr(py_instance, "kv_cache")) {
            has_kv_cache = !py_instance.attr("kv_cache").is_none();
        }
        if (!has_kv_cache) {
            RTP_LLM_LOG_WARNING("HIP graph capture is enabled but kv_cache is not available. "
                                "Skipping decode graph capture for this instance.");
        }
        return !has_kv_cache;
    };
    callbacks.before_capture_stream = [](py::object py_instance, int key, const char* key_type) {
        (void)py_instance;
        py::gil_scoped_acquire gil;
        try {
            py::module_ torch_dist = py::module_::import("torch.distributed");
            if (torch_dist.attr("is_initialized")().cast<bool>()) {
                RTP_LLM_LOG_INFO("Executing torch.distributed.barrier() before graph capture for %s %d", key_type, key);
                torch_dist.attr("barrier")();
                RTP_LLM_LOG_INFO("torch.distributed.barrier() completed for %s %d", key_type, key);
            }
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("Failed to execute torch.distributed.barrier(): %s", e.what());
        }
    };

    callbacks.enter_capture = [nccl_capture_ctx = nccl_capture_ctx_](py::object py_instance) {
        (void)py_instance;
        rocm::CaptureCheck::in_hip_graph_capture = true;
        if (nccl_capture_ctx->comm_handle != 0) {
            try {
                py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
                collective_torch.attr("enter_graph_capture_mode")(
                    nccl_capture_ctx->comm_handle, nccl_capture_ctx->world_size, nccl_capture_ctx->rank);
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
            }
        }
    };
    callbacks.exit_capture = [nccl_capture_ctx = nccl_capture_ctx_](py::object py_instance) {
        (void)py_instance;
        rocm::CaptureCheck::in_hip_graph_capture = false;
        if (nccl_capture_ctx->comm_handle != 0) {
            try {
                py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
                collective_torch.attr("exit_graph_capture_mode")();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("Failed to exit graph capture mode: %s", e.what());
            }
        }
    };
    callbacks.kv_block_cols = [](int max_seq_len, int seq_size_per_block) {
        return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block;
    };
    callbacks.sequence_lengths_plus_one_tensor = [](int max_bs, const at::TensorOptions& device_int_opts) {
        return torch::full({max_bs}, 2, device_int_opts);
    };

    runner_ = std::make_unique<GraphBaseRunner>(
        params, py_instance_, model_data_type, num_tokens_per_bs, is_prefill_hip_graph_mode, std::move(callbacks));
}

HipGraphRunner::~HipGraphRunner() {
    RTP_LLM_LOG_INFO("Release HipGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release HipGraphRunner Successfully");
}

void HipGraphRunner::captureDecode() {
    runner_->captureDecode();
}

void HipGraphRunner::capturePrefill() {
    runner_->capturePrefill();
}

void HipGraphRunner::captureDecodeOneBatchSize(int bs) {
    runner_->captureDecodeOneBatchSize(bs);
}

void HipGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    runner_->capturePrefillOneSeqLen(seq_len);
}

void HipGraphRunner::prepareInputs(PyModelInputs& inputs) {
    runner_->prepareInputs(inputs);
}

bool HipGraphRunner::canRun(PyModelInputs& inputs) {
    return runner_->canRun(inputs);
}

void HipGraphRunner::replayGraph(int key) {
    runner_->replayGraph(key);
}

void HipGraphRunner::replayDecode(int bs) {
    runner_->replayDecode(bs);
}

void HipGraphRunner::replayPrefill(int seq_len) {
    runner_->replayPrefill(seq_len);
}

void HipGraphRunner::setMaxPrefillHipGraphLen(int max_prefill_hip_graph_len) {
    runner_->setMaxPrefillGraphLen(max_prefill_hip_graph_len);
}

py::object HipGraphRunner::normalForward(PyModelInputs& inputs) {
    return runner_->normalForward(inputs);
}

int HipGraphRunner::getCurrentRealGraphBs() {
    return runner_->getCurrentRealGraphBs();
}

PyModelOutputs HipGraphRunner::forward(PyModelInputs& inputs) {
    return runner_->forward(inputs);
}

void HipGraphRunner::initCapture() {
    runner_->initCapture();
}

void HipGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    runner_->setPositionEncoding(position_encoding);
}

void HipGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    runner_->setTokenTypeEmbedding(token_type_embedding);
}

void HipGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    runner_->setInputEmbeddingScalar(input_embedding_scalar);
}

}  // namespace rtp_llm
