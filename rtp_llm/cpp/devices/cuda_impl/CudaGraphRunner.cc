#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"

#include <cstring>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphUtils.h"

namespace rtp_llm {

CudaGraphRunner::CudaGraphRunner(const DeviceInitParams& params,
                                 py::object              py_instance,
                                 c10::ScalarType         model_data_type,
                                 int                     num_tokens_per_bs,
                                 bool                    is_prefill_cuda_graph_mode):
    GraphBase(std::move(py_instance)) {
    GraphBackendCallbacks callbacks;
    callbacks.event_device_type = c10::DeviceType::CUDA;
    callbacks.debug_file_prefix = "cuda_graph_tokens";

    callbacks.memcpy_async = [](const torch::Tensor& src, torch::Tensor& dst, size_t size) {
        if (!src.defined() || src.numel() <= 0) {
            return;
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        if (src.is_cuda() && dst.is_cuda()) {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToDevice, stream));
        } else if (!src.is_cuda() && !dst.is_cuda()) {
            std::memcpy(dst.data_ptr(), src.data_ptr(), size);
        } else if (src.is_cuda() && !dst.is_cuda()) {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyDeviceToHost, stream));
        } else {
            check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cudaMemcpyHostToDevice, stream));
        }
    };
    callbacks.device_synchronize         = []() { check_cuda_value(cudaDeviceSynchronize()); };
    callbacks.record_forward_event       = [](torch::Event& event) { event.record(at::cuda::getCurrentCUDAStream()); };
    callbacks.synchronize_forward_stream = []() {};

    auto capture_stream           = at::cuda::getStreamFromPool(true);
    callbacks.with_capture_stream = [capture_stream](const std::function<void()>& fn) {
        CudaGraphStreamLife stream_life(capture_stream);
        fn();
    };

    callbacks.should_skip_decode_capture = [](py::object py_instance, bool is_prefill_mode) {
        (void)py_instance;
        (void)is_prefill_mode;
        return false;
    };
    callbacks.before_capture_stream = [](py::object py_instance, int key, const char* key_type) {
        (void)py_instance;
        (void)key;
        (void)key_type;
    };
    callbacks.enter_capture = [](py::object py_instance) {
        (void)py_instance;
        CaptureCheck::in_cuda_graph_capture = true;
    };
    callbacks.exit_capture = [](py::object py_instance) {
        (void)py_instance;
        CaptureCheck::in_cuda_graph_capture = false;
    };
    callbacks.kv_block_cols = [](int max_seq_len, int seq_size_per_block) {
        return (max_seq_len + seq_size_per_block - 1) / seq_size_per_block + 1;
    };
    callbacks.sequence_lengths_plus_one_tensor = [](int max_bs, const at::TensorOptions& device_int_opts) {
        return torch::zeros({max_bs}, device_int_opts);
    };

    runner_ = std::make_unique<GraphBaseRunner>(
        params, py_instance_, model_data_type, num_tokens_per_bs, is_prefill_cuda_graph_mode, std::move(callbacks));
}

CudaGraphRunner::~CudaGraphRunner() {
    RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
}

void CudaGraphRunner::captureDecode() {
    runner_->captureDecode();
}

void CudaGraphRunner::capturePrefill() {
    runner_->capturePrefill();
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    runner_->captureDecodeOneBatchSize(bs);
}

void CudaGraphRunner::capturePrefillOneSeqLen(int seq_len) {
    runner_->capturePrefillOneSeqLen(seq_len);
}

void CudaGraphRunner::prepareInputs(PyModelInputs& inputs) {
    runner_->prepareInputs(inputs);
}

bool CudaGraphRunner::canRun(PyModelInputs& inputs) {
    return runner_->canRun(inputs);
}

void CudaGraphRunner::replayGraph(int key) {
    runner_->replayGraph(key);
}

void CudaGraphRunner::replayDecode(int bs) {
    runner_->replayDecode(bs);
}

void CudaGraphRunner::replayPrefill(int seq_len) {
    runner_->replayPrefill(seq_len);
}

void CudaGraphRunner::setMaxPrefillCudaGraphLen(int max_prefill_cuda_graph_len) {
    runner_->setMaxPrefillGraphLen(max_prefill_cuda_graph_len);
}

py::object CudaGraphRunner::normalForward(PyModelInputs& inputs) {
    return runner_->normalForward(inputs);
}

int CudaGraphRunner::getCurrentRealGraphBs() {
    return runner_->getCurrentRealGraphBs();
}

PyModelOutputs CudaGraphRunner::forward(PyModelInputs& inputs) {
    return runner_->forward(inputs);
}

void CudaGraphRunner::initCapture() {
    runner_->initCapture();
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    runner_->setPositionEncoding(position_encoding);
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    runner_->setTokenTypeEmbedding(token_type_embedding);
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    runner_->setInputEmbeddingScalar(input_embedding_scalar);
}

}  // namespace rtp_llm
