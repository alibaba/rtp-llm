#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;
namespace rtp_llm {

// Single wrapper for both prefill and decode tests; init_prefill / init_decode
// build GraphParams and call CudaGraphRunner factory methods.
// Plain pybind11 class (no torch::jit::CustomClassHolder) so the module loads without
// depending on torch's registered CustomClassHolder type.
class CudaGraphTestRunner {
public:
    void init_prefill(py::object       py_instance,
                      int64_t          max_context_batch_size,
                      int64_t          max_seq_len,
                      int64_t          tokens_per_block,
                      int64_t          kernel_tokens_per_block,
                      std::vector<int> prefill_capture_seq_lens,
                      int64_t          hidden_size) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs            = static_cast<int>(max_seq_len);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);
        params.kv_cache_layer_to_group      = {};  // test: no hybrid kv cache
        params.kv_cache_group_num           = 0;

        runner_ = CudaGraphRunner::createForPrefill(std::move(py_instance), std::move(params));
    }

    void init_generation_prefill(py::object       py_instance,
                                 int64_t          hidden_size,
                                 int64_t          max_context_batch_size,
                                 int64_t          max_seq_len,
                                 int64_t          tokens_per_block,
                                 int64_t          kernel_tokens_per_block,
                                 std::vector<int> prefill_capture_seq_lens,
                                 int64_t          mori_max_tokens = 0) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = true;
        params.graph_mode                   = GraphMode::GenerationPrefill;
        params.lazy_capture                 = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs            = 1;
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);
        params.mori_max_tokens              = static_cast<int>(mori_max_tokens);
        params.model_data_type              = c10::ScalarType::Half;

        runner_ = CudaGraphRunner::createForPrefill(std::move(py_instance), std::move(params));
    }

    void init_decode(py::object       py_instance,
                     int64_t          hidden_size,
                     int64_t          max_seq_len,
                     int64_t          tokens_per_block,
                     int64_t          kernel_tokens_per_block,
                     std::vector<int> decode_capture_batch_sizes,
                     bool             lazy_capture = false) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.lazy_capture                 = lazy_capture;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs            = 1;
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::Half;
        params.max_context_batch_size       = 128;
        params.decode_capture_batch_sizes   = std::move(decode_capture_batch_sizes);
        params.kv_cache_layer_to_group      = {};
        params.kv_cache_group_num           = 0;

        runner_ = CudaGraphRunner::createForDecode(std::move(py_instance), std::move(params));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    std::string plan(torch_ext::PyModelInputs& inputs) {
        if (runner_ == nullptr) {
            return "Eager";
        }
        switch (runner_->plan(inputs, state_)) {
            case GraphRunDecision::Replay:
                return "Replay";
            case GraphRunDecision::CaptureAfterEager:
                return "CaptureAfterEager";
            case GraphRunDecision::Eager:
            default:
                return "Eager";
        }
    }

    bool captureCurrentBucket() {
        return runner_ != nullptr && runner_->captureCurrentBucket(state_);
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        return runner_->forward(inputs, state_);
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphBs(state_) : 0;
    }

    void close() {
        reset_runner();
    }

    ~CudaGraphTestRunner() {
        reset_runner();
    }

private:
    void reset_runner() {
        if (runner_ != nullptr) {
            delete runner_;
            runner_ = nullptr;
        }
    }

    CudaGraphRunner* runner_ = nullptr;
    CudaGraphState   state_{};
};

}  // namespace rtp_llm

PYBIND11_MODULE(libtest_cuda_graph_runner, m) {
    using namespace rtp_llm;
    py::class_<CudaGraphTestRunner>(m, "CudaGraphRunner")
        .def(py::init<>())
        .def("init_prefill",
             &CudaGraphTestRunner::init_prefill,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"))
        .def("init_generation_prefill",
             &CudaGraphTestRunner::init_generation_prefill,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("mori_max_tokens") = 0)
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("decode_capture_batch_sizes"),
             py::arg("lazy_capture") = false)
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("plan", &CudaGraphTestRunner::plan)
        .def("captureCurrentBucket", &CudaGraphTestRunner::captureCurrentBucket)
        .def("forward", &CudaGraphTestRunner::forward)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize)
        .def("close", &CudaGraphTestRunner::close);
}
