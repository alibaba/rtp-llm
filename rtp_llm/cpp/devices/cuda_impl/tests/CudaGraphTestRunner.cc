#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
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
                      int64_t          max_prefill_cuda_graph_len,
                      std::vector<int> prefill_capture_seq_lens,
                      int64_t          hidden_size) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.num_tokens_per_bs            = static_cast<int>(max_seq_len);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);
        params.max_prefill_cuda_graph_len   = static_cast<int>(max_prefill_cuda_graph_len);
        params.kv_cache_layer_to_group      = {};  // test: no hybrid kv cache
        params.kv_cache_group_num           = 0;

        runner_ = CudaGraphRunner::createForPrefill(std::move(py_instance), std::move(params));
    }

    void init_decode(py::object       py_instance,
                     int64_t          hidden_size,
                     int64_t          max_seq_len,
                     int64_t          tokens_per_block,
                     std::vector<int> decode_capture_batch_sizes) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.num_tokens_per_bs            = 1;
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::Half;
        params.concurrency_limit            = 128;
        params.decode_capture_batch_sizes   = std::move(decode_capture_batch_sizes);
        params.kv_cache_layer_to_group      = {};  // test: no hybrid kv cache
        params.kv_cache_group_num           = 0;

        runner_ = CudaGraphRunner::createForDecode(std::move(py_instance), std::move(params));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        return runner_->forward(inputs, state_);
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphBs(state_) : 0;
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
             py::arg("max_prefill_cuda_graph_len"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"))
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("decode_capture_batch_sizes"))
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("forward", &CudaGraphTestRunner::forward)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize);
}
