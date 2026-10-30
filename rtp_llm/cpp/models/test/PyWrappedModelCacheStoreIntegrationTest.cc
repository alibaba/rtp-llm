#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/models/HiddenStateCapturePublisher.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/cpp/models/models_weight/Weights.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/RuntimeDevice.h"

namespace py = pybind11;

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace test {

// The fixture runs under the GIL. Invalid-input tests must throw rather than
// abort the interpreter, without changing the process default after the call.
struct ScopedAssertionExceptions {
    ScopedAssertionExceptions(): previous(std::exchange(StaticConfig::user_ft_core_dump_on_exception, false)) {}
    ~ScopedAssertionExceptions() {
        StaticConfig::user_ft_core_dump_on_exception = previous;
    }
    const bool previous;
};

// Keep the real publisher across forwards to exercise broken/recoverable state.
// BF16 payloads may be CPU tensors; the real TorchSpec FP8 quantizer uses CUDA.
class CapturePublisherFixture {
public:
    CapturePublisherFixture(bool owner, bool fp8):
        publisher_(2,
                   2,
                   fp8 ? HiddenStateCaptureDtype::FP8_E4M3 : HiddenStateCaptureDtype::BF16,
                   torch::kFloat16,
                   owner,
                   0,
                   nullptr,
                   HiddenStateCaptureLimits{},
                   {0, 1}) {}

    ~CapturePublisherFixture() {
        publisher_.flushAndClose();
    }

    py::dict
    forward(torch::Tensor packed, torch::Tensor ids, torch::Tensor lengths, torch::Tensor requests, bool wait) {
        // The fixture reserves before submission, like request admission. Only
        // this test helper optionally waits; PyWrappedModel never does.
        if (lengths.dim() == 1 && requests.dim() == 1 && lengths.numel() == requests.numel()
            && lengths.scalar_type() == torch::kInt32 && requests.scalar_type() == torch::kInt64) {
            auto host_lengths  = lengths.cpu().contiguous();
            auto host_requests = requests.cpu().contiguous();
            for (int64_t i = 0; i < lengths.numel(); ++i) {
                (void)publisher_.tryReserve(host_requests.data_ptr<int64_t>()[i], host_lengths.data_ptr<int32_t>()[i]);
            }
        }
        const bool accepted = publisher_.trySubmit(packed, ids, lengths, requests, ids.numel(), "test capture");
        if (wait) {
            (void)publisher_.flushPending();
        }
        const auto stats = publisher_.stats();
        py::dict   result;
        result["accepted"]          = accepted;
        result["failures"]          = stats.failure_count;
        result["broken_rejections"] = stats.broken_rejection_count;
        return result;
    }

    bool reserve(int64_t request_id, int64_t valid_tokens) {
        return publisher_.tryReserve(request_id, valid_tokens);
    }

    void cancel(int64_t request_id) {
        publisher_.cancelReserved(request_id);
    }

    bool flush() {
        return publisher_.flushPending();
    }

    bool enabled() {
        return publisher_.shouldPublish();
    }

    py::object takeError() {
        const auto error = publisher_.takeDeferredError();
        return error ? py::cast(*error) : py::none();
    }

    void closeWithoutGil() {
        py::gil_scoped_release release;
        publisher_.flushAndClose();
    }

private:
    HiddenStateCapturePublisher publisher_;
};

void initTestRuntime(int local_device_id) {
    if (local_device_id < 0) {
        throw std::invalid_argument("fixture local_device_id must be non-negative");
    }
    static std::once_flag runtime_once;
    std::call_once(runtime_once, [local_device_id] {
        if (!isRuntimeInitialized()) {
            initRuntime(local_device_id, false, false, MlaOpsType::AUTO);
        }
    });
    // Runtime device is process-global. Never silently reuse GPU0 in a GPU1
    // worker (or reinitialize a sequential single-GPU fixture for a TP rank).
    if (getDeviceId() != local_device_id) {
        throw std::runtime_error("fixture runtime is already initialized on a different local_device_id");
    }
}

// Transport tests obtain each peer's tokens from the production CP planner,
// rather than reconstructing its zigzag/padding arithmetic in Python.
py::dict splitCPInputs(
    torch::Tensor input_ids, const std::vector<int32_t>& lengths, int tp_size, int tp_rank, int local_device_id) {
    ScopedAssertionExceptions assertion_exceptions;
    initTestRuntime(local_device_id);
    ParallelismConfig parallelism;
    parallelism.tp_size                  = tp_size;
    parallelism.tp_rank                  = tp_rank;
    parallelism.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    ZigZagProcessor processor(parallelism);
    GptModelInputs  inputs{};
    inputs.combo_tokens     = input_ids;
    inputs.input_lengths    = torch::tensor(lengths, torch::kInt32).to(input_ids.device());
    inputs.sequence_lengths = torch::empty({0}, inputs.input_lengths.options());
    torch_ext::PyContextParallelParams cp_params;
    processor.handleInputs(inputs, cp_params);
    py::dict result;
    result["input_ids"]        = inputs.combo_tokens;
    result["input_lengths"]    = inputs.input_lengths;
    result["original_lengths"] = cp_params.prefill_actual_input_lengths_cpu;
    return result;
}

// Base fixture: one ordinary grouped block table, no main-only tag-cache,
// generation-prefill runner, custom output or DSpark APIs. Python only supplies
// layer outputs; preparation, CP, micro-batch merging and post-layers are real.
py::dict runModel(py::object                  py_model,
                  bool                        capture,
                  bool                        skip_lm_head,
                  bool                        micro_batch,
                  bool                        cp,
                  bool                        warmup,
                  bool                        fake,
                  int                         tp_rank,
                  const std::vector<int32_t>& lengths,
                  int                         tp_size,
                  bool                        configured,
                  bool                        final_norm,
                  bool                        fp8,
                  bool                        decode,
                  const std::vector<bool>&    forward_captures,
                  const std::vector<int64_t>& capture_layer_ids,
                  bool                        input_ids_on_cuda,
                  int64_t                     hc_mult,
                  int                         local_device_id,
                  bool                        need_all_hidden_states,
                  bool                        ffn_disaggregate,
                  const std::vector<int64_t>& capture_limits) {
    ScopedAssertionExceptions assertion_exceptions;
    initTestRuntime(local_device_id);

    Weights weights;
    weights.layers.resize(2);
    auto lm_head    = std::make_shared<DenseWeights>();
    lm_head->kernel = torch::ones({2, 2}, torch::TensorOptions(torch::kFloat16).device(getTorchCudaDevice()));
    weights.lm_head = lm_head;
    if (final_norm) {
        auto norm               = std::make_shared<LayerNormWeights>();
        norm->gamma             = torch::tensor({0.5f, 1.5f}, lm_head->kernel.options());
        weights.final_layernorm = norm;
    }
    GptModelDescription description{};
    description.data_type                    = DataType::TYPE_FP16;
    description.norm_type                    = NormType::rmsnorm;
    description.attention_conf.head_num      = 1;
    description.attention_conf.kv_head_num   = 1;
    description.attention_conf.size_per_head = 2;
    GptModelInitParams params{weights, description, std::nullopt};
    params.hidden_size                                                        = 2;
    params.hc_mult                                                            = hc_mult;
    params.max_seq_len                                                        = 64;
    params.tokens_per_block                                                   = 8;
    params.kernel_tokens_per_block                                            = 8;
    params.parallelism_config.tp_rank                                         = tp_rank;
    params.parallelism_config.local_rank                                      = local_device_id;
    params.parallelism_config.tp_size                                         = tp_size;
    params.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = ffn_disaggregate;
    params.device_resource_config.enable_layer_micro_batch                    = micro_batch ? 1 : 0;
    params.parallelism_config.prefill_cp_config.method = cp ? CPRotateMethod::ALL_GATHER : CPRotateMethod::DISABLED;
    params.hidden_state_capture_layer_ids              = configured ? capture_layer_ids : std::vector<int64_t>{};
    params.hidden_state_capture_dtype = fp8 ? HiddenStateCaptureDtype::FP8_E4M3 : HiddenStateCaptureDtype::BF16;
    if (!capture_limits.empty()) {
        if (capture_limits.size() != 5) {
            throw std::invalid_argument("capture_limits requires five values");
        }
        params.hidden_state_capture_max_inflight_samples = capture_limits[0];
        params.hidden_state_capture_max_inflight_bytes   = capture_limits[1];
        params.hidden_state_capture_max_inflight_tokens  = capture_limits[2];
        params.hidden_state_capture_max_pending_batches  = capture_limits[3];
        params.hidden_state_capture_poll_interval_ms     = capture_limits[4];
    }

    const int64_t  batch  = lengths.size();
    const int64_t  count  = decode ? batch : std::accumulate(lengths.begin(), lengths.end(), int64_t{0});
    auto           pinned = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    GptModelInputs inputs{};
    inputs.combo_tokens      = torch::arange(1, count + 1, pinned);
    inputs.input_lengths     = torch::tensor(lengths, torch::kInt32).pin_memory();
    inputs.sequence_lengths  = decode ? inputs.input_lengths.clone() : torch::empty({0}, pinned);
    inputs.prefix_lengths    = torch::zeros({decode ? 0 : batch}, pinned);
    inputs.lm_output_indexes = (decode ? torch::arange(batch, pinned) : inputs.input_lengths.cumsum(0) - 1)
                                   .to(torch::kInt32)
                                   .to(getTorchCudaDevice());
    inputs.kv_cache_block_id         = torch::zeros({1, batch, 8}, pinned).to(getTorchCudaDevice());
    inputs.kv_cache_kernel_block_id  = inputs.kv_cache_block_id.clone();
    inputs.request_id                = torch::arange(100, 100 + batch, torch::kInt64).pin_memory();
    inputs.request_pd_separation     = torch::ones({batch}, torch::kBool).pin_memory();
    inputs.seq_size_per_block        = 8;
    inputs.kernel_seq_size_per_block = 8;
    inputs.capture_hidden_states     = capture;
    inputs.skip_lm_head              = skip_lm_head;
    inputs.warmup                    = warmup;
    inputs.is_fake_stream            = fake;
    inputs.pd_separation             = true;
    inputs.need_all_hidden_states    = need_all_hidden_states;
    if (input_ids_on_cuda) {
        inputs.combo_tokens  = inputs.combo_tokens.to(getTorchCudaDevice());
        inputs.input_lengths = inputs.input_lengths.to(getTorchCudaDevice());
    }

    py::dict result;
    py::list forwards;
    {
        PyWrappedModel model(params, py_model);
        const auto     forward_count = std::max(size_t{1}, forward_captures.size());
        for (size_t i = 0; i < forward_count; ++i) {
            auto forward_inputs = inputs;
            // CP mutates host lengths in place; each forward needs the original batch.
            forward_inputs.input_lengths         = inputs.input_lengths.clone();
            forward_inputs.capture_hidden_states = forward_captures.empty() ? capture : forward_captures[i];
            forward_inputs.request_id            = inputs.request_id + static_cast<int64_t>(i) * batch;
            if (forward_inputs.capture_hidden_states && !warmup && !fake && tp_rank == 0) {
                bool admitted = true;
                for (int64_t request = 0; request < batch; ++request) {
                    admitted = model.tryReserveHiddenStateCapture(100 + static_cast<int64_t>(i) * batch + request,
                                                                  lengths[request])
                               && admitted;
                }
                if (!admitted) {
                    for (int64_t request = 0; request < batch; ++request) {
                        model.cancelHiddenStateCapture(100 + static_cast<int64_t>(i) * batch + request);
                    }
                }
                forward_inputs.capture_hidden_states = admitted;
            }
            const auto output           = model.forward(forward_inputs);
            result                      = py::dict();
            result["capture_requested"] = forward_inputs.capture_hidden_states;
            result["hidden"]            = output.all_hidden_states;
            result["logits"]            = output.logits.defined() ? py::cast(output.logits) : py::none();
            const auto error            = model.takeDeferredHiddenStateCaptureError();
            result["error"]             = error ? py::cast(*error) : py::none();
            const auto second_error     = model.takeDeferredHiddenStateCaptureError();
            result["second_error"]      = second_error ? py::cast(*second_error) : py::none();
            const auto stats            = model.hiddenStateCaptureStats();
            result["failures"]          = stats.failure_count;
            result["broken_rejections"] = stats.broken_rejection_count;
            forwards.append(result);
            model.releaseBuffers();
        }
    }
    // Copy before adding the history to avoid a dict/list reference cycle.
    py::dict summary(result.attr("copy")());
    summary["forwards"] = forwards;
    return summary;
}

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_pywrapped_model_cache_store_integration_test, m) {
    torch_ext::registerPyOpDefs(m);
    rtp_llm::registerExecCtxOps(m);
    py::class_<rtp_llm::test::CapturePublisherFixture>(m, "CapturePublisher")
        .def(py::init<bool, bool>(), py::arg("owner") = true, py::arg("fp8") = false)
        .def("forward",
             &rtp_llm::test::CapturePublisherFixture::forward,
             py::arg("packed"),
             py::arg("ids"),
             py::arg("lengths"),
             py::arg("requests"),
             py::arg("wait") = true)
        .def("reserve", &rtp_llm::test::CapturePublisherFixture::reserve)
        .def("cancel", &rtp_llm::test::CapturePublisherFixture::cancel)
        .def("flush", &rtp_llm::test::CapturePublisherFixture::flush)
        .def("enabled", &rtp_llm::test::CapturePublisherFixture::enabled)
        .def("take_error", &rtp_llm::test::CapturePublisherFixture::takeError)
        .def("close_without_gil", &rtp_llm::test::CapturePublisherFixture::closeWithoutGil);
    m.def("split_cp_inputs",
          &rtp_llm::test::splitCPInputs,
          py::arg("input_ids"),
          py::arg("lengths"),
          py::arg("tp_size"),
          py::arg("tp_rank"),
          py::arg("local_device_id") = 0);
    m.def("run_model",
          &rtp_llm::test::runModel,
          py::arg("model"),
          py::arg("capture")                = true,
          py::arg("skip_lm_head")           = true,
          py::arg("micro_batch")            = false,
          py::arg("cp")                     = false,
          py::arg("warmup")                 = false,
          py::arg("fake")                   = false,
          py::arg("tp_rank")                = 0,
          py::arg("lengths")                = std::vector<int32_t>{2, 3},
          py::arg("tp_size")                = 1,
          py::arg("configured")             = true,
          py::arg("final_norm")             = false,
          py::arg("fp8")                    = false,
          py::arg("decode")                 = false,
          py::arg("forward_captures")       = std::vector<bool>{},
          py::arg("capture_layer_ids")      = std::vector<int64_t>{0, 1},
          py::arg("input_ids_on_cuda")      = false,
          py::arg("hc_mult")                = 1,
          py::arg("local_device_id")        = 0,
          py::arg("need_all_hidden_states") = true,
          py::arg("ffn_disaggregate")       = false,
          py::arg("capture_limits")         = std::vector<int64_t>{});
}
