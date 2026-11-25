#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"

#include <limits>

#if USING_CUDA
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <torch/version.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <map>
#include <mutex>
#include <tuple>
#include <array>
#include <dlfcn.h>
#include <iterator>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/common/kernels/vocab_prune/mapping.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/temperature_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling_host.h"
#include <cstddef>
#include <random>
#include <memory>
#endif

using namespace std;

namespace rtp_llm {

namespace {

#if USING_CUDA || USING_ROCM

void checkRejectionSamplingTensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t dim) {
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "%s must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.is_cuda(), "%s must be on CUDA/HIP device", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == dtype, "%s dtype mismatch", name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == dim,
                            "%s must be %ld-D, got %ld-D",
                            name,
                            static_cast<long>(dim),
                            static_cast<long>(tensor.dim()));
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
}

void checkSameDevice(const torch::Tensor& tensor, const char* name, const c10::Device& device) {
    RTP_LLM_CHECK_WITH_INFO(tensor.device() == device, "%s must be on the same device as draft_token_ids_d", name);
}

RejectionSamplingLaunchConfig validateRejectionSamplingParams(const RejectionSamplingParams& params) {
    /**
     * Point-mass proposals may leave draft_probs_d undefined because the kernel
     * reconstructs q from draft_token_ids_d. Requiring a dense q tensor here
     * would reject valid proposals, including DSpARK handoff padding.
     * Derive launch dimensions from draft_token_ids_d and the target tensors;
     * validate draft_probs_d only when the caller supplies a dense proposal q.
     */
    const bool point_mass = params.draft_probs_point_mass;

    checkRejectionSamplingTensor(params.draft_token_ids_d, "draft_token_ids_d", torch::kInt32, 2);
    const auto device = params.draft_token_ids_d.device();

    if (!point_mass) {
        checkRejectionSamplingTensor(params.draft_probs_d, "draft_probs_d", torch::kFloat32, 3);
        checkSameDevice(params.draft_probs_d, "draft_probs_d", device);
    }
    checkRejectionSamplingTensor(params.uniform_samples_d, "uniform_samples_d", torch::kFloat32, 2);
    checkRejectionSamplingTensor(params.target_probs_d, "target_probs_d", torch::kFloat32, 3);
    checkRejectionSamplingTensor(params.target_token_ids_d, "target_token_ids_d", torch::kInt32, 2);
    checkRejectionSamplingTensor(params.output_token_ids_d, "output_token_ids_d", torch::kInt32, 2);
    checkRejectionSamplingTensor(params.output_accepted_token_num_d, "output_accepted_token_num_d", torch::kInt32, 1);
    checkRejectionSamplingTensor(params.do_sample_d, "do_sample_d", torch::kBool, 1);

    checkSameDevice(params.uniform_samples_d, "uniform_samples_d", device);
    checkSameDevice(params.target_probs_d, "target_probs_d", device);
    checkSameDevice(params.target_token_ids_d, "target_token_ids_d", device);
    checkSameDevice(params.output_token_ids_d, "output_token_ids_d", device);
    checkSameDevice(params.output_accepted_token_num_d, "output_accepted_token_num_d", device);
    checkSameDevice(params.do_sample_d, "do_sample_d", device);

    const int64_t batch_size             = params.draft_token_ids_d.size(0);
    const int64_t num_speculative_tokens = params.draft_token_ids_d.size(1);
    const int64_t target_vocab_size      = params.target_probs_d.size(2);
    const int64_t target_token_stride    = params.target_token_ids_d.size(1);

    RTP_LLM_CHECK_WITH_INFO(target_vocab_size > 0, "target_vocab_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(target_token_stride > 0, "target_token_ids_d stride dimension must be positive");
    RTP_LLM_CHECK_WITH_INFO(batch_size <= std::numeric_limits<int>::max(), "batch_size too large");
    RTP_LLM_CHECK_WITH_INFO(num_speculative_tokens <= std::numeric_limits<int>::max(),
                            "num_speculative_tokens too large");
    RTP_LLM_CHECK_WITH_INFO(target_vocab_size <= std::numeric_limits<int>::max(), "target_vocab_size too large");
    RTP_LLM_CHECK_WITH_INFO(target_token_stride <= std::numeric_limits<int>::max(), "target_token_stride too large");

    const int64_t target_token_rows = batch_size * (num_speculative_tokens + 1);

    if (!point_mass) {
        RTP_LLM_CHECK_WITH_INFO(params.draft_probs_d.size(0) == batch_size, "draft_probs_d shape[0] mismatch");
        RTP_LLM_CHECK_WITH_INFO(params.draft_probs_d.size(1) == num_speculative_tokens,
                                "draft_probs_d shape[1] mismatch");
        RTP_LLM_CHECK_WITH_INFO(params.draft_probs_d.size(2) == target_vocab_size, "draft_probs_d shape[2] mismatch");
    }
    RTP_LLM_CHECK_WITH_INFO(params.uniform_samples_d.size(0) == batch_size, "uniform_samples_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.uniform_samples_d.size(1) == num_speculative_tokens + 1,
                            "uniform_samples_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(0) == batch_size, "target_probs_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(1) == num_speculative_tokens + 1,
                            "target_probs_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_probs_d.size(2) == target_vocab_size, "target_probs_d shape[2] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.target_token_ids_d.size(0) == target_token_rows,
                            "target_token_ids_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_token_ids_d.size(0) == batch_size, "output_token_ids_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_token_ids_d.size(1) == num_speculative_tokens + 1,
                            "output_token_ids_d shape[1] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.output_accepted_token_num_d.size(0) == batch_size,
                            "output_accepted_token_num_d shape[0] mismatch");
    RTP_LLM_CHECK_WITH_INFO(params.do_sample_d.size(0) == batch_size, "do_sample_d shape[0] mismatch");

    return {static_cast<int>(batch_size),
            static_cast<int>(num_speculative_tokens),
            static_cast<int>(target_vocab_size),
            static_cast<int>(target_token_stride)};
}

#endif  // USING_CUDA || USING_ROCM

}  // anonymous namespace

#if USING_CUDA

using SamplerT = float;

namespace {

struct SamplingRouteState {
    bool    initialized  = false;
    bool    ready        = false;
    int64_t warmed_vocab = 0;
};

bool samplingRuntimeReady(int64_t vocab_size = 0) {
    if (!Py_IsInitialized()) {
        return false;
    }
    pybind11::gil_scoped_acquire             gil;
    static std::mutex                        mutex;
    static std::map<int, SamplingRouteState> states;
    std::unique_lock<std::mutex>             guard(mutex, std::defer_lock);
    {
        pybind11::gil_scoped_release release;
        guard.lock();
    }
    auto& state = states[at::cuda::current_device()];
    if (!state.initialized || (state.ready && vocab_size > 0 && vocab_size != state.warmed_vocab)) {
        const int64_t requested_vocab = vocab_size > 0 ? vocab_size : 8193;
        std::string   failure;
        try {
            failure = pybind11::module_::import("rtp_llm.models_py.sampling.runtime")
                          .attr("warmup")(requested_vocab)
                          .cast<std::string>();
        } catch (pybind11::error_already_set& error) {
            if (!error.matches(PyExc_ImportError)) {
                throw;
            }
            failure = error.what();
        }
        state.ready       = failure.empty();
        state.initialized = true;
        if (state.ready) {
            state.warmed_vocab = requested_vocab;
        }
        RTP_LLM_LOG_INFO("deferred rejection sampling backend=%s warmup=%s",
                         state.ready ? "vllm" : "legacy",
                         state.ready ? "passed" : failure.c_str());
    }
    return state.ready;
}

// This private entry point only replays our deterministic Triton graph. Its
// random values come from explicit seed inputs, not captured Torch RNG state.
at::cuda::CUDAStream uploadDeferredInputs(const torch::Tensor&              addresses,
                                          const std::vector<torch::Tensor>& inputs,
                                          const torch::Tensor&              tokens,
                                          const torch::Tensor&              lengths,
                                          const std::vector<torch::Tensor>& extra_inputs = {}) {
    const int64_t prefix = 8 + extra_inputs.size();
    TORCH_CHECK(inputs.size() == 5 || inputs.size() == 6, "invalid deferred Graph input count");
    TORCH_CHECK(addresses.is_cuda() && addresses.scalar_type() == torch::kInt64 && addresses.numel() == prefix
                    && addresses.is_contiguous(),
                "invalid deferred Graph address table");
    const auto device = addresses.get_device();
    TORCH_CHECK(at::cuda::current_device() == device, "deferred Graph device mismatch");
    const auto    stream     = at::cuda::getCurrentCUDAStream(device);
    const bool    packed_rng = !inputs[2].is_cuda();
    const int64_t batch      = inputs[0].size(0);
    if (packed_rng) {
        TORCH_CHECK(inputs[2].scalar_type() == torch::kInt64 && inputs[2].is_pinned() && inputs[2].is_contiguous()
                        && inputs[2].numel() == batch && inputs[2].storage_offset() == prefix && !inputs[3].is_cuda()
                        && inputs[3].scalar_type() == torch::kInt64 && inputs[3].is_contiguous()
                        && inputs[3].numel() == batch
                        && inputs[3].data_ptr<int64_t>() == inputs[2].data_ptr<int64_t>() + batch,
                    "invalid deferred Graph packed RNG state");
    }
    auto          host     = packed_rng ? inputs[2] :
                                          torch::empty({prefix}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    auto          pointers = host.data_ptr<int64_t>() - (packed_rng ? prefix : 0);
    const int64_t count    = packed_rng ? prefix + 2 * batch : prefix;
    TORCH_CHECK(addresses.storage().nbytes() >= (addresses.storage_offset() + count) * sizeof(int64_t),
                "deferred address storage is too small");
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (packed_rng && (i == 2 || i == 3)) {
            pointers[i] = reinterpret_cast<int64_t>(addresses.data_ptr<int64_t>() + prefix + (i - 2) * batch);
        } else {
            TORCH_CHECK(inputs[i].is_cuda() && inputs[i].get_device() == device,
                        "deferred Graph input device mismatch");
            pointers[i] = reinterpret_cast<int64_t>(inputs[i].data_ptr());
        }
    }
    if (inputs.size() == 5) {
        pointers[5] = 0;
    }
    TORCH_CHECK(tokens.is_cuda() && lengths.is_cuda() && tokens.get_device() == device
                    && lengths.get_device() == device,
                "deferred Graph output device mismatch");
    pointers[6] = reinterpret_cast<int64_t>(tokens.data_ptr());
    pointers[7] = reinterpret_cast<int64_t>(lengths.data_ptr());
    for (size_t i = 0; i < extra_inputs.size(); ++i) {
        TORCH_CHECK(extra_inputs[i].is_cuda() && extra_inputs[i].get_device() == device,
                    "compact sampler input device mismatch");
        pointers[8 + i] = reinterpret_cast<int64_t>(extra_inputs[i].data_ptr());
        c10::cuda::CUDACachingAllocator::recordStream(extra_inputs[i].storage().data_ptr(), stream);
    }
    // Register the asynchronous use with the pinned allocator, just as ATen
    // copy_ does, without constructing temporary views and dispatching a copy op.
    C10_CUDA_CHECK(cudaMemcpyAsync(
        addresses.data_ptr(), pointers, count * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream()));
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 11)
    const bool recorded = at::getHostAllocator(at::kCUDA)->record_event(
        pointers, host.storage().data_ptr().get_context(), stream.unwrap());
#else
    const bool recorded =
        at::cuda::CachingHostAllocator_recordEvent(pointers, host.storage().data_ptr().get_context(), stream);
#endif
    TORCH_CHECK(recorded, "deferred upload requires a managed pinned allocation");
    for (const auto& input : inputs) {
        if (input.is_cuda()) {
            c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
        }
    }
    c10::cuda::CUDACachingAllocator::recordStream(tokens.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(lengths.storage().data_ptr(), stream);
    return stream;
}

void replayDeferredGraph(uint64_t                          graph_exec,
                         uint64_t                          ready_event,
                         const torch::Tensor&              addresses,
                         const std::vector<torch::Tensor>& inputs,
                         const torch::Tensor&              tokens,
                         const torch::Tensor&              lengths) {
    TORCH_CHECK(graph_exec != 0 && ready_event != 0, "invalid deferred Graph handles");
    const auto stream = uploadDeferredInputs(addresses, inputs, tokens, lengths);
    auto       status = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec), stream.stream());
    TORCH_CHECK(status == cudaSuccess, "deferred Graph launch failed: ", cudaGetErrorString(status));
    status = cudaEventRecord(reinterpret_cast<cudaEvent_t>(ready_event), stream.stream());
    TORCH_CHECK(status == cudaSuccess, "deferred Graph event record failed: ", cudaGetErrorString(status));
}

using DeferredPlanKey = std::tuple<int, uint64_t, bool, int64_t, int64_t, int64_t, bool>;

struct DeferredGraphPlan {
    pybind11::weakref owner;
    uint64_t          graph_exec;
    uint64_t          ready_event;
    torch::Tensor     addresses;
};

std::mutex& deferredGraphMutex() {
    static std::mutex mutex;
    return mutex;
}

// All accesses hold the GIL and native guard. The guard also covers Python's
// ready-query/eviction sequence, whose CUDA query may temporarily release GIL.
// Only Entry's capsule owns a plan; the registry cannot retain Graph buffers.
std::map<DeferredPlanKey, std::weak_ptr<DeferredGraphPlan>>& deferredGraphPlans() {
    static std::map<DeferredPlanKey, std::weak_ptr<DeferredGraphPlan>> plans;
    return plans;
}

template<typename Key, typename Plan>
pybind11::capsule storeDeferredPlan(std::map<Key, std::weak_ptr<Plan>>& plans,
                                    const Key&                          key,
                                    std::shared_ptr<Plan>               plan) {
    for (auto it = plans.begin(); it != plans.end();) {
        it = it->second.expired() ? plans.erase(it) : std::next(it);
    }
    constexpr size_t max_plan_keys = 64;
    if (plans.size() >= max_plan_keys && plans.find(key) == plans.end()) {
        plans.erase(plans.begin());
    }
    plans[key] = plan;
    return pybind11::capsule(new std::shared_ptr<Plan>(std::move(plan)),
                             [](void* pointer) { delete static_cast<std::shared_ptr<Plan>*>(pointer); });
}

pybind11::capsule registerDeferredGraphPlan(const pybind11::object& owner,
                                            uint64_t                graph_exec,
                                            uint64_t                ready_event,
                                            const torch::Tensor&    addresses,
                                            const DeferredPlanKey&  key) {
    TORCH_CHECK(graph_exec && ready_event && addresses.is_cuda(), "invalid deferred Graph registration");
    TORCH_CHECK(std::get<0>(key) == addresses.get_device(), "deferred Graph registration device mismatch");
    auto plan = std::make_shared<DeferredGraphPlan>(
        DeferredGraphPlan{pybind11::weakref(owner), graph_exec, ready_event, addresses});
    return storeDeferredPlan(deferredGraphPlans(), key, std::move(plan));
}

bool supportsDeferredPlan(const RejectionSamplingParams& params) {
    const auto& logits = params.target_logits_d;
    if (!params.draft_probs_point_mass || params.cum_log_probs.defined() || !logits.is_cuda() || logits.dim() != 3
        || logits.scalar_type() != torch::kFloat32 || !logits.is_contiguous()) {
        return false;
    }
    const auto batch = logits.size(0);
    const auto rows  = logits.size(1);
    if (batch < 8 || batch > 128 || rows < 2 || rows > 9) {
        return false;
    }
    const auto& forced = params.force_accept;
    if (!forced.empty()
        && (forced.size() != static_cast<size_t>(batch)
            || std::find(forced.begin(), forced.end(), !forced.front()) != forced.end())) {
        return false;
    }
    const auto device  = logits.get_device();
    auto       matches = [device](const torch::Tensor& tensor, at::ScalarType dtype, c10::IntArrayRef shape) {
        return tensor.defined() && tensor.is_cuda() && tensor.get_device() == device && tensor.scalar_type() == dtype
               && tensor.sizes() == shape && tensor.is_contiguous();
    };
    return matches(params.draft_token_ids_d, torch::kInt32, {batch, rows - 1})
           && matches(params.do_sample_d, torch::kBool, {batch})
           && matches(params.output_token_ids_d, torch::kInt32, {batch, rows})
           && matches(params.output_accepted_token_num_d, torch::kInt32, {batch})
           && (!params.acceptance_cap.defined() || matches(params.acceptance_cap, torch::kInt32, {batch}));
}

std::optional<DeferredPlanKey> deferredSamplingLayout(const RejectionSamplingParams& params, bool capacity_bucket) {
    const auto& logits = params.target_logits_d;
    const auto  device = logits.get_device();
    if (at::cuda::current_device() != device) {
        return std::nullopt;
    }
    const auto              stream = at::cuda::getCurrentCUDAStream(device);
    cudaStreamCaptureStatus capture_status;
    const auto              status = cudaStreamIsCapturing(stream.stream(), &capture_status);
    TORCH_CHECK(status == cudaSuccess, "deferred Graph capture query failed: ", cudaGetErrorString(status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return std::nullopt;
    }
    const auto batch  = logits.size(0);
    const bool forced = !params.force_accept.empty() && params.force_accept.front();
    int64_t    bucket = batch;
    if (capacity_bucket) {
        bucket = 8;
        while (bucket < batch) {
            bucket *= 2;
        }
    }
    return DeferredPlanKey{device, reinterpret_cast<uint64_t>(stream.stream()), forced, bucket, logits.size(1),
                           logits.size(2), params.acceptance_cap.defined()};
}

bool tryReplayDeferredGraph(const RejectionSamplingParams& params,
                            const torch::Tensor&           seeds,
                            const torch::Tensor&           positions) {
    if (params.target_candidate_ids_d.defined())
        return false;
    const auto layout = deferredSamplingLayout(params, false);
    if (!layout) {
        return false;
    }
    const auto&                  key = *layout;
    std::unique_lock<std::mutex> guard(deferredGraphMutex(), std::try_to_lock);
    if (!guard.owns_lock()) {
        return false;
    }
    auto& plans = deferredGraphPlans();
    auto  found = plans.find(key);
    if (found == plans.end()) {
        return false;
    }
    auto plan = found->second.lock();
    if (!plan) {
        plans.erase(found);
        return false;
    }
    auto owner = plan->owner();
    if (owner.is_none()) {
        plans.erase(found);
        return false;
    }
    std::vector<torch::Tensor> inputs{
        params.target_logits_d, params.draft_token_ids_d, seeds, positions, params.do_sample_d};
    if (params.acceptance_cap.defined()) {
        inputs.push_back(params.acceptance_cap);
    }
    replayDeferredGraph(plan->graph_exec,
                        plan->ready_event,
                        plan->addresses,
                        inputs,
                        params.output_token_ids_d,
                        params.output_accepted_token_num_d);
    return true;
}

union DeferredLaunchValue {
    uint64_t pointer;
    int32_t  i32;
    uint32_t u32;
    int64_t  i64;
    uint64_t u64;
    float    f32;
    double   f64;
};

struct DeferredKernel {
    uint64_t                         function;
    unsigned                         shared;
    unsigned                         warps;
    unsigned                         grid_factor;
    unsigned                         grid_divisor;
    unsigned                         grid_y;
    std::vector<int>                 types;
    std::vector<DeferredLaunchValue> values;
    std::vector<int64_t>             factors;

    explicit DeferredKernel(const pybind11::tuple& descriptor) {
        TORCH_CHECK(descriptor.size() == 9, "invalid native sampler launch descriptor");
        function     = descriptor[0].cast<uint64_t>();
        shared       = descriptor[1].cast<unsigned>();
        warps        = descriptor[2].cast<unsigned>();
        grid_factor  = descriptor[3].cast<unsigned>();
        grid_divisor = descriptor[4].cast<unsigned>();
        grid_y       = descriptor[5].cast<unsigned>();
        types        = descriptor[6].cast<std::vector<int>>();
        auto args    = descriptor[7].cast<pybind11::list>();
        factors      = descriptor[8].cast<std::vector<int64_t>>();
        TORCH_CHECK(function && warps && grid_factor && grid_divisor && grid_y && types.size() <= 64
                        && args.size() == types.size() && factors.size() == types.size(),
                    "invalid native sampler launch ABI");
        values.resize(types.size());
        for (size_t i = 0; i < types.size(); ++i) {
            switch (types[i]) {
                case 0:
                    values[i].pointer = args[i].cast<uint64_t>();
                    break;
                case 1:
                    values[i].i32 = args[i].cast<int32_t>();
                    break;
                case 2:
                    values[i].u32 = args[i].cast<uint32_t>();
                    break;
                case 3:
                    values[i].i64 = args[i].cast<int64_t>();
                    break;
                case 4:
                    values[i].u64 = args[i].cast<uint64_t>();
                    break;
                case 5:
                    values[i].f32 = args[i].cast<float>();
                    break;
                case 6:
                    values[i].f64 = args[i].cast<double>();
                    break;
                default:
                    TORCH_CHECK(false, "unsupported native sampler argument type");
            }
            TORCH_CHECK(factors[i] == 0 || (factors[i] > 0 && types[i] >= 1 && types[i] <= 4),
                        "invalid native sampler batch argument");
        }
    }

    void launch(int64_t batch, cudaStream_t stream) {
        using Launch =
            int (*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
        static Launch launch_kernel = [] {
            void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
            TORCH_CHECK(library, "Cannot load CUDA driver: ", dlerror());
            auto function = reinterpret_cast<Launch>(dlsym(library, "cuLaunchKernel"));
            TORCH_CHECK(function, "Cannot resolve cuLaunchKernel");
            return function;
        }();
        std::array<void*, 66> arguments;
        for (size_t i = 0; i < values.size(); ++i) {
            if (factors[i]) {
                const auto value = batch * factors[i];
                if (types[i] == 1)
                    values[i].i32 = static_cast<int32_t>(value);
                else if (types[i] == 2)
                    values[i].u32 = static_cast<uint32_t>(value);
                else if (types[i] == 3)
                    values[i].i64 = value;
                else
                    values[i].u64 = static_cast<uint64_t>(value);
            }
            arguments[i] = &values[i];
        }
        // Triton 3.6 appends two scratch pointers; the Python gate requires
        // zero scratch sizes and excludes cluster/cooperative/PDL launches.
        void* scratch                = nullptr;
        arguments[values.size()]     = &scratch;
        arguments[values.size() + 1] = &scratch;
        const unsigned grid_x        = (batch * grid_factor + grid_divisor - 1) / grid_divisor;
        const int      status        = launch_kernel(reinterpret_cast<void*>(function),
                                         grid_x,
                                         grid_y,
                                         1,
                                         warps * 32,
                                         1,
                                         1,
                                         shared,
                                         reinterpret_cast<void*>(stream),
                                         arguments.data(),
                                         nullptr);
        TORCH_CHECK(status == 0, "native sampler cuLaunchKernel failed: ", status);
    }
};

using DeferredKernelKey = std::tuple<int, uint64_t, bool, int64_t, int64_t, int64_t, bool, int64_t, bool>;

struct DeferredKernelPlan {
    pybind11::weakref           owner;
    uint64_t                    ready_event;
    torch::Tensor               addresses;
    std::vector<DeferredKernel> kernels;
};

std::map<DeferredKernelKey, std::weak_ptr<DeferredKernelPlan>>& deferredKernelPlans() {
    static std::map<DeferredKernelKey, std::weak_ptr<DeferredKernelPlan>> plans;
    return plans;
}

pybind11::capsule registerDeferredKernelPlan(const pybind11::object&  owner,
                                             uint64_t                 ready_event,
                                             const torch::Tensor&     addresses,
                                             const DeferredKernelKey& key,
                                             const pybind11::list&    descriptors) {
    TORCH_CHECK(ready_event && (descriptors.size() == 5 || descriptors.size() == 1) && addresses.is_cuda()
                    && addresses.get_device() == std::get<0>(key),
                "invalid native sampler registration");
    auto plan =
        std::make_shared<DeferredKernelPlan>(DeferredKernelPlan{pybind11::weakref(owner), ready_event, addresses, {}});
    for (const auto& descriptor : descriptors) {
        plan->kernels.emplace_back(pybind11::cast<pybind11::tuple>(descriptor));
    }
    return storeDeferredPlan(deferredKernelPlans(), key, std::move(plan));
}

pybind11::object pyDeferredRejectionFn();

bool trySubmitDeferredKernels(const RejectionSamplingParams& params,
                              const torch::Tensor&           seeds,
                              const torch::Tensor&           positions) {
    const auto layout = deferredSamplingLayout(params, true);
    if (!layout)
        return false;
    const bool compact     = params.target_candidate_ids_d.defined();
    int64_t    dense_vocab = 0;
    if (compact) {
        const auto& ids   = params.target_candidate_ids_d;
        const auto& dense = params.target_dense_fallback_d;
        const auto& flags = params.target_fallback_rows_d;
        if (params.target_logits_d.size(2) != 1024 || !ids.is_cuda() || ids.scalar_type() != torch::kInt32
            || ids.sizes() != params.target_logits_d.sizes() || !ids.is_contiguous() || !dense.is_cuda()
            || dense.scalar_type() != torch::kFloat32 || dense.dim() != 3 || !dense.is_contiguous()
            || dense.size(0) != ids.size(0) || dense.size(1) != ids.size(1) || dense.size(2) <= 1024 || !flags.is_cuda()
            || flags.scalar_type() != torch::kInt32 || !flags.is_contiguous()
            || flags.numel() != ids.size(0) * ids.size(1))
            return false;
        const auto& normalizer = params.target_log_normalizer_d;
        if (normalizer.defined()
            && (!normalizer.is_cuda() || normalizer.device() != ids.device()
                || normalizer.scalar_type() != torch::kFloat32 || !normalizer.is_contiguous()
                || normalizer.numel() != ids.size(0) * ids.size(1)))
            return false;
        dense_vocab = dense.size(2);
    }
    const DeferredKernelKey key =
        std::tuple_cat(*layout, std::make_tuple(dense_vocab, compact && params.target_log_normalizer_d.defined()));
    std::unique_lock<std::mutex> guard(deferredGraphMutex(), std::try_to_lock);
    if (!guard.owns_lock())
        return false;
    auto&            plans = deferredKernelPlans();
    auto             found = plans.find(key);
    auto             plan  = found == plans.end() ? nullptr : found->second.lock();
    pybind11::object owner = plan ? plan->owner() : pybind11::none();
    if (!plan || owner.is_none()) {
        // Compilation never launches or reserves RNG. Release the guard while
        // the Python builder acquires it for workspace eviction/registration.
        guard.unlock();
        pyDeferredRejectionFn();
        auto builder = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_native").attr("build_plan");
        if (!builder(key).cast<bool>() || !guard.try_lock())
            return false;
        found = plans.find(key);
        plan  = found == plans.end() ? nullptr : found->second.lock();
        owner = plan ? plan->owner() : pybind11::none();
        if (!plan || owner.is_none())
            return false;
    }
    std::vector<torch::Tensor> inputs{
        params.target_logits_d, params.draft_token_ids_d, seeds, positions, params.do_sample_d};
    if (params.acceptance_cap.defined())
        inputs.push_back(params.acceptance_cap);
    std::vector<torch::Tensor> extra = compact ? std::vector<torch::Tensor>{params.target_candidate_ids_d,
                                                                            params.target_dense_fallback_d,
                                                                            params.target_fallback_rows_d} :
                                                 std::vector<torch::Tensor>{};
    if (compact && params.target_log_normalizer_d.defined())
        extra.push_back(params.target_log_normalizer_d);
    const auto stream = uploadDeferredInputs(
        plan->addresses, inputs, params.output_token_ids_d, params.output_accepted_token_num_d, extra);
    for (auto& kernel : plan->kernels)
        kernel.launch(params.target_logits_d.size(0), stream.stream());
    const auto status = cudaEventRecord(reinterpret_cast<cudaEvent_t>(plan->ready_event), stream.stream());
    TORCH_CHECK(status == cudaSuccess, "native sampler completion event failed: ", cudaGetErrorString(status));
    return true;
}

pybind11::object pyDeferredRejectionFn() {
    auto native_module = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_native");
    if (native_module.attr("_register_plan").is_none()) {
        auto graph_module                   = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_graph");
        graph_module.attr("_native_replay") = pybind11::cpp_function(&replayDeferredGraph);
        graph_module.attr("_register_native_plan") = pybind11::cpp_function(&registerDeferredGraphPlan);
        graph_module.attr("_try_native_guard") =
            pybind11::cpp_function([]() { return deferredGraphMutex().try_lock(); });
        graph_module.attr("_release_native_guard") = pybind11::cpp_function([]() { deferredGraphMutex().unlock(); });
        native_module.attr("_register_plan")       = pybind11::cpp_function(&registerDeferredKernelPlan);
    }
    return pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("rejection_sample_logits");
}

torch::Tensor reserveSamplingSeedOffsets(const std::vector<at::Generator>& generators,
                                         int64_t                           batch_size,
                                         int                               increment,
                                         int64_t                           prefix = 0) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto state_h =
        torch::empty({prefix + 2 * batch_size}, options).narrow(0, prefix, 2 * batch_size).view({2, batch_size});
    auto seed_ptr = state_h.data_ptr<int64_t>();
    auto off_ptr  = seed_ptr + batch_size;

    for (int64_t i = 0; i < batch_size; ++i) {
        // Do not resolve the default CUDA generator for explicit per-request
        // generators. Keep PyTorch's capture checks and offset reservation intact.
        const auto& generator = (i < static_cast<int64_t>(generators.size()) && generators[i].defined()) ?
                                    generators[i] :
                                    at::cuda::detail::getDefaultCUDAGenerator();
        auto*       impl      = at::check_generator<at::CUDAGeneratorImpl>(generator);
        std::lock_guard<std::mutex> lock(impl->mutex_);
        const auto                  state = impl->philox_cuda_state(increment);
        seed_ptr[i]                       = static_cast<int64_t>(state.seed_.val);
        off_ptr[i]                        = static_cast<int64_t>(state.offset_.val);
    }

    return state_h;
}

bool requiresTokenHistory(const GreedyParams& params) {
    const auto batch_size = params.logits.size(0);
    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto* repetition = params.repetition_penalty->data_ptr<float>();
        const auto* presence   = params.presence_penalty->data_ptr<float>();
        const auto* frequency  = params.frequency_penalty->data_ptr<float>();
        if (std::any_of(repetition, repetition + batch_size, [](float value) { return value != 1.0f; })
            || std::any_of(presence, presence + batch_size, [](float value) { return value != 0.0f; })
            || std::any_of(frequency, frequency + batch_size, [](float value) { return value != 0.0f; })) {
            return true;
        }
    }
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    if (decoder_batch_size && params.no_repeat_ngram_size.has_value()) {
        const auto* ngram = params.no_repeat_ngram_size->data_ptr<int32_t>();
        return std::any_of(ngram, ngram + decoder_batch_size, [](int32_t value) { return value != 0; });
    }
    return false;
}

torch::Tensor getSamplingHostBufferSlice(const torch::Tensor& buffer, int64_t batch_size, const char* buffer_name) {
    RTP_LLM_CHECK_WITH_INFO(buffer.defined(), "%s is not initialized", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.device().is_cpu(), "%s must be a CPU tensor", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.scalar_type() == torch::kInt64, "%s must be an int64 tensor", buffer_name);
    RTP_LLM_CHECK_WITH_INFO(buffer.numel() >= batch_size,
                            "%s capacity [%ld] is smaller than batch size [%ld]",
                            buffer_name,
                            buffer.numel(),
                            batch_size);
    return buffer.narrow(0, 0, batch_size);
}

std::pair<torch::Tensor, torch::Tensor> makeSamplingSeedOffsetTensors(const std::vector<at::Generator>& generators,
                                                                      int64_t                           batch_size,
                                                                      int                               increment,
                                                                      GreedySamplingBuffers* sampling_buffers) {
    RTP_LLM_CHECK_WITH_INFO(generators.size() >= static_cast<size_t>(batch_size),
                            "sampling generators size [%lu] is smaller than batch size [%ld]",
                            generators.size(),
                            batch_size);
    const bool    use_persistent_buffers = sampling_buffers != nullptr;
    auto          options                = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    torch::Tensor seed_h;
    torch::Tensor offset_h;
    if (use_persistent_buffers) {
        seed_h   = getSamplingHostBufferSlice(sampling_buffers->seed_host, batch_size, "sampling seed buffer");
        offset_h = getSamplingHostBufferSlice(sampling_buffers->offset_host, batch_size, "sampling offset buffer");
    } else {
        seed_h   = torch::empty({batch_size}, options);
        offset_h = torch::empty({batch_size}, options);
    }
    auto seed_ptr = seed_h.data_ptr<int64_t>();
    auto off_ptr  = offset_h.data_ptr<int64_t>();

    for (int64_t i = 0; i < batch_size; ++i) {
        // Undefined generator entries intentionally use PyTorch's default CUDA generator.
        auto generator      = (i < static_cast<int64_t>(generators.size()) && generators[i].defined()) ?
                                  std::make_optional(generators[i]) :
                                  std::nullopt;
        auto [seed, offset] = get_seed_and_offset(increment, generator);
        seed_ptr[i]         = static_cast<int64_t>(seed);
        off_ptr[i]          = static_cast<int64_t>(offset);
    }

    auto seed_d   = seed_h.to(torch::kCUDA, /*non_blocking=*/use_persistent_buffers).contiguous();
    auto offset_d = offset_h.to(torch::kCUDA, /*non_blocking=*/use_persistent_buffers).contiguous();
    return {seed_d, offset_d};
}

bool canUseCompactSamplerOutput(const GreedyParams& params) {
    return params.token_ids.is_cuda() && params.token_ids.is_contiguous() && params.token_ids.dim() == 2
           && params.token_ids.scalar_type() == torch::kInt32 && params.token_ids.size(0) == params.logits.size(0)
           && params.token_ids.size(0) > 1 && params.token_ids.size(1) > 1;
}

void processLogits(const GreedyParams& params, const torch::Tensor& device_tokens, torch::Tensor& transposed_tokens) {
    const auto vocab_size_padded  = params.logits.size(1);
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    const auto batch_size         = params.logits.size(0);
    const auto step               = params.step;
    auto       cur_stream         = at::cuda::getCurrentCUDAStream().stream();

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA, true);
        applySamplerTemperature(params.logits, temperature_gpu, cur_stream);
    }

    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            // Only history-dependent penalties need a full transposed token buffer.
            if (!transposed_tokens.defined()) {
                transposed_tokens = device_tokens.transpose(0, 1).contiguous();
            }
            // Build sequence_lengths: clone input_lengths, then overwrite first decoder_batch_size entries
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA, true);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA, true), true);
            }
            auto penalty_ws = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA, true);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA, true);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA, true);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
            // NOTE: here step is max_len - 1
        }
    }

    if (decoder_batch_size && params.no_repeat_ngram_size.has_value()) {
        const auto& no_repeat_ngram_size = params.no_repeat_ngram_size.value();
        if (any_of(no_repeat_ngram_size.data_ptr<int32_t>(),
                   no_repeat_ngram_size.data_ptr<int32_t>() + decoder_batch_size,
                   [](auto s) { return s != 0; })) {
            auto no_repeat_ngram_size_gpu = no_repeat_ngram_size.to(torch::kCUDA, true);
            // Build array of pointers to each batch's token ids
            const bool    use_persistent_buffers = params.sampling_buffers != nullptr;
            torch::Tensor output_ids_ptrs;
            if (use_persistent_buffers) {
                output_ids_ptrs = getSamplingHostBufferSlice(params.sampling_buffers->output_ids_ptrs_host,
                                                             decoder_batch_size,
                                                             "sampling output ids ptrs buffer");
            } else {
                output_ids_ptrs = torch::empty({(int64_t)decoder_batch_size},
                                               torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
            }
            for (int64_t i = 0; i < (int64_t)decoder_batch_size; i++) {
                output_ids_ptrs.data_ptr<int64_t>()[i] = (int64_t)(device_tokens.data_ptr<int32_t>() + i * (step + 1));
            }
            auto output_ids_ptrs_gpu = output_ids_ptrs.to(torch::kCUDA, use_persistent_buffers);
            // SamplerInputs stores real sequence lengths, while the imported
            // TensorRT-LLM no-repeat kernel still expects the last valid
            // token index and adds one internally.
            auto sequence_last_indexes_gpu = params.sequence_lengths.to(torch::kCUDA, true).sub(1);

            tensorrt_llm::kernels::invokeBanRepeatNgram(params.logits.data_ptr<float>(),
                                                        (int32_t const**)(output_ids_ptrs_gpu.data_ptr()),
                                                        nullptr,  // finished_buf
                                                        nullptr,  // parent_ids_buf
                                                        nullptr,  // batch_slot
                                                        sequence_last_indexes_gpu.data_ptr<int32_t>(),
                                                        decoder_batch_size,
                                                        1,  // beam_width
                                                        step + 1,
                                                        no_repeat_ngram_size_gpu.data_ptr<int32_t>(),
                                                        vocab_size_padded,
                                                        step + 1,
                                                        cur_stream);
        }
    }
}

}  // anonymous namespace

static float normalizeDeferredTopP(float value) {
    return std::abs(value) < 1e-7f || std::abs(value - 1.0f) < 1e-7f ? 1.0f : value;
}

static GreedyOutput tryCompactDeferredSampling(const GreedyParams& params, float temperature = 1.0f) {
    const auto batch_size = params.logits.size(0);
    const auto* requested_k = params.top_k.data_ptr<int32_t>();
    const auto* requested_p = params.top_p.data_ptr<float>();
    const float compact_p = normalizeDeferredTopP(requested_p[0]);
    if (params.allow_compact_deferred_output && requested_k[0] >= 2 && requested_k[0] <= 1024
        && std::all_of(
            requested_k, requested_k + batch_size, [requested_k](int32_t value) { return value == requested_k[0]; })
        && std::all_of(requested_p, requested_p + batch_size, [compact_p](float value) {
               return normalizeDeferredTopP(value) == compact_p;
           })) {
        torch::Tensor log_normalizer;
        auto          compact = torch_ext::deferred_compact_topk_topp(
            params.logits, requested_k[0], compact_p, temperature, &log_normalizer);
        if (std::get<0>(compact).defined()) {
            torch::Tensor prepared_rng;
            if (params.deferred_rejection_rows > 0) {
                const int64_t rows = params.deferred_rejection_rows;
                TORCH_CHECK(batch_size % rows == 0, "deferred rejection rows must divide target rows");
                const int64_t requests = batch_size / rows;
                TORCH_CHECK(static_cast<int64_t>(params.deferred_rejection_generators.size()) <= requests,
                            "too many deferred rejection generators");
                // The GPU can filter while the CPU reserves the exact same
                // per-request counters that rejection would reserve later.
                prepared_rng = reserveSamplingSeedOffsets(params.deferred_rejection_generators,
                                                          requests,
                                                          std::max<int64_t>(32, ((rows + 3) / 4) * 4),
                                                          log_normalizer.defined() ? 12 : 11);
            }
            return GreedyOutput{torch::Tensor(),
                                true,
                                std::get<0>(compact),
                                std::get<1>(compact),
                                params.logits,
                                std::get<2>(compact),
                                prepared_rng,
                                log_normalizer};
        }
    }
    return {};
}

static GreedyOutput prepareDeferredTarget(const GreedyParams& params) {
    const auto batch_size = params.logits.size(0);
    auto       cur_stream = at::cuda::getCurrentCUDAStream().stream();
    if (batch_size == 0) {
        return GreedyOutput{torch::Tensor(), true};
    }
    const auto* requested_k = params.top_k.data_ptr<int32_t>();
    const auto* requested_p = params.top_p.data_ptr<float>();
    const auto  vocab       = params.logits.size(1);
    bool        unfiltered  = true;
    for (int64_t row = 0; row < batch_size; ++row) {
        // Greedy rows ignore TopP, including non-finite values, just as
        // the normalization below does before applying any filters.
        if (requested_k[row] != 1
            && !((requested_k[row] <= 0 || requested_k[row] >= vocab)
                 && normalizeDeferredTopP(requested_p[row]) == 1.0f)) {
            unfiltered = false;
            break;
        }
    }
    if (unfiltered) {
        return GreedyOutput{torch::Tensor(), true};
    }
    const float compact_p = normalizeDeferredTopP(requested_p[0]);
    auto        prepared  = tryCompactDeferredSampling(params);
    if (prepared.deferred_token_sampling)
        return prepared;
    if (requested_k[0] == 1024
        && std::all_of(requested_k, requested_k + batch_size, [](int32_t value) { return value == 1024; })
        && std::all_of(requested_p,
                       requested_p + batch_size,
                       [compact_p](float value) { return normalizeDeferredTopP(value) == compact_p; })
        && torch_ext::deferred_topk_topp_filter(params.logits, compact_p)) {
        return GreedyOutput{torch::Tensor(), true};
    }
    auto top_k = params.top_k.clone();
    auto top_p = params.top_p.clone();
    auto k     = top_k.data_ptr<int32_t>();
    auto p     = top_p.data_ptr<float>();
    for (int64_t row = 0; row < batch_size; ++row) {
        if (k[row] == 1) {
            k[row] = 0;
            p[row] = 1.0f;
        }
    }
    std::transform(p, p + batch_size, p, [](float value) { return std::abs(value) < 1e-7f ? 1.0f : value; });
    const bool no_k = std::all_of(k, k + batch_size, [vocab](int32_t value) { return value <= 0 || value >= vocab; });
    const bool no_p = std::all_of(p, p + batch_size, [](float value) { return std::abs(value - 1.0f) < 1e-7f; });
    auto probs_t = torch::softmax(params.logits, -1);
    auto output  = torch::empty_like(probs_t);
    if (no_k) {
        top_p_renorm_probs(probs_t, output, top_p, 1.0, (int64_t)cur_stream);
    } else {
        std::transform(k, k + batch_size, k, [](int32_t value) { return value <= 0 ? 1 << 30 : value; });
        if (no_p) {
            top_k_renorm_probs(probs_t, output, top_k, 0, (int64_t)cur_stream);
        } else {
            auto temporary = torch::empty_like(probs_t);
            top_k_renorm_probs(probs_t, temporary, top_k, 0, (int64_t)cur_stream);
            top_p_renorm_probs(temporary, output, top_p, 1.0, (int64_t)cur_stream);
        }
    }
    params.logits.copy_(output.log(), true);
    check_cuda_error();
    return GreedyOutput{torch::Tensor(), true};
}

static GreedyOutput flashinferSampleGreedy(const GreedyParams& params, const torch::Tensor& samples_t) {
    const auto batch_size = params.logits.size(0);
    auto       cur_stream = at::cuda::getCurrentCUDAStream().stream();

    // [batch_size, vocab_size] — compute softmax probabilities.
    // Copy result back to logits to preserve the in-place behavior of the original kernel,
    // since callers may reuse the logits tensor across iterations.
    auto probs_t = samplerSoftmaxInplace(params.logits);
    auto success = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    constexpr bool deterministic       = true;
    constexpr int  max_sampling_rounds = 32;
    auto [seed_t, offset_t] =
        makeSamplingSeedOffsetTensors(params.generator, batch_size, max_sampling_rounds, params.sampling_buffers);

    torch::Tensor success_t = success;
    torch::Tensor top_k_t   = params.top_k;
    torch::Tensor top_p_t   = params.top_p;
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }
    const auto output_width = output_all_probs_t.defined() ? output_all_probs_t.size(1) : probs_t.size(1);
    RTP_LLM_CHECK_WITH_INFO(output_width > 0 && output_width <= probs_t.size(1),
                            "output probability width must be in (0, padded vocab size]");
    const bool need_full_sampling_probs = output_all_probs_t.defined() && output_width < probs_t.size(1);
    auto       sampling_probs_t         = need_full_sampling_probs ? torch::zeros_like(probs_t) : output_all_probs_t;

    // top_k/top_p are CPU tensors with int32/float32 dtype
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    auto top_p_ptr = params.top_p.data_ptr<float>();

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    const bool need_renorm_probs  = sampling_probs_t.defined() && !params.return_original_all_probs;
    const bool all_top_k_one      = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; });
    const bool all_top_k_no_limit = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
    const bool all_top_p_one =
        std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });

    if (all_top_k_one) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);
        success = torch::Tensor();  // mark as undefined — all succeeded
        if (need_renorm_probs) {
            top_k_renorm_probs(probs_t, sampling_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else if (all_top_k_no_limit) {
        const bool exact_top_p_one = std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return t == 1.0f; });
        if (exact_top_p_one) {
            // The unfiltered case needs no nucleus threshold search. Preserve
            // the existing per-request Philox reservation above (32), the
            // logits/probability outputs and every filtered/greedy fallback.
            sampling_from_probs(probs_t,
                                samples_t,
                                success_t,
                                std::nullopt,
                                deterministic,
                                seed_t,
                                0,
                                offset_t,
                                0,
                                (int64_t)cur_stream);
        } else {
            top_p_sampling_from_probs(probs_t,
                                      samples_t,
                                      success_t,
                                      std::nullopt,
                                      top_p_t,
                                      1.0,
                                      deterministic,
                                      seed_t,
                                      0,
                                      offset_t,
                                      0,
                                      (int64_t)cur_stream);
        }
        if (need_renorm_probs) {
            if (exact_top_p_one) {
                // Softmax probabilities are nonnegative and have positive row
                // sums. Renormalize explicitly; returning the original softmax
                // would change the distribution consumed by rejection sampling.
                torch::div_out(sampling_probs_t, probs_t, probs_t.sum(-1, /*keepdim=*/true));
            } else {
                top_p_renorm_probs(probs_t, sampling_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
            }
        }
    } else {
        // top_k<=0 means "no limit" in RTP config. The combined FlashInfer
        // kernel takes a top_k array, so normalize mixed batches after the
        // pure top-p route has been selected.
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [](auto t) { return t <= 0 ? 1 << 30 : t; });
        if (all_top_p_one) {
            top_k_sampling_from_probs(probs_t,
                                      samples_t,
                                      success_t,
                                      std::nullopt,
                                      top_k_t,
                                      0,
                                      deterministic,
                                      seed_t,
                                      0,
                                      offset_t,
                                      0,
                                      (int64_t)cur_stream);
            if (need_renorm_probs) {
                top_k_renorm_probs(probs_t, sampling_probs_t, top_k_t, 0, (int64_t)cur_stream);
            }
        } else {
            top_k_top_p_sampling_from_probs(probs_t,
                                            samples_t,
                                            success_t,
                                            std::nullopt,
                                            top_k_t,
                                            0,
                                            top_p_t,
                                            1.0,
                                            deterministic,
                                            seed_t,
                                            0,
                                            offset_t,
                                            0,
                                            (int64_t)cur_stream);
            if (need_renorm_probs) {
                torch::Tensor temp_t = torch::zeros_like(sampling_probs_t);
                /** Joint sampling applies top-p to the original distribution; return the same support. */
                top_p_renorm_probs(probs_t, temp_t, top_p_t, 1.0, (int64_t)cur_stream);
                top_k_renorm_probs(temp_t, sampling_probs_t, top_k_t, 0, (int64_t)cur_stream);
            }
        }
    }

    if (params.return_original_all_probs && sampling_probs_t.defined()) {
        sampling_probs_t.copy_(probs_t);
    }
    if (need_full_sampling_probs) {
        output_all_probs_t.copy_(sampling_probs_t.slice(1, 0, output_width));
    }

    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        auto token_probs_t   = sampling_probs_t.gather(1, samples_t.unsqueeze(1).to(torch::kLong)).squeeze(1);
        cum_log_probs_t.add_(token_probs_t.log().to(cum_log_probs_t.device()));
    }

    return {success};
}

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size        = params.logits.size(0);
    bool       has_not_do_sample = params.do_sample.has_value()
                                   && std::any_of(params.do_sample.value().data_ptr<bool>(),
                                                  params.do_sample.value().data_ptr<bool>() + batch_size,
                                                  [&](auto t) { return !t; });
    bool       need_do_sample    = (!params.do_sample.has_value())
                                   || std::any_of(params.do_sample.value().data_ptr<bool>(),
                                                  params.do_sample.value().data_ptr<bool>() + batch_size,
                                                  [&](auto t) { return t; });
    const bool deferred = params.defer_token_sampling && Py_IsInitialized();
    if (deferred && batch_size > 0 && params.allow_compact_deferred_output
        && params.top_k.data_ptr<int32_t>()[0] == 1024 && need_do_sample && !has_not_do_sample) {
        const auto* temperatures = params.temperature.data_ptr<float>();
        const float temperature  = temperatures[0];
        // Mixed temperatures retain the batch-wide epsilon scaling in processLogits.
        if (temperature != 1.0f && std::isfinite(temperature) && temperature > 0.0f
            && std::all_of(
                temperatures, temperatures + batch_size, [temperature](float value) { return value == temperature; })
            && !requiresTokenHistory(params)) {
            auto output = tryCompactDeferredSampling(params, temperature);
            if (output.deferred_token_sampling)
                return output;
        }
    }
    torch::Tensor device_tokens;
    torch::Tensor transposed_tokens;
    if (!deferred || (need_do_sample && requiresTokenHistory(params))) {
        device_tokens = params.token_ids.to(torch::kCUDA, true);
        if (!deferred && !canUseCompactSamplerOutput(params)) {
            transposed_tokens = device_tokens.transpose(0, 1).contiguous();
        }
    }
    if (need_do_sample) {
        torch::Tensor selected_logits;
        torch::Tensor mask_tensor;
        if (has_not_do_sample) {
            auto do_sample_gpu = params.do_sample.value().to(torch::kCUDA, true);
            mask_tensor        = do_sample_gpu.reshape({(int64_t)batch_size, 1}).logical_not();
            selected_logits    = params.logits.masked_select(mask_tensor);
        }
        processLogits(params, device_tokens, transposed_tokens);
        if (has_not_do_sample) {
            params.logits.masked_scatter_(mask_tensor, selected_logits);
        }
    }

    if (deferred) {
        return prepareDeferredTarget(params);
    }

    // Preserve the initial output slot, including unsuccessful-sample behavior.
    auto         samples_t = transposed_tokens.defined() ? transposed_tokens.select(0, transposed_tokens.size(0) - 1) :
                                                           device_tokens.select(1, device_tokens.size(1) - 1).contiguous();
    GreedyOutput output;
    // fast path for topk = 1
    // TODO: Support output_all_probs and cum_log_probs in this fast path.
    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value() && !params.cum_log_probs.has_value()) {
        samples_t.copy_(torch::argmax(params.logits, -1, /*keepdim=*/false), true);
    } else {
        output = flashinferSampleGreedy(params, samples_t);
    }

    // Sampling only changes the output column; every history prefix stays in place.
    params.token_ids.select(1, params.token_ids.size(1) - 1).copy_(samples_t, true);
    check_cuda_error();
    return output;
}

torch::Tensor sampleFromProbs(const torch::Tensor& probabilities) {
    RTP_LLM_CHECK_WITH_INFO(probabilities.defined() && probabilities.is_cuda() && probabilities.dim() == 2
                                && probabilities.scalar_type() == torch::kFloat32 && probabilities.is_contiguous(),
                            "probability sampling expects contiguous CUDA FP32 [batch,vocab] probabilities");
    const auto          batch_size = probabilities.size(0);
    at::cuda::CUDAGuard device_guard(probabilities.device());
    auto [seed, offset] = get_seed_and_offset(/*increment_size=*/4);

    auto token_ids =
        torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(probabilities.device()));
    auto valid = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kBool).device(probabilities.device()));
    sampling_from_probs(probabilities,
                        token_ids,
                        valid,
                        std::nullopt,
                        /*deterministic=*/true,
                        std::nullopt,
                        seed,
                        std::nullopt,
                        offset);
    return token_ids;
}

void rejectionSampling(const RejectionSamplingParams& params) {
    if (params.deferred_token_sampling) {
        TORCH_CHECK(Py_IsInitialized(), "deferred verification requires the Python Gumbel runtime");
        const int64_t batch_size = params.target_logits_d.size(0);
        const int64_t rows       = params.target_logits_d.size(1);
        // The compact producer may already have reserved RNG while filtering
        // ran on the GPU. Native and fallback dispatch consume the same packet.
        const int64_t prefix =
            params.target_candidate_ids_d.defined() ? (params.target_log_normalizer_d.defined() ? 12 : 11) : 8;
        auto state_h = params.prepared_rejection_rng;
        if (state_h.defined()) {
            TORCH_CHECK(state_h.device().is_cpu() && state_h.scalar_type() == torch::kInt64 && state_h.dim() == 2
                            && state_h.size(0) == 2 && state_h.size(1) == batch_size && state_h.is_contiguous()
                            && state_h.is_pinned() && state_h.storage_offset() == prefix,
                        "invalid prepared rejection RNG packet");
        } else {
            state_h = reserveSamplingSeedOffsets(
                params.generator, batch_size, std::max<int64_t>(32, ((rows + 3) / 4) * 4), prefix);
        }
        pybind11::gil_scoped_acquire gil;
        if (supportsDeferredPlan(params)) {
            const auto seeds_h     = state_h.select(0, 0);
            const auto positions_h = state_h.select(0, 1);
            if (trySubmitDeferredKernels(params, seeds_h, positions_h)
                || tryReplayDeferredGraph(params, seeds_h, positions_h)) {
                check_cuda_error();
                return;
            }
        }
        auto state_d = state_h.to(torch::kCUDA, /*non_blocking=*/true);
        if (params.buffer_holder != nullptr) {
            params.buffer_holder->hold_host(state_h);
        }
        auto             seeds     = state_d.select(0, 0);
        auto             positions = state_d.select(0, 1);
        pybind11::object draft =
            params.draft_probs_point_mass ? pybind11::none() : pybind11::cast(params.draft_probs_d);
        pybind11::object cap =
            params.acceptance_cap.defined() ? pybind11::cast(params.acceptance_cap) : pybind11::none();
        pybind11::object cumulative =
            params.cum_log_probs.defined() ? pybind11::cast(params.cum_log_probs) : pybind11::none();
        torch::Tensor fallback_logits = params.target_logits_d;
        if (params.target_candidate_ids_d.defined()) {
            fallback_logits = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_compact")
                                  .attr("materialize")(params.target_logits_d,
                                                       params.target_candidate_ids_d,
                                                       params.target_dense_fallback_d,
                                                       params.target_fallback_rows_d)
                                  .cast<torch::Tensor>();
        }
        pyDeferredRejectionFn()(fallback_logits,
                                draft,
                                params.draft_token_ids_d,
                                seeds,
                                positions,
                                params.do_sample_d,
                                params.force_accept,
                                cap,
                                cumulative,
                                params.output_token_ids_d,
                                params.output_accepted_token_num_d,
                                true);
        check_cuda_error();
        return;
    }
    auto config = validateRejectionSamplingParams(params);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    check_cuda_value(dispatchRejectionSampling(params, config, stream));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        auto* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();
        if (params.tokens.dtype() == torch::kInt32) {
            auto* tokens_ptr = params.tokens.data_ptr<int32_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = static_cast<int32_t>(d2t_map_ptr[token_id]);
                    }
                }
            }
        } else if (params.tokens.dtype() == torch::kInt64) {
            auto* tokens_ptr = params.tokens.data_ptr<int64_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = d2t_map_ptr[token_id];
                    }
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else if (params.tokens.is_cuda() && params.d2t_map.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (params.tokens.dtype() == torch::kInt32) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int32_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else if (params.tokens.dtype() == torch::kInt64) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int64_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "tokens and d2t_map must be on the same device");
    }
}

#elif USING_ROCM

}  // namespace rtp_llm — temporarily close for includes

// Forward-declare penalty kernels to avoid transitive amd_bfloat16.h breakage
// (sampling_penalty_kernels.h → cuda_shims.h → amd_bfloat16.h is broken in .cc on ROCm 6.4.3 + clang 19)
namespace rtp_llm {
template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        int          batch_size,
                                        int          vocab_size,
                                        int          vocab_size_padd,
                                        hipStream_t  stream);
template<typename T>
void invokeBatchApplyRepetitionPenalty(T*           logits,
                                       int*         penalty_ws,
                                       const float* repetition_penalty,
                                       const float* presence_penalty,
                                       const float* frequency_penalty,
                                       const int*   output_ids,
                                       int          batch_size,
                                       int          local_batch_size,
                                       int          vocab_size,
                                       const int*   input_lengths,
                                       int          max_input_length,
                                       int          step,
                                       hipStream_t  stream);

template<typename IdType>
void invokeMappingDraft2Target(IdType*     tokens,
                               int         batch_size,
                               int         token_offset,
                               int         token_stride,
                               int64_t*    d2t_map,
                               int         d2t_map_size,
                               hipStream_t stream);
}  // namespace rtp_llm

#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/kernels/sampling/sampling.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"

namespace rtp_llm {  // reopen

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size         = params.logits.size(0);
    const auto vocab_size_padded  = params.logits.size(1);
    const auto step               = params.step;
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    auto       cur_stream         = at::hip::getCurrentHIPStream().stream();

    // [batch_size, step + 1] — clone to GPU
    // On ROCm, hipMemcpyAsync from pageable memory is truly async (unlike CUDA where it
    // falls back to sync). Use blocking transfer to avoid memory access faults.
    auto device_tokens = params.token_ids.to(torch::kCUDA);
    // [step + 1, batch_size]
    auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();

    // 1. Apply temperature penalty
    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
    }

    // 2. Apply repetition/presence/frequency penalty
    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        if (std::any_of(repetition_penalty.data_ptr<float>(),
                        repetition_penalty.data_ptr<float>() + batch_size,
                        [&](auto t) { return t != 1.0f; })
            || std::any_of(presence_penalty.data_ptr<float>(),
                           presence_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })
            || std::any_of(frequency_penalty.data_ptr<float>(),
                           frequency_penalty.data_ptr<float>() + batch_size,
                           [&](auto t) { return t != 0.0f; })) {
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA));
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
                                           torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            auto repetition_penalty_gpu = repetition_penalty.to(torch::kCUDA);
            auto presence_penalty_gpu   = presence_penalty.to(torch::kCUDA);
            auto frequency_penalty_gpu  = frequency_penalty.to(torch::kCUDA);
            invokeBatchApplyRepetitionPenalty(params.logits.data_ptr<float>(),
                                              penalty_ws.data_ptr<int32_t>(),
                                              repetition_penalty_gpu.data_ptr<float>(),
                                              presence_penalty_gpu.data_ptr<float>(),
                                              frequency_penalty_gpu.data_ptr<float>(),
                                              transposed_tokens.data_ptr<int32_t>(),
                                              batch_size,
                                              batch_size,  // local_batch_size
                                              vocab_size_padded,
                                              sequence_lengths_gpu.data_ptr<int32_t>(),
                                              step + 1,  // max_input_length
                                              step + 1,  // step
                                              cur_stream);
        }
    }

    // 3. Fast path for topk = 1
    auto       top_k_ptr     = params.top_k.data_ptr<int32_t>();
    const bool all_top_k_one = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; });
    if (all_top_k_one && !params.cum_log_probs.has_value() && !params.return_original_all_probs
        && (!params.output_all_probs.has_value() || params.output_all_probs->size(1) == vocab_size_padded)) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor selected_tokens = torch::argmax(params.logits, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (params.output_all_probs.has_value()) {
            params.output_all_probs->zero_().scatter_(1, selected_tokens.unsqueeze(1), 1.0);
        }

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);

        return GreedyOutput{};
    }

    // 4. Compute softmax probabilities
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // 5. Prepare sampling parameters
    constexpr bool deterministic = true;
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).flatten();
    auto top_p_ptr = params.top_p.data_ptr<float>();

    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }
    const auto output_width = output_all_probs_t.defined() ? output_all_probs_t.size(1) : probs_t.size(1);
    RTP_LLM_CHECK_WITH_INFO(output_width > 0 && output_width <= probs_t.size(1),
                            "output probability width must be in (0, padded vocab size]");
    const bool need_full_sampling_probs = output_all_probs_t.defined() && output_width < probs_t.size(1);
    auto       sampling_probs_t         = need_full_sampling_probs ? torch::zeros_like(probs_t) : output_all_probs_t;
    const bool need_renorm_probs        = sampling_probs_t.defined() && !params.return_original_all_probs;
    if (params.return_original_all_probs && sampling_probs_t.defined()) {
        sampling_probs_t.copy_(probs_t);
    }

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });
    const bool all_top_k_no_limit = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
    const bool all_top_p_one =
        std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });

    // The ROCm sampling kernels consume these arrays on the GPU. Passing the pinned host
    // tensors directly happens to work on some single-GPU setups, but can fault when the
    // active device is not the one which mapped the host allocation. Stage every array on
    // the logits device before issuing batched sampling launches.
    if (!all_top_k_one && !all_top_k_no_limit) {
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [](auto t) { return t <= 0 ? 1 << 30 : t; });
    }
    auto top_k_t = params.top_k.to(params.logits.device(), /*non_blocking=*/true).contiguous();
    auto top_p_t = params.top_p.to(params.logits.device(), /*non_blocking=*/true).contiguous();

    // 6. Sample. Kernels write the per-row status directly, without a comparison launch.
    auto success =
        all_top_k_one ? torch::Tensor() : torch::empty({(int64_t)batch_size}, probs_t.options().dtype(torch::kBool));
    if (all_top_k_one) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_renorm_probs) {
            top_k_renorm_probs(probs_t, sampling_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
        }
    } else {
        const bool    use_persistent_buffers = params.sampling_buffers != nullptr;
        auto          host_options           = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
        torch::Tensor seed_h;
        torch::Tensor offset_h;
        if (use_persistent_buffers) {
            RTP_LLM_CHECK_WITH_INFO(params.sampling_buffers->seed_host.numel() >= batch_size,
                                    "sampling seed buffer capacity is smaller than batch size");
            RTP_LLM_CHECK_WITH_INFO(params.sampling_buffers->offset_host.numel() >= batch_size,
                                    "sampling offset buffer capacity is smaller than batch size");
            seed_h   = params.sampling_buffers->seed_host.narrow(0, 0, batch_size);
            offset_h = params.sampling_buffers->offset_host.narrow(0, 0, batch_size);
        } else {
            seed_h   = torch::empty({(int64_t)batch_size}, host_options);
            offset_h = torch::empty({(int64_t)batch_size}, host_options);
        }
        constexpr int max_sampling_rounds = 32;
        for (int64_t i = 0; i < (int64_t)batch_size; ++i) {
            auto generator      = i < static_cast<int64_t>(params.generator.size()) && params.generator[i].defined() ?
                                      std::make_optional(params.generator[i]) :
                                      std::nullopt;
            auto [seed, offset] = get_seed_and_offset(max_sampling_rounds, generator);
            seed_h.data_ptr<int64_t>()[i]   = static_cast<int64_t>(seed);
            offset_h.data_ptr<int64_t>()[i] = static_cast<int64_t>(offset);
        }
        auto seed_t   = seed_h.to(params.logits.device(), /*non_blocking=*/true).contiguous();
        auto offset_t = offset_h.to(params.logits.device(), /*non_blocking=*/true).contiguous();

        if (all_top_k_no_limit) {
            top_p_sampling_from_probs(probs_t,
                                      samples_t,
                                      std::nullopt,
                                      top_p_t,
                                      1.0,
                                      deterministic,
                                      seed_t,
                                      offset_t,
                                      reinterpret_cast<uintptr_t>(cur_stream),
                                      success);
            if (need_renorm_probs) {
                top_p_renorm_probs(probs_t, sampling_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(cur_stream));
            }
        } else if (all_top_p_one) {
            top_k_sampling_from_probs(probs_t,
                                      samples_t,
                                      std::nullopt,
                                      top_k_t,
                                      0,
                                      deterministic,
                                      seed_t,
                                      offset_t,
                                      reinterpret_cast<uintptr_t>(cur_stream),
                                      success);
            if (need_renorm_probs) {
                top_k_renorm_probs(probs_t, sampling_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
            }
        } else {
            // Match the CUDA kernel's joint top-k/top-p semantics. Construct the
            // corresponding normalized intersection only when the caller needs it.
            top_k_top_p_sampling_from_probs(probs_t,
                                            samples_t,
                                            std::nullopt,
                                            top_k_t,
                                            0,
                                            top_p_t,
                                            1.0,
                                            deterministic,
                                            seed_t,
                                            offset_t,
                                            reinterpret_cast<uintptr_t>(cur_stream),
                                            success);
            if (need_renorm_probs) {
                auto top_k_probs_t = torch::zeros_like(sampling_probs_t);
                top_k_renorm_probs(probs_t, top_k_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
                top_p_renorm_probs(probs_t, sampling_probs_t, top_p_t, 1.0, reinterpret_cast<uintptr_t>(cur_stream));
                sampling_probs_t.masked_fill_(top_k_probs_t.eq(0), 0.0);
                sampling_probs_t.div_(sampling_probs_t.sum(-1, /*keepdim=*/true).clamp_min_(1e-10));
            }
        }
    }

    torch::Tensor selected_log_probs;
    if (success.defined() && sampling_probs_t.defined()) {
        if (params.cum_log_probs.has_value()) {
            selected_log_probs = torch::empty({(int64_t)batch_size}, probs_t.options());
        }
        // Healthy rows skip the vocabulary entirely unless a selected log probability
        // is needed. Failed rows are zeroed and never gathered through their -1 token.
        finalize_sampling_probs(sampling_probs_t,
                                samples_t,
                                success,
                                selected_log_probs.defined() ? std::make_optional(selected_log_probs) : std::nullopt,
                                reinterpret_cast<uintptr_t>(cur_stream));
    }

    if (need_full_sampling_probs) {
        output_all_probs_t.copy_(sampling_probs_t.slice(1, 0, output_width));
    }

    // 7. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        if (!selected_log_probs.defined()) {
            selected_log_probs = sampling_probs_t.gather(1, samples_t.to(torch::kLong).unsqueeze(1)).squeeze(1).log();
        }
        cum_log_probs_t.add_(selected_log_probs.to(cum_log_probs_t.device()));
    }

    // 8. Copy results back
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);
    return {success};
}

torch::Tensor sampleFromProbs(const torch::Tensor& probabilities) {
    RTP_LLM_CHECK_WITH_INFO(probabilities.defined() && probabilities.is_cuda() && probabilities.dim() == 2
                                && probabilities.scalar_type() == torch::kFloat32 && probabilities.is_contiguous(),
                            "probability sampling expects contiguous CUDA/HIP FP32 [batch,vocab] probabilities");

    // torch::multinomial dispatches to the active HIP stream and uses PyTorch's
    // process-local default generator. The regular ROCm sampler above uses the
    // same fallback because FlashInfer sampling is not stable for TP > 1.
    return torch::multinomial(probabilities, 1, /*replacement=*/false).squeeze(-1).to(torch::kInt32);
}

}  // namespace rtp_llm

// NOTE: intentionally at global scope (not inside namespace rtp_llm) because the
// ROCm kernel in rtp_llm/models_py/bindings/rocm/speculative_sampling/sampling.cu
// is also defined at global scope; the call site below uses ::invokeRejectionSampling.
template<typename DType, typename IdType>
hipError_t invokeRejectionSampling(DType*      draft_probs,
                                   IdType*     draft_token_ids,
                                   DType*      uniform_samples,
                                   DType*      target_probs,
                                   IdType*     target_token_ids,
                                   int         target_token_stride,
                                   IdType*     output_token_ids,
                                   IdType*     output_accepted_token_num,
                                   bool*       do_sample,
                                   int         batch_size,
                                   int         num_speculative_tokens,
                                   int         target_vocab_size,
                                   hipStream_t stream,
                                   bool        draft_probs_point_mass);

namespace rtp_llm {

void rejectionSampling(const RejectionSamplingParams& params) {
    auto config = validateRejectionSamplingParams(params);
    auto stream = at::hip::getCurrentHIPStream().stream();

    // A point-mass draft distribution passes no draft_probs buffer at all; the
    // kernel substitutes q(draft_token) == 1.
    float*     draft_probs = params.draft_probs_point_mass ? nullptr : params.draft_probs_d.data_ptr<float>();
    hipError_t err         = ::invokeRejectionSampling(draft_probs,
                                               params.draft_token_ids_d.data_ptr<int32_t>(),
                                               params.uniform_samples_d.data_ptr<float>(),
                                               params.target_probs_d.data_ptr<float>(),
                                               params.target_token_ids_d.data_ptr<int32_t>(),
                                               config.target_token_stride,
                                               params.output_token_ids_d.data_ptr<int32_t>(),
                                               params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                               params.do_sample_d.data_ptr<bool>(),
                                               config.batch_size,
                                               config.num_speculative_tokens,
                                               config.target_vocab_size,
                                               stream,
                                               params.draft_probs_point_mass);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess, "invokeRejectionSampling failed: %s", hipGetErrorString(err));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        auto* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();
        if (params.tokens.dtype() == torch::kInt32) {
            auto* tokens_ptr = params.tokens.data_ptr<int32_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = static_cast<int32_t>(d2t_map_ptr[token_id]);
                    }
                }
            }
        } else if (params.tokens.dtype() == torch::kInt64) {
            auto* tokens_ptr = params.tokens.data_ptr<int64_t>();
            for (int i = 0; i < params.batch_size; ++i) {
                for (int j = params.token_offset; j < params.token_stride; ++j) {
                    const int     index    = i * params.token_stride + j;
                    const int64_t token_id = tokens_ptr[index];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[index] = d2t_map_ptr[token_id];
                    }
                }
            }
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else if (params.tokens.is_cuda() && params.d2t_map.is_cuda()) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        if (params.tokens.dtype() == torch::kInt32) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int32_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else if (params.tokens.dtype() == torch::kInt64) {
            invokeMappingDraft2Target(params.tokens.data_ptr<int64_t>(),
                                      params.batch_size,
                                      params.token_offset,
                                      params.token_stride,
                                      params.d2t_map.data_ptr<int64_t>(),
                                      d2t_map_size,
                                      stream);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "Unsupported token dtype for mappingDraft2Target");
        }
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "tokens and d2t_map must be on the same device");
    }
}

#else  // Generic PyTorch fallback

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size        = params.logits.size(0);
    const auto vocab_size_padded = params.logits.size(1);
    const auto device            = params.logits.device();

    auto          device_tokens     = params.token_ids.to(device);
    auto          transposed_tokens = device_tokens.transpose(0, 1).contiguous();
    torch::Tensor original_logits;
    torch::Tensor do_sample_device;
    if (params.do_sample.has_value()) {
        RTP_LLM_CHECK_WITH_INFO(params.do_sample.value().numel() == batch_size,
                                "do_sample length must match batch size");
        do_sample_device = params.do_sample.value().to(device).reshape({batch_size, 1});
        if (!params.do_sample.value().all().item<bool>()) {
            original_logits = params.logits.clone();
        }
    }

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [](auto temperature) { return temperature != 1.0f; })) {
        auto temperature = params.temperature.to(device).reshape({batch_size, 1});
        params.logits.div_(temperature);
    }

    if (params.repetition_penalty.has_value()) {
        RTP_LLM_CHECK(params.presence_penalty.has_value() && params.frequency_penalty.has_value());
        const auto& repetition_penalty = params.repetition_penalty.value();
        const auto& presence_penalty   = params.presence_penalty.value();
        const auto& frequency_penalty  = params.frequency_penalty.value();
        const bool  has_penalty        = std::any_of(repetition_penalty.data_ptr<float>(),
                                             repetition_penalty.data_ptr<float>() + batch_size,
                                             [](auto penalty) { return penalty != 1.0f; })
                                 || std::any_of(presence_penalty.data_ptr<float>(),
                                                presence_penalty.data_ptr<float>() + batch_size,
                                                [](auto penalty) { return penalty != 0.0f; })
                                 || std::any_of(frequency_penalty.data_ptr<float>(),
                                                frequency_penalty.data_ptr<float>() + batch_size,
                                                [](auto penalty) { return penalty != 0.0f; });
        if (has_penalty) {
            auto       effective_lengths  = params.input_lengths.clone();
            const auto decoder_batch_size = params.sequence_lengths.size(0);
            if (decoder_batch_size > 0) {
                effective_lengths.narrow(0, 0, decoder_batch_size).copy_(params.sequence_lengths);
            }

            auto history = device_tokens.slice(1, 0, params.step + 1).to(torch::kLong);
            auto positions =
                torch::arange(history.size(1), torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(0);
            auto lengths = effective_lengths.to(device).to(torch::kLong).unsqueeze(1);
            auto valid   = positions.lt(lengths).logical_and(history.ge(0)).logical_and(history.lt(vocab_size_padded));
            auto safe_history = history.clamp(0, vocab_size_padded - 1);
            auto counts       = torch::zeros({batch_size, vocab_size_padded},
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));
            counts.scatter_add_(1, safe_history, valid.to(torch::kInt32));

            auto seen                 = counts.gt(0);
            auto repetition_penalty_d = repetition_penalty.to(device).reshape({batch_size, 1});
            auto repeated_logits      = torch::where(
                params.logits.lt(0), params.logits * repetition_penalty_d, params.logits / repetition_penalty_d);
            params.logits.copy_(torch::where(seen, repeated_logits, params.logits));
            params.logits.sub_(presence_penalty.to(device).reshape({batch_size, 1}) * seen.to(params.logits.dtype())
                               + frequency_penalty.to(device).reshape({batch_size, 1})
                                     * counts.to(params.logits.dtype()));
        }
    }

    const auto decoder_batch_size = params.sequence_lengths.size(0);
    if (decoder_batch_size > 0 && params.no_repeat_ngram_size.has_value()) {
        RTP_LLM_CHECK_WITH_INFO(params.no_repeat_ngram_size.value().numel() >= decoder_batch_size,
                                "no_repeat_ngram_size length is smaller than decoder batch size");
        RTP_LLM_CHECK_WITH_INFO(params.sequence_lengths.device().is_cpu()
                                    && params.no_repeat_ngram_size.value().device().is_cpu()
                                    && device_tokens.device().is_cpu(),
                                "generic no-repeat n-gram fallback requires CPU tensors");
        auto       sequence_lengths_ptr = params.sequence_lengths.data_ptr<int32_t>();
        auto       ngram_sizes_ptr      = params.no_repeat_ngram_size.value().data_ptr<int32_t>();
        auto       tokens_ptr           = device_tokens.data_ptr<int32_t>();
        const auto token_stride         = device_tokens.size(1);

        for (int64_t batch = 0; batch < decoder_batch_size; ++batch) {
            const int64_t ngram_size = ngram_sizes_ptr[batch];
            // sequence_lengths already counts valid history tokens; the final column is the output slot.
            const int64_t length = sequence_lengths_ptr[batch];
            RTP_LLM_CHECK_WITH_INFO(length >= 0 && length < token_stride,
                                    "sequence length is outside the token history");
            if (ngram_size <= 0 || length < ngram_size) {
                continue;
            }
            const auto* row = tokens_ptr + batch * token_stride;
            for (int64_t start = 0; start <= length - ngram_size; ++start) {
                bool matches_suffix = true;
                for (int64_t offset = 0; offset < ngram_size - 1; ++offset) {
                    if (row[start + offset] != row[length - ngram_size + 1 + offset]) {
                        matches_suffix = false;
                        break;
                    }
                }
                if (matches_suffix) {
                    const auto banned_token = row[start + ngram_size - 1];
                    if (banned_token >= 0 && banned_token < vocab_size_padded) {
                        auto logits_row = params.logits.select(0, batch);
                        logits_row.index_put_({banned_token}, -std::numeric_limits<float>::infinity());
                    }
                }
            }
        }
    }

    if (original_logits.defined()) {
        params.logits.copy_(torch::where(do_sample_device, params.logits, original_logits));
    }

    auto top_k_ptr = params.top_k.data_ptr<int32_t>();
    // TODO: Support output_all_probs and cum_log_probs in this top-k=1 fast path.
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto top_k) { return top_k == 1; })
        && !params.output_all_probs.has_value() && !params.cum_log_probs.has_value()) {
        auto samples = transposed_tokens.select(0, transposed_tokens.size(0) - 1);
        samples.copy_(torch::argmax(params.logits, -1, /*keepdim=*/false));
        params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
        return GreedyOutput{};
    }

    auto probs = torch::softmax(params.logits, -1);
    params.logits.copy_(probs);
    auto filtered_probs = probs.clone();
    auto top_p_ptr      = params.top_p.data_ptr<float>();
    std::transform(
        top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [](auto top_p) { return std::abs(top_p) < 1e-7 ? 1.0f : top_p; });

    int64_t max_bounded_top_k = 0;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t top_k = top_k_ptr[batch];
        if (top_k > 0 && top_k < vocab_size_padded) {
            max_bounded_top_k = std::max(max_bounded_top_k, top_k);
        }
    }
    if (max_bounded_top_k > 0) {
        auto top_k_device = params.top_k.to(device).to(torch::kLong);
        auto bounded_rows = top_k_device.gt(0).logical_and(top_k_device.lt(vocab_size_padded));
        auto row_indices  = torch::nonzero(bounded_rows).squeeze(1);
        auto active_probs = filtered_probs.index_select(0, row_indices);
        auto active_top_k = top_k_device.index_select(0, row_indices);

        // PyTorch topk accepts one k for the whole tensor. Compute the largest requested
        // k once, then gather each row's actual kth value and filter the active rows in one
        // batched operation. Using a threshold preserves the old fallback's tie behavior.
        auto topk_values   = std::get<0>(active_probs.topk(max_bounded_top_k, /*dim=*/-1));
        auto kth_positions = active_top_k.sub(1).unsqueeze(1);
        auto kth_values    = topk_values.gather(1, kth_positions);
        active_probs.masked_fill_(active_probs.lt(kth_values), 0.0f);
        filtered_probs.index_copy_(0, row_indices, active_probs);
    }

    const bool has_top_p =
        !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto top_p) { return std::abs(top_p - 1.0f) < 1e-7; });
    if (has_top_p) {
        auto top_p_device = params.top_p.to(device);
        auto active_rows  = top_p_device.sub(1.0f).abs().ge(1e-7);
        auto row_indices  = torch::nonzero(active_rows).squeeze(1);
        auto active_probs = filtered_probs.index_select(0, row_indices);
        auto active_top_p = top_p_device.index_select(0, row_indices).unsqueeze(1);

        auto [sorted_probs, sorted_indices] = active_probs.sort(/*dim=*/-1, /*descending=*/true);
        auto cumulative_probs               = sorted_probs.cumsum(-1);
        auto outside_nucleus                = cumulative_probs - sorted_probs > active_top_p;
        sorted_probs.masked_fill_(outside_nucleus, 0.0f);
        active_probs.zero_().scatter_(1, sorted_indices, sorted_probs);
        filtered_probs.index_copy_(0, row_indices, active_probs);
    }

    auto row_sums = filtered_probs.sum(-1, /*keepdim=*/true);
    filtered_probs.div_(row_sums.clamp_min(1e-10));
    auto selected = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kLong).device(device));
    // TODO: Use batched multinomial sampling when no per-request generator is specified.
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        c10::optional<at::Generator> generator = c10::nullopt;
        if (batch < static_cast<int64_t>(params.generator.size()) && params.generator[batch].defined()) {
            generator = params.generator[batch];
        }
        auto sample = at::multinomial(filtered_probs.select(0, batch),
                                      /*num_samples=*/1,
                                      /*replacement=*/false,
                                      generator);
        selected.select(0, batch).copy_(sample.select(0, 0));
    }
    transposed_tokens.select(0, transposed_tokens.size(0) - 1).copy_(selected);

    torch::Tensor probability_output;
    if (params.output_all_probs.has_value()) {
        probability_output = params.output_all_probs.value();
    } else if (params.cum_log_probs.has_value()) {
        probability_output = torch::empty_like(probs);
    }
    if (probability_output.defined()) {
        probability_output.copy_(params.return_original_all_probs ? probs : filtered_probs);
    }
    if (params.cum_log_probs.has_value()) {
        auto selected_probs = probability_output.gather(1, selected.to(torch::kLong).unsqueeze(1)).squeeze(1);
        params.cum_log_probs.value().add_(selected_probs.log());
    }

    params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
    return GreedyOutput{};
}

void rejectionSampling(const RejectionSamplingParams&) {
    RTP_LLM_CHECK_WITH_INFO(false, "rejection sampling is not supported on this device");
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");
    RTP_LLM_CHECK_WITH_INFO(params.tokens.device() == params.d2t_map.device(),
                            "mappingDraft2Target: tokens and d2t_map must be on the same device");

    auto token_rows    = params.tokens.reshape({-1, params.token_stride});
    auto active_tokens = token_rows.narrow(0, 0, params.batch_size).slice(1, params.token_offset, params.token_stride);
    auto valid_tokens  = active_tokens.ge(0).logical_and(active_tokens.lt(params.d2t_map.numel()));
    auto safe_tokens   = active_tokens.to(torch::kLong).clamp(0, params.d2t_map.numel() - 1);
    auto mapped_tokens = params.d2t_map.index_select(0, safe_tokens.flatten()).reshape_as(active_tokens);
    active_tokens.copy_(torch::where(valid_tokens, mapped_tokens.to(active_tokens.dtype()), active_tokens));
}

#endif  // USING_CUDA / USING_ROCM / generic fallback

bool supportsDeferredRejectionSampling() {
#if USING_CUDA
    return samplingRuntimeReady();
#else
    return false;
#endif
}

void initializeSamplingRuntime(int64_t vocab_size) {
#if USING_CUDA
    samplingRuntimeReady(vocab_size);
#endif
}

}  // namespace rtp_llm
