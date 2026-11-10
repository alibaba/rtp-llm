#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <torch/version.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/common/kernels/vocab_prune/mapping.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>
#include <memory>
#include <map>
#include <mutex>
#include <tuple>
#include <array>
#include <dlfcn.h>
#include <iterator>
#endif

using namespace std;

namespace rtp_llm {

#if USING_CUDA

using SamplerT = float;

namespace {

bool samplingEnabled(bool rejection) {
    const char* value = rejection ? std::getenv("RTP_LLM_USE_VLLM_REJECTION_SAMPLING") : nullptr;
    if (value == nullptr) {
        value = std::getenv("RTP_LLM_USE_GUMBEL_SAMPLING");
    }
    return value == nullptr || std::strcmp(value, "0") != 0;
}

struct SamplingRouteState {
    bool    initialized  = false;
    bool    ready        = false;
    int64_t warmed_vocab = 0;
};

bool samplingRuntimeReady(bool rejection, int64_t vocab_size = 0) {
    if (!Py_IsInitialized() || !samplingEnabled(rejection)) {
        return false;
    }
    pybind11::gil_scoped_acquire                              gil;
    static std::mutex                                         mutex;
    static std::map<std::pair<int, bool>, SamplingRouteState> states;
    std::unique_lock<std::mutex>                              guard(mutex, std::defer_lock);
    {
        pybind11::gil_scoped_release release;
        guard.lock();
    }
    auto& state = states[{at::cuda::current_device(), rejection}];
    if (!state.initialized || (state.ready && vocab_size > 0 && vocab_size != state.warmed_vocab)) {
        const int64_t requested_vocab = vocab_size > 0 ? vocab_size : 8193;
        std::string   failure;
        try {
            failure = pybind11::module_::import("rtp_llm.models_py.sampling.runtime")
                          .attr("warmup")(requested_vocab, rejection)
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
        RTP_LLM_LOG_INFO("sampling route=%s backend=%s warmup=%s",
                         rejection ? "mtp_rejection" : "ordinary_random",
                         state.ready ? "vllm" : "legacy",
                         state.ready ? "passed" : failure.c_str());
    }
    return state.ready;
}

// Cached handles to flashinfer.sampling Python entry points. The static
// initialization is guarded by C++17 thread-safe statics + GIL, and the
// caller must hold the GIL when invoking these helpers.
pybind11::object& pyTopKRenormFn() {
    static pybind11::object fn = pybind11::module_::import("flashinfer.sampling").attr("top_k_renorm_probs");
    return fn;
}

pybind11::object& pyTopPRenormFn() {
    static pybind11::object fn = pybind11::module_::import("flashinfer.sampling").attr("top_p_renorm_probs");
    return fn;
}

pybind11::object& pyGumbelSampleFn() {
    static pybind11::object fn =
        pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("sample");
    return fn;
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

pybind11::capsule registerDeferredGraphPlan(const pybind11::object& owner,
                                            uint64_t                graph_exec,
                                            uint64_t                ready_event,
                                            const torch::Tensor&    addresses,
                                            const DeferredPlanKey&  key) {
    TORCH_CHECK(graph_exec && ready_event && addresses.is_cuda(), "invalid deferred Graph registration");
    TORCH_CHECK(std::get<0>(key) == addresses.get_device(), "deferred Graph registration device mismatch");
    auto& plans = deferredGraphPlans();
    for (auto it = plans.begin(); it != plans.end();) {
        it = it->second.expired() ? plans.erase(it) : std::next(it);
    }
    constexpr size_t max_plan_keys = 64;
    if (plans.size() >= max_plan_keys && plans.find(key) == plans.end()) {
        plans.erase(plans.begin());
    }
    auto plan = std::make_shared<DeferredGraphPlan>(
        DeferredGraphPlan{pybind11::weakref(owner), graph_exec, ready_event, addresses});
    plans[key] = plan;
    return pybind11::capsule(new std::shared_ptr<DeferredGraphPlan>(std::move(plan)),
                             [](void* pointer) { delete static_cast<std::shared_ptr<DeferredGraphPlan>*>(pointer); });
}

std::optional<DeferredPlanKey> deferredSamplingLayout(const RejectionSamplingParams& params, bool capacity_bucket) {
    const auto& logits = params.target_logits_d;
    if (!logits.is_cuda() || logits.dim() != 3 || logits.scalar_type() != torch::kFloat32 || !logits.is_contiguous()) {
        return std::nullopt;
    }
    const auto batch = logits.size(0);
    const auto rows  = logits.size(1);
    const auto vocab = logits.size(2);
    if (batch < 8 || batch > 128 || rows < 2 || rows > 9) {
        return std::nullopt;
    }
    bool forced = false;
    if (!params.force_accept.empty()) {
        if (params.force_accept.size() != static_cast<size_t>(batch)) {
            return std::nullopt;
        }
        forced = params.force_accept.front();
        if (std::any_of(params.force_accept.begin(), params.force_accept.end(), [forced](bool value) {
                return value != forced;
            })) {
            return std::nullopt;
        }
    }
    const auto device  = logits.get_device();
    auto       matches = [device](const torch::Tensor& tensor, at::ScalarType dtype, c10::IntArrayRef shape) {
        return tensor.defined() && tensor.is_cuda() && tensor.get_device() == device && tensor.scalar_type() == dtype
               && tensor.sizes() == shape && tensor.is_contiguous();
    };
    const bool has_cap = params.acceptance_cap.defined();
    if (!matches(params.draft_token_ids_d, torch::kInt32, {batch, rows - 1})
        || !matches(params.do_sample_d, torch::kBool, {batch})
        || !matches(params.output_token_ids_d, torch::kInt32, {batch, rows})
        || !matches(params.output_accepted_token_num_d, torch::kInt32, {batch})
        || (has_cap && !matches(params.acceptance_cap, torch::kInt32, {batch}))
        || at::cuda::current_device() != device) {
        return std::nullopt;
    }
    const auto              stream = at::cuda::getCurrentCUDAStream(device);
    cudaStreamCaptureStatus capture_status;
    const auto              status = cudaStreamIsCapturing(stream.stream(), &capture_status);
    TORCH_CHECK(status == cudaSuccess, "deferred Graph capture query failed: ", cudaGetErrorString(status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return std::nullopt;
    }
    int64_t bucket = batch;
    if (capacity_bucket) {
        bucket = 8;
        while (bucket < batch) {
            bucket *= 2;
        }
    }
    return DeferredPlanKey{device, reinterpret_cast<uint64_t>(stream.stream()), forced, bucket, rows, vocab, has_cap};
}

bool tryReplayDeferredGraph(const RejectionSamplingParams& params,
                            const torch::Tensor&           seeds,
                            const torch::Tensor&           positions) {
    if (params.target_candidate_ids_d.defined())
        return false;
    const char* enabled = std::getenv("RTP_LLM_GUMBEL_REJECTION_GRAPH");
    if ((enabled && std::strcmp(enabled, "1") != 0) || !params.draft_point_mass || params.cum_log_probs.defined()) {
        return false;
    }
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

using DeferredKernelKey = std::tuple<int, uint64_t, bool, int64_t, int64_t, int64_t, bool, int64_t>;

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
    auto& plans = deferredKernelPlans();
    for (auto it = plans.begin(); it != plans.end();) {
        it = it->second.expired() ? plans.erase(it) : std::next(it);
    }
    if (plans.size() >= 64 && plans.find(key) == plans.end())
        plans.erase(plans.begin());
    plans[key] = plan;
    return pybind11::capsule(new std::shared_ptr<DeferredKernelPlan>(std::move(plan)),
                             [](void* pointer) { delete static_cast<std::shared_ptr<DeferredKernelPlan>*>(pointer); });
}

pybind11::object& pyDeferredRejectionFn();

bool trySubmitDeferredKernels(const RejectionSamplingParams& params,
                              const torch::Tensor&           seeds,
                              const torch::Tensor&           positions) {
    const char* enabled = std::getenv("RTP_LLM_GUMBEL_REJECTION_NATIVE");
    if ((enabled && std::strcmp(enabled, "1") != 0) || !params.draft_point_mass || params.cum_log_probs.defined()) {
        return false;
    }
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
        dense_vocab = dense.size(2);
    }
    const DeferredKernelKey      key = std::tuple_cat(*layout, std::make_tuple(dense_vocab));
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
    const std::vector<torch::Tensor> extra  = compact ? std::vector<torch::Tensor>{params.target_candidate_ids_d,
                                                                                   params.target_dense_fallback_d,
                                                                                   params.target_fallback_rows_d} :
                                                        std::vector<torch::Tensor>{};
    const auto                       stream = uploadDeferredInputs(
        plan->addresses, inputs, params.output_token_ids_d, params.output_accepted_token_num_d, extra);
    for (auto& kernel : plan->kernels)
        kernel.launch(params.target_logits_d.size(0), stream.stream());
    const auto status = cudaEventRecord(reinterpret_cast<cudaEvent_t>(plan->ready_event), stream.stream());
    TORCH_CHECK(status == cudaSuccess, "native sampler completion event failed: ", cudaGetErrorString(status));
    return true;
}

pybind11::object& pyDeferredRejectionFn() {
    static pybind11::object fn = []() -> pybind11::object {
        auto graph_module                   = pybind11::module_::import("rtp_llm.models_py.sampling.rejection_graph");
        graph_module.attr("_native_replay") = pybind11::cpp_function(&replayDeferredGraph);
        graph_module.attr("_register_native_plan") = pybind11::cpp_function(&registerDeferredGraphPlan);
        graph_module.attr("_try_native_guard") =
            pybind11::cpp_function([]() { return deferredGraphMutex().try_lock(); });
        graph_module.attr("_release_native_guard") = pybind11::cpp_function([]() { deferredGraphMutex().unlock(); });
        pybind11::module_::import("rtp_llm.models_py.sampling.rejection_native").attr("_register_plan") =
            pybind11::cpp_function(&registerDeferredKernelPlan);
        return pybind11::module_::import("rtp_llm.models_py.sampling.adapter").attr("rejection_sample_logits");
    }();
    return fn;
}

// Python FlashInfer returns a new tensor; copy back into renorm_probs to
// preserve the original in-place C++ contract.
void pyTopKRenormProbs(const torch::Tensor& probs, torch::Tensor& renorm_probs, const torch::Tensor& top_k) {
    pybind11::gil_scoped_acquire gil;
    auto                         result = pyTopKRenormFn()(probs, top_k).cast<torch::Tensor>();
    renorm_probs.copy_(result, /*non_blocking=*/true);
}

void pyTopPRenormProbs(const torch::Tensor& probs, torch::Tensor& renorm_probs, const torch::Tensor& top_p) {
    pybind11::gil_scoped_acquire gil;
    auto                         result = pyTopPRenormFn()(probs, top_p).cast<torch::Tensor>();
    renorm_probs.copy_(result, /*non_blocking=*/true);
}

torch::Tensor reserveSamplingSeedOffsets(const std::vector<at::Generator>& generators,
                                         int64_t                           batch_size,
                                         int                               increment,
                                         int64_t                           prefix = 0) {
    auto options  = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
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

std::pair<torch::Tensor, torch::Tensor> makeSamplingSeedOffsetTensors(const std::vector<at::Generator>& generators,
                                                                      int64_t                           batch_size,
                                                                      int                               increment,
                                                                      TensorHolder*                     buffer_holder) {
    auto state_h = reserveSamplingSeedOffsets(generators, batch_size, increment);
    auto state_d = state_h.to(torch::kCUDA, /*non_blocking=*/true);
    if (buffer_holder != nullptr) {
        buffer_holder->hold_host(state_h);
    }
    return {state_d.select(0, 0), state_d.select(0, 1)};
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

void processLogits(const GreedyParams&  params,
                   const torch::Tensor& device_tokens,
                   const torch::Tensor& transposed_tokens) {
    const auto vocab_size_padded  = params.logits.size(1);
    const auto decoder_batch_size = params.sequence_lengths.size(0);
    const auto batch_size         = params.logits.size(0);
    const auto step               = params.step;
    auto       cur_stream         = at::cuda::getCurrentCUDAStream().stream();

    if (std::any_of(params.temperature.data_ptr<float>(),
                    params.temperature.data_ptr<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; })) {
        auto temperature_gpu = params.temperature.to(torch::kCUDA, true);
        invokeBatchApplyTemperaturePenalty(params.logits.data_ptr<float>(),
                                           (float*)nullptr,  // embedding_bias
                                           temperature_gpu.data_ptr<float>(),
                                           batch_size,
                                           vocab_size_padded,
                                           vocab_size_padded,
                                           cur_stream);
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
            // Build sequence_lengths: clone input_lengths, then overwrite first decoder_batch_size entries
            auto sequence_lengths_gpu = params.input_lengths.to(torch::kCUDA, true);
            if (decoder_batch_size > 0) {
                auto dst_slice = sequence_lengths_gpu.slice(0, 0, decoder_batch_size);
                dst_slice.copy_(params.sequence_lengths.to(torch::kCUDA, true), true);
            }
            auto penalty_ws             = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size_padded},
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
            auto output_ids_ptrs =
                torch::empty({(int64_t)decoder_batch_size}, torch::TensorOptions().dtype(torch::kInt64)).pin_memory();
            for (int64_t i = 0; i < (int64_t)decoder_batch_size; i++) {
                output_ids_ptrs.data_ptr<int64_t>()[i] = (int64_t)(device_tokens.data_ptr<int32_t>() + i * (step + 1));
            }
            if (params.buffer_holder) {
                params.buffer_holder->hold_host(output_ids_ptrs);
            }
            auto output_ids_ptrs_gpu  = output_ids_ptrs.to(torch::kCUDA, true);
            auto sequence_lengths_gpu = params.sequence_lengths.to(torch::kCUDA, true);

            if (params.sequence_lengths_are_token_counts) {
                sequence_lengths_gpu = sequence_lengths_gpu - 1;
            }

            tensorrt_llm::kernels::invokeBanRepeatNgram(params.logits.data_ptr<float>(),
                                                        (int32_t const**)(output_ids_ptrs_gpu.data_ptr()),
                                                        nullptr,  // finished_buf
                                                        nullptr,  // parent_ids_buf
                                                        nullptr,  // batch_slot
                                                        sequence_lengths_gpu.data_ptr<int32_t>(),
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

static GreedyOutput
flashinferSampleGreedy(const GreedyParams& params, const torch::Tensor& transposed_tokens, bool mixed_batch = false) {
    const auto batch_size = params.logits.size(0);
    auto       cur_stream = at::cuda::getCurrentCUDAStream().stream();
    const bool compact_output = params.output_all_probs_indices.has_value();

    // [batch_size, vocab_size] — compute softmax probabilities.
    // Copy result back to logits to preserve the in-place behavior of the original kernel,
    // since callers may reuse the logits tensor across iterations.
    if (params.defer_token_sampling && Py_IsInitialized()) {
        const auto* requested_k = params.top_k.data_ptr<int32_t>();
        const auto* requested_p = params.top_p.data_ptr<float>();
        const auto  vocab       = params.logits.size(1);
        bool        unfiltered  = true;
        for (int64_t row = 0; row < batch_size; ++row) {
            // Greedy rows ignore TopP, including non-finite values, just as
            // the normalization below does before applying any filters.
            if (requested_k[row] != 1
                && !((requested_k[row] <= 0 || requested_k[row] >= vocab)
                     && (std::abs(requested_p[row]) < 1e-7f || std::abs(requested_p[row] - 1.0f) < 1e-7f))) {
                unfiltered = false;
                break;
            }
        }
        if (unfiltered) {
            return GreedyOutput{torch::Tensor(), true};
        }
        const char* filter_env  = std::getenv("RTP_LLM_GUMBEL_LOGIT_FILTER");
        const auto  normalize_p = [](float value) {
            return std::abs(value) < 1e-7f || std::abs(value - 1.0f) < 1e-7f ? 1.0f : value;
        };
        const float compact_p = normalize_p(requested_p[0]);
        if (params.allow_compact_deferred_output && (!filter_env || std::string(filter_env) == "1")
            && requested_k[0] == 1024
            && std::all_of(requested_k, requested_k + batch_size, [](int32_t value) { return value == 1024; })
            && std::all_of(requested_p, requested_p + batch_size, [compact_p, normalize_p](float value) {
                   return normalize_p(value) == compact_p;
               })) {
            auto compact = torch_ext::deferred_compact_topk_topp(params.logits, 1024, compact_p);
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
                                                              11);
                }
                return GreedyOutput{torch::Tensor(),
                                    true,
                                    std::get<0>(compact),
                                    std::get<1>(compact),
                                    params.logits,
                                    std::get<2>(compact),
                                    prepared_rng};
            }
        }
        if ((!filter_env || std::string(filter_env) == "1") && requested_k[0] == 1024
            && std::all_of(requested_k, requested_k + batch_size, [](int32_t value) { return value == 1024; })
            && std::all_of(requested_p,
                           requested_p + batch_size,
                           [compact_p, normalize_p](float value) { return normalize_p(value) == compact_p; })
            && torch_ext::deferred_topk_topp_filter(params.logits, 1024, compact_p)) {
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
        const bool no_k =
            std::all_of(k, k + batch_size, [vocab](int32_t value) { return value <= 0 || value >= vocab; });
        const bool no_p = std::all_of(p, p + batch_size, [](float value) { return std::abs(value - 1.0f) < 1e-7f; });
        if (no_k && no_p) {
            return GreedyOutput{torch::Tensor(), true};
        }
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
    const auto* requested_k  = params.top_k.data_ptr<int32_t>();
    const auto* requested_p  = params.top_p.data_ptr<float>();
    const bool  greedy_batch = std::all_of(requested_k, requested_k + batch_size, [](int32_t k) { return k == 1; });
    const bool  mixed =
        std::any_of(requested_k, requested_k + batch_size, [&](int32_t k) { return k != requested_k[0]; })
        || std::any_of(requested_p, requested_p + batch_size, [&](float p) { return p != requested_p[0]; });
    const bool pure_top_p =
        std::all_of(requested_k, requested_k + batch_size, [](int32_t k) { return k <= 0; })
        && std::any_of(requested_p, requested_p + batch_size, [](float p) { return p > 1e-7f && p < 1.0f - 1e-7f; });
    const int force_topk = params.force_topk > 0 && params.force_topk < params.logits.size(1)
                                   && !params.cum_log_probs.has_value() && !params.output_log_probs.has_value()
                                   && !greedy_batch && (compact_output || mixed_batch || mixed || pure_top_p) ?
                               params.force_topk :
                               0;
    TORCH_CHECK(!compact_output
                    || (force_topk > 0 && params.output_all_probs.has_value() && params.output_all_probs->dim() == 2
                        && params.output_all_probs->size(0) == batch_size
                        && params.output_all_probs->size(1) == force_topk
                        && params.output_all_probs_indices->sizes() == params.output_all_probs->sizes()
                        && params.output_all_probs_indices->scalar_type() == torch::kInt32
                        && params.output_all_probs_indices->device() == params.logits.device()),
                "compact target probabilities require matching [batch, force_topk] probabilities and int32 IDs");
    torch::Tensor topk_indices;
    torch::Tensor probs_t;
    if (force_topk > 0) {
        auto selected = torch_ext::deepselect_fp32_topk(params.logits, force_topk);
        topk_indices  = std::get<1>(selected);
        probs_t       = torch::softmax(std::get<0>(selected), -1);
        if (!params.skip_logits_probs_restore)
            params.logits.zero_().scatter_(1, topk_indices, probs_t);
    } else {
        probs_t = torch::softmax(params.logits, -1);
        params.logits.copy_(probs_t, true);
    }
    auto success = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // [1, batch_size] — last row of transposed_tokens
    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0));

    constexpr bool       deterministic    = true;
    constexpr int        max_sampling_rounds = 32;
    auto [seed_t, offset_t] =
        makeSamplingSeedOffsetTensors(
            params.generator, batch_size, static_cast<int>(max_sampling_rounds), params.buffer_holder);

    torch::Tensor success_t = success;
    torch::Tensor top_k_t   = force_topk > 0 ? params.top_k.clone() : params.top_k;
    torch::Tensor top_p_t   = params.top_p;
    torch::Tensor output_all_probs_t;
    if (params.output_all_probs.has_value()) {
        output_all_probs_t =
            force_topk > 0 && !compact_output ? torch::empty_like(probs_t) : params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    // top_k/top_p are CPU tensors with int32/float32 dtype
    auto top_k_ptr = top_k_t.data_ptr<int32_t>();
    auto top_p_ptr = params.top_p.data_ptr<float>();

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });
    const bool all_top_k_one      = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t == 1; });
    const bool all_top_k_no_limit = std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
    const bool all_top_p_one =
        std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });
    auto renorm_top_p = [&](const torch::Tensor& input) {
        if (all_top_p_one) {
            output_all_probs_t.copy_(input);
            return;
        }
        top_p_renorm_probs(input, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        // AirTopP's histogram threshold can drop mass at p=1 due to reduction rounding.
        std::vector<int64_t> unfiltered_rows;
        for (int64_t row = 0; row < batch_size; ++row) {
            if (std::abs(top_p_ptr[row] - 1.0f) < 1e-7f)
                unfiltered_rows.push_back(row);
        }
        if (!unfiltered_rows.empty()) {
            auto rows = torch::tensor(unfiltered_rows, torch::kInt64).to(input.device());
            output_all_probs_t.index_copy_(0, rows, input.index_select(0, rows));
        }
    };

    if (force_topk > 0 || (params.use_optimized_sampling && !all_top_k_no_limit && !all_top_p_one)) {
        if (!output_all_probs_t.defined())
            output_all_probs_t = torch::empty_like(probs_t);
        if (all_top_k_no_limit) {
            renorm_top_p(probs_t);
        } else {
            const int limit = force_topk > 0 ? force_topk : static_cast<int>(probs_t.size(1));
            std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [limit](int32_t k) {
                return k <= 0 ? limit : std::min(k, limit);
            });
            if (all_top_p_one) {
                top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
            } else {
                auto temp = torch::empty_like(probs_t);
                top_k_renorm_probs(probs_t, temp, top_k_t, 0, (int64_t)cur_stream);
                renorm_top_p(temp);
            }
        }
        sampling_from_probs(
            output_all_probs_t, samples_t.squeeze(0), success_t, deterministic, seed_t, offset_t, (int64_t)cur_stream);
    } else if (all_top_k_one) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);
        success = torch::Tensor();  // mark as undefined — all succeeded
        if (output_all_probs_t.defined()) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
        }
    } else if (all_top_k_no_limit) {
        top_p_sampling_from_probs(probs_t,
                                  samples_t.squeeze(0),
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
        if (output_all_probs_t.defined()) {
            top_p_renorm_probs(probs_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
        }
    } else {
        // top_k<=0 means "no limit" in RTP config. The combined FlashInfer
        // kernel takes a top_k array, so normalize mixed batches after the
        // pure top-p route has been selected.
        std::transform(top_k_ptr, top_k_ptr + batch_size, top_k_ptr, [force_topk](int32_t k) {
            return force_topk > 0 ? (k <= 0 ? force_topk : std::min(k, force_topk)) : (k <= 0 ? 1 << 30 : k);
        });
        if (all_top_p_one) {
            top_k_sampling_from_probs(probs_t,
                                      samples_t.squeeze(0),
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
            if (output_all_probs_t.defined()) {
                top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, (int64_t)cur_stream);
            }
        } else {
            top_k_top_p_sampling_from_probs(probs_t,
                                            samples_t.squeeze(0),
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
            if (output_all_probs_t.defined()) {
                torch::Tensor temp_t = torch::zeros_like(output_all_probs_t);
                top_k_renorm_probs(probs_t, temp_t, top_k_t, 1.0, (int64_t)cur_stream);
                top_p_renorm_probs(temp_t, output_all_probs_t, top_p_t, 1.0, (int64_t)cur_stream);
            }
        }
    }

    if (force_topk > 0) {
        if (compact_output) {
            params.output_all_probs_indices->copy_(topk_indices);
        } else if (params.output_all_probs.has_value()) {
            params.output_all_probs->zero_().scatter_(1, topk_indices, output_all_probs_t);
        }
        auto candidate_ids = samples_t.transpose(0, 1).to(torch::kLong);
        auto valid         = candidate_ids.ge(0).logical_and(candidate_ids.lt(force_topk));
        auto token_ids     = topk_indices.gather(1, candidate_ids.clamp(0, force_topk - 1));
        samples_t.copy_(torch::where(valid, token_ids, torch::full_like(token_ids, -1)).transpose(0, 1));
    }

    if (params.cum_log_probs.has_value()) {

        // [batch_size]
        auto cum_log_probs_t = params.cum_log_probs.value();
        // [batch_size]
        auto token_probs_t     = output_all_probs_t.gather(1, samples_t.transpose(1, 0).to(torch::kLong)).squeeze(1);
        auto token_probs_t_log = token_probs_t.log();
        cum_log_probs_t.add_(token_probs_t_log.to(cum_log_probs_t.device()));
    }

    // Copy results back: transpose and copy to token_ids
    auto transposed_t = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(transposed_t, true);
    check_cuda_error();
    return {success};
}

GreedyOutput sampleGreedy(const GreedyParams& params) {
    const auto batch_size        = params.logits.size(0);
    bool       has_not_do_sample = params.do_sample.has_value()
                             && std::any_of(params.do_sample.value().data_ptr<bool>(),
                                            params.do_sample.value().data_ptr<bool>() + batch_size,
                                            [&](auto t) { return !t; });
    bool need_do_sample = (!params.do_sample.has_value())
                          || std::any_of(params.do_sample.value().data_ptr<bool>(),
                                         params.do_sample.value().data_ptr<bool>() + batch_size,
                                         [&](auto t) { return t; });
    const bool    deferred = params.defer_token_sampling && Py_IsInitialized();
    torch::Tensor device_tokens;
    torch::Tensor transposed_tokens;
    if (!deferred || (need_do_sample && requiresTokenHistory(params))) {
        device_tokens     = params.token_ids.to(torch::kCUDA, true);
        transposed_tokens = device_tokens.transpose(0, 1).contiguous();
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
        return flashinferSampleGreedy(params, transposed_tokens);
    }

    const auto* top_p_ptr = params.top_p.data_ptr<float>();
    if (params.use_optimized_sampling && std::any_of(top_p_ptr, top_p_ptr + batch_size, [](float p) {
            return !std::isfinite(p) || (p < 0 && std::abs(p) >= 1e-7f) || p > 1;
        })) {
        TORCH_CHECK(!params.output_all_probs_indices.has_value(), "invalid top_p for compact target probabilities");
        auto fallback                   = params;
        fallback.force_topk             = 0;
        fallback.use_optimized_sampling = false;
        return flashinferSampleGreedy(fallback, transposed_tokens);
    }
    const auto* top_k_ptr = params.top_k.data_ptr<int32_t>();
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [](int32_t k) { return k == 1; })
        && (params.use_optimized_sampling || !params.output_all_probs.has_value())) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens, true);
        if (params.output_all_probs.has_value()) {
            if (params.output_all_probs_indices.has_value()) {
                params.output_all_probs->zero_().select(1, 0).fill_(1.0);
                params.output_all_probs_indices->fill_(-1).select(1, 0).copy_(selected_tokens);
            } else {
                params.output_all_probs->zero_().scatter_(1, selected_tokens.unsqueeze(1), 1.0);
            }
        }

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens, true);

        return GreedyOutput{};
    }

    if (params.use_optimized_sampling
        && std::any_of(top_k_ptr, top_k_ptr + batch_size, [](int32_t k) { return k == 1; })) {
        std::vector<int64_t> greedy_rows, random_rows;
        for (int64_t row = 0; row < batch_size; ++row)
            (top_k_ptr[row] == 1 ? greedy_rows : random_rows).push_back(row);
        auto greedy_cpu = torch::tensor(greedy_rows, torch::kInt64);
        auto random_cpu = torch::tensor(random_rows, torch::kInt64);
        auto greedy_gpu = greedy_cpu.to(params.logits.device());
        auto random_gpu = random_cpu.to(params.logits.device());
        auto greedy_ids = torch::argmax(params.logits.index_select(0, greedy_gpu), -1, true);
        transposed_tokens.select(0, transposed_tokens.size(0) - 1)
            .index_copy_(0, greedy_gpu, greedy_ids.squeeze(1).to(transposed_tokens.scalar_type()));

        GreedyParams random_params = params;
        random_params.logits       = params.logits.index_select(0, random_gpu);
        random_params.token_ids    = device_tokens.index_select(0, random_gpu);
        random_params.top_k        = params.top_k.index_select(0, random_cpu);
        random_params.top_p        = params.top_p.index_select(0, random_cpu);
        random_params.generator.clear();
        if (!params.generator.empty()) {
            for (auto row : random_rows)
                random_params.generator.push_back(
                    row < static_cast<int64_t>(params.generator.size()) ? params.generator[row] : at::Generator{});
        }
        if (params.output_all_probs.has_value()) {
            const auto output_width = params.output_all_probs->size(1);
            auto       greedy_probs =
                torch::zeros({static_cast<int64_t>(greedy_rows.size()), output_width}, params.logits.options());
            if (params.output_all_probs_indices.has_value()) {
                greedy_probs.select(1, 0).fill_(1.0);
                auto greedy_candidates =
                    torch::full_like(greedy_probs, -1, greedy_probs.options().dtype(torch::kInt32));
                greedy_candidates.select(1, 0).copy_(greedy_ids.squeeze(1));
                params.output_all_probs_indices->index_copy_(0, greedy_gpu, greedy_candidates);
                random_params.output_all_probs_indices =
                    torch::empty({static_cast<int64_t>(random_rows.size()), output_width}, greedy_candidates.options());
            } else {
                greedy_probs.scatter_(1, greedy_ids, 1.0);
            }
            params.output_all_probs->index_copy_(0, greedy_gpu, greedy_probs);
            random_params.output_all_probs =
                torch::empty({static_cast<int64_t>(random_rows.size()), output_width}, params.logits.options());
        }
        if (params.cum_log_probs.has_value()) {
            const auto indices          = params.cum_log_probs->is_cuda() ? random_gpu : random_cpu;
            random_params.cum_log_probs = params.cum_log_probs->index_select(0, indices);
        }
        auto random_tokens = transposed_tokens.index_select(1, random_gpu);
        auto output        = flashinferSampleGreedy(random_params, random_tokens, true);
        transposed_tokens.index_copy_(1, random_gpu, random_tokens);
        params.token_ids.copy_(transposed_tokens.transpose(0, 1));
        if (!params.skip_logits_probs_restore)
            params.logits.index_copy_(0, random_gpu, random_params.logits);
        if (params.output_all_probs.has_value())
            params.output_all_probs->index_copy_(0, random_gpu, *random_params.output_all_probs);
        if (params.output_all_probs_indices.has_value())
            params.output_all_probs_indices->index_copy_(0, random_gpu, *random_params.output_all_probs_indices);
        if (params.cum_log_probs.has_value()) {
            const auto indices = params.cum_log_probs->is_cuda() ? random_gpu : random_cpu;
            params.cum_log_probs->index_copy_(0, indices, *random_params.cum_log_probs);
        }
        auto success = torch::ones({batch_size}, params.logits.options().dtype(torch::kBool));
        if (output.success.defined())
            success.index_copy_(0, random_gpu, output.success);
        return {success};
    }

    // Ordinary categorical sampling needs only a token ID. Keep probability
    // consumers (including stochastic speculative verification) and truncated
    // distributions on the existing FlashInfer path.
    const bool use_gumbel = samplingEnabled(false);
    const auto vocab_size = params.logits.size(1);

    const bool unfiltered =
        std::all_of(top_k_ptr, top_k_ptr + batch_size, [vocab_size](auto k) {
            return static_cast<int32_t>(k) <= 0 || k >= vocab_size;
        })
        && std::all_of(top_p_ptr, top_p_ptr + batch_size, [](float p) {
               // top_p=0 is the existing RTP "no cutoff" convention.
               return std::abs(p) < 1e-7f || std::abs(p - 1.0f) < 1e-7f;
           });
    if (use_gumbel && batch_size > 0 && unfiltered && !has_not_do_sample && !params.output_all_probs.has_value()
        && !params.cum_log_probs.has_value() && !params.output_log_probs.has_value()
        && params.logits.scalar_type() == torch::kFloat32 && params.logits.is_contiguous()
        && samplingRuntimeReady(false)) {
        auto samples = transposed_tokens.select(0, transposed_tokens.size(0) - 1);
        // Reserve a fresh generator offset on every call, as for FlashInfer.
        auto [seeds, offsets] = makeSamplingSeedOffsetTensors(params.generator, batch_size, 32, params.buffer_holder);
        {
            pybind11::gil_scoped_acquire gil;
            auto selected = pyGumbelSampleFn()(params.logits, seeds, offsets).cast<torch::Tensor>();
            samples.copy_(selected, true);
        }
        params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous(), true);
        check_cuda_error();
        return GreedyOutput{};
    }

    return flashinferSampleGreedy(params, transposed_tokens);
}

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    chain_speculative_sampling(params.draft_probs_d,
                               params.draft_token_ids_d,
                               params.uniform_samples_d,
                               params.target_probs_d,
                               params.output_token_ids_d,
                               params.output_accepted_token_num_d,
                               params.output_emitted_token_num_d,
                               true,
                               int64_t(stream));
}

void rejectionSampling(const RejectionSamplingParams& params) {
    if (params.deferred_token_sampling) {
        TORCH_CHECK(Py_IsInitialized(), "deferred verification requires the Python Gumbel runtime");
        const int64_t batch_size = params.target_logits_d.size(0);
        const int64_t rows       = params.target_logits_d.size(1);
        // The compact producer may already have reserved RNG while filtering
        // ran on the GPU. Native and fallback dispatch consume the same packet.
        const int64_t prefix  = params.target_candidate_ids_d.defined() ? 11 : 8;
        auto          state_h = params.prepared_rejection_rng;
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
        if (trySubmitDeferredKernels(params, state_h.select(0, 0), state_h.select(0, 1))
            || tryReplayDeferredGraph(params, state_h.select(0, 0), state_h.select(0, 1))) {
            check_cuda_error();
            return;
        }
        auto state_d = state_h.to(torch::kCUDA, /*non_blocking=*/true);
        if (params.buffer_holder != nullptr) {
            params.buffer_holder->hold_host(state_h);
        }
        auto             seeds     = state_d.select(0, 0);
        auto             positions = state_d.select(0, 1);
        pybind11::object draft = params.draft_point_mass ? pybind11::none() : pybind11::cast(params.draft_probs_d);
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
    RTP_LLM_CHECK(params.draft_probs_d.is_cuda());
    RTP_LLM_CHECK(params.draft_token_ids_d.is_cuda());
    RTP_LLM_CHECK(params.target_probs_d.is_cuda());

    int  batch_size             = params.draft_probs_d.size(0);
    int  num_speculative_tokens = params.draft_probs_d.size(1);
    int  target_vocab_size      = params.target_probs_d.size(2);
    int  target_token_stride    = params.target_token_ids_d.size(1);
    auto stream                 = at::cuda::getCurrentCUDAStream().stream();

    RTP_LLM_CHECK(params.draft_probs_d.dim() == 3);
    RTP_LLM_CHECK(params.draft_token_ids_d.dim() == 2);
    RTP_LLM_CHECK(params.target_probs_d.dim() == 3);
    RTP_LLM_CHECK(params.draft_token_ids_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.draft_token_ids_d.size(1) == num_speculative_tokens);
    RTP_LLM_CHECK(params.target_probs_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.target_probs_d.size(1) == num_speculative_tokens + 1);
    RTP_LLM_CHECK(params.draft_probs_d.size(2) == target_vocab_size);

    check_cuda_value(invokeRejectionSampling(params.draft_probs_d.data_ptr<float>(),
                                             params.draft_token_ids_d.data_ptr<int32_t>(),
                                             params.uniform_samples_d.data_ptr<float>(),
                                             params.target_probs_d.data_ptr<float>(),
                                             params.target_token_ids_d.data_ptr<int32_t>(),
                                             target_token_stride,
                                             params.output_token_ids_d.data_ptr<int32_t>(),
                                             params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                             params.do_sample_d.data_ptr<bool>(),
                                             batch_size,
                                             num_speculative_tokens,
                                             target_vocab_size,
                                             stream));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        int*     tokens_ptr  = params.tokens.data_ptr<int32_t>();
        int64_t* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();

        for (int i = 0; i < params.batch_size; i++) {
            for (int j = params.token_offset; j < params.token_stride; j++) {
                int idx      = i * params.token_stride + j;
                int token_id = tokens_ptr[idx];
                if (token_id >= 0 && token_id < d2t_map_size) {
                    tokens_ptr[idx] = d2t_map_ptr[token_id];
                }
            }
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

#else  // !USING_CUDA — ROCm platform

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

// Forward-declare vocab-prune mapping kernel (same reason — header pulls amd_bfloat16.h via cuda_shims.h)
template<typename IdType>
void invokeMappingDraft2Target(IdType*     tokens,
                               int         batch_size,
                               int         token_offset,
                               int         token_stride,
                               int64_t*    d2t_map,
                               int         d2t_map_size,
                               hipStream_t stream);
}  // namespace rtp_llm

// Forward-declare rejection-sampling kernel from rtp_llm/models_py/bindings/rocm/speculative_sampling/sampling.cu
// (same amd_bfloat16.h transitive-include hazard — kept out of the include graph).
// NOTE: This declaration is intentionally at global scope (not inside namespace rtp_llm)
// because the ROCm kernel implementation is also defined at global scope. The call site
// uses ::invokeRejectionSampling to match. This compiles and links correctly.
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
                                   hipStream_t stream);

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
    auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })
        && !params.output_all_probs.has_value()) {
        torch::Tensor samples_t =
            transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).squeeze(0);
        torch::Tensor probs_t         = params.logits;
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);

        auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
        params.token_ids.copy_(output_tokens);

        return GreedyOutput{};
    }

    // 4. Compute softmax probabilities
    auto probs_t = torch::softmax(params.logits, -1);
    params.logits.copy_(probs_t);

    // 5. Prepare sampling parameters
    constexpr bool deterministic = true;
    auto           seed_h        = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    auto           offset_h      = torch::empty({(int64_t)batch_size}, torch::TensorOptions().dtype(torch::kInt64));
    for (int64_t i = 0; i < (int64_t)batch_size; i++) {
        auto [sd, ofst] = get_seed_and_offset(
            batch_size * 32, params.generator[i].defined() ? std::make_optional(params.generator[i]) : std::nullopt);
        seed_h.data_ptr<int64_t>()[i]   = static_cast<int64_t>(sd);
        offset_h.data_ptr<int64_t>()[i] = static_cast<int64_t>(ofst);
    }

    auto samples_t = transposed_tokens.slice(0, transposed_tokens.size(0) - 1, transposed_tokens.size(0)).flatten();
    auto top_k_t   = params.top_k;
    auto top_p_t   = params.top_p;
    auto top_p_ptr = params.top_p.data_ptr<float>();

    bool          need_output_all_probs = params.output_all_probs.has_value();
    torch::Tensor output_all_probs_t;
    if (need_output_all_probs) {
        output_all_probs_t = params.output_all_probs.value();
    }
    if (params.cum_log_probs.has_value() && !output_all_probs_t.defined()) {
        output_all_probs_t = torch::zeros_like(probs_t);
    }

    std::transform(top_p_ptr, top_p_ptr + batch_size, top_p_ptr, [&](auto t) { return std::abs(t) < 1e-7 ? 1.0 : t; });

    // 6. Sample
    if (std::all_of(top_k_ptr, top_k_ptr + batch_size, [&](auto t) { return t == 1; })) {
        torch::Tensor selected_tokens = torch::argmax(probs_t, -1, /*keepdim=*/false);
        samples_t.copy_(selected_tokens);
        if (need_output_all_probs) {
            top_k_renorm_probs(probs_t, output_all_probs_t, top_k_t, 0, reinterpret_cast<uintptr_t>(cur_stream));
        }
    } else {
        // Use pure PyTorch sampling instead of FlashInfer ROCm kernels.
        // FlashInfer's ROCm sampling kernels (top_p_sampling_from_probs,
        // top_k_sampling_from_probs, top_k_top_p_sampling_from_probs) crash
        // with GPU memory access faults in multi-GPU (TP>1) configurations.
        // torch::multinomial is well-tested and handles all cases correctly.
        //
        // Apply top_k filtering if needed
        auto filtered_probs = probs_t;
        bool has_top_k      = !std::all_of(top_k_ptr, top_k_ptr + batch_size, [](auto t) { return t <= 0; });
        if (has_top_k) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                int k = top_k_ptr[b] <= 0 ? vocab_size_padded : top_k_ptr[b];
                if ((int64_t)k < vocab_size_padded) {
                    auto row                    = filtered_probs[b];
                    auto [topk_vals, topk_inds] = row.topk(k);
                    auto min_val                = topk_vals[-1];
                    row.masked_fill_(row < min_val, 0.0f);
                }
            }
        }
        // Apply top_p (nucleus) filtering if needed
        bool has_top_p =
            !std::all_of(top_p_ptr, top_p_ptr + batch_size, [](auto t) { return std::abs(t - 1.0f) < 1e-7; });
        if (has_top_p) {
            for (int64_t b = 0; b < (int64_t)batch_size; b++) {
                float p = top_p_ptr[b];
                if (std::abs(p - 1.0f) >= 1e-7) {
                    auto row                            = filtered_probs[b];
                    auto [sorted_probs, sorted_indices] = row.sort(/*descending=*/true);
                    auto cumsum                         = sorted_probs.cumsum(0);
                    auto mask                           = cumsum - sorted_probs > p;
                    sorted_probs.masked_fill_(mask, 0.0f);
                    row.scatter_(0, sorted_indices, sorted_probs);
                }
            }
        }
        // Re-normalize and sample
        auto row_sums  = filtered_probs.sum(-1, /*keepdim=*/true);
        filtered_probs = filtered_probs / row_sums.clamp_min(1e-10);
        auto selected  = torch::multinomial(filtered_probs, 1, /*replacement=*/false).squeeze(-1);
        samples_t.copy_(selected);
        if (need_output_all_probs) {
            output_all_probs_t.copy_(filtered_probs);
        }
    }

    // 7. Update cum_log_probs
    if (params.cum_log_probs.has_value()) {
        auto cum_log_probs_t = params.cum_log_probs.value();
        cum_log_probs_t.add_(probs_t.log());
    }

    // 8. Copy results back
    auto output_tokens = transposed_tokens.transpose(0, 1).contiguous();
    params.token_ids.copy_(output_tokens);
    return GreedyOutput{};
}

}  // namespace rtp_llm

// Forward-declare in global namespace (matches rtp_llm/models_py/bindings/rocm/speculative_sampling/sampling.cu)
void chain_speculative_sampling(at::Tensor draft_probs,
                                at::Tensor draft_token_ids,
                                at::Tensor uniform_samples,
                                at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num,
                                bool       deterministic,
                                int64_t    hip_stream);

namespace rtp_llm {

void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    ::chain_speculative_sampling(params.draft_probs_d,
                                 params.draft_token_ids_d,
                                 params.uniform_samples_d,
                                 params.target_probs_d,
                                 params.output_token_ids_d,
                                 params.output_accepted_token_num_d,
                                 params.output_emitted_token_num_d,
                                 true,
                                 int64_t(stream));
}

void rejectionSampling(const RejectionSamplingParams& params) {
    RTP_LLM_CHECK(params.draft_probs_d.is_cuda());
    RTP_LLM_CHECK(params.draft_token_ids_d.is_cuda());
    RTP_LLM_CHECK(params.target_probs_d.is_cuda());

    int  batch_size             = params.draft_probs_d.size(0);
    int  num_speculative_tokens = params.draft_probs_d.size(1);
    int  target_vocab_size      = params.target_probs_d.size(2);
    int  target_token_stride    = params.target_token_ids_d.size(1);
    auto stream                 = at::hip::getCurrentHIPStream().stream();

    RTP_LLM_CHECK(params.draft_probs_d.dim() == 3);
    RTP_LLM_CHECK(params.draft_token_ids_d.dim() == 2);
    RTP_LLM_CHECK(params.target_probs_d.dim() == 3);
    RTP_LLM_CHECK(params.draft_token_ids_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.draft_token_ids_d.size(1) == num_speculative_tokens);
    RTP_LLM_CHECK(params.target_probs_d.size(0) == batch_size);
    RTP_LLM_CHECK(params.target_probs_d.size(1) == num_speculative_tokens + 1);
    RTP_LLM_CHECK(params.draft_probs_d.size(2) == target_vocab_size);

    hipError_t err = ::invokeRejectionSampling(params.draft_probs_d.data_ptr<float>(),
                                               params.draft_token_ids_d.data_ptr<int32_t>(),
                                               params.uniform_samples_d.data_ptr<float>(),
                                               params.target_probs_d.data_ptr<float>(),
                                               params.target_token_ids_d.data_ptr<int32_t>(),
                                               target_token_stride,
                                               params.output_token_ids_d.data_ptr<int32_t>(),
                                               params.output_accepted_token_num_d.data_ptr<int32_t>(),
                                               params.do_sample_d.data_ptr<bool>(),
                                               batch_size,
                                               num_speculative_tokens,
                                               target_vocab_size,
                                               stream);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess, "invokeRejectionSampling failed: %s", hipGetErrorString(err));
}

void mappingDraft2Target(const MappingDraft2TargetParams& params) {
    if (!params.d2t_map.defined() || params.d2t_map.numel() == 0) {
        return;
    }

    const int d2t_map_size = static_cast<int>(params.d2t_map.numel());
    RTP_LLM_CHECK_WITH_INFO(params.d2t_map.dtype() == torch::kInt64, "mappingDraft2Target: d2t_map must be int64");

    if (!params.tokens.is_cuda() && !params.d2t_map.is_cuda()) {
        int64_t* d2t_map_ptr = params.d2t_map.data_ptr<int64_t>();
        if (params.tokens.dtype() == torch::kInt32) {
            int32_t* tokens_ptr = params.tokens.data_ptr<int32_t>();
            for (int i = 0; i < params.batch_size; i++) {
                for (int j = params.token_offset; j < params.token_stride; j++) {
                    int     idx      = i * params.token_stride + j;
                    int64_t token_id = static_cast<int64_t>(tokens_ptr[idx]);
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[idx] = static_cast<int32_t>(d2t_map_ptr[token_id]);
                    }
                }
            }
        } else if (params.tokens.dtype() == torch::kInt64) {
            int64_t* tokens_ptr = params.tokens.data_ptr<int64_t>();
            for (int i = 0; i < params.batch_size; i++) {
                for (int j = params.token_offset; j < params.token_stride; j++) {
                    int     idx      = i * params.token_stride + j;
                    int64_t token_id = tokens_ptr[idx];
                    if (token_id >= 0 && token_id < d2t_map_size) {
                        tokens_ptr[idx] = d2t_map_ptr[token_id];
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

#endif  // USING_CUDA

bool supportsDeferredRejectionSampling() {
#if USING_CUDA
    return samplingRuntimeReady(true);
#else
    return false;
#endif
}

void initializeSamplingRuntime(int64_t vocab_size) {
#if USING_CUDA
    const bool ordinary  = samplingRuntimeReady(false, vocab_size);
    const bool rejection = samplingRuntimeReady(true, vocab_size);
    RTP_LLM_LOG_INFO("sampling selection ordinary_vllm=%d rejection_vllm=%d", ordinary, rejection);
#endif
}

}  // namespace rtp_llm
