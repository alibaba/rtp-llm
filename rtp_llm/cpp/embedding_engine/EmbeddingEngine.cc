#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <c10/core/InferenceMode.h>
#include <c10/util/Exception.h>
#include <exception>
#include <utility>

#if defined(USING_CUDA) && USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#elif defined(USING_ROCM) && USING_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace std;
namespace rtp_llm {

namespace {

std::string toLowerCase(const std::string& input) {
    std::string lower = input;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower;
}

bool containsAnyMarker(const std::string& haystack, const char* const* markers, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (haystack.find(markers[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool isRecoverableGpuOomMessage(const std::string& error_msg) {
    static constexpr std::array<const char*, 10> kOomMarkers = {
        "cuda out of memory",
        "cuda error: out of memory",
        "cuda runtime error: out of memory",
        "cudaerrormemoryallocation",
        "cublas_status_alloc_failed",
        "hip out of memory",
        "hip error: out of memory",
        "hip runtime error: out of memory",
        "hiperroroutofmemory",
        "trackerallocator failed to allocate",
    };
    return containsAnyMarker(toLowerCase(error_msg), kOomMarkers.data(), kOomMarkers.size());
}

bool isRecoverableGpuOomException(const std::exception& e) {
    if (dynamic_cast<const c10::OutOfMemoryError*>(&e) != nullptr) {
        return true;
    }
    return isRecoverableGpuOomMessage(e.what() ? e.what() : "");
}

bool isFatalGpuRuntimeError(const std::string& error_msg) {
    static constexpr std::array<const char*, 5> kFatalMarkers = {
        "CUDA Driver error",
        "CUDA error",
        "CUDA runtime error",
        "HIP error",
        "HIP runtime error",
    };
    return containsAnyMarker(error_msg, kFatalMarkers.data(), kFatalMarkers.size());
}

void clearStickyGpuOomState() {
#if defined(USING_CUDA) && USING_CUDA
    constexpr int kMaxStickyErrorsToDrain = 16;
    for (int i = 0; i < kMaxStickyErrorsToDrain; ++i) {
        const cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            break;
        }
        if (err != cudaErrorMemoryAllocation) {
            RTP_LLM_LOG_ERROR("detect non-OOM sticky CUDA error after OOM (%d: %s), do abort",
                              static_cast<int>(err),
                              cudaGetErrorString(err));
            abort();
        }
    }
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
    } else if (sync_err != cudaSuccess) {
        RTP_LLM_LOG_ERROR("cudaDeviceSynchronize failed after OOM (%d: %s), do abort",
                          static_cast<int>(sync_err),
                          cudaGetErrorString(sync_err));
        abort();
    }
    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
    } catch (const std::exception& cache_err) {
        RTP_LLM_LOG_WARNING("CUDACachingAllocator::emptyCache failed after OOM: %s", cache_err.what());
    }
#elif defined(USING_ROCM) && USING_ROCM
    constexpr int kMaxStickyErrorsToDrain = 16;
    for (int i = 0; i < kMaxStickyErrorsToDrain; ++i) {
        const hipError_t err = hipGetLastError();
        if (err == hipSuccess) {
            break;
        }
        if (err != hipErrorOutOfMemory) {
            RTP_LLM_LOG_ERROR("detect non-OOM sticky HIP error after OOM (%d: %s), do abort",
                              static_cast<int>(err),
                              hipGetErrorString(err));
            abort();
        }
    }
    const hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err == hipErrorOutOfMemory) {
        (void)hipGetLastError();
    } else if (sync_err != hipSuccess) {
        RTP_LLM_LOG_ERROR("hipDeviceSynchronize failed after OOM (%d: %s), do abort",
                          static_cast<int>(sync_err),
                          hipGetErrorString(sync_err));
        abort();
    }
#endif
}

bool handleStepFailure(const std::list<EmbeddingStreamPtr>& streams,
                       const std::string&                   error_msg,
                       bool                                 recoverable_oom) {
    RTP_LLM_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), error_msg.c_str());
    for (const auto& stream : streams) {
        stream->setError(error_msg);
        RTP_LLM_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
    }
    if (recoverable_oom) {
        RTP_LLM_LOG_WARNING("detect recoverable GPU OOM in embedding step, isolating failed batch and resetting runtime state");
        clearStickyGpuOomState();
        return true;
    }
    if (isFatalGpuRuntimeError(error_msg)) {
        RTP_LLM_LOG_ERROR("detect fatal GPU runtime error, do abort: %s", error_msg.c_str());
        abort();
    }
    return false;
}

}  // namespace

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    concurrency_config(params.concurrency_config),
    metrics_reporter_(params.metrics_reporter),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.world_rank) {
    {
        size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
        rtp_llm::initRuntime(device_id);
    }
    warmupNoBlockCopy();
    executor_.reset(new EmbeddingExecutor(params, handler));
    scheduler_.reset(new EmbeddingScheduler(
        model_config_, concurrency_config, params.runtime_config, metrics_reporter_, params.fmha_config));
    step_profiler_.configureFromConfig(params.profiling_debug_logging_config);

    (void)startLoop();
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
}

void EmbeddingEngine::startTimelineProfiling(const std::string& trace_name,
                                             int                start_step,
                                             int                num_steps,
                                             bool               enable_all_rank) {
    if (enable_all_rank && parallelism_config.tp_size > 1) {
        throw std::invalid_argument("embedding profiling does not support broadcasting to all TP ranks");
    }
    step_profiler_.configure(true, trace_name, start_step, num_steps);
}

absl::Status EmbeddingEngine::startLoop() {
    RTP_LLM_LOG_INFO("start embedding engine");
    running_     = true;
    loop_thread_ = std::thread(&EmbeddingEngine::loop, this);
    return absl::OkStatus();
}

absl::Status EmbeddingEngine::stop() {
    RTP_LLM_LOG_INFO("stop embedding engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void EmbeddingEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    c10::InferenceMode inference_guard(true);
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(th::Tensor                       token_ids,
                                                         th::Tensor                       token_type_ids,
                                                         th::Tensor                       input_lengths,
                                                         int64_t                          request_id,
                                                         std::optional<MultimodalFeature> multimodal_features,
                                                         std::optional<th::Tensor>        input_embeddings,
                                                         EmbeddingProfileConfig           profile_config) {
    auto input = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_features, input_embeddings);
    return decode(input, std::move(profile_config));
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(std::shared_ptr<EmbeddingInput> input,
                                                         EmbeddingProfileConfig          profile_config) {
    auto embedding_stream = std::make_shared<EmbeddingStream>(input, std::move(profile_config));
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return embedding_stream->embeddingOutput();
}

absl::Status EmbeddingEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status EmbeddingEngine::enqueue(EmbeddingStreamPtr streams) {
    return scheduler_->enqueue(streams);
}

absl::Status EmbeddingEngine::step() {
    cudaSyncAndCheck();
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
    if (!step_profiler_.enabled()) {
        for (const auto& stream : streams) {
            if (stream && stream->genTimeline()) {
                const auto& cfg       = stream->profileConfig();
                const int   num_steps = cfg.profile_step > 0 ? cfg.profile_step : 1;
                step_profiler_.configure(true, cfg.profile_trace_name, 0, num_steps);
                break;
            }
        }
    }

    {
        [[maybe_unused]] auto profile_step = step_profiler_.stepScope();
        try {
            auto status = executor_->process(streams);
            if (!status.ok()) {
                const std::string error_msg = status.ToString();
                if (handleStepFailure(streams, error_msg, isRecoverableGpuOomMessage(error_msg))) {
                    return absl::OkStatus();
                }
            }
        } catch (const exception& e) {
            const std::string error_msg = e.what() ? e.what() : "";
            if (handleStepFailure(streams, error_msg, isRecoverableGpuOomException(e))) {
                return absl::OkStatus();
            }
        }
        cudaSyncAndCheck();
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
