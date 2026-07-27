#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/GatherBatchScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <thread>
#include <random>

#if USING_CUDA
#include "c10/cuda/CUDACachingAllocator.h"
#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"
#endif

#ifdef __linux__
#include <malloc.h>
#endif

using namespace std;
namespace rtp_llm {

namespace {
// 释放glibc缓存的host内存，将其归还给操作系统
// 在模型加载完成后调用，可以显著减少常驻内存占用
void releaseHostMemoryCache() {
#ifdef __linux__
    // malloc_trim(0) 会释放所有可以释放的内存回操作系统
    // 这对于checkpoint加载后释放临时分配的大量CPU内存很重要
    int result = malloc_trim(0);
    RTP_LLM_LOG_INFO("Released host memory cache to OS (malloc_trim returned %d)", result);
#else
    RTP_LLM_LOG_DEBUG("malloc_trim not available on this platform");
#endif
}

bool shouldUseCudaMallocKVCacheBacking(const PDSepConfig& pd_sep_config, const CacheStoreConfig& cache_store_config) {
    // Only PD cache-store RDMA registers KV cache as user MR.  Keep the
    // raw cudaMalloc backing out of direct KVCacheManager users and non-RDMA
    // paths so PyTorch allocator behavior is unchanged elsewhere.
    const bool pd_role = pd_sep_config.role_type == RoleType::PREFILL || pd_sep_config.role_type == RoleType::DECODE;
    const bool has_cache_store_server = pd_sep_config.cache_store_listen_port > 0
                                        || pd_sep_config.cache_store_rdma_listen_port > 0
                                        || pd_sep_config.remote_rpc_server_port > 0;
    return pd_role && pd_sep_config.cache_store_rdma_mode
           && (cache_store_config.cache_store_rdma_mode || has_cache_store_server);
}

bool cacheStatusSnapshotEnabled() {
    const char* env = std::getenv("RTP_LLM_CACHE_STATUS_SNAPSHOT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

bool shouldRefreshCacheStatusSnapshot(RoleType role_type, const std::list<GenerateStreamPtr>& streams) {
    if (!cacheStatusSnapshotEnabled() || (role_type != RoleType::PREFILL && role_type != RoleType::PDFUSION)) {
        return false;
    }
    return std::any_of(streams.begin(), streams.end(), [](const GenerateStreamPtr& stream) {
        return stream && !stream->isFakeStream() && stream->isContextStream();
    });
}
}  // anonymous namespace

NormalEngine::NormalEngine(const EngineInitParams&                       params,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params):
    EngineBase(params),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    runtime_config(params.runtime_config),
    eplb_config(params.eplb_config),
    pd_sep_config(params.pd_sep_config),
    profiling_debug_logging_config(params.profiling_debug_logging_config),
    kv_cache_config(params.kv_cache_config),
    cache_store_config(params.cache_store_config),
    ffn_disaggregate_config(params.ffn_disaggregate_config),
    model_specific_config(params.model_specific_config),
    sp_config(params.sp_config),
    metrics_reporter_(params.metrics_reporter),
    propose_params_(std::move(propose_params)),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank) {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    if (propose_params_) {
        reserve_step_ = propose_params_->gen_num_per_circle + 1;
    } else {
        reserve_step_ = 0;
    }
    RTP_LLM_LOG_INFO("normal engine speculative reserve_step is %d", reserve_step_);
#if !USING_CUDA
    // On ROCm, this constructor runs on a gRPC handler thread that defaults to
    // GPU 0. Set the correct device so all GPU allocations (KV cache, etc.) go
    // to the right device.  The guard is scoped to the constructor body.
    c10::DeviceGuard ctor_device_guard(
        c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(parallelism_config.local_rank)));
    RTP_LLM_LOG_INFO("ROCm NormalEngine ctor: set device to %d", parallelism_config.local_rank);
#endif

    std::optional<WarmUpResult> warm_up_result = std::nullopt;
#if USING_CUDA
    if (runtime_config.warm_up && (!model_config_.mm_model_config.is_multimodal)
        && !ffn_disaggregate_config.enable_ffn_disaggregate) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         runtime_config.fifo_scheduler_config.max_context_batch_size,
                         model_config_.max_seq_len,
                         int(runtime_config.warm_up_with_loss));
        warm_up_result = warmUp(params);
        RTP_LLM_LOG_INFO(
            "warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
            warm_up_result->max_used_memory,
            warm_up_result->max_used_memory / 1024 / 1024,
            warm_up_result->device_reserved_bytes,
            warm_up_result->device_reserved_bytes / 1024 / 1024);
    } else {
        RTP_LLM_LOG_INFO("skip warm up.");
    }
#else
    RTP_LLM_LOG_INFO("skip warm up on non-CUDA platform.");
#endif
    initCacheManager(warm_up_result);
    RTP_LLM_LOG_INFO("create cache manager done");

    initExecutor(params, propose_params_);

    RTP_LLM_LOG_INFO("create normal executor done");

    // 释放模型加载过程中使用的临时host内存
    // 此时checkpoint已加载完成，可以将glibc缓存的内存归还给操作系统
    releaseHostMemoryCache();

    initScheduler();
    (void)startLoop();
}

void NormalEngine::initExecutor(const EngineInitParams&                        params,
                                std::unique_ptr<ProposeModelEngineInitParams>& propose_params) {
    if (propose_params_) {
        executor_.reset(new MtpExecutor(params,
                                        propose_params,
                                        resource_context_.cache_manager,
                                        mla_ops_type_,
                                        kv_cache_group_num_,
                                        kv_cache_layer_to_group_));
    } else {
        executor_.reset(new NormalExecutor(
            params,
            resource_context_.cache_manager,
            false,
            false,
            0,
            mla_ops_type_,
            kv_cache_group_num_,
            kv_cache_layer_to_group_,
            [this]() { step_profiler_.startStep(); },
            [this]() { step_profiler_.finishStep(); }));
    }
}

void NormalEngine::initScheduler() {
    if (runtime_config.use_batch_decode_scheduler) {
        scheduler_.reset(new BatchDecodeScheduler(
            runtime_config, resource_context_.cache_manager, metrics_reporter_, parallelism_config.dp_rank));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else if (runtime_config.use_gather_batch_scheduler) {
        scheduler_.reset(new GatherBatchScheduler(runtime_config,
                                                  model_config_,
                                                  pd_sep_config,
                                                  parallelism_config,
                                                  model_specific_config,
                                                  resource_context_.cache_manager,
                                                  metrics_reporter_));
        RTP_LLM_LOG_INFO("create gather batch scheduler done");
    } else {
        scheduler_.reset(new FIFOScheduler(runtime_config,
                                           model_config_,
                                           pd_sep_config,
                                           parallelism_config,
                                           model_specific_config,
                                           resource_context_.cache_manager,
                                           metrics_reporter_));
        RTP_LLM_LOG_INFO("create fifo scheduler done");
    }
}

NormalEngine::~NormalEngine() {
    RTP_LLM_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                       preRunMode                            mode) {
    auto stream = std::make_shared<NormalGenerateStream>(generate_input,
                                                         model_config_,
                                                         runtime_config,
                                                         resource_context_,
                                                         nullptr,
                                                         0,
                                                         mode == preRunMode::prefill_warm_up);
    stream->setReserveStep(reserve_step_);
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        size_t seq_size_per_block = model_config_.attn_config.tokens_per_block;
        size_t reserved_blocks    = (stream->seqLength() + seq_size_per_block - 1) / seq_size_per_block + reserve_step_;
        stream->fakeInitKVBlock(reserved_blocks);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUS_ERROR(stream->initKVBlock());
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return stream;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    if (runtime_config.use_batch_decode_scheduler) {
        if (runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0) {
            return decodeWarmUp(params);
        } else {
            return prefillWarmUp(params);
        }
    }
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::PREFILL) {
        return prefillWarmUp(params);
    } else if (pd_sep_config.role_type == RoleType::DECODE) {
        return decodeWarmUp(params);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
        return {};
    }
}

std::shared_ptr<GenerateInput> NormalEngine::makeFakeInput(size_t seq_len) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->generate_config               = make_shared<GenerateConfig>();
    size_t token_size                         = model_config_.embedding_size ?
                                                    std::min(model_config_.embedding_size, model_config_.vocab_size) :
                                                    model_config_.vocab_size;
    fake_input->input_ids              = torch::randint(0, (int64_t)token_size, {(int64_t)seq_len}, torch::kInt32);
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->generate_config->top_k = 1;

    return fake_input;
}

size_t NormalEngine::getWarmUpInputLength() const {
    const auto max_seq_len  = static_cast<size_t>(model_config_.max_seq_len);
    const auto reserve_step = reserve_step_ > 0 ? static_cast<size_t>(reserve_step_) : 0;
    if (reserve_step > 0) {
        RTP_LLM_CHECK_WITH_INFO(max_seq_len > reserve_step,
                                "max_seq_len [%zu] should be greater than speculative reserve_step [%zu]",
                                max_seq_len,
                                reserve_step);
        const auto input_len = max_seq_len - reserve_step;
        RTP_LLM_LOG_INFO("framework warm up input len adjusted by speculative reserve_step, "
                         "max_seq_len=%zu, reserve_step=%zu, input_len=%zu",
                         max_seq_len,
                         reserve_step,
                         input_len);
        return input_len;
    }
    RTP_LLM_CHECK_WITH_INFO(max_seq_len > 1, "max_seq_len [%zu] should be greater than 1", max_seq_len);
    return max_seq_len - 1;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("prefillWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput(getWarmUpInputLength());
    fake_input->generate_config->num_return_sequences = runtime_config.fifo_scheduler_config.max_context_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);
    // Snapshot hybrid-cache group info from a basic cache_config before
    // constructing the warmup NormalExecutor — see decodeWarmUp comment.
    // Without this, CudaGraphRunner inside the warmup executor sees
    // kv_cache_group_num_=0 and skips per-group block_table setup.
    {
        const int cache_gen_num_per_cycle =
            sp_config.type != SP_TYPE_NONE ? static_cast<int>(sp_config.gen_num_per_cycle) : 0;
        auto cfg = CacheConfigCreator::createBasicConfig(
            model_config_, parallelism_config, kv_cache_config, false, cache_gen_num_per_cycle);
        kv_cache_group_num_      = cfg.groupNums();
        kv_cache_layer_to_group_ = cfg.layer_to_group_id;
    }
    executor_.reset(new NormalExecutor(
        params, nullptr, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("decodeWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput(getWarmUpInputLength());
    fake_input->generate_config->num_return_sequences = runtime_config.max_generate_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);

    const int cache_gen_num_per_cycle =
        sp_config.type != SP_TYPE_NONE ? static_cast<int>(sp_config.gen_num_per_cycle) : 0;
    auto cache_config = CacheConfigCreator::createBasicConfig(
        model_config_, parallelism_config, kv_cache_config, false, cache_gen_num_per_cycle);
    // Do NOT override seq_size_per_block here. createBasicConfig already
    // returns the correct value: model_config.attn_config.tokens_per_block
    // for non-DSV4 (via SingleConfigCreator / HybridConfigCreator), and the
    // 256-token physical block for DSV4 (via DSV4CacheConfigHelper). Forcing
    // it back to attn_config.tokens_per_block would clobber DSV4's promoted
    // value when the user passed --seq_size_per_block < 256.
    cache_config.block_num = 5;
    // Snapshot hybrid-cache group info from the warmup cache_config so that the
    // NormalExecutor → PyWrappedModel → CudaGraphRunner created below sees the
    // real kv_cache_group_num_ / kv_cache_layer_to_group_.  Without this the
    // members are still default-0 (real values get set in initCacheManager
    // AFTER warmUp returns — see line 331/348), which makes CudaGraphRunner
    // skip per-group block_table construction in initCaptureAttentionInputs
    // (the `if (kv_cache_group_num_ > 1)` branch), and the DSv4 FP8 decode
    // path then asserts on pool_block_tables=None during the dtype-check
    // forward inside initCapture.
    kv_cache_group_num_      = cache_config.groupNums();
    kv_cache_layer_to_group_ = cache_config.layer_to_group_id;
    ParallelismConfig temp_parallelism_config;
    RuntimeConfig     temp_runtime_config;
    auto              cache_manager = make_shared<KVCacheManager>(
        cache_config, true, nullptr, KVCacheConfig{}, temp_parallelism_config, temp_runtime_config);
    if (!cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed in decodeWarmUp");
    }
    // Under engine-sleep (torch_memory_saver preload) the throwaway warmup
    // executor must NOT capture CUDA graphs. decodeWarmUp builds a real
    // cache_manager for valid block geometry, so PyWrappedModel's
    // "DeepSeekV4 warmup" guard (keyed on !kv_cache_layer_layout.has_value())
    // does not fire and the executor would capture graphs under the
    // 'cuda_graph' VMM tag. Those graphs are discarded immediately and the
    // post-warmup emptyCache() then faults releasing the tagged-but-unmapped
    // VMM ranges. The persistent graphs are (re)captured by the real executor
    // in initExecutor() after CacheManager init. Only skip capture when VMM is
    // active so the non-sleep path (accurate warmup memory accounting) is
    // unchanged. Restored right after so initExecutor still captures normally.
    const bool skip_warmup_graph = VmmBackend().isAvailable() && params.hw_kernel_config.enable_cuda_graph;
    auto&      mutable_hw_kernel = const_cast<HWKernelConfig&>(params.hw_kernel_config);
    if (skip_warmup_graph) {
        RTP_LLM_LOG_INFO("decodeWarmUp: VMM active, disabling CUDA graph capture for the throwaway warmup "
                         "executor; real executor captures after CacheManager init");
        mutable_hw_kernel.enable_cuda_graph = false;
    }
    executor_.reset(new NormalExecutor(
        params, cache_manager, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
    if (skip_warmup_graph) {
        mutable_hw_kernel.enable_cuda_graph = true;
    }
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

std::shared_ptr<GenerateStream> NormalEngine::createMinFakeStream(int32_t max_new_tokens) {
    RTP_LLM_LOG_DEBUG("create min fake query");
    auto fake_input                             = makeFakeInput(1);
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->fake_query                      = true;
    auto stream                                 = makeStream(fake_input);
    stream->setIsFakeStream(true);
    stream->setMetricsReporter(nullptr);
    stream->fakeInitKVBlock();
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::DECODE) {
        auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

        StreamUpdateInfo update_info{new_tokens,
                                     1,
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     false};
        stream->update(update_info);
        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .epoch                 = 0,
            .last_sample_token_gpu = torch::zeros({1}, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32),
            .last_real_seq_len     = stream->seqLength(),
            .next_real_seq_len     = stream->seqLength(),
        });
    }
    return stream;
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    const bool use_cuda_malloc_block_pool = shouldUseCudaMallocKVCacheBacking(pd_sep_config, cache_store_config);
    if (propose_params_ && propose_params_->draftModel()) {
        auto config = CacheConfigCreator::createSpConfig(model_config_,
                                                         propose_params_->getEngineInitParams().model_config_,
                                                         parallelism_config,
                                                         runtime_config,
                                                         kv_cache_config,
                                                         sp_config,
                                                         warm_up_result,
                                                         isMTPEagle(),
                                                         isEagle());

        resource_context_.cache_manager = make_shared<KVCacheManager>(config,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      kv_cache_config,
                                                                      parallelism_config,
                                                                      runtime_config,
                                                                      sp_config,
                                                                      pd_sep_config,
                                                                      cache_store_config,
                                                                      use_cuda_malloc_block_pool);
        resource_context_.role_type     = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }

        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
    } else {
        auto result = CacheConfigCreator::createConfig(
            model_config_, parallelism_config, runtime_config, kv_cache_config, warm_up_result, sp_config);
        RTP_LLM_LOG_INFO("create cache manager with config %s", result.debugString().c_str());
        RTP_LLM_LOG_INFO("create cache manager with block nums %d, block size %ld KB",
                         result.block_num,
                         result.block_size_bytes / 1024);
        RTP_LLM_LOG_INFO("create cache manager with linear step %d", result.linear_step);
        resource_context_.cache_manager = make_shared<KVCacheManager>(result,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      kv_cache_config,
                                                                      parallelism_config,
                                                                      runtime_config,
                                                                      SpeculativeExecutionConfig{},
                                                                      pd_sep_config,
                                                                      cache_store_config,
                                                                      use_cuda_malloc_block_pool);
        resource_context_.role_type     = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
    }
}

absl::Status NormalEngine::initSystemPrompt() {
    resource_context_.initCacheConfig(kv_cache_config, runtime_config.fifo_scheduler_config, model_config_.max_seq_len);

    if (!kv_cache_config.multi_task_prompt_tokens.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(
            system_prompt_param,
            SystemPromptConstructor::construct(
                kv_cache_config, this, resource_context_.cache_manager.get(), parallelism_config.tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }

    return absl::OkStatus();
}

KVCacheInfo NormalEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

absl::Status NormalEngine::startLoop() {
    if (parallelism_config.tp_rank == 0) {
        RTP_LLM_LOG_INFO("start init system prompt");
        THROW_IF_STATUS_ERROR(initSystemPrompt());
        RTP_LLM_LOG_INFO("init system prompt done");
    }
    RTP_LLM_LOG_INFO("start normal engine loop");
    running_     = true;
    loop_thread_ = autil::Thread::createThread(std::bind(&NormalEngine::loop, this), "normal_engine_loop");
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    RTP_LLM_LOG_INFO("stop normal engine");
    running_ = false;
    restart();
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    loop_thread_->join();
    return absl::OkStatus();
}

void NormalEngine::pause() {
    {
        // Bump the epoch under pause_mutex_ so it is published together with pause_=true.
        // A quiesce reader (enterPausedState) that observes pause_=true then loads the epoch
        // under the same lock is guaranteed to see the bumped value, so it can never
        // acknowledge a stale epoch and strand pauseAndWaitQuiesced() until timeout.
        // Bumping the epoch is also the sole "reset" of the quiesce ack: quiesced_pause_epoch_
        // stays below this new epoch until a real quiesce records it, with no separate reset
        // step that a concurrent acknowledgement could clobber.
        std::lock_guard<std::mutex> lock(pause_mutex_);
        bool                        expected = false;
        if (pause_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            auto epoch = pause_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
            RTP_LLM_LOG_INFO("normal engine pause requested, epoch=%lu", epoch);
        }
    }
    if (scheduler_) {
        scheduler_->wake();
    }
}

void NormalEngine::restart() {
    // Clear pause_ under pause_mutex_ so the store cannot slip into the window
    // between a waiter's predicate check and its wait: holding the lock forces
    // the waiter to either observe pause_=false before parking or receive the
    // notify while parked. A bare store+notify here could be lost.
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        pause_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();
}

void NormalEngine::invalidateCudaGraphs() {
    if (executor_) {
        executor_->invalidateCudaGraphs();
    }
}

void NormalEngine::recaptureCudaGraphs() {
    if (executor_) {
        executor_->recaptureCudaGraphs();
    }
}

void NormalEngine::recordPauseQuiesceFailure(uint64_t pause_epoch, const absl::Status& status) {
    if (status.ok()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        if (failed_pause_epoch_ != pause_epoch) {
            failed_pause_epoch_  = pause_epoch;
            failed_pause_status_ = status;
        }
    }
    pause_cv_.notify_all();
}

void NormalEngine::completePauseQuiesce(uint64_t pause_epoch, const absl::Status& status) {
    if (!status.ok()) {
        recordPauseQuiesceFailure(pause_epoch, status);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pause_mutex_);
        // A failure wins over a later successful retry for the same epoch. The coordinator
        // must roll the failed pause back instead of releasing resources after partial drain.
        if (failed_pause_epoch_ != pause_epoch && pause_epoch > quiesced_pause_epoch_) {
            quiesced_pause_epoch_ = pause_epoch;
        }
    }
    pause_cv_.notify_all();
}

void NormalEngine::enterPausedState() {
    while (running_.load()) {
        uint64_t active_epoch = 0;
        {
            std::lock_guard<std::mutex> lock(pause_mutex_);
            if (!pause_.load(std::memory_order_acquire)) {
                break;
            }
            active_epoch = pause_epoch_.load(std::memory_order_acquire);
        }

        // The engine loop owns executor quiescence. Ack this epoch only after all host
        // runners have joined and their device work has completed.
        completePauseQuiesce(active_epoch, quiesceExecutorForPause(active_epoch));

        std::unique_lock<std::mutex> lock(pause_mutex_);
        pause_cv_.wait(lock, [this, active_epoch] {
            return !pause_.load(std::memory_order_acquire)
                   || pause_epoch_.load(std::memory_order_acquire) > active_epoch || !running_.load();
        });
    }
}

bool NormalEngine::collectiveSleepQuiesceEnabled() const {
    // The decision whether step() performs the extra pause-wave all-reduce MUST be
    // rank-symmetric: if it diverges across ranks, some ranks call execAllReduce
    // while others don't and the collective deadlocks during *normal* serving, not
    // just during sleep. So gate it on enabled() (the static enable_sleep_mode launch
    // config, set in initRuntime() before startLoop() and identical on every rank),
    // NOT on effective(): effective() folds in runtimeSupported(), a per-rank VMM
    // availability flag that is set only after the loop thread has started
    // (LocalRpcServer::installSleepHooks) and may legitimately differ across ranks.
    // A rank without VMM support still participates in the quiesce collective (safe,
    // it just no-ops the local memory release); a sleep request there fails cleanly
    // with DISABLED and the coordinator aborts, rather than hanging the fleet.
    return sleep_controller_.enabled() && parallelism_config.world_size > 1
           && (parallelism_config.dp_size > 1 || parallelism_config.ep_size > 1);
}

// Why the sleep-quiesce consensus needs a collective at all, and why it now costs the steady
// serving path nothing:
//
// Under DP/EP (world_size > 1 with dp_size > 1 or ep_size > 1) sleep cannot be a rank-local
// decision. The pause request is broadcast to every rank's engine independently
// (grpc_client_wrapper.sleep_serving fans out to every control rank), but arrives
// asynchronously: rank A may observe pause_ several steps before rank B. If a rank simply
// stopped as soon as it saw pause_, its peers' next forward-pass collective (TP all-reduce /
// EP all-to-all) would block forever on the departed rank -- a fleet-wide hang, not a clean
// sleep. The ranks must instead agree, from a value every rank observes identically, that
// "everyone has received the pause AND everyone is drained" before any of them stops.
//
// That agreement is an all-reduce, but it is issued ASYNCHRONOUSLY and ONLY WHILE A SLEEP IS
// ARMED, on a DEDICATED communicator (ParallelMode::SLEEP_QUIESCE) that carries no other
// traffic:
//   * Steady state (no pause armed): this function returns after a single relaxed atomic
//     load. It issues NO collective, acquires NO GIL, does NO device work -- the decode hot
//     path pays nothing. This is what the earlier every-step blocking all-reduce could not do.
//   * Armed: the rank latches "engaged" and drives one async all-reduce at a time. Because it
//     is async_op=True the rank KEEPS participating in its forward collectives while the
//     consensus is in flight, so an early-arming rank never strands its peers -- the deadlock
//     that forces a *blocking* consensus to run every step cannot occur.
//   * Dedicated comm: an async round can complete a variable number of engine steps after it
//     was issued. On the shared DP_AND_TP communicator that lag would let the consensus
//     interleave with EPLB's periodic all-reduce/broadcast (ExpertBalancer, also on
//     DP_AND_TP) and desync the collective stream. A private communicator carrying only the
//     consensus makes the k-th round unambiguous on every rank.
//
// Round matching / termination: a rank issues round k+1 only after round k completes, so all
// ranks run the same 1..K rounds. A shared ready verdict starts terminal alignment; it does not
// immediately park a rank. Ranks first exchange the maximum host-enqueued forward sequence,
// compensate to that sequence, and confirm that every peer has enqueued it before synchronizing
// CUDA. This closes the poll skew where one rank observes round K one engine step later than a
// peer and has already launched one more peer-symmetric MegaMoE forward. (The single-rank /
// pure-TP deployment escapes all of this: enterPausedState() drives its pause wave and
// executor drain locally, without a cross-rank sleep-quiesce collective.)
absl::Status NormalEngine::maybeReachCollectiveSleepQuiesce() {
    if (!collectiveSleepQuiesceEnabled()) {
        return absl::OkStatus();
    }

    if (collective_quiesce_alignment_.terminalAlignmentActive()) {
        return driveCollectiveSleepTerminalAlignment();
    }

    const bool pending = pause_.load(std::memory_order_acquire);

    // Steady serving fast path: nothing armed and no round in flight -> issue no collective.
    // engaged_ and handle_ are touched only on the engine loop thread (this function runs
    // from step()), so they need no synchronization.
    if (!collective_quiesce_engaged_ && collective_quiesce_handle_ == 0 && !pending) {
        return absl::OkStatus();
    }

    // First observation of pause_ arms this rank. Once engaged we stay engaged (still issuing
    // rounds) even if pause_ later clears via wake, so a cancel converges to a shared CANCELLED
    // verdict rather than leaving peers waiting on a round this rank stopped issuing.
    if (pending && !collective_quiesce_engaged_) {
        // Arm transition (once per sleep). Flush any outstanding stream-async work BEFORE we
        // stop issuing forwards and start the consensus rounds. With MTP the last pre-pause
        // step leaves cross-step prepare/verify runners in flight; left un-drained across a
        // torch_memory_saver release/restore they corrupt NCCL/CUDA state so the NEXT sleep's
        // SLEEP_QUIESCE all-reduce never completes (repeat-sleep hang, MTP-only). Draining here
        // is on the engine loop thread, exactly like process()'s own start-of-step sync.
        if (executor_) {
            const auto status = executor_->drainAsyncRunners();
            if (!status.ok()) {
                const auto epoch = pause_epoch_.load(std::memory_order_acquire);
                RTP_LLM_LOG_ERROR("collective sleep quiesce async runner drain failed, epoch=%lu: %s",
                                  epoch,
                                  status.ToString().c_str());
                recordPauseQuiesceFailure(epoch, status);
            }
        }
    }
    if (pending) {
        collective_quiesce_engaged_ = true;
    }

    bool reached = false;
    {
        // Guard the reusable buffer against a concurrent execute-path caller. The async round
        // reduces IN-PLACE on collective_quiesce_state_; because we never refill or read the
        // buffer until the in-flight round completes, its contents are stable while in flight.
        std::lock_guard<std::mutex> lock(collective_quiesce_state_mutex_);
        if (!collective_quiesce_state_.defined()) {
            // CPU-resident on purpose: the SLEEP_QUIESCE consensus runs on the gloo (host)
            // group, so the reduce executes on the CPU and never competes with the co-stepped
            // fake-decode's DeepEP kernels for SMs (a GPU/NCCL reduce here starves behind the
            // full-grid DeepEP low-latency kernels and never completes -> /sleep hangs). A CPU
            // buffer also means no device->host copy before reading the verdict below.
            const auto options        = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            collective_quiesce_state_ = torch::empty({2}, options);
        }

        if (collective_quiesce_handle_ == 0) {
            // Drain the just-completed co-step forward NOW, before issuing the round, while
            // every rank is still active and co-stepping so their matched DeepEP all-to-all
            // retires together. If this drain were deferred to the reached branch, the rank
            // that wins the (async) consensus race parks first; the loser's park-time
            // cudaDeviceSynchronize would then block on a DeepEP low-latency kernel from the
            // last MTP fake-decode forward whose peer has already left the EP group -> GPU
            // spins at 100% and /sleep times out (observed: MTP tp1/dp2, winner/loser role
            // flips per run). Synchronizing here makes the device idle before the round is
            // even enqueued, so once the round resolves the park touches no in-flight EP.
            // Armed-only (handle==0 && pending/engaged), so steady serving never reaches this.
#if USING_CUDA
            if (pending || collective_quiesce_engaged_) {
                if (auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    // A failed drain means the device is already in a bad state (e.g. a peer's
                    // mega-MoE NVLink barrier trapped -> CUDA_ERROR_LAUNCH_FAILED). Do NOT enqueue a
                    // consensus round on a poisoned context; clear the sticky error and bail this
                    // step. The coordinator's pauseAndWaitQuiesced timeout bounds the retry and rolls
                    // the sleep back cleanly. A non-OK return here is fatal (loop() ->
                    // THROW_IF_STATUS_ERROR kills the engine thread), so we stay OK and simply refuse
                    // to progress the quiesce.
                    RTP_LLM_LOG_ERROR("collective sleep quiesce pre-round drain sync failed, skipping round: %s",
                                      cudaGetErrorString(err));
                    recordPauseQuiesceFailure(
                        pause_epoch_.load(std::memory_order_acquire),
                        absl::InternalError(std::string("collective sleep quiesce device sync failed: ")
                                            + cudaGetErrorString(err)));
                    cudaGetLastError();  // clear sticky so it cannot resurface on an unrelated later call
                    return absl::OkStatus();
                }
            }
#endif
            // No round in flight: fill this rank's contribution and enqueue an async round.
            //   data[0] = 1 if this rank currently holds the pause request, else 0.
            //   data[1] = unfinished streams on this rank (only tp_rank 0 can observe them).
            const int64_t pending_flag = pending ? 1 : 0;
            const int64_t unfinished   = (pending && parallelism_config.tp_rank == 0 && scheduler_) ?
                                             std::max<int64_t>(0, scheduler_->onflightStreams()) :
                                             0;
            collective_quiesce_state_.select(0, 0).fill_(pending_flag);
            collective_quiesce_state_.select(0, 1).fill_(unfinished);
            collective_quiesce_handle_ =
                execAllReduceAsync({collective_quiesce_state_, ReduceOp::Sum, false, ParallelMode::SLEEP_QUIESCE});
            // handle 0 => async path unavailable (no callback / degenerate group). Nothing to
            // wait for; retry next step. (Should not happen once the group is built.)
            return absl::OkStatus();
        }

        // A round is in flight. Poll without blocking the loop; if not done, keep serving and
        // participating in forward collectives (the whole point of async arm-on-demand).
        if (!pollAsyncComm(collective_quiesce_handle_)) {
            return absl::OkStatus();
        }
        collective_quiesce_handle_ = 0;

        // Round complete: read the summed verdict. The device->host copy only ever runs on the
        // armed path (never in steady serving). pollAsyncComm() already waited on the collective,
        // so the reduced buffer is safe to read here.
        auto reduced = collective_quiesce_state_;
        if (reduced.device().is_cuda()) {
            reduced = reduced.cpu();
        }
        auto          reduced_data    = reduced.data_ptr<int64_t>();
        const int64_t pending_sum     = reduced_data[0];
        const int64_t not_ready_count = reduced_data[1];

        if (pending_sum == parallelism_config.world_size && not_ready_count == 0) {
            reached = true;  // all ranks armed AND drained on the same round -> sleep now
        } else if (pending_sum == 0) {
            // Every rank has left the pause window (sleep cancelled / woken before quiesce):
            // disengage together on this shared verdict and resume normal serving at zero cost.
            collective_quiesce_engaged_ = false;
            if (scheduler_) {
                scheduler_->setForcePoll(false);  // stop the arm-time poll; back to blocking schedule()
            }
            RTP_LLM_LOG_INFO("normal engine collective sleep quiesce cancelled before reaching, world_size=%ld",
                             parallelism_config.world_size);
        }
        // else: partial (some ranks still arming, or still draining) -> stay engaged and issue
        // the next round on a later step.
    }

    if (reached) {
        collective_quiesce_alignment_.beginTargetExchange();
        RTP_LLM_LOG_INFO("collective sleep quiesce entering terminal alignment, local_forward_seq=%ld",
                         collective_quiesce_alignment_.forward_sequence);
        return driveCollectiveSleepTerminalAlignment();
    }
    return absl::OkStatus();
}

absl::Status NormalEngine::runCollectiveSleepCompensationForward() {
    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        mayAddFakeStream(streams);
    }

    const auto before = collective_quiesce_alignment_.forward_sequence;
    auto       status = executor_->process(streams, autil::TimeUtility::currentTimeInMicroSeconds());
    if (!status.ok()) {
        return status;
    }
    collective_quiesce_alignment_.noteForwardEnqueued();
    RTP_LLM_LOG_INFO("collective sleep quiesce compensated forward, seq=%ld->%ld, target=%ld",
                     before,
                     collective_quiesce_alignment_.forward_sequence,
                     collective_quiesce_alignment_.target_sequence);
    return absl::OkStatus();
}

absl::Status NormalEngine::finishCollectiveSleepQuiesce(bool should_park) {
    const auto aligned_sequence = collective_quiesce_alignment_.target_sequence;
    if (should_park) {
        // The host barrier proves every rank has enqueued exactly aligned_sequence forwards.
        // Only now is a device drain safe: every peer-symmetric invocation already has its
        // matching launch. enterPausedState() performs runner drain followed by device sync,
        // keeping the acknowledgement invariant identical for every topology.
    }

    collective_quiesce_alignment_.resetToConsensus();
    collective_quiesce_engaged_ = false;
    if (scheduler_) {
        scheduler_->setForcePoll(false);
    }

    if (!should_park) {
        RTP_LLM_LOG_INFO("normal engine collective sleep quiesce cancelled during terminal alignment, "
                         "forward_seq=%ld, world_size=%ld",
                         aligned_sequence,
                         parallelism_config.world_size);
        return absl::OkStatus();
    }

    const auto pause_epoch = pause_epoch_.load(std::memory_order_acquire);
    RTP_LLM_LOG_INFO("normal engine collective sleep quiesce reached, epoch=%lu, world_size=%ld, forward_seq=%ld",
                     pause_epoch,
                     parallelism_config.world_size,
                     aligned_sequence);
    enterPausedState();
    return absl::OkStatus();
}

absl::Status NormalEngine::driveCollectiveSleepTerminalAlignment() {
    using Phase = CollectiveQuiesceAlignment::Phase;

    if (collective_quiesce_alignment_.phase == Phase::COMPENSATE_FORWARDS) {
        if (collective_quiesce_alignment_.needsCompensationForward()) {
            return runCollectiveSleepCompensationForward();
        }
        if (collective_quiesce_alignment_.readyForBarrier()) {
            collective_quiesce_alignment_.beginReadyBarrier();
        }
    }

    bool finish      = false;
    bool should_park = false;
    {
        std::lock_guard<std::mutex> lock(collective_quiesce_state_mutex_);
        if (collective_quiesce_alignment_.phase == Phase::TARGET_EXCHANGE) {
            if (collective_quiesce_handle_ == 0) {
                collective_quiesce_state_.select(0, 0).fill_(collective_quiesce_alignment_.forward_sequence);
                collective_quiesce_state_.select(0, 1).zero_();
                collective_quiesce_handle_ =
                    execAllReduceAsync({collective_quiesce_state_, ReduceOp::Max, false, ParallelMode::SLEEP_QUIESCE});
                return absl::OkStatus();
            }
            if (!pollAsyncComm(collective_quiesce_handle_)) {
                return absl::OkStatus();
            }
            collective_quiesce_handle_ = 0;
            auto reduced               = collective_quiesce_state_;
            if (reduced.device().is_cuda()) {
                reduced = reduced.cpu();
            }
            const auto target_sequence = reduced.data_ptr<int64_t>()[0];
            if (!collective_quiesce_alignment_.acceptTargetSequence(target_sequence)) {
                return absl::InternalError("collective sleep quiesce target sequence regressed from "
                                           + std::to_string(collective_quiesce_alignment_.forward_sequence) + " to "
                                           + std::to_string(target_sequence));
            }
            RTP_LLM_LOG_INFO("collective sleep quiesce target aligned, local_forward_seq=%ld, target=%ld",
                             collective_quiesce_alignment_.forward_sequence,
                             target_sequence);
            return absl::OkStatus();
        }

        if (collective_quiesce_alignment_.phase != Phase::READY_BARRIER) {
            return absl::OkStatus();
        }

        if (collective_quiesce_handle_ == 0) {
            collective_quiesce_state_.select(0, 0).fill_(1);
            collective_quiesce_state_.select(0, 1).fill_(pause_.load(std::memory_order_acquire) ? 1 : 0);
            collective_quiesce_handle_ =
                execAllReduceAsync({collective_quiesce_state_, ReduceOp::Sum, false, ParallelMode::SLEEP_QUIESCE});
            return absl::OkStatus();
        }
        if (!pollAsyncComm(collective_quiesce_handle_)) {
            return absl::OkStatus();
        }
        collective_quiesce_handle_ = 0;
        auto reduced               = collective_quiesce_state_;
        if (reduced.device().is_cuda()) {
            reduced = reduced.cpu();
        }
        const auto* reduced_data    = reduced.data_ptr<int64_t>();
        const auto  participant_sum = reduced_data[0];
        const auto  pending_sum     = reduced_data[1];
        if (participant_sum != parallelism_config.world_size) {
            return absl::InternalError("collective sleep quiesce terminal barrier participant mismatch: "
                                       + std::to_string(participant_sum)
                                       + " != " + std::to_string(parallelism_config.world_size));
        }
        if (pending_sum == parallelism_config.world_size) {
            finish      = true;
            should_park = true;
        } else if (pending_sum == 0) {
            finish = true;
        }
        // A partial cancel is sampled identically by every rank. Keep issuing decision rounds
        // without launching forwards until the coordinator converges all ranks to park or cancel.
    }

    if (finish) {
        return finishCollectiveSleepQuiesce(should_park);
    }
    return absl::OkStatus();
}

absl::Status NormalEngine::quiesceExecutorForPause(uint64_t pause_epoch) {
    if (processed_pause_epoch_.load(std::memory_order_acquire) >= pause_epoch) {
        return absl::OkStatus();
    }
    if (!executor_) {
        return absl::FailedPreconditionError("pause quiesce requires an initialized executor");
    }

    absl::Status status = absl::OkStatus();
    // In pure TP, rank 0 drives the empty pause wave from the engine loop. Keeping all
    // executor calls on this thread avoids racing an RPC-thread processForPause() against
    // a normal model step. Worker ranks receive the marker, finish their current process(),
    // and enter this same quiesce boundary.
    if (!collectiveSleepQuiesceEnabled() && parallelism_config.tp_size > 1 && parallelism_config.tp_rank == 0) {
        RTP_LLM_LOG_INFO("normal engine pause: run one empty TP sync step, epoch=%lu", pause_epoch);
        status = executor_->processForPause();
        (void)executor_->consumeLastPauseSignal();
    }

    auto drain_status = executor_->drainAsyncRunners();
    if (status.ok() && !drain_status.ok()) {
        status = drain_status;
    }

#if USING_CUDA
    if (auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
        auto sync_status =
            absl::InternalError(std::string("pause quiesce device sync failed: ") + cudaGetErrorString(err));
        if (status.ok()) {
            status = std::move(sync_status);
        }
        cudaGetLastError();
    }
#endif

    if (status.ok()) {
        processed_pause_epoch_.store(pause_epoch, std::memory_order_release);
    }
    return status;
}

absl::Status NormalEngine::pauseAndWaitQuiesced(int64_t timeout_ms) {
    constexpr int64_t kDefaultPauseQuiesceTimeoutMs = 60000;
    const int64_t     effective_timeout_ms          = timeout_ms > 0 ? timeout_ms : kDefaultPauseQuiesceTimeoutMs;

    pause();
    const auto pause_epoch = pause_epoch_.load(std::memory_order_acquire);
    if (!running_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }

    std::unique_lock<std::mutex> lock(pause_mutex_);
    if (failed_pause_epoch_ == pause_epoch) {
        return failed_pause_status_;
    }
    if (quiesced_pause_epoch_ >= pause_epoch || !pause_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(effective_timeout_ms);
    if (!pause_cv_.wait_until(lock, deadline, [this, pause_epoch] {
            return failed_pause_epoch_ == pause_epoch || quiesced_pause_epoch_ >= pause_epoch
                   || !pause_.load(std::memory_order_acquire) || !running_.load();
        })) {
        return absl::Status(absl::StatusCode::kDeadlineExceeded,
                            "normal engine pause quiesce timeout after " + std::to_string(effective_timeout_ms)
                                + " ms");
    }
    if (failed_pause_epoch_ == pause_epoch) {
        return failed_pause_status_;
    }
    return absl::OkStatus();
}

void NormalEngine::armCollectiveSleepQuiesce() {
    // DP/EP: the sleep-quiesce consensus (maybeReachCollectiveSleepQuiesce) must be armed on
    // EVERY rank when it enters DRAINING -- i.e. BEFORE the per-rank drain wait, not after it
    // (as the quiesceEngine hook / pauseAndWaitQuiesced does). Drain is rank-asymmetric: only
    // the rank holding the in-flight request blocks in DrainManager.waitDrained, so if arming
    // waited for local drain the busy rank would keep pause_=false and never arm, while its
    // idle peers (already drained, pause_=true) would issue unmatched SLEEP_QUIESCE all-reduce
    // rounds. Those never-matched collective rounds pin GPU collective resources indefinitely
    // and desync the EP all-to-all until the busy rank's real forward hits a DeepEP CPU recv
    // timeout -- the request errors, drain never completes, and the worker rank dies.
    //
    // Arming here calls pause() (single CAS: sets pause_ + bumps pause_epoch_), so the busy
    // rank co-steps the consensus while it drains and data[1]=onflightStreams carries its drain
    // progress into the shared verdict; all ranks reach REACHED together once the global
    // on-flight count is zero. The drain-time quiesceEngine hook's pause() then becomes a no-op
    // CAS (epoch unchanged) and simply waits for the consensus to record that epoch.
    //
    // No-op unless the collective quiesce path is enabled (multi-rank DP/EP): for single-rank
    // pause() belongs AFTER drain (inside pauseAndWaitQuiesced), because setting pause_ early
    // would park the loop (step()) before the in-flight request completes, stranding drain.
    if (collectiveSleepQuiesceEnabled()) {
        pause();
        // Keep the scheduler polling (not blocking on an empty queue) for the duration of the
        // consensus. The async SLEEP_QUIESCE rounds only advance when the engine loop cycles; once
        // the rank's own work has drained, schedule() would otherwise block indefinitely (the
        // fake-stream 10ms poll is need_fill_fake_stream_ = dp_size>1 && tp_rank==0 only, so a
        // dp_size==1 EP deployment has no such poll), stalling the empty co-steps its peers need to
        // reach the terminal round. Cleared at the consensus verdict in maybeReachCollectiveSleepQuiesce
        // (REACHED or CANCELLED), NOT in restart(): during a cancel, restart() runs WHILE the consensus
        // is still converging to CANCELLED, so clearing the flag there would stop the very polling that
        // carries the cancel to its shared verdict and strand peers.
        if (scheduler_) {
            scheduler_->setForcePoll(true);
        }
    }
}

void NormalEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    cudaPreRun(getDeviceId());
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    stream->setReserveStep(reserve_step_);
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    stream->setReserveStep(reserve_step_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

std::vector<std::shared_ptr<GenerateStream>>
NormalEngine::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    std::vector<std::shared_ptr<GenerateStream>> streams;
    streams.reserve(inputs.size());
    for (auto& inp : inputs) {
        auto stream = std::make_shared<NormalGenerateStream>(
            inp, model_config_, runtime_config, resource_context_, metrics_reporter_);
        stream->setReserveStep(reserve_step_);
        streams.push_back(stream);
    }
    return scheduler_->batchEnqueue(streams);
}

absl::Status NormalEngine::step() {
    RTP_LLM_PROFILE_SCOPE("engine.normal.step_work");
    // Cache the feature flag once so disabled serving keeps the same single atomic load it
    // already paid for the collective gate. Pure-TP workers also use it to consume a pause
    // marker that can arrive before their local control RPC.
    const bool sleep_mode_enabled       = sleep_controller_.enabled();
    const bool collective_sleep_quiesce = sleep_mode_enabled && parallelism_config.world_size > 1
                                          && (parallelism_config.dp_size > 1 || parallelism_config.ep_size > 1);
    if (pause_.load(std::memory_order_acquire) && !collective_sleep_quiesce
        && (parallelism_config.tp_size <= 1 || parallelism_config.tp_rank == 0)) {
        enterPausedState();
    }

    // stop() wakes a paused loop by clearing pause_ (via restart()) with running_
    // already false. Without this guard the woken loop would fall through to a
    // real schedule()/execute step while the KV backing is still released
    // (sleeping) -- for TP>1 that empty step also blocks in tpSyncModelInputs
    // against ranks that have already torn down. Bail out promptly on shutdown.
    if (!running_.load(std::memory_order_acquire)) {
        return absl::OkStatus();
    }

    // Once a rank has observed the shared ready verdict it must stop normal scheduling.
    // Peers may observe that verdict one or more host steps later, so terminal alignment is
    // driven separately and only launches the exact number of fake forwards needed to reach
    // the shared maximum sequence.
    if (collective_sleep_quiesce && collective_quiesce_alignment_.terminalAlignmentActive()) {
        return driveCollectiveSleepTerminalAlignment();
    }

    // Sleep-quiesce, decode (tp1) DP/EP path: DO NOT skip the fake-decode forward while a
    // consensus round is in flight. This model's MoE is DeepGEMM ``fp8_fp4_mega_moe`` -- a
    // peer-symmetric NVLink-barrier collective (symm-mem), NOT DeepEP low-latency. Every rank
    // in the group MUST enter the mega kernel together every forward, or the surviving peers
    // hit an NVLink barrier timeout (deep_gemm/comm/barrier.cuh) -> asm("trap") after 30 s ->
    // CUDA_ERROR_LAUNCH_FAILED. That hard per-forward barrier is exactly what keeps the ranks'
    // fake-decode forwards ALIGNED: as long as every rank keeps co-stepping, forward index N is
    // reached simultaneously on all ranks. A forward-skip here (the earlier "DeepEP" fix) breaks
    // that: the first-armed rank starts skipping while a not-yet-armed peer still runs its fake
    // forward -> the peer's mega kernel strands with no partner -> NVLink-barrier trap (this is
    // the repeated-sleep arm-skew hang). So we keep co-stepping through the whole drain. The
    // terminal alignment state machine handles the remaining host poll skew after the shared
    // ready verdict; armed-only path, steady serving is untouched.
    int64_t                 tps_schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.schedule(reserve_step=%d)", reserve_step_);
            CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        }
        if (parallelism_config.dp_size > 1 || (collective_sleep_quiesce && parallelism_config.ep_size > 1)) {
            RTP_LLM_PROFILE_SCOPE("engine.normal.may_add_fake_stream_work");
            mayAddFakeStream(streams);
        }
        // When TP > 1, all ranks must enter process() together so that
        // tpSyncModelInputs (collective broadcast) does not deadlock.
        // The skip_run flag inside process() handles the "no work" case.
        if (streams.empty() && parallelism_config.tp_size <= 1) {
            return absl::OkStatus();
        }
    }

    if (pause_.load(std::memory_order_acquire) && !collective_sleep_quiesce && parallelism_config.tp_rank == 0) {
        enterPausedState();
        return absl::OkStatus();
    }

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = absl::OkStatus();
    if (propose_params_) {
        step_profiler_.tick();
    }
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        const bool refresh_cache_status_snapshot =
            resource_context_.cache_manager && shouldRefreshCacheStatusSnapshot(pd_sep_config.role_type, streams);
        status = executor_->process(streams, tps_schedule_time_us);
        if (status.ok() && sleep_mode_enabled && executor_->consumeLastPauseSignal()) {
            pause();
        }
        if (status.ok() && collective_sleep_quiesce) {
            collective_quiesce_alignment_.noteForwardEnqueued();
        }
        if (status.ok() && refresh_cache_status_snapshot) {
            RTP_LLM_PROFILE_SCOPE("engine.normal.refresh_cache_status_snapshot");
            resource_context_.cache_manager->refreshKVCacheInfoSnapshot();
        }
    }

    if (status.ok() && collective_sleep_quiesce) {
        status = maybeReachCollectiveSleepQuiesce();
    } else if (status.ok() && pause_.load(std::memory_order_acquire)) {
        enterPausedState();
    }

    // loop() is a no-sleep tight loop and with TP>1 every iteration enters
    // process() to drive the collective tpSync even with empty streams —
    // without this gate the gauge gets diluted to ~0 by idle iterations.
    if (parallelism_config.tp_rank == 0 && !streams.empty()) {
        RTP_LLM_PROFILE_SCOPE("engine.normal.report_metrics_work");
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({step_latency});
    }

    return status;
}

bool NormalEngine::updateEplbConfig(const EPLBConfig& config) {
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
}

void NormalEngine::startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {
    step_profiler_.configure(true, trace_name, start_step, num_steps);
}

bool NormalEngine::isTimelineProfilingEnabled() const {
    return step_profiler_.enabled();
}

bool NormalEngine::isMTPEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_MTP || propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

bool NormalEngine::isEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

void NormalEngine::mayAddFakeStream(std::list<GenerateStreamPtr>& streams) {
    if (isMTPEagle()) {
        int propose_step   = sp_config.gen_num_per_cycle;
        int mtp_vocab_size = propose_params_->getEngineInitParams().model_config_.vocab_size;
        switch (pd_sep_config.role_type) {
            case RoleType::PREFILL:
                if (streams.empty()) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                break;
            case RoleType::DECODE:
                if (streams.empty()) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, mtp_vocab_size));
                }
                break;
            case RoleType::PDFUSION: {
                bool has_prefill = false;
                bool has_decode  = false;
                for (auto& stream : streams) {
                    if (stream->isContextStream()) {
                        has_prefill = true;
                    } else {
                        has_decode = true;
                    }
                }
                if (!has_prefill && !runtime_config.use_batch_decode_scheduler) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                if (!has_decode) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_, mtp_vocab_size));
                }
                break;
            }
            default:
                RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
                break;
        }
    } else {
        if (streams.empty()) {
            streams.emplace_back(createMinFakeStream(1));
        }
    }
}

}  // namespace rtp_llm
