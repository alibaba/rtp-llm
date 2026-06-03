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
#include <memory>
#include <thread>
#include <random>

#if USING_CUDA
#include "c10/cuda/CUDACachingAllocator.h"
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
    executor_.reset(new NormalExecutor(
        params, cache_manager, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
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
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    loop_thread_->join();
    return absl::OkStatus();
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

void NormalEngine::batchEnqueue(std::vector<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        stream->setReserveStep(reserve_step_);
    }
    (void)scheduler_->batchEnqueue(streams);
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
    while (pause_) {
        // wait 50ms if system paused.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    int64_t                 tps_schedule_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.schedule(reserve_step=%d)", reserve_step_);
            CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        }
        if (parallelism_config.dp_size > 1) {
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

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = absl::OkStatus();
    if (propose_params_) {
        step_profiler_.tick();
    }
    {
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        status = executor_->process(streams, tps_schedule_time_us);
    }

    // report step metrics
    if (parallelism_config.tp_rank == 0) {
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
