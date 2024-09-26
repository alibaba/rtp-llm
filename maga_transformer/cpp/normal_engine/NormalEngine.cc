#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include "autil/TimeUtility.h"
#include <memory>
#include <thread>

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const EngineInitParams& params) :
    EngineBase(params),
    params_(params.gpt_init_parameter),
    metrics_reporter_(params.metrics_reporter)
{
    FT_LOG_INFO(__PRETTY_FUNCTION__);
    size_t max_left_free_bytes = 0;
    if (params_.warm_up_) {
        // warm up
        FT_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin", params_.max_context_batch_size_, params_.max_seq_len_, int(params_.warm_up_with_loss_));
        max_left_free_bytes = warmUp(params);
        size_t max_runtime_buffer = device_->getDeviceStatus().device_memory_status.preserved_bytes - max_left_free_bytes;
        size_t other_reserve_mem = 0;
        size_t sample_need_mem = (size_t)params_.max_generate_batch_size_ * params_.vocab_size_ * 4 * 8; // just estimated value
        if (sample_need_mem > max_runtime_buffer) {
            other_reserve_mem = std::min(sample_need_mem - max_runtime_buffer, (size_t)2048 * 1024 * 1024); // not allow to large than 2G
        }
        if (max_left_free_bytes > 1024L * 1024 * 1024) {
            if (params_.is_multimodal_) {
                other_reserve_mem += 1024L * 1024 * 1024; // just reserve 1024M for vit
            } else {
                other_reserve_mem += (size_t)128 * 1024 * 1024; // just reserve 128M for other, maybe can rm
            }
        }
        FT_CHECK_WITH_INFO(max_left_free_bytes > other_reserve_mem, "max_left_free_bytes %ld need to be larger than other_reserve_mem %ld", max_left_free_bytes, other_reserve_mem);
        max_left_free_bytes -= other_reserve_mem;
        FT_LOG_INFO("warm up done, max left free bytes: %ld, max runtime buffer bytes %ld", max_left_free_bytes, other_reserve_mem + max_runtime_buffer);
    }
    initCacheManager(max_left_free_bytes);
    FT_LOG_INFO("create cache manager done");
    executor_.reset(new NormalExecutor(params, resource_context_.cache_manager, device_, getLoraManager()));
    FT_LOG_INFO("create normal executor done");
    scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
    FT_LOG_INFO("create fifo scheduler done");
    (void)startLoop();
    if (device_->getDeviceProperties().tp_rank == 0) {
        initLoadBalance();
    }
}

NormalEngine::~NormalEngine() {
    FT_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(generate_input, params_, resource_context_, nullptr);
    if (mode == preRunMode::warm_up) {
        stream->setPerfTest(true);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUSOR_ERROR(stream->initKVBlock(0, 0));
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return streams.front();
}

size_t NormalEngine::warmUp(const EngineInitParams& params) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)params_.max_seq_len_ - 1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = params_.max_context_batch_size_;
    fake_input->generate_config->calculate_loss = int(params_.warm_up_with_loss_);
    device_->setTraceMemory(true);
    executor_.reset(new NormalExecutor(params, nullptr, device_, nullptr, true));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::warm_up));
    size_t min_preserved_bytes = device_->getDeviceStatus().device_memory_status.min_preserved_bytes;
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return min_preserved_bytes;
}

void NormalEngine::initLoadBalance() {
    FT_LOG_INFO("init load balance start");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = 3;
    fake_input->generate_config->top_k = 1;
    auto stream = enqueue(fake_input);
    while(!stream->finished() && !stream->stopped()) {
        FT_LOG_INFO("wait load balance int run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    FT_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)", step_recorder_.getStepPerMin(), step_recorder_.getStepLatency());
}

void NormalEngine::initCacheManager(size_t kv_cache_mem_size) {
    auto result = CacheConfigCreator::createConfig(params_);
    THROW_IF_STATUS_ERROR(result.status());
    if (kv_cache_mem_size) {
        uint32_t new_block_nums = kv_cache_mem_size / result.value().block_size;
        FT_LOG_INFO("try adjust block num %d to %d for warm up test result", result.value().block_nums, new_block_nums);
        result.value().block_nums = std::min(new_block_nums, result.value().block_nums); // choose min when both warm up and set reserve_runtime_mem_mb
    }
    resource_context_.cache_manager = make_shared<CacheManager>(result.value(), device_, metrics_reporter_);
}

absl::Status NormalEngine::initSystemPrompt() {
    if (device_->getDeviceProperties().tp_rank != 0) {
        return absl::OkStatus();
    }
    resource_context_.reuse_cache = params_.reuse_cache_;
    CHECK_AND_RETURN_REF(system_prompt_param, SystemPromptConstructor::construct(params_, this, resource_context_.cache_manager.get()));
    if (!system_prompt_param.empty()) {
        resource_context_.reuse_cache = true;
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }
    return absl::OkStatus();
}

LoadBalanceInfo NormalEngine::getLoadBalanceInfo() {
    auto kv_cache_info = resource_context_.cache_manager->getKVCacheInfo();
    return LoadBalanceInfo{
        (int64_t)step_recorder_.getStepLatency(),
        (int64_t)step_recorder_.getStepCount(),
        (int64_t)step_recorder_.getStepPerMin(),
        (int64_t)kv_cache_info.available_kv_cache,
        (int64_t)kv_cache_info.total_kv_cache
    };
}

absl::Status NormalEngine::startLoop() {
    FT_LOG_INFO("start normal engine loop");
    running_ = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    FT_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt()); // system prompt constructor depends on engine startup
    FT_LOG_INFO("init system prompt done");
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    FT_LOG_INFO("stop normal engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void NormalEngine::loop() {
    FT_LOG_INFO("loop begin");
    device_->preRun();
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(input, params_, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

absl::Status NormalEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        if (scheduler_->empty() || step_recorder_.empty()) {
            step_recorder_.reset();
            step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds());
        }
        CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }
    int64_t step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto status = executor_->process(streams);
    if (device_->getDeviceProperties().tp_rank == 0) {
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({false, false, step_latency});
        for (auto& stream: streams) {
            if (stream->finished()) {
                step_recorder_.addStepCount(stream->iterCount());
            }
        }
        step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds());
    }
    return status;
}

}  // namespace rtp_llm
