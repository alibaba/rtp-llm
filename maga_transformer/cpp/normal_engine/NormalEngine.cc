#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/utils/logger.h"
#include "autil/TimeUtility.h"
#include <memory>

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
        if (max_left_free_bytes > 1024L * 1024 * 1024) {
            if (params_.is_multimodal_) {
                max_left_free_bytes -= 1024L * 1024 * 1024; // just reserve 1024M for vit
            } else {
                max_left_free_bytes -= 128L * 1024 * 1024; // just reserve 128M for other, maybe can rm
            }
        }
        FT_LOG_INFO("warm up done, max left free bytes: %ld, max runtime buffer bytes %ld", max_left_free_bytes, device_->getDeviceStatus().device_memory_status.preserved_bytes - max_left_free_bytes);
    }
    initCacheManager(max_left_free_bytes);
    FT_LOG_INFO("create cache manager done");
    executor_.reset(new NormalExecutor(params, resource_context_.cache_manager, device_, getLoraManager()));
    FT_LOG_INFO("create normal executor done");
    scheduler_.reset(new FIFOScheduler(params_, resource_context_.cache_manager, metrics_reporter_));
    FT_LOG_INFO("create fifo scheduler done");
    (void)startLoop();
}

NormalEngine::~NormalEngine() {
    FT_LOG_INFO("destory normal engine");
    (void)stop();
}

size_t NormalEngine::warmUp(const EngineInitParams& params) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)params_.max_seq_len_ - 1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = params_.max_context_batch_size_;
    fake_input->generate_config->calculate_loss = int(params_.warm_up_with_loss_);
    device_->setTraceMemory(true);
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(fake_input, params_, resource_context_, nullptr);
    stream->setPerfTest(true);
    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream);
    executor_.reset(new NormalExecutor(params, nullptr, device_, nullptr, true));
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    size_t min_preserved_bytes = device_->getDeviceStatus().device_memory_status.min_preserved_bytes;
    device_->setTraceMemory(false);
    (void)executor_.reset(nullptr);
    return min_preserved_bytes;
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

void NormalEngine::initSystemPrompt() {
    if (device_->getDeviceProperties().tp_rank != 0) {
        return;
    }
    resource_context_.reuse_cache = params_.reuse_cache_;
    auto system_prompt_param = SystemPromptConstructor::construct(params_, this, resource_context_.cache_manager.get());
    if (!system_prompt_param.empty()) {
        resource_context_.reuse_cache = true;
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }
}

KVCacheInfo NormalEngine::getKVCacheInfo() const {
    return resource_context_.cache_manager->getKVCacheInfo();
}

absl::Status NormalEngine::startLoop() {
    FT_LOG_INFO("start normal engine loop");
    running_ = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    FT_LOG_INFO("start init system prompt");
    initSystemPrompt(); // system prompt constructor depends on engine startup
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
        int64_t step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
        reportMetrics({false, false, autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us});
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
        CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }
    return executor_->process(streams);
}

}  // namespace rtp_llm
