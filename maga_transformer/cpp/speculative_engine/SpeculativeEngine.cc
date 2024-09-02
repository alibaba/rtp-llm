#include <cstdint>

#include "maga_transformer/cpp/speculative_engine/SpeculativeEngine.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeOnlineAdaptor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/utils/logger.h"

using namespace std;
namespace rtp_llm {

SpeculativeEngine::SpeculativeEngine(const EngineInitParams&                       engine_init_params,
                                     std::unique_ptr<ProposeModelEngineInitParams> propose_model_engine_init_params):
    EngineBase(engine_init_params),
    metrics_reporter_(engine_init_params.metrics_reporter),
    propose_model_params_(std::move(propose_model_engine_init_params)),
    score_model_params_(std::move(engine_init_params)) {}

SpeculativeEngine::~SpeculativeEngine() {
    FT_LOG_INFO("destory speculative engine");
    (void)stop();
}

absl::Status SpeculativeEngine::init() {
    FT_LOG_INFO(__PRETTY_FUNCTION__);
    size_t max_left_free_bytes = 0;
    if (score_model_params_.gpt_init_parameter.warm_up_) {
        // warm up
        const ft::GptInitParameter& score_gpt_params = score_model_params_.gpt_init_parameter;
        FT_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin", score_gpt_params.max_context_batch_size_, score_gpt_params.max_seq_len_, int(score_gpt_params.warm_up_with_loss_));
        max_left_free_bytes = warmUp();
        if (max_left_free_bytes > 1024L * 1024 * 1024) {
            max_left_free_bytes -= 128L * 1024 * 1024; // just reserve 128M for other, maybe can rm
        }
        FT_LOG_INFO("warm up done, max left free bytes: %ld, max runtime buffer bytes %ld", max_left_free_bytes, device_->getDeviceStatus().device_memory_status.preserved_bytes - max_left_free_bytes);
    }
    RETURN_IF_STATUS_ERROR(initCacheManager());
    FT_LOG_INFO("create cache manager done");
    propose_executor_ =
        createProposeExecutor(propose_model_params_, device_, resource_context_.propose_cache_manager, getLoraManager());
    FT_LOG_INFO("create speculative executor done");
    score_executor_.reset(
        new ScoreExecutor(score_model_params_, device_, resource_context_.cache_manager, getLoraManager()));

    scheduler_.reset(
        new FIFOScheduler(score_model_params_.gpt_init_parameter, resource_context_.cache_manager, metrics_reporter_));
    FT_LOG_INFO("create fifo scheduler done");
    online_adaptor_.reset(new SpeculativeOnlineAdaptor());
    FT_LOG_INFO("create online adaptor");
    speculative_sampler_ = createSpeculativeSampler(propose_model_params_, device_);
    FT_LOG_INFO("create speculative sampler");
    speculative_updater_.reset(new SpeculativeUpdater(resource_context_, createSpeculativeUpdaterConfig(propose_model_params_)));
    return startLoop();
}

absl::StatusOr<GenerateStreamPtr> SpeculativeEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) {
    std::shared_ptr<GenerateStream> score_stream = std::make_shared<NormalGenerateStream>(generate_input, score_model_params_.gpt_init_parameter, resource_context_, nullptr);
    std::shared_ptr<GenerateStream> propose_stream = nullptr;
    if (mode == preRunMode::warm_up) {
        score_stream->setPerfTest(true);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUSOR_ERROR(score_stream->initKVBlock(0, 0));
    };

    if (propose_model_params_->gpt_model()) {
        propose_stream = std::make_shared<NormalGenerateStream>(*score_stream);
    }

    std::list<GenerateStreamPtr> score_streams{score_stream};
    THROW_IF_STATUS_ERROR(score_executor_->process(score_streams));

    if (propose_model_params_->gpt_model()) {
        THROW_IF_STATUS_ERROR(propose_executor_->process({propose_stream}));
    }

    return score_streams.front();
}

absl::Status SpeculativeEngine::initCacheManager() {
    if (propose_model_params_->gpt_model()) {
        CHECK_AND_RETURN_CONST_REF(
            config,
            CacheConfigCreator::createSpConfig(score_model_params_.gpt_init_parameter,
                                               propose_model_params_->vanilla_model_params->gpt_init_parameter));
        auto scorer_cache_config        = std::get<0>(config);
        auto proposer_cache_config      = std::get<1>(config);
        resource_context_.cache_manager = make_shared<CacheManager>(scorer_cache_config, device_, metrics_reporter_);
        resource_context_.propose_cache_manager = make_shared<CacheManager>(proposer_cache_config, device_, metrics_reporter_);
    } else {
        CHECK_AND_RETURN_CONST_REF(config, CacheConfigCreator::createConfig(score_model_params_.gpt_init_parameter));
        resource_context_.cache_manager = make_shared<CacheManager>(config, device_, metrics_reporter_);
    }
    return absl::OkStatus();
}

size_t SpeculativeEngine::warmUp() {
    const ft::GptInitParameter& socre_gpt_params = score_model_params_.gpt_init_parameter; 
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)socre_gpt_params.max_seq_len_ - 1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config               = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = socre_gpt_params.max_context_batch_size_;
    fake_input->generate_config->calculate_loss = int(socre_gpt_params.warm_up_with_loss_);
    device_->setTraceMemory(true);

    score_executor_.reset(new ScoreExecutor(score_model_params_, device_, nullptr, nullptr, true));
    if (propose_model_params_->gpt_model()) {
        propose_executor_.reset(new VanillaExecutor(propose_model_params_, device_, nullptr, nullptr, true));
    }
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::warm_up));
    size_t min_preserved_bytes = device_->getDeviceStatus().device_memory_status.min_preserved_bytes;
    device_->setTraceMemory(false);
    (void)score_executor_.reset(nullptr);
    if (propose_model_params_->gpt_model()) {
        (void)propose_executor_.reset(nullptr);
    }
    return min_preserved_bytes;
}

absl::Status SpeculativeEngine::initSystemPrompt() {
    if (device_->getDeviceProperties().tp_rank != 0) {
        return absl::OkStatus();
    }
    resource_context_.reuse_cache = score_model_params_.gpt_init_parameter.reuse_cache_;
    CHECK_AND_RETURN_REF(system_prompt_param, SystemPromptConstructor::construct(
        score_model_params_.gpt_init_parameter, this, resource_context_.cache_manager.get()));
    if (!system_prompt_param.empty()) {
        resource_context_.reuse_cache = true;
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }
    return absl::OkStatus();
}

KVCacheInfo SpeculativeEngine::getKVCacheInfo() const {
    return resource_context_.cache_manager->getKVCacheInfo();
}

absl::Status SpeculativeEngine::startLoop() {
    FT_LOG_INFO("start speculative engine loop");
    running_     = true;
    loop_thread_ = std::thread(&SpeculativeEngine::loop, this);
    FT_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt());  // system prompt constructor depends on engine startup
    FT_LOG_INFO("init system prompt done");
    return absl::OkStatus();
}

absl::Status SpeculativeEngine::stop() {
    FT_LOG_INFO("stop speculative engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::loop() {
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

absl::Status SpeculativeEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> SpeculativeEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, score_model_params_.gpt_init_parameter, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

absl::Status SpeculativeEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // dynamic adjust propose_executor
    online_adaptor_->dynamicUpdateProposerConfig(propose_executor_, scheduler_);

    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        CHECK_AND_ASSIGN(streams, scheduler_->schedule(propose_executor_->reserveStep()));
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }
    
    for (auto& stream: streams) {
        FT_LOG_DEBUG("pre stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    CHECK_AND_RETURN_REF(propose_output, propose_executor_->propose(streams));
    FT_LOG_DEBUG("propose_output: %s", propose_output.debugString().c_str());
    
    int64_t score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    CHECK_AND_RETURN_REF(score_output, score_executor_->score(streams, propose_output));
    FT_LOG_DEBUG("score_output: %s", score_output.debugString().c_str());

    if (device_->getDeviceProperties().tp_rank == 0) {
        int64_t sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams, propose_output, score_output));
        FT_LOG_DEBUG("sampler_output: %s", sampler_output.debugString().c_str());

        int64_t update_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        RETURN_IF_STATUS_ERROR(speculative_updater_->update(streams, sampler_output));

        for (auto& stream: streams) {
           FT_LOG_DEBUG("post stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
        }
        reportMetrics(sampler_output, propose_begin_time_us, score_begin_time_us, sampler_begin_time_us, update_begin_time_us);
    }

    return absl::OkStatus();
}

void SpeculativeEngine::reportMetrics(const SpeculativeSamplerOutput& sampler_output, int64_t propose_begin_time_us, int64_t score_begin_time_us, int64_t sampler_begin_time_us, int64_t update_begin_time_us) {
    if (metrics_reporter_ == nullptr) {
        return;
    }

    int64_t        total_propose_token_num = 0;
    int64_t        total_accepted_token_num = 0;

    for (const auto& output : sampler_output.outputs) {
        total_propose_token_num += output.propose_step;
        total_accepted_token_num += output.accepted_token_nums;
    }

    int64_t current_time = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t propose_time = score_begin_time_us - propose_begin_time_us;
    int64_t score_time = sampler_begin_time_us - score_begin_time_us;
    int64_t sampler_time = update_begin_time_us - sampler_begin_time_us;
    int64_t update_time = current_time - update_begin_time_us;
    int64_t total_step_time = current_time - propose_begin_time_us;
    RtpLLMSpeculativeEngineMetricsCollector collector{total_step_time, propose_time, score_time, sampler_time, update_time, total_propose_token_num, total_accepted_token_num};
    metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(nullptr, &collector);
}

}  // namespace rtp_llm