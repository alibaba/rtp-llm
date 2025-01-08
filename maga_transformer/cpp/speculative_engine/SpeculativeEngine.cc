#include <algorithm>
#include <cstdint>

#include "maga_transformer/cpp/speculative_engine/SpeculativeEngine.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/stream/StreamCacheResource.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "maga_transformer/cpp/cache/CacheConfigCreator.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeOnlineAdaptor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeScheduler.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "maga_transformer/cpp/utils/Logger.h"

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
    if (score_model_params_.gpt_init_parameter.warm_up_) {
        // warm up
        const ft::GptInitParameter& score_gpt_params = score_model_params_.gpt_init_parameter;
        FT_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                    score_gpt_params.max_context_batch_size_,
                    score_gpt_params.max_seq_len_,
                    int(score_gpt_params.warm_up_with_loss_));
        const auto result = warmUp();
        FT_LOG_INFO("warm up done, max runtime used bytes %ld", result.max_used_memory);
    }
    RETURN_IF_STATUS_ERROR(initCacheManager());
    FT_LOG_INFO("create cache manager done");
    propose_executor_ = createProposeExecutor(score_model_params_,
        propose_model_params_, device_, resource_context_.propose_cache_manager, getLoraManager());
    FT_LOG_INFO("create speculative executor done");
    score_executor_.reset(
        new ScoreExecutor(score_model_params_, device_, resource_context_.cache_manager, getLoraManager()));

    scheduler_.reset(
        new SpeculativeScheduler(score_model_params_.gpt_init_parameter, resource_context_.cache_manager, metrics_reporter_));
    FT_LOG_INFO("create fifo scheduler done");
    online_adaptor_.reset(new SpeculativeOnlineAdaptor());
    FT_LOG_INFO("create online adaptor");
    speculative_sampler_ = createSpeculativeSampler(propose_model_params_, device_);
    FT_LOG_INFO("create speculative sampler");
    speculative_updater_.reset(
        new SpeculativeUpdater(resource_context_, createSpeculativeUpdaterConfig(propose_model_params_)));
    RETURN_IF_STATUS_ERROR(startLoop());
    if (device_->getDeviceProperties().tp_rank == 0) {
        initLoadBalance();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::initLoadBalance() {
    FT_LOG_INFO("init load balance start");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config                 = make_shared<GenerateConfig>();
    fake_input->generate_config->max_new_tokens = 3;
    fake_input->generate_config->top_k          = 1;
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto stream                                 = enqueue(fake_input);
    while (!stream->finished() && !stream->stopped()) {
        FT_LOG_INFO("wait load balance int run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    FT_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)",
                step_recorder_.getStepPerMin(),
                step_recorder_.getStepLatency());
}

absl::StatusOr<GenerateStreamPtr> SpeculativeEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                            preRunMode                            mode) {
    std::shared_ptr<GenerateStream> score_stream = std::make_shared<NormalGenerateStream>(
        generate_input, score_model_params_.gpt_init_parameter, resource_context_, nullptr);
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
    THROW_IF_STATUS_ERROR(score_executor_->normalProcess(score_streams));

    if (propose_model_params_->gpt_model()) {
        THROW_IF_STATUS_ERROR(propose_executor_->normalProcess({propose_stream}));
    }

    return score_streams.front();
}

absl::Status SpeculativeEngine::initCacheManager() {
    if (propose_model_params_->gpt_model()) {
        const auto& config = CacheConfigCreator::createSpConfig(
            score_model_params_.gpt_init_parameter,
            propose_model_params_->vanilla_model_params->gpt_init_parameter);
        auto scorer_cache_config        = std::get<0>(config);
        auto proposer_cache_config      = std::get<1>(config);
        resource_context_.cache_manager = make_shared<CacheManager>(scorer_cache_config, device_, metrics_reporter_);
        resource_context_.propose_cache_manager =
            make_shared<CacheManager>(proposer_cache_config, device_, metrics_reporter_);
    } else {
        const auto& config = CacheConfigCreator::createConfig(score_model_params_.gpt_init_parameter);
        resource_context_.cache_manager = make_shared<CacheManager>(config, device_, metrics_reporter_);
    }
    return absl::OkStatus();
}

WarmUpResult SpeculativeEngine::warmUp() {
    const ft::GptInitParameter&    socre_gpt_params = score_model_params_.gpt_init_parameter;
    std::shared_ptr<GenerateInput> fake_input       = make_shared<GenerateInput>();
    fake_input->input_ids                           = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)socre_gpt_params.max_seq_len_ - 1}, ft::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config                       = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = socre_gpt_params.max_context_batch_size_;
    fake_input->generate_config->calculate_loss       = int(socre_gpt_params.warm_up_with_loss_);
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);

    score_executor_.reset(new ScoreExecutor(score_model_params_, device_, nullptr, nullptr, true));
    if (propose_model_params_->gpt_model()) {
        propose_executor_.reset(new VanillaExecutor(propose_model_params_, device_, nullptr, nullptr, true));
    }
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::warm_up));
    device_->setTraceMemory(false);
    (void)score_executor_.reset(nullptr);
    if (propose_model_params_->gpt_model()) {
        (void)propose_executor_.reset(nullptr);
    }
    return WarmUpResult();
}

absl::Status SpeculativeEngine::initSystemPrompt() {
    if (device_->getDeviceProperties().tp_rank == 0) {
        resource_context_.reuse_cache = score_model_params_.gpt_init_parameter.reuse_cache_;
        if (!score_model_params_.gpt_init_parameter.multi_task_prompt_tokens_.empty()) {
            resource_context_.reuse_cache = true;
            CHECK_AND_RETURN_REF(system_prompt_param,
                            SystemPromptConstructor::construct(
                                score_model_params_.gpt_init_parameter, this, resource_context_.cache_manager.get()));
            resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
        }
    } else {
        std::list<GenerateStreamPtr> streams;
        THROW_IF_STATUS_ERROR(score_executor_->normalProcess(streams));
    }
    return absl::OkStatus();
}

LoadBalanceInfo SpeculativeEngine::getLoadBalanceInfo() {
    auto kv_cache_info = resource_context_.cache_manager->getKVCacheInfo();
    return LoadBalanceInfo{(int64_t)step_recorder_.getStepLatency(),
                           (int64_t)step_recorder_.getStepCount(),
                           (int64_t)step_recorder_.getStepPerMin(),
                           (int64_t)kv_cache_info.available_kv_cache,
                           (int64_t)kv_cache_info.total_kv_cache};
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

void SpeculativeEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> SpeculativeEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, score_model_params_.gpt_init_parameter, resource_context_, metrics_reporter_);
    (void)scheduler_->enqueue(stream);
    return stream;
}

void SpeculativeEngine::tpSyncDisableSPRun(bool& all_streams_disable_sp_run) {
    if (device_->getDeviceProperties().tp_size <= 1) {
        return;
    }
    auto disable_sp_run = device_->allocateBuffer({ft::DataType::TYPE_INT32, {1}, ft::AllocationType::HOST});
    auto disable_sp_run_ptr = disable_sp_run->data<int32_t>();
    disable_sp_run_ptr[(size_t)0] = all_streams_disable_sp_run;

    device_->broadcast({{disable_sp_run}, 0});
    device_->syncCommunication(false);
    device_->syncAndCheck();
    all_streams_disable_sp_run = disable_sp_run_ptr[(size_t)0];
}

absl::Status SpeculativeEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // dynamic adjust propose_executor
    online_adaptor_->dynamicUpdateProposerConfig(propose_executor_, scheduler_);

    list<GenerateStreamPtr> streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        if (scheduler_->empty() || step_recorder_.empty()) {
            step_recorder_.reset();
            step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds(), propose_executor_->reserveStep() / 2);
        }
        CHECK_AND_ASSIGN(streams, scheduler_->schedule(propose_executor_->reserveStep()));
        if (streams.empty()) {
            return absl::OkStatus();
        }
    }

    for (auto& stream : streams) {
        FT_LOG_DEBUG("pre stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    bool all_streams_disable_sp_run = !streams.empty() && std::all_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->disableSpRun(); });
    tpSyncDisableSPRun(all_streams_disable_sp_run);
    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t score_begin_time_us = 0;
    int64_t sampler_begin_time_us = 0;
    int64_t update_begin_time_us = 0;
    int64_t total_propose_token_num  = 0;
    int64_t total_accepted_token_num = 0;

    ProposeOutput propose_output;
    if (!all_streams_disable_sp_run) {
        CHECK_AND_ASSIGN(propose_output, propose_executor_->propose(streams));
        FT_LOG_DEBUG("propose_output: %s", propose_output.debugString().c_str());
    }

    // fast path for no propose and all_streams_disable_sp_run
    if (all_streams_disable_sp_run || propose_output.hasNoPropose()) {
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        THROW_IF_STATUS_ERROR(score_executor_->normalProcess(streams));
        sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        update_begin_time_us = sampler_begin_time_us;
        total_propose_token_num = 0;
        total_accepted_token_num = streams.size();
        for (auto& stream : streams) {
            stream->setReuseLength(stream->seqLength() - 1);
            stream->setFallbackPrefixLength(stream->reuseLength());
            stream->setSpEditRun(false);
            FT_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_tokens = 0, accept_tokens = 1",
                    stream->streamId(),
                    stream->generateConfig()->top_k,
                    stream->generateConfig()->top_p);
        }
    } else {
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(score_output, score_executor_->score(streams, propose_output));
        FT_LOG_DEBUG("score_output: %s", score_output.debugString().c_str());

        if (device_->getDeviceProperties().tp_rank == 0) {
            sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams, propose_output, score_output));
            FT_LOG_DEBUG("sampler_output: %s", sampler_output.debugString().c_str());

            update_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            RETURN_IF_STATUS_ERROR(speculative_updater_->update(streams, sampler_output));

            for (const auto& output : sampler_output.outputs) {
                total_propose_token_num += output.propose_step;
                total_accepted_token_num += output.accepted_token_nums;
            }
        }
    }

    for (auto& stream : streams) {
        FT_LOG_DEBUG("post stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    if (device_->getDeviceProperties().tp_rank == 0) {
        reportMetrics(propose_begin_time_us,
                    score_begin_time_us,
                    sampler_begin_time_us,
                    update_begin_time_us,
                    total_propose_token_num,
                    total_accepted_token_num);

        for (auto& stream : streams) {
            if (stream->finished()) {
                step_recorder_.addStepCount(stream->iterCount());
            }
        }
        step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds(), total_accepted_token_num / streams.size());
    }

    return absl::OkStatus();
}

void SpeculativeEngine::reportMetrics(int64_t                         propose_begin_time_us,
                                      int64_t                         score_begin_time_us,
                                      int64_t                         sampler_begin_time_us,
                                      int64_t                         update_begin_time_us,
                                      int64_t                         total_propose_token_num,
                                      int64_t                         total_accepted_token_num) {
    if (!metrics_reporter_) {
        return;
    }

    int64_t                                 current_time    = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t                                 propose_time    = score_begin_time_us - propose_begin_time_us;
    int64_t                                 score_time      = sampler_begin_time_us - score_begin_time_us;
    int64_t                                 sampler_time    = update_begin_time_us - sampler_begin_time_us;
    int64_t                                 update_time     = current_time - update_begin_time_us;
    int64_t                                 total_step_time = current_time - propose_begin_time_us;
    FT_LOG_DEBUG("total_step_time: %ld, propose_time: %ld, score_time: %ld, sampler_time: %ld, update_time: %ld",
                total_step_time, propose_time, score_time, sampler_time, update_time);
    RtpLLMSpeculativeEngineMetricsCollector collector{total_step_time,
                                                      propose_time,
                                                      score_time,
                                                      sampler_time,
                                                      update_time,
                                                      total_propose_token_num,
                                                      total_accepted_token_num};
    metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(nullptr,
                                                                                                       &collector);
}

}  // namespace rtp_llm
