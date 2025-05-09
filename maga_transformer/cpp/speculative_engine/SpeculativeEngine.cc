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
#include "maga_transformer/cpp/speculative_engine/propose_executor/MTPExecutor.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"

using namespace std;


namespace rtp_llm {

SpeculativeEngine::SpeculativeEngine(const EngineInitParams&                       engine_init_params,
                                     std::unique_ptr<ProposeModelEngineInitParams> propose_model_engine_init_params):
    EngineBase(engine_init_params),
    metrics_reporter_(engine_init_params.metrics_reporter),
    propose_model_params_(std::move(propose_model_engine_init_params)),
    score_model_params_(std::move(engine_init_params)),
    sp_type_(propose_model_params_->sp_type) {};

SpeculativeEngine::~SpeculativeEngine() {
    RTP_LLM_LOG_INFO("destory speculative engine");
    (void)stop();
}

std::shared_ptr<GenerateStream> SpeculativeEngine::enqueueMinFakeQuery(int32_t max_new_tokens, bool fake_hidden_states) {
    RTP_LLM_LOG_DEBUG("enqueue min fake query");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)1}, rtp_llm::AllocationType::HOST});

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, score_model_params_.gpt_init_parameter.vocab_size_ - 1);
    for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
        *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    }

    fake_input->generate_config               = make_shared<GenerateConfig>();
    if (fake_hidden_states) {
        fake_input->generate_config->max_new_tokens = max_new_tokens + 1;
    } else {
        fake_input->generate_config->max_new_tokens = max_new_tokens;
    }
    fake_input->generate_config->top_k = 1;
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query = true;
    auto stream = makeStream(fake_input);
    stream->setMetricsReporter(nullptr);

    if (fake_hidden_states) {
        auto dtype = rtp_llm::getDataType(score_model_params_.gpt_init_parameter.data_type_);
        auto fake_hidden_states = device_->allocateBuffer(
            {dtype, {1, (size_t)score_model_params_.gpt_init_parameter.hidden_size_}, rtp_llm::AllocationType::DEVICE});
        stream->setReturnLastHiddenStates(true);
        BufferPtr new_tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {(size_t)1, 1}, rtp_llm::AllocationType::HOST});
        *new_tokens->dataWithOffset<int32_t>(0) = distribution(generator);
        StreamUpdateInfo update_info{new_tokens, (int)1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, fake_hidden_states};
        stream->update(update_info);
        stream->setIsContextStream(false);
    }

    enqueue(stream);
    return stream;
}

absl::Status SpeculativeEngine::init() {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
    if (score_model_params_.gpt_init_parameter.warm_up_) {
        // warm up
        const rtp_llm::GptInitParameter& score_gpt_params = score_model_params_.gpt_init_parameter;
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                    score_gpt_params.max_context_batch_size_,
                    score_gpt_params.max_seq_len_,
                    int(score_gpt_params.warm_up_with_loss_));
        warm_up_result = warmUp();
        RTP_LLM_LOG_INFO("warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
                    warm_up_result->max_used_memory,
                    warm_up_result->max_used_memory / 1024 / 1024,
                    warm_up_result->device_reserved_bytes,
                    warm_up_result->device_reserved_bytes / 1024 / 1024);
    }
    RETURN_IF_STATUS_ERROR(initCacheManager(warm_up_result));
    RTP_LLM_LOG_INFO("create cache manager done");
    propose_executor_ = createProposeExecutor(score_model_params_,
        propose_model_params_, device_,
        resource_context_.propose_cache_manager,
        resource_context_.mtp_cache_managers,
        getLoraManager());
    RTP_LLM_LOG_INFO("create speculative executor done");
    score_executor_.reset(
        new ScoreExecutor(score_model_params_, device_, resource_context_.cache_manager, getLoraManager()));

    scheduler_.reset(
        new SpeculativeScheduler(score_model_params_.gpt_init_parameter, resource_context_.cache_manager, metrics_reporter_));
    RTP_LLM_LOG_INFO("create fifo scheduler done");
    online_adaptor_.reset(new SpeculativeOnlineAdaptor());
    RTP_LLM_LOG_INFO("create online adaptor");
    speculative_sampler_ = createSpeculativeSampler(propose_model_params_, device_);
    RTP_LLM_LOG_INFO("create speculative sampler");
    speculative_updater_.reset(
        new SpeculativeUpdater(resource_context_, createSpeculativeUpdaterConfig(propose_model_params_)));
    RETURN_IF_STATUS_ERROR(startLoop());
    if (device_->getDeviceProperties().tp_rank == 0) {
        initLoadBalance();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::initLoadBalance() {
    RTP_LLM_LOG_INFO("init load balance start");
    auto stream = enqueueMinFakeQuery(3,  false);
    while(!stream->finished() && !stream->stopped()) {
        RTP_LLM_LOG_INFO("wait load balance init run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    RTP_LLM_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)",
            step_recorder_.getStepPerMin(), step_recorder_.getStepLatency());
}

absl::StatusOr<GenerateStreamPtr> SpeculativeEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                            preRunMode                            mode) {
    std::shared_ptr<GenerateStream> score_stream = std::make_shared<NormalGenerateStream>(
        generate_input, score_model_params_.gpt_init_parameter, resource_context_, nullptr);
    std::shared_ptr<GenerateStream> propose_stream = nullptr;
    if (mode == preRunMode::prefill_warm_up) {
        score_stream->setPerfTest(true);
    } else if (mode == preRunMode::decode_warm_up) {
        score_stream->setIsContextStream(false);
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

absl::Status SpeculativeEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    if (propose_model_params_->gpt_model()) {
        const auto& config = CacheConfigCreator::createSpConfig(
            score_model_params_.gpt_init_parameter,
            propose_model_params_->getGptInitParameter(),
            warm_up_result,
            propose_model_params_->isMTP());
        auto scorer_cache_config        = std::get<0>(config);
        auto proposer_cache_config      = std::get<1>(config);
        resource_context_.cache_manager = make_shared<CacheManager>(scorer_cache_config, device_, false, metrics_reporter_);
        if (propose_model_params_->isMTP()) {
            auto layer_num = propose_model_params_->getGptInitParameter().num_layers_;
            RTP_LLM_LOG_INFO("mtp cache manager init use layer num : %d", layer_num);
            for (int i = 0; i < layer_num; i++) {
                RTP_LLM_CHECK(proposer_cache_config.layer_num == 1);
                resource_context_.mtp_cache_managers.push_back(
                    std::make_shared<CacheManager>(proposer_cache_config, device_, false, metrics_reporter_)
                );
            }
        } else {
            resource_context_.propose_cache_manager =
                make_shared<CacheManager>(proposer_cache_config, device_, false, metrics_reporter_);
        }

    } else {
        const auto& config = CacheConfigCreator::createConfig(score_model_params_.gpt_init_parameter, warm_up_result);
        resource_context_.cache_manager = make_shared<CacheManager>(config, device_, false, metrics_reporter_);
    }
    return absl::OkStatus();
}

WarmUpResult SpeculativeEngine::warmUp() {
    const rtp_llm::GptInitParameter&    socre_gpt_params = score_model_params_.gpt_init_parameter;
    std::shared_ptr<GenerateInput> fake_input       = make_shared<GenerateInput>();
    fake_input->input_ids                           = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)socre_gpt_params.max_seq_len_ - 1}, rtp_llm::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config                       = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = socre_gpt_params.max_context_batch_size_;
    fake_input->generate_config->calculate_loss       = int(socre_gpt_params.warm_up_with_loss_);
    fake_input->generate_config->top_k                = 2;
    fake_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);

    score_executor_.reset(new ScoreExecutor(score_model_params_, device_, nullptr, nullptr, true));
    if (propose_model_params_->isVanilla()) {
        propose_executor_.reset(new VanillaExecutor(propose_model_params_, device_, nullptr, nullptr, true));
    } else if (propose_model_params_->isMTP()) {
        propose_executor_.reset(new MTPExecutor(propose_model_params_, device_, {nullptr}, nullptr, true));
    }
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)score_executor_.reset(nullptr);
    if (propose_model_params_->gpt_model()) {
        (void)propose_executor_.reset(nullptr);
    }
    return WarmUpResult({
        device_status.device_memory_status.preserved_bytes,
        device_status.device_memory_status.max_consumed_bytes});
}

absl::Status SpeculativeEngine::initSystemPrompt() {
    resource_context_.reuse_cache = score_model_params_.gpt_init_parameter.reuse_cache_;

    if (!score_model_params_.gpt_init_parameter.multi_task_prompt_tokens_.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(system_prompt_param,
                        SystemPromptConstructor::construct(
                            score_model_params_.gpt_init_parameter, this, resource_context_.cache_manager.get(), device_->getDeviceProperties().tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }
    return absl::OkStatus();
}

LoadBalanceInfo SpeculativeEngine::getLoadBalanceInfo() {
    auto kv_cache_info = resource_context_.cache_manager->getKVCacheInfo();
    return LoadBalanceInfo{(int64_t)step_recorder_.getStepLatency(),
                           (int64_t)step_recorder_.getStepCount(),
                           (int64_t)step_recorder_.getStepPerMin(),
                           (int64_t)kv_cache_info.available_kv_cache,
                           (int64_t)kv_cache_info.total_kv_cache,
                           (int64_t)scheduler_->onflightStreams()};
}

absl::Status SpeculativeEngine::startLoop() {
    RTP_LLM_LOG_INFO("start init system prompt");
    THROW_IF_STATUS_ERROR(initSystemPrompt());
    RTP_LLM_LOG_INFO("init system prompt done");
    RTP_LLM_LOG_INFO("start speculative engine loop");
    running_     = true;
    loop_thread_ = std::thread(&SpeculativeEngine::loop, this);
    return absl::OkStatus();
}

absl::Status SpeculativeEngine::stop() {
    RTP_LLM_LOG_INFO("stop speculative engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::loop() {
    RTP_LLM_LOG_INFO("loop begin");
    device_->preRun();
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status SpeculativeEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> SpeculativeEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(input, score_model_params_.gpt_init_parameter, resource_context_, metrics_reporter_);
    return stream;
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
    auto disable_sp_run = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST});
    auto disable_sp_run_ptr = disable_sp_run->data<int32_t>();
    disable_sp_run_ptr[(size_t)0] = all_streams_disable_sp_run;

    device_->broadcast({{disable_sp_run}, 0});
    device_->syncCommunication(false);
    device_->syncAndCheck();
    all_streams_disable_sp_run = disable_sp_run_ptr[(size_t)0];
}

void SpeculativeEngine::dpAndTpSyncNeedHiddenStates(bool& need_hidden_states) {
    const auto properties = device_->getDeviceProperties();
    size_t world_size = properties.dp_size;
    if (world_size <= 1) {
        return;
    }
    size_t local_rank = properties.dp_rank;
    RTP_LLM_LOG_DEBUG("local_rank is %d", local_rank);
    auto flag = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {world_size}, rtp_llm::AllocationType::HOST});
    auto flag_ptr = flag->data<int32_t>();
    flag_ptr[(size_t)local_rank] = need_hidden_states;
    printBufferData(*flag, "before dp flag");

    device_->allGather({{flag}, ParallelMode::DP});
    device_->syncCommunication(false);
    device_->syncAndCheck();
    printBufferData(*flag, "after dp flag");
    need_hidden_states = (std::accumulate(flag_ptr, flag_ptr + world_size, 0) >= 1);
}

absl::Status SpeculativeEngine::step() {

    list<GenerateStreamPtr> streams;

    if (device_->getDeviceProperties().tp_rank == 0) {
        if (scheduler_->empty() || step_recorder_.empty()) {
            step_recorder_.reset();
            step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds(), propose_executor_->reserveStep() / 2);
        }
        auto reserve_step = propose_executor_->reserveStep() + 1;

        CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step));

        if (streams.empty()) {
            if (score_model_params_.gpt_init_parameter.dp_size_ > 1) {
                if (score_model_params_.gpt_init_parameter.pd_separation_ == 1) {
                    enqueueMinFakeQuery(1, false);
                } else {
                    enqueueMinFakeQuery(1, true);
                }
            }
            return absl::OkStatus();
        }
        if (score_model_params_.gpt_init_parameter.dp_size_ > 1 &&
            score_model_params_.gpt_init_parameter.pd_separation_ == 0)
        {
            bool has_hidden_states = false;
            for (auto stream : streams) {
                if (stream->getLastHiddenStates() != nullptr) {
                    has_hidden_states = true;
                    break;
                }
            }
            if (!has_hidden_states) {
                enqueueMinFakeQuery(1, true);
                return absl::OkStatus();
            }
        }
    }

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("pre stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    bool all_streams_disable_sp_run = !streams.empty() && std::all_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->disableSpRun(); });
    tpSyncDisableSPRun(all_streams_disable_sp_run);

    if (all_streams_disable_sp_run) {
        if (sp_type_ == "mtp") {
            for (auto& stream : streams) {
                stream->setReturnLastHiddenStates(true);
            }
        }
        return normStep(streams);
    }
    if (sp_type_ == "mtp") {
        // Make sure each stream is able to save the hidden states value in each calculation result
        for (auto& stream : streams) {
            stream->setReturnLastHiddenStates(true);
        }
        return noPrefillProposeStep(streams);
    } else {
        return prefillProposeStep(streams);
    }

}


absl::Status SpeculativeEngine::normStep(std::list<GenerateStreamPtr>& streams) {
    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t score_begin_time_us = 0;
    int64_t sampler_begin_time_us = 0;
    int64_t update_begin_time_us = 0;
    int64_t total_propose_token_num  = 0;
    int64_t total_accepted_token_num = 0;

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
        RTP_LLM_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_tokens = 0, accept_tokens = 1",
                stream->streamId(),
                stream->generateConfig()->top_k,
                stream->generateConfig()->top_p);
    }
    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("post stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
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

absl::Status SpeculativeEngine::prefillProposeStep(std::list<GenerateStreamPtr>& streams) {
    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t score_begin_time_us = 0;
    int64_t sampler_begin_time_us = 0;
    int64_t update_begin_time_us = 0;
    int64_t total_propose_token_num  = 0;
    int64_t total_accepted_token_num = 0;
    ProposeOutput propose_output;
    CHECK_AND_ASSIGN(propose_output, propose_executor_->propose(streams));
    RTP_LLM_LOG_DEBUG("propose_output: %s", propose_output.debugString().c_str());
    if (propose_output.hasNoPropose()) {
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
            RTP_LLM_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_tokens = 0, accept_tokens = 1",
                    stream->streamId(),
                    stream->generateConfig()->top_k,
                    stream->generateConfig()->top_p);
        }
    } else {
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(score_output, score_executor_->score(streams, propose_output));
        RTP_LLM_LOG_DEBUG("score_output: %s", score_output.debugString().c_str());

        if (device_->getDeviceProperties().tp_rank == 0) {
            sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams, propose_output, score_output));
            RTP_LLM_LOG_DEBUG("sampler_output: %s", sampler_output.debugString().c_str());

            update_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            RETURN_IF_STATUS_ERROR(speculative_updater_->update(streams, sampler_output));

            for (const auto& output : sampler_output.outputs) {
                total_propose_token_num += output.propose_step;
                total_accepted_token_num += output.accepted_token_nums;
            }
        }
    }

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("post stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
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

bool SpeculativeEngine::checkAllHasHiddenStates(std::list<GenerateStreamPtr>& streams) {
    bool flag = true;
    for (auto& stream : streams) {
        if (stream->getLastHiddenStates() == nullptr) {
            flag = false;
        }
    }
    flag = !streams.empty() && flag;
    tpSyncDisableSPRun(flag);
    return flag;
};

std::list<GenerateStreamPtr> SpeculativeEngine::extractFirstPrefillStreams(std::list<GenerateStreamPtr>& streams) {
    std::list<GenerateStreamPtr> need_prefill;
    for (auto& stream : streams) {
        if (stream->getLastHiddenStates() == nullptr) {
            need_prefill.push_back(stream);
        }
    }
   return need_prefill;
}


absl::Status SpeculativeEngine::noPrefillProposeStep(std::list<GenerateStreamPtr>& streams) {
    std::list<GenerateStreamPtr> propose_streams;
    std::list<GenerateStreamPtr> prefill_streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        for (auto& stream: streams) {
            if (stream->getLastHiddenStates() != nullptr) {
                propose_streams.emplace_back(stream);
            } else {
                prefill_streams.emplace_back(stream);
            }
        }

        for (auto& stream : propose_streams) {
            RTP_LLM_LOG_DEBUG("pre propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
        }


        for (auto& stream : prefill_streams) {
            RTP_LLM_LOG_DEBUG("pre prefill stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
        }

    }

    // base model generate current hidden states.
    // mtp model according to last hidden states from base model,
    // generate one token.
    // TODO(lidongjin) support multi mtp model.
    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t total_propose_token_num  = 0;
    int64_t total_accepted_token_num = 0;
    int64_t score_begin_time_us      = 0;
    int64_t sampler_begin_time_us  = 0;
    int64_t update_begin_time_us  = 0;

    ProposeOutput propose_output;
    {
        bool skip_propose = propose_streams.empty();
        tpSyncDisableSPRun(skip_propose);
        if (!skip_propose) {
            RTP_LLM_LOG_DEBUG("propose step");
            CHECK_AND_ASSIGN(propose_output, propose_executor_->propose(propose_streams));
            RTP_LLM_LOG_DEBUG("propose_output: %s", propose_output.debugString().c_str());
        } else {
            RTP_LLM_LOG_DEBUG("skip propose");
        }


        for (const GenerateStreamPtr& stream : prefill_streams) {
            size_t stream_id = stream->streamId();
            propose_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
            propose_output.outputs[stream_id]->propose_step = 0;

        }
    }

    // base model score propose new tokens.
    {
        RTP_LLM_LOG_DEBUG("score step");
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(score_output, score_executor_->score(streams, propose_output));
        RTP_LLM_LOG_DEBUG("score_output: %s", score_output.debugString().c_str());

        if (device_->getDeviceProperties().tp_rank == 0) {
            RTP_LLM_LOG_DEBUG("sample step");
            sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams, propose_output, score_output));
            RTP_LLM_LOG_DEBUG("sampler_output: %s", sampler_output.debugString().c_str());
            RTP_LLM_LOG_DEBUG("update step");
            update_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            RETURN_IF_STATUS_ERROR(speculative_updater_->update(streams, sampler_output));
            for (const auto& output : sampler_output.outputs) {
                total_propose_token_num += output.propose_step;
                total_accepted_token_num += output.accepted_token_nums;
            }
        }

        for (auto& stream : streams) {
            RTP_LLM_LOG_DEBUG("post stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
        }
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
    RTP_LLM_LOG_DEBUG("total_step_time: %ld, propose_time: %ld, score_time: %ld, sampler_time: %ld, update_time: %ld",
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

bool SpeculativeEngine::updateEplbConfig(const EplbConfig& config)
{
    if (score_executor_ && propose_executor_) {
        return score_executor_->updateEplbConfig(config) &&
               propose_executor_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
