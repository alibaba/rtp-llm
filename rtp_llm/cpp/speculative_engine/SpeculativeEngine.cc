#include <algorithm>
#include <cstdint>

#include "rtp_llm/cpp/speculative_engine/SpeculativeEngine.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/EagleExecutor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPStream.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeScheduler.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPExecutor.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace std;

namespace rtp_llm {

size_t CudaProfiler_E::count = 0;

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

std::shared_ptr<GenerateStream> SpeculativeEngine::enqueueMinFakeQuery(int32_t max_new_tokens,
                                                                       bool    fake_hidden_states) {
    RTP_LLM_LOG_DEBUG("enqueue min fake query");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1}, rtp_llm::AllocationType::HOST});

    std::default_random_engine         generator;
    std::uniform_int_distribution<int> distribution(0, score_model_params_.gpt_init_parameter.vocab_size_ - 1);
    for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
        *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    }

    fake_input->generate_config = make_shared<GenerateConfig>();
    if (fake_hidden_states) {
        fake_input->generate_config->max_new_tokens = max_new_tokens + 1;
    } else {
        fake_input->generate_config->max_new_tokens = max_new_tokens;
    }
    fake_input->generate_config->top_k = 1;
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->fake_query             = true;
    auto stream                        = makeStream(fake_input);
    stream->setIsDummyStream(true);
    stream->setMetricsReporter(nullptr);

    if (fake_hidden_states) {
        auto      dtype = score_model_params_.gpt_init_parameter.data_type_;
        BufferPtr fake_hidden_states;
        if (sp_type_ == "eagle3") {
            fake_hidden_states =
                device_->allocateBuffer({dtype,
                                         {1, (size_t)score_model_params_.gpt_init_parameter.hidden_size_ * 3},
                                         rtp_llm::AllocationType::DEVICE});
        } else {
            fake_hidden_states =
                device_->allocateBuffer({dtype,
                                         {1, (size_t)score_model_params_.gpt_init_parameter.hidden_size_},
                                         rtp_llm::AllocationType::DEVICE});
        }
        // avoid logits nan
        device_->bufMemset(*fake_hidden_states, 0);
        stream->setReturnLastHiddenStates(true);
        BufferPtr new_tokens =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1, 1}, rtp_llm::AllocationType::HOST});
        *new_tokens->dataWithOffset<int32_t>(0) = distribution(generator);
        StreamUpdateInfo update_info{new_tokens,
                                     (int)1,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     fake_hidden_states,
                                     false};
        stream->update(update_info);
        stream->setIsContextStream(false);
        stream->setReuseLength(1);
        stream->setFallbackPrefixLength(1);
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
        RTP_LLM_LOG_INFO(
            "warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
            warm_up_result->max_used_memory,
            warm_up_result->max_used_memory / 1024 / 1024,
            warm_up_result->device_reserved_bytes,
            warm_up_result->device_reserved_bytes / 1024 / 1024);
    }
    RETURN_IF_STATUS_ERROR(initCacheManager(warm_up_result));
    RTP_LLM_LOG_INFO("create cache manager done");
    propose_executor_ = createProposeExecutor(score_model_params_,
                                              propose_model_params_,
                                              device_,
                                              resource_context_.propose_cache_manager,
                                              resource_context_.mtp_cache_managers,
                                              getLoraManager());
    RTP_LLM_LOG_INFO("create speculative executor done");
    score_executor_.reset(
        new ScoreExecutor(score_model_params_, device_, resource_context_.cache_manager, getLoraManager()));

    scheduler_.reset(new SpeculativeScheduler(score_model_params_.gpt_init_parameter,
                                              resource_context_.cache_manager,
                                              metrics_reporter_,
                                              propose_model_params_->genNumPerCircle() + 1));
    RTP_LLM_LOG_INFO("create fifo scheduler done");
    speculative_sampler_ = std::make_unique<SpeculativeSampler>(device_);
    RTP_LLM_LOG_INFO("create speculative sampler");
    RETURN_IF_STATUS_ERROR(startLoop());
    if (device_->getDeviceProperties().tp_rank == 0) {
        initLoadBalance();
    }
    return absl::OkStatus();
}

void SpeculativeEngine::initLoadBalance() {
    RTP_LLM_LOG_INFO("init load balance start");
    std::shared_ptr<GenerateStream> stream;
    if (score_model_params_.gpt_init_parameter.role_type_ == RoleType::PREFILL) {
        stream = enqueueMinFakeQuery(1, false);
    } else {
        stream = enqueueMinFakeQuery(1, true);
    }
    while (!stream->finished() && !stream->stopped()) {
        RTP_LLM_LOG_INFO("wait load balance init run over for 1s");
        this_thread::sleep_for(std::chrono::seconds(1));
    }
    RTP_LLM_LOG_INFO("init load balance done and (StepPerMin: %ld , StepLatencyUs: %ld)",
                     step_recorder_.getStepPerMin(),
                     step_recorder_.getStepLatency());
}

absl::StatusOr<GenerateStreamPtr> SpeculativeEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                            preRunMode                            mode) {
    std::shared_ptr<GenerateStream> score_stream =
        std::make_shared<NormalGenerateStream>(generate_input,
                                               score_model_params_.gpt_init_parameter,
                                               resource_context_,
                                               nullptr,
                                               0,
                                               mode == preRunMode::prefill_warm_up);
    std::shared_ptr<GenerateStream> propose_stream = nullptr;
    if (mode == preRunMode::decode_warm_up) {
        score_stream->setIsContextStream(false);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUSOR_ERROR(score_stream->initKVBlock(0, 0));
    };

    if (propose_model_params_->draftModel()) {
        propose_stream = std::make_shared<NormalGenerateStream>(*score_stream);
    }

    std::list<GenerateStreamPtr> score_streams{score_stream};
    THROW_IF_STATUS_ERROR(score_executor_->normalProcess(score_streams));

    if (propose_model_params_->draftModel()) {
        THROW_IF_STATUS_ERROR(propose_executor_->normalProcess({propose_stream}));
    }

    return score_streams.front();
}

absl::Status SpeculativeEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result) {
    if (propose_model_params_->draftModel()) {
        const auto& config                = CacheConfigCreator::createSpConfig(score_model_params_.gpt_init_parameter,
                                                                propose_model_params_->getGptInitParameter(),
                                                                warm_up_result,
                                                                isMTPEagle(),
                                                                isEagle());
        auto        scorer_cache_config   = std::get<0>(config);
        auto        proposer_cache_config = std::get<1>(config);
        resource_context_.cache_manager   = make_shared<CacheManager>(
            scorer_cache_config, device_, false, metrics_reporter_, score_model_params_.gpt_init_parameter);
        if (isMTPEagle()) {
            auto layer_num = propose_model_params_->getGptInitParameter().gen_num_per_circle_;
            if (isEagle()) {
                layer_num = 1;
            }
            RTP_LLM_LOG_INFO("mtp cache manager init use layer num : %d", layer_num);
            for (int i = 0; i < layer_num; i++) {
                RTP_LLM_CHECK(proposer_cache_config.layer_num == 1);
                resource_context_.mtp_cache_managers.push_back(std::make_shared<CacheManager>(
                    proposer_cache_config, device_, false, metrics_reporter_, score_model_params_.gpt_init_parameter));
            }
        } else {
            resource_context_.propose_cache_manager = make_shared<CacheManager>(
                proposer_cache_config, device_, false, metrics_reporter_, score_model_params_.gpt_init_parameter);
        }

    } else {
        const auto& config = CacheConfigCreator::createConfig(score_model_params_.gpt_init_parameter, warm_up_result);
        resource_context_.cache_manager = make_shared<CacheManager>(
            config, device_, false, metrics_reporter_, score_model_params_.gpt_init_parameter);
    }
    return absl::OkStatus();
}

WarmUpResult SpeculativeEngine::warmUp() {
    const rtp_llm::GptInitParameter& socre_gpt_params = score_model_params_.gpt_init_parameter;
    std::shared_ptr<GenerateInput>   fake_input       = make_shared<GenerateInput>();
    fake_input->input_ids                             = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)socre_gpt_params.max_seq_len_ - 1}, rtp_llm::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config                       = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences = socre_gpt_params.max_context_batch_size_;
    fake_input->generate_config->calculate_loss       = int(socre_gpt_params.warm_up_with_loss_);
    fake_input->generate_config->top_k                = 2;
    fake_input->begin_time_us                         = autil::TimeUtility::currentTimeInMicroSeconds();
    device_->setTraceMemory(true);

    score_executor_.reset(new ScoreExecutor(score_model_params_, device_, nullptr, nullptr, true));
    if (isVanilla()) {
        propose_executor_.reset(new VanillaExecutor(propose_model_params_, device_, nullptr, nullptr, true));
    } else if (isMTPEagle()) {
        if (isEagle()) {
            propose_executor_.reset(
                new EagleExecutor(sp_type_, propose_model_params_, device_, {nullptr}, nullptr, true));
        } else {
            propose_executor_.reset(
                new MTPExecutor(sp_type_, propose_model_params_, device_, {nullptr}, nullptr, true));
        }
    }

    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto device_status = device_->getDeviceStatus();
    device_->setTraceMemory(false);
    (void)score_executor_.reset(nullptr);
    if (propose_model_params_->draftModel()) {
        (void)propose_executor_.reset(nullptr);
    }
    return WarmUpResult(
        {device_status.device_memory_status.preserved_bytes, device_status.device_memory_status.max_consumed_bytes});
}

absl::Status SpeculativeEngine::initSystemPrompt() {
    resource_context_.reuse_cache = score_model_params_.gpt_init_parameter.reuse_cache_;
    resource_context_.enable_3fs  = score_model_params_.gpt_init_parameter.kv_cache_config.enable_3fs;

    if (!score_model_params_.gpt_init_parameter.multi_task_prompt_tokens_.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(system_prompt_param,
                             SystemPromptConstructor::construct(score_model_params_.gpt_init_parameter,
                                                                this,
                                                                resource_context_.cache_manager.get(),
                                                                device_->getDeviceProperties().tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }
    return absl::OkStatus();
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
    std::shared_ptr<GenerateStream> stream =
        std::make_shared<NormalGenerateStream>(input,
                                               score_model_params_.gpt_init_parameter,
                                               resource_context_,
                                               metrics_reporter_,
                                               propose_model_params_->gen_num_per_circle);
    if (isMTPEagle()) {
        stream->setReturnLastHiddenStates(true);
    }
    return stream;
}

void SpeculativeEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    (void)scheduler_->enqueue(stream);
}

std::shared_ptr<GenerateStream> SpeculativeEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = makeStream(input);
    (void)scheduler_->enqueue(stream);
    return stream;
}

void SpeculativeEngine::tpSyncDisableSPRun(bool& all_streams_disable_sp_run) {
    if (device_->getDeviceProperties().tp_size <= 1) {
        return;
    }
    auto disable_sp_run = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1}, rtp_llm::AllocationType::HOST});
    auto disable_sp_run_ptr       = disable_sp_run->data<int32_t>();
    disable_sp_run_ptr[(size_t)0] = all_streams_disable_sp_run;

    device_->broadcast({{disable_sp_run}, 0});
    device_->syncCommunication(false);
    device_->syncAndCheck();
    all_streams_disable_sp_run = disable_sp_run_ptr[(size_t)0];
}

absl::Status SpeculativeEngine::step() {

    while (pause_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    list<GenerateStreamPtr> streams;

    if (device_->getDeviceProperties().tp_rank == 0) {
        if (scheduler_->empty() || step_recorder_.empty()) {
            step_recorder_.reset();
            step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds(),
                                        propose_executor_->reserveStep() / 2);
        }
        auto reserve_step = propose_executor_->reserveStep() + 1;

        CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step));

        if (streams.empty()) {
            if (score_model_params_.gpt_init_parameter.dp_size_ > 1) {
                CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step));
                if (streams.empty()) {
                    if (score_model_params_.gpt_init_parameter.role_type_ == RoleType::PREFILL) {
                        enqueueMinFakeQuery(1, false);
                    } else {
                        enqueueMinFakeQuery(1, true);
                    }
                    return absl::OkStatus();
                }
            } else {
                return absl::OkStatus();
            }
        }
        if (score_model_params_.gpt_init_parameter.dp_size_ > 1
            && score_model_params_.gpt_init_parameter.role_type_ != RoleType::PREFILL) {
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
        RTP_LLM_LOG_DEBUG("pre stream[%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    bool all_streams_disable_sp_run =
        !streams.empty()
        && std::all_of(streams.begin(), streams.end(), [](const auto& stream) { return stream->disableSpRun(); });
    bool gen_timeline = !streams.empty() && std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
        return stream->genTimeline();
    });
    profiler_step_--;
    if (profiler_step_ <= 0) {
        profiler_.reset();
        profiler_step_ = 0;
    }
    if (gen_timeline && profiler_step_ <= 0) {
        auto stream_group = StreamGroups(streams);
        auto world_rank   = device_->getDeviceProperties().dp_rank * device_->getDeviceProperties().tp_size
                          + device_->getDeviceProperties().tp_rank;
        auto profiler_prefix = autil::StringUtil::formatString("sp_profiler_wr%d_b%d_s%d_prefill%d_",
                                                               world_rank,
                                                               stream_group.totalModelBatchSize(),
                                                               stream_group.maxSeqLen(),
                                                               int(stream_group.totalContextBatchSize() > 0));
        profiler_            = std::make_shared<CudaProfiler_E>(profiler_prefix);
        profiler_->start();
        auto it        = std::max_element(streams.begin(), streams.end(), [](const auto& a, const auto& b) {
            return a->profileStep() < b->profileStep();
        });
        profiler_step_ = (*it)->profileStep();
    }
    tpSyncDisableSPRun(all_streams_disable_sp_run);

    absl::Status status;
    if (all_streams_disable_sp_run) {
        status = normStep(streams);
    } else if (isMTPEagle()) {
        status = mtpStep(streams);
    } else {
        status = spStep(streams);
    }

    if (device_->getDeviceProperties().tp_rank == 0) {
        reportMetrics();
        for (auto& stream : streams) {
            if (stream->finished()) {
                step_recorder_.addStepCount(stream->iterCount());
            }
        }

        step_recorder_.registerStep(autil::TimeUtility::currentTimeInMicroSeconds(),
                                    metrics_.accept_token_num / streams.size());

        metrics_.reset();
    }

    return status;
}

absl::Status SpeculativeEngine::normStep(std::list<GenerateStreamPtr>& streams) {
    int64_t score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("before normal process stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    THROW_IF_STATUS_ERROR(score_executor_->normalProcess(streams));

    // stream post process
    for (auto& stream : streams) {
        stream->setReuseLength(stream->seqLength() - 1);
        stream->setFallbackPrefixLength(stream->reuseLength());
        stream->setSpEditRun(false);
        RTP_LLM_LOG_DEBUG("stream [%ld], topk = [%d], topp = [%f], propose_tokens = 0, accept_tokens = 1",
                          stream->streamId(),
                          stream->generateConfig()->top_k,
                          stream->generateConfig()->top_p);

        RTP_LLM_LOG_DEBUG("after normal process stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    metrics_.score_time_us    = autil::TimeUtility::currentTimeInMicroSeconds() - score_begin_time_us;
    metrics_.accept_token_num = streams.size();

    return absl::OkStatus();
}

absl::Status SpeculativeEngine::spStep(std::list<GenerateStreamPtr>& streams) {
    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t score_begin_time_us   = 0;
    int64_t sampler_begin_time_us = 0;

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("before sp step stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    THROW_IF_STATUS_ERROR(propose_executor_->propose(streams));

    score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();

    THROW_IF_STATUS_ERROR(score_executor_->score(streams));

    if (device_->getDeviceProperties().tp_rank == 0) {
        sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams));
        RTP_LLM_LOG_DEBUG("speculative sample done");

        metrics_.propose_token_num += sampler_output.propose_token_num;
        metrics_.accept_token_num += sampler_output.accept_token_num;
        metrics_.stream_num += sampler_output.stream_num;
    }

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("post sp step stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    metrics_.propose_time_us = score_begin_time_us - propose_begin_time_us;
    metrics_.score_time_us   = sampler_begin_time_us - score_begin_time_us;
    metrics_.sampler_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - sampler_begin_time_us;

    return absl::OkStatus();
}

absl::Status SpeculativeEngine::prefillMtpStep(std::list<GenerateStreamPtr>& streams) {
    int64_t propose_begin_time_us = 0;
    int64_t score_begin_time_us   = 0;

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("before mtp prefill stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    {
        RTP_LLM_LOG_DEBUG("score model prefill");
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        THROW_IF_STATUS_ERROR(score_executor_->score(streams, true));

        RTP_LLM_LOG_DEBUG("update stream");
        for (GenerateStreamPtr& stream : streams) {
            SpeculativeExecutorStreamOutputPtr score_output = stream->getScoreStream()->getSPOutputBuffer();
            StreamUpdateInfo                   update_info{score_output->tokens,
                                         (int)1,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         score_output->hidden_states,
                                         false,
                                         true};
            stream->update(update_info);
            stream->setIsContextStream(true);
        }

        for (auto& stream : streams) {
            RTP_LLM_LOG_DEBUG("after update stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
        }

        propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        RTP_LLM_LOG_DEBUG("propose model prefill");
        THROW_IF_STATUS_ERROR(propose_executor_->propose(streams, true));

        for (const GenerateStreamPtr& stream : streams) {
            BufferPtr   propose_tokens = stream->getProposeStream()->getSPOutputBuffer()->tokens;
            vector<int> propose_tokens_vec;
            for (int i = 0; i < propose_tokens->shape()[1]; ++i) {
                propose_tokens_vec.push_back(propose_tokens->data<int>()[i]);
            }
            RTP_LLM_CHECK_WITH_INFO(propose_tokens_vec.size() > 0, "propose token size should not be empty");
            stream->setProposeToken(propose_tokens_vec);
            stream->setReuseLength(stream->seqLength() - 1);
            stream->setFallbackPrefixLength(stream->reuseLength());
            stream->setSpEditRun(false);
            stream->setLastHiddenStates(nullptr);
            stream->setSPOutputBuffer(nullptr);
            auto score_stream   = stream->getScoreStream();
            auto propose_stream = stream->getProposeStream();
            if (score_stream) {
                score_stream->setLastHiddenStates(nullptr);
                score_stream->setSPOutputBuffer(nullptr);
            }
            if (propose_stream) {
                propose_stream->setLastHiddenStates(nullptr);
                propose_stream->setSPOutputBuffer(nullptr);
            }
            if (stream->queryPdSep()) {
                RTP_LLM_LOG_DEBUG("stream [%ld] set setNeedRemoteGenerate", stream->streamId());
                stream->setNeedRemoteGenerate(true);
            }
        }
    }

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("post mtp prefill stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    metrics_.propose_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - propose_begin_time_us;
    metrics_.score_time_us   = propose_begin_time_us - score_begin_time_us;

    return absl::OkStatus();
}

absl::Status SpeculativeEngine::mtpStep(std::list<GenerateStreamPtr>& streams) {
    if (score_model_params_.gpt_init_parameter.role_type_ == RoleType::PREFILL) {
        return prefillMtpStep(streams);
    }

    int64_t propose_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t score_begin_time_us   = 0;
    int64_t sampler_begin_time_us = 0;

    std::list<GenerateStreamPtr> propose_streams;
    std::list<GenerateStreamPtr> prefill_streams;
    std::list<GenerateStreamPtr> pre_propose_streams;
    if (device_->getDeviceProperties().tp_rank == 0) {
        for (auto& stream : streams) {
            if (stream->getContainProposeToken()) {
                pre_propose_streams.emplace_back(stream);
            } else if (stream->getLastHiddenStates() != nullptr) {
                propose_streams.emplace_back(stream);
            } else {
                prefill_streams.emplace_back(stream);
            }
        }

        for (auto& stream : propose_streams) {
            RTP_LLM_LOG_DEBUG("begin propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
        }

        for (auto& stream : prefill_streams) {
            RTP_LLM_LOG_DEBUG("begin prefill stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
        }

        for (auto& stream : pre_propose_streams) {
            RTP_LLM_LOG_DEBUG("begin pre propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
        }
    }

    {
        bool skip_propose = propose_streams.empty();
        tpSyncDisableSPRun(skip_propose);
        if (!skip_propose) {
            RTP_LLM_LOG_DEBUG("propose step");
            THROW_IF_STATUS_ERROR(propose_executor_->propose(propose_streams));
        } else {
            RTP_LLM_LOG_DEBUG("skip propose");
        }

        for (const GenerateStreamPtr& stream : prefill_streams) {
            size_t            propose_step   = 0;
            GenerateStreamPtr propose_stream = makeMTPStream(stream, propose_step);

            SpeculativeExecutorStreamOutputPtr sp_output_buffer_ = propose_stream->getSPOutputBuffer();
            sp_output_buffer_->propose_step                      = 0;
            sp_output_buffer_->tokens                            = nullptr;

            stream->setProposeStream(propose_stream);
        }

        for (const GenerateStreamPtr& stream : pre_propose_streams) {
            size_t            propose_step   = 1;
            GenerateStreamPtr propose_stream = makeMTPStream(stream, propose_step);

            SpeculativeExecutorStreamOutputPtr sp_output_buffer_ = propose_stream->getSPOutputBuffer();
            sp_output_buffer_->propose_step                      = propose_step;
            vector<int> propose_tokens                           = stream->getProposeToken();
            sp_output_buffer_->tokens                            = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, propose_tokens.size()}, rtp_llm::AllocationType::HOST}, {});
            memcpy(sp_output_buffer_->tokens->data(), propose_tokens.data(), sizeof(int) * propose_tokens.size());

            stream->setProposeStream(propose_stream);
        }
    }

    // base model score propose new tokens.
    {
        RTP_LLM_LOG_DEBUG("score step");
        score_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        THROW_IF_STATUS_ERROR(score_executor_->score(streams));

        if (device_->getDeviceProperties().tp_rank == 0) {
            RTP_LLM_LOG_DEBUG("sample step");
            sampler_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
            CHECK_AND_RETURN_REF(sampler_output, speculative_sampler_->sample(streams));
            RTP_LLM_LOG_DEBUG("speculative sample done");

            metrics_.propose_token_num += sampler_output.propose_token_num;
            metrics_.accept_token_num += sampler_output.accept_token_num;
            metrics_.stream_num += sampler_output.stream_num;
        }

        for (auto& stream : pre_propose_streams) {
            stream->setContainProposeToken(false);
        }

        for (auto& stream : streams) {
            RTP_LLM_LOG_DEBUG("post stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
        }
    }

    metrics_.propose_time_us = score_begin_time_us - propose_begin_time_us;
    metrics_.score_time_us   = sampler_begin_time_us - score_begin_time_us;
    metrics_.sampler_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - sampler_begin_time_us;

    return absl::OkStatus();
}

void SpeculativeEngine::reportMetrics() {
    RTP_LLM_LOG_DEBUG("propose_token_num: %d, accept_token_num: %d, stream_num: %d",
                      metrics_.propose_token_num,
                      metrics_.accept_token_num,
                      metrics_.stream_num);

    if (!metrics_reporter_) {
        return;
    }

    int64_t total_step_time = metrics_.propose_time_us + metrics_.score_time_us + metrics_.sampler_time_us;
    RTP_LLM_LOG_DEBUG("total_step_time: %ld, propose_time: %ld, score_time: %ld, sampler_time: %ld",
                      total_step_time,
                      metrics_.propose_time_us,
                      metrics_.score_time_us,
                      metrics_.sampler_time_us);
    RtpLLMSpeculativeEngineMetricsCollector collector{total_step_time,
                                                      metrics_.propose_time_us,
                                                      metrics_.score_time_us,
                                                      metrics_.sampler_time_us,
                                                      metrics_.propose_token_num,
                                                      metrics_.accept_token_num,
                                                      metrics_.stream_num};
    metrics_reporter_->report<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>(nullptr,
                                                                                                       &collector);
}

bool SpeculativeEngine::updateEplbConfig(const EplbConfig& config) {
    if (score_executor_ && propose_executor_) {
        return score_executor_->updateEplbConfig(config) && propose_executor_->updateEplbConfig(config);
    }
    return true;
}

KVCacheInfo SpeculativeEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) const {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

}  // namespace rtp_llm
