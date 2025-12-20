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
#include "rtp_llm/cpp/speculative_engine/SpeculativeGatherBatchScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPExecutor.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

size_t CudaProfiler_E::count = 0;

SpeculativeEngine::SpeculativeEngine(const EngineInitParams&                       engine_init_params,
                                     std::unique_ptr<ProposeModelEngineInitParams> propose_model_engine_init_params):
    EngineBase(engine_init_params),
    metrics_reporter_(engine_init_params.metrics_reporter),
    propose_model_params_(std::move(propose_model_engine_init_params)),
    score_model_params_(std::move(engine_init_params)),
    sp_type_(SpeculativeExecutionConfig::to_string(propose_model_params_->sp_type)) {};

SpeculativeEngine::~SpeculativeEngine() {
    RTP_LLM_LOG_INFO("destory speculative engine");
    (void)stop();
}

std::shared_ptr<GenerateStream> SpeculativeEngine::createMinFakeStream(int32_t max_new_tokens,
                                                                       bool    fake_hidden_states) {
    RTP_LLM_LOG_DEBUG("create sp min fake query");
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1}, rtp_llm::AllocationType::HOST});

    // std::default_random_engine         generator;
    // std::uniform_int_distribution<int> distribution(0, score_model_params_.model_config_.vocab_size - 1);
    // for (size_t i = 0; i < fake_input->input_ids->size(); ++i) {
    //     *fake_input->input_ids->dataWithOffset<int32_t>(i) = distribution(generator);
    // }
    device_->bufMemset(*fake_input->input_ids, 0);
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
    stream->setIsFakeStream(true);
    stream->setMetricsReporter(nullptr);
    stream->fakeInitKVBlock();

    if (fake_hidden_states) {
        auto      dtype = score_model_params_.model_config_.data_type;
        BufferPtr fake_hidden_states;
        if (sp_type_ == "eagle3") {
            fake_hidden_states =
                device_->allocateBuffer({dtype,
                                         {1, (size_t)score_model_params_.model_config_.hidden_size * 3},
                                         rtp_llm::AllocationType::DEVICE});
        } else {
            fake_hidden_states = device_->allocateBuffer(
                {dtype, {1, (size_t)score_model_params_.model_config_.hidden_size}, rtp_llm::AllocationType::DEVICE});
        }
        // avoid logits nan
        device_->bufMemset(*fake_hidden_states, 0);
        stream->setReturnLastHiddenStates(true);
        BufferPtr new_tokens =
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1, 1}, rtp_llm::AllocationType::HOST});
        *new_tokens->dataWithOffset<int32_t>(0) = 0;
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
    }

    return stream;
}

absl::Status SpeculativeEngine::init() {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
    std::optional<WarmUpResult> warm_up_result = std::nullopt;
    if (score_model_params_.runtime_config.warm_up) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         score_model_params_.runtime_config.fifo_scheduler_config.max_context_batch_size,
                         score_model_params_.model_config_.max_seq_len,
                         int(score_model_params_.runtime_config.warm_up_with_loss));
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

    if (score_model_params_.runtime_config.use_batch_decode_scheduler) {
        RTP_LLM_LOG_INFO("create speculative batch decode scheduler");
        scheduler_.reset(new BatchDecodeScheduler(
            score_model_params_.runtime_config, resource_context_.cache_manager, metrics_reporter_, device_));
    } else if (score_model_params_.runtime_config.use_gather_batch_scheduler) {
        RTP_LLM_LOG_INFO("create speculative gather batch scheduler");
        scheduler_.reset(new SpeculativeGatherBatchScheduler(score_model_params_.runtime_config,
                                                             score_model_params_.model_config_,
                                                             score_model_params_.pd_sep_config,
                                                             score_model_params_.parallelism_config,
                                                             score_model_params_.model_specific_config,
                                                             resource_context_.cache_manager,
                                                             metrics_reporter_,
                                                             propose_model_params_->genNumPerCircle() + 1));
    } else {
        RTP_LLM_LOG_INFO("create speculative scheduler");
        scheduler_.reset(new SpeculativeScheduler(score_model_params_.runtime_config,
                                                  score_model_params_.model_config_,
                                                  score_model_params_.pd_sep_config,
                                                  score_model_params_.parallelism_config,
                                                  score_model_params_.model_specific_config,
                                                  resource_context_.cache_manager,
                                                  metrics_reporter_,
                                                  propose_model_params_->genNumPerCircle() + 1));
    }
    speculative_sampler_ = std::make_unique<SpeculativeSampler>(device_);
    RTP_LLM_LOG_INFO("create speculative sampler");
    RETURN_IF_STATUS_ERROR(startLoop());
    return absl::OkStatus();
}

absl::StatusOr<GenerateStreamPtr> SpeculativeEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                            preRunMode                            mode) {
    std::shared_ptr<GenerateStream> score_stream =
        std::make_shared<NormalGenerateStream>(generate_input,
                                               score_model_params_.model_config_,
                                               score_model_params_.runtime_config,
                                               resource_context_,
                                               nullptr,
                                               0,
                                               mode == preRunMode::prefill_warm_up);
    std::shared_ptr<GenerateStream> propose_stream = nullptr;
    if (mode == preRunMode::decode_warm_up) {
        score_stream->setIsContextStream(false);
    } else if (mode == preRunMode::build_system_prompt) {
        THROW_IF_STATUS_ERROR(score_stream->initKVBlock());
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
        const auto& propose_params           = propose_model_params_->getEngineInitParams();
        const auto& config                   = CacheConfigCreator::createSpConfig(score_model_params_.model_config_,
                                                                propose_params.model_config_,
                                                                score_model_params_.parallelism_config,
                                                                score_model_params_.runtime_config,
                                                                score_model_params_.kv_cache_config,
                                                                score_model_params_.sp_config,
                                                                warm_up_result,
                                                                isMTPEagle(),
                                                                isEagle());
        auto        scorer_cache_config      = std::get<0>(config);
        auto        proposer_cache_config    = std::get<1>(config);
        scorer_cache_config.mtp_model_type   = "score_model";
        proposer_cache_config.mtp_model_type = "propose_model";
        resource_context_.cache_manager      = make_shared<KVCacheManager>(scorer_cache_config,
                                                                      device_,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      score_model_params_.kv_cache_config,
                                                                      score_model_params_.parallelism_config,
                                                                      score_model_params_.runtime_config);
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
        if (isMTPEagle()) {
            auto layer_num = propose_model_params_->genNumPerCircle();
            if (isEagle()) {
                layer_num = 1;
            }
            RTP_LLM_LOG_INFO("mtp cache manager init use layer num : %d", layer_num);
            for (int i = 0; i < layer_num; i++) {
                RTP_LLM_CHECK(proposer_cache_config.layer_num == 1);
                resource_context_.mtp_cache_managers.push_back(
                    std::make_shared<KVCacheManager>(proposer_cache_config,
                                                     device_,
                                                     false,
                                                     metrics_reporter_,
                                                     score_model_params_.kv_cache_config,
                                                     score_model_params_.parallelism_config,
                                                     score_model_params_.runtime_config));
                if (!resource_context_.mtp_cache_managers.back()->init()) {
                    RTP_LLM_FAIL("init mtp kv cache manager failed");
                }
            }
        } else {
            resource_context_.propose_cache_manager =
                make_shared<KVCacheManager>(proposer_cache_config,
                                            device_,
                                            false,
                                            metrics_reporter_,
                                            score_model_params_.kv_cache_config,
                                            score_model_params_.parallelism_config,
                                            score_model_params_.runtime_config);
            if (!resource_context_.propose_cache_manager->init()) {
                RTP_LLM_FAIL("init propose kv cache manager failed");
            }
        }

    } else {
        const auto& config              = CacheConfigCreator::createConfig(score_model_params_.model_config_,
                                                              score_model_params_.parallelism_config,
                                                              score_model_params_.runtime_config,
                                                              score_model_params_.kv_cache_config,
                                                              warm_up_result,
                                                              score_model_params_.sp_config);
        resource_context_.cache_manager = make_shared<KVCacheManager>(config,
                                                                      device_,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      score_model_params_.kv_cache_config,
                                                                      score_model_params_.parallelism_config,
                                                                      score_model_params_.runtime_config);
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
    }
    return absl::OkStatus();
}

WarmUpResult SpeculativeEngine::warmUp() {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->input_ids                     = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                                         {(size_t)score_model_params_.model_config_.max_seq_len - 1},
                                                                         rtp_llm::AllocationType::HOST});
    std::memset(fake_input->input_ids->data(), 0, fake_input->input_ids->sizeBytes());
    fake_input->generate_config = make_shared<GenerateConfig>();
    fake_input->generate_config->num_return_sequences =
        score_model_params_.runtime_config.fifo_scheduler_config.max_context_batch_size;
    fake_input->generate_config->calculate_loss = int(score_model_params_.runtime_config.warm_up_with_loss);
    fake_input->generate_config->top_k          = 2;
    fake_input->begin_time_us                   = autil::TimeUtility::currentTimeInMicroSeconds();
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
    resource_context_.reuse_cache               = score_model_params_.kv_cache_config.reuse_cache;
    resource_context_.enable_3fs                = score_model_params_.kv_cache_config.enable_3fs;
    resource_context_.enable_memory_block_cache = score_model_params_.kv_cache_config.memory_block_cache_size_mb > 0;

    if (!score_model_params_.kv_cache_config.multi_task_prompt_tokens.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(system_prompt_param,
                             SystemPromptConstructor::construct(score_model_params_.kv_cache_config,
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
                                               score_model_params_.model_config_,
                                               score_model_params_.runtime_config,
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
        auto reserve_step = propose_executor_->reserveStep() + 1;

        CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step));

        if (streams.empty()) {
            if (score_model_params_.parallelism_config.dp_size > 1) {
                CHECK_AND_ASSIGN(streams, scheduler_->schedule(reserve_step));
                if (streams.empty()) {
                    if (score_model_params_.pd_sep_config.role_type == RoleType::PREFILL) {
                        streams.emplace_back(createMinFakeStream(1, false));
                    } else {
                        streams.emplace_back(createMinFakeStream(1, true));
                    }
                }
            } else {
                return absl::OkStatus();
            }
        }

        preparePerfStreams(streams);

        if (score_model_params_.parallelism_config.dp_size > 1
            && score_model_params_.pd_sep_config.role_type != RoleType::PREFILL) {
            bool has_hidden_states = false;
            for (auto stream : streams) {
                if (stream->getLastHiddenStates() != nullptr) {
                    has_hidden_states = true;
                    break;
                }
            }
            if (!has_hidden_states) {
                streams.emplace_back(createMinFakeStream(1, true));
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
            stream->setSpEditRun(false);
            stream->setLastHiddenStates(nullptr);
            stream->setSPOutputBuffer(nullptr);
            // 前面stream的状态准备完了，可以先将remote generate设置为true然后让另外的线程开始发送kv
            // cache，最后再清理一些不需要的资源
            if (stream->queryPdSep()) {
                RTP_LLM_LOG_DEBUG("stream [%ld] set setNeedRemoteGenerate", stream->streamId());
                stream->setNeedRemoteGenerate(true);
            }
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
        }
    }

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("post mtp prefill stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    metrics_.propose_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - propose_begin_time_us;
    metrics_.score_time_us   = propose_begin_time_us - score_begin_time_us;

    return absl::OkStatus();
}

void SpeculativeEngine::preparePerfStreams(std::list<GenerateStreamPtr>& streams) {
    for (auto& stream : streams) {
        if (stream->getScoreStream() == nullptr && !stream->isContextStream() && stream->isPerfTest()) {
            int       input_len = stream->inputLength();
            BufferPtr new_tokens =
                device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)1, 1}, rtp_llm::AllocationType::HOST});
            *new_tokens->dataWithOffset<int32_t>(0) = 0;

            auto propose_stream = makeMTPStream(stream, 0);
            stream->setProposeStream(propose_stream);

            StreamUpdateInfo update_info{
                new_tokens, (int)1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false};
            stream->update(update_info);

            auto decode_hidden = device_->allocateBuffer({score_model_params_.model_config_.data_type,
                                                          {1, (size_t)score_model_params_.model_config_.hidden_size}});

            device_->bufMemset(*decode_hidden, 0);
            stream->setLastHiddenStates(decode_hidden);
            stream->setSeqLength(input_len + 1);
            stream->setMtpTokenIndex(0);
            stream->setReuseLength(input_len);

            propose_stream->setMtpTokenIndex(input_len - 1);
            propose_stream->setSeqLength(input_len);
        }
    }
}

absl::Status SpeculativeEngine::mtpStep(std::list<GenerateStreamPtr>& streams) {
    if (score_model_params_.pd_sep_config.role_type == RoleType::PREFILL) {
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
        // dp 情况下不允许skip propose，有可能步骤不同步
        RTP_LLM_CHECK_WITH_INFO(score_model_params_.parallelism_config.dp_size <= 1 || !skip_propose,
                                "skip propose not allowed now");
        if (!skip_propose) {
            RTP_LLM_LOG_DEBUG("propose step");
            THROW_IF_STATUS_ERROR(propose_executor_->propose(propose_streams));
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
            // set output token to zero when steam is fake query and can debug easily
            if (stream->isFakeStream()) {
                device_->bufMemset(*(sp_output_buffer_->tokens), 0);
            }
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

bool SpeculativeEngine::updateEplbConfig(const EPLBConfig& config) {
    if (score_executor_ && propose_executor_) {
        return score_executor_->updateEplbConfig(config) && propose_executor_->updateEplbConfig(config);
    }
    return true;
}

KVCacheInfo SpeculativeEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

}  // namespace rtp_llm
