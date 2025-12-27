#pragma once

#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPStream.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/EagleStream.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "rtp_llm/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "kmonitor/client/MetricsReporter.h"
#include "torch/csrc/autograd/profiler_kineto.h"

namespace rtp_llm {

namespace tap = torch::autograd::profiler;
namespace tpi = torch::profiler::impl;
class CudaProfiler_E {
public:
    CudaProfiler_E(const std::string& prefix): prefix_(prefix) {
        tap::prepareProfiler(config_, activities_);
    }
    ~CudaProfiler_E() {
        if (!stoped_) {
            stoped_ = true;
            stop();
        }
    }
    void start() {
        count += 1;
        stoped_ = false;
        tap::enableProfiler(config_, activities_);
    }
    void stop() {
        std::unique_ptr<tap::ProfilerResult> res       = tap::disableProfiler();
        std::string                          file_name = prefix_ + std::to_string(count) + ".json";
        res->save(file_name);
        stoped_ = true;
    }

protected:
    static size_t               count;
    std::string                 prefix_;
    tpi::ProfilerConfig         config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO);
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CUDA};
    bool                        stoped_ = true;
};

struct SpeculativeEngineStepMetrics {
    void reset() {
        propose_time_us   = 0;
        score_time_us     = 0;
        sampler_time_us   = 0;
        propose_token_num = 0;
        accept_token_num  = 0;
        stream_num        = 0;
    }
    int64_t propose_time_us   = 0;
    int64_t score_time_us     = 0;
    int64_t sampler_time_us   = 0;
    int64_t propose_token_num = 0;
    int64_t accept_token_num  = 0;
    int64_t stream_num        = 0;
};

class SpeculativeEngine: public EngineBase {
public:
    explicit SpeculativeEngine(const EngineInitParams&                       engine_init_params,
                               std::unique_ptr<ProposeModelEngineInitParams> propose_model_engine_init_params);
    ~SpeculativeEngine();
    absl::Status                      init();
    std::shared_ptr<GenerateStream>   makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream>   enqueue(const std::shared_ptr<GenerateInput>& input) override;
    void                              enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;
    KVCacheInfo                       getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    GenerateStreamPtr                 makeMTPStream(const GenerateStreamPtr& stream, size_t propose_step) {
        if (isEagle()) {
            return std::make_shared<EagleStream>(*stream, propose_step);
        } else {
            return std::make_shared<MTPStream>(*stream, propose_step);
        }
    }

    bool isMTPEagle() override {
        return sp_type_ == "mtp" || isEagle();
    }

    bool isEagle() {
        return sp_type_ == "eagle" || sp_type_ == "eagle3";
    }

    bool isVanilla() {
        return sp_type_ == "vanilla";
    }

private:
    WarmUpResult warmUp();
    void         initLoadBalance();
    absl::Status step();

    // do not walk through speculative process.
    absl::Status normStep(std::list<GenerateStreamPtr>& streams);

    absl::Status mtpStep(std::list<GenerateStreamPtr>& streams);

    absl::Status prefillMtpStep(std::list<GenerateStreamPtr>& streams);

    absl::Status spStep(std::list<GenerateStreamPtr>& streams);

    void preparePerfStreams(std::list<GenerateStreamPtr>& streams);

    std::list<GenerateStreamPtr> extractPrefillStreams(std::list<GenerateStreamPtr>& streams) {
        std::list<GenerateStreamPtr> need_prefill_streams;
        streams.erase(std::remove_if(streams.begin(),
                                     streams.end(),
                                     [&](GenerateStreamPtr stream) {
                                         if (stream->getLastHiddenStates() == nullptr) {
                                             need_prefill_streams.push_back(stream);
                                             return true;
                                         } else {
                                             return false;
                                         }
                                     }),
                      streams.end());
        return need_prefill_streams;
    };

    absl::Status startLoop();
    void         loop();
    absl::Status trySaveStepError() const;
    absl::Status initCacheManager(std::optional<WarmUpResult> warm_up_result);
    absl::Status initSystemPrompt();
    void         tpSyncDisableSPRun(bool& all_streams_disable_sp_run);
    void         reportMetrics();

    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens, bool fake_hidden_states = false);

    std::list<GenerateStreamPtr> extractFirstPrefillStreams(std::list<GenerateStreamPtr>& streams);

    bool updateEplbConfig(const EPLBConfig& config) override;

private:
    kmonitor::MetricsReporterPtr                  metrics_reporter_ = nullptr;
    std::unique_ptr<ProposeModelEngineInitParams> propose_model_params_;
    const EngineInitParams                        score_model_params_;

    std::unique_ptr<ProposeExecutor>    propose_executor_    = nullptr;
    std::unique_ptr<ScoreExecutor>      score_executor_      = nullptr;
    std::unique_ptr<SpeculativeSampler> speculative_sampler_ = nullptr;
    std::shared_ptr<SystemPrompt>       system_prompt_       = nullptr;

    SpeculativeEngineStepMetrics metrics_;

    const std::string               sp_type_;
    std::thread                     loop_thread_;
    std::atomic<bool>               running_{false};
    std::shared_ptr<CudaProfiler_E> profiler_;
    int                             profiler_step_ = 0;
};

}  // namespace rtp_llm
