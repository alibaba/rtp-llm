#include <any>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "autil/EnvUtil.h"
#include "torch/all.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#undef protected
#undef private
#include "gtest/gtest.h"

extern char** environ;

namespace rtp_llm {
namespace {

class GatedDecodeStream: public NormalGenerateStream {
public:
    using NormalGenerateStream::NormalGenerateStream;

    void updateOutput(const StreamUpdateInfo& update_info) override {
        // Generation reaches the outer AsyncRunner on decode; prefill-only finishes on its first update.
        const int gated_update = generateConfig()->isPrefillOnly() ? 1 : 2;
        if (++update_count_ == gated_update) {
            decode_started.set_value();
            release.wait();
            NormalGenerateStream::updateOutput(update_info);
            decode_completed = true;
        } else {
            NormalGenerateStream::updateOutput(update_info);
        }
    }

    std::promise<void>       decode_started;
    std::atomic<bool>        decode_completed{false};
    std::shared_future<void> release;

private:
    int update_count_ = 0;
};

class EngineStopTest: public DeviceTestBase, public ::testing::WithParamInterface<int> {
protected:
    void checkStopWaitsForDispatch(int max_new_tokens);
};

// Separate Bazel targets set RTP_LLM_STREAM_ASYNC before process startup because
// NormalExecutor caches this flag. Both modes also exercise worker counts 0/2.
void EngineStopTest::checkStopWaitsForDispatch(int max_new_tokens) {
    const char* async_env = std::getenv("RTP_LLM_STREAM_ASYNC");
    ASSERT_NE(async_env, nullptr);
    const bool stream_async = std::string(async_env) == "1";

    CustomConfig config;
    config.output_dispatcher_worker_count = GetParam();
    auto  engine                          = createMockEngine(config);
    auto* executor                        = dynamic_cast<NormalExecutor*>(engine->executor_.get());
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(executor->useStreamAsync(), stream_async);
    EXPECT_EQ(executor->batch_stream_processor_->output_dispatcher_->thread_pool_ != nullptr, GetParam() > 0);

    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = max_new_tokens;
    query->generate_config->is_streaming   = false;

    std::promise<void> release;
    auto               stream = std::make_shared<GatedDecodeStream>(
        query, engine->model_config_, engine->runtime_config, engine->resourceContext(), nullptr);
    stream->release                 = release.get_future().share();
    auto              started       = stream->decode_started.get_future();
    GenerateStreamPtr queued_stream = stream;
    engine->enqueue(queued_stream);

    // No fatal assertions while the gate is held: always unblock the worker
    // before destroying the stop future or engine, including failure paths.
    EXPECT_EQ(started.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async && max_new_tokens > 0);
    EXPECT_FALSE(stream->decode_completed.load());
    std::promise<void> stop_started;
    auto               stopping = stop_started.get_future();
    auto               stopped  = std::async(std::launch::async, [&]() {
        stop_started.set_value();
        return engine->stop();
    });
    EXPECT_EQ(stopping.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(stream->decode_completed.load());
    EXPECT_EQ(stream->hasPendingAsyncBookkeeping(), stream_async && max_new_tokens > 0);
    release.set_value();

    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_TRUE(stopped.get().ok());
    EXPECT_TRUE(stream->decode_completed.load());
    EXPECT_FALSE(stream->hasPendingAsyncBookkeeping());
    EXPECT_EQ(engine->executor_, nullptr);
    // stop() cancels scheduled streams, so nextOutput() need not succeed.
    // A stopped prefill-only request must not manufacture a generated token.
    EXPECT_EQ(stream->seqLength(), 2 + max_new_tokens);
}

TEST_P(EngineStopTest, StopWaitsForInFlightDecodeDispatch) {
    checkStopWaitsForDispatch(2);
}

TEST_P(EngineStopTest, StopWaitsForInFlightPrefillOnlyDispatch) {
    checkStopWaitsForDispatch(0);
}

TEST_P(EngineStopTest, StopWakesIdleSchedulerWhilePaused) {
    CustomConfig config;
    config.output_dispatcher_worker_count = GetParam();
    auto engine                           = createMockEngine(config);
    engine->pause();
    auto stopped = std::async(std::launch::async, [&] { return engine->stop(); });
    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(stopped.get().ok());
    EXPECT_FALSE(engine->running_.load());
    EXPECT_EQ(engine->executor_, nullptr);
    EXPECT_TRUE(engine->stop().ok());
}

TEST_P(EngineStopTest, Tp1DpAndFfnStopNeedNoWorldCollective) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    // No communication callbacks are registered in this fixture. Generic DP
    // and FFN roles must never enter the SM120 WORLD scheduling protocol.
    for (bool ffn : {false, true}) {
        engine->ffn_disaggregate_config.enable_ffn_disaggregate = ffn;
        for (int tp_size : (ffn ? std::vector<int>{1, 2} : std::vector<int>{1})) {
            engine->parallelism_config.dp_size = 4;
            engine->parallelism_config.tp_size = tp_size;
            for (bool stop : {false, true}) {
                engine->stop_requested_ = stop;
                EXPECT_EQ(engine->synchronizeStopRequested(), stop);
            }
        }
    }
}

TEST_P(EngineStopTest, ActiveTpStepDoesNotAddAControlCollective) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    engine->parallelism_config.tp_size = 2;
    engine->parallelism_config.dp_size = 1;
    engine->stop_requested_            = false;
    for (int rank : {0, 1}) {
        engine->parallelism_config.tp_rank = rank;
        EXPECT_FALSE(engine->synchronizeStopRequested());
    }
}

TEST_P(EngineStopTest, OtherDevicesDoNotAddExpertStopVotes) {
    auto engine = createMockEngine(CustomConfig{});
    ASSERT_TRUE(engine->stop().ok());
    engine->sm120_ep_stop_sync_        = false;
    engine->model_config_.expert_num   = 4;
    engine->parallelism_config.dp_size = 4;
    engine->parallelism_config.ep_size = 4;
    // No communication callback is registered. Any added DP vote fails this
    // production call, including TP1 and the TP leaders of mixed TP/DP/EP.
    for (int tp_size : {1, 2}) {
        engine->parallelism_config.tp_size = tp_size;
        for (int rank = 0; rank < tp_size; ++rank) {
            engine->parallelism_config.tp_rank = rank;
            engine->stop_requested_            = false;
            EXPECT_FALSE(engine->synchronizeStopRequested());
        }
    }
    engine->parallelism_config.tp_size = 1;
    engine->parallelism_config.tp_rank = 0;
    engine->stop_requested_            = true;
    EXPECT_TRUE(engine->synchronizeStopRequested());
}

class TerminalFrameExecutor: public Executor {
public:
    explicit TerminalFrameExecutor(const ParallelismConfig& config): config_(config) {}

    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
        GptModelInputs inputs;
        tpSyncModelInputs(inputs, config_);
        execution_stopped_ = inputs.stop_execution;
        return absl::OkStatus();
    }

private:
    ParallelismConfig config_;
};

class EngineTpStopChild: public DeviceTestBase {};

TEST_F(EngineTpStopChild, DISABLED_RunRank) {
    const auto rank_env = std::getenv("RTP_ENGINE_TP_TEST_RANK");
    const auto base_env = std::getenv("RTP_ENGINE_TP_TEST_BASE");
    ASSERT_NE(rank_env, nullptr);
    ASSERT_NE(base_env, nullptr);
    ::alarm(90);
    const int rank        = std::atoi(rank_env);
    auto&     broadcaster = CpuTpBroadcaster::instance();
    broadcaster.initialize(rank, 2, base_env);
    for (int dp_size : {1, 2}) {
        auto engine = createMockEngine(CustomConfig{});
        ASSERT_TRUE(engine->stop().ok());
        auto& config            = engine->parallelism_config;
        config.tp_size          = 2;
        config.tp_rank          = rank;
        config.dp_size          = dp_size;
        config.world_size       = 2 * dp_size;
        engine->executor_       = std::make_unique<TerminalFrameExecutor>(config);
        engine->running_        = true;
        engine->stop_requested_ = rank == 0;
        // Only the leader requests stop. The follower must leave its real
        // step after receiving the terminal frame on this TP group's channel.
        EXPECT_TRUE(engine->step().ok());
        EXPECT_TRUE(engine->executor_->executionStopped());
        EXPECT_TRUE(engine->stop_requested_.load());
        EXPECT_FALSE(engine->running_.load());
        EXPECT_TRUE(engine->stop().ok());
    }
    broadcaster.reset();
    ::alarm(0);
}

TEST(EngineTpStopTest, LeaderStopExitsTpFollowersWithAndWithoutDp) {
    const auto        tmpdir  = std::getenv("TMPDIR");
    std::string       pattern = std::string(tmpdir ? tmpdir : "/tmp") + "/engine_tp.XXXXXX";
    std::vector<char> temp_path(pattern.begin(), pattern.end());
    temp_path.push_back('\0');
    ASSERT_NE(::mkdtemp(temp_path.data()), nullptr);
    const std::string  base = std::string(temp_path.data()) + "/b";
    std::vector<pid_t> children;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<std::string> env_strings;
        for (char** entry = environ; *entry != nullptr; ++entry) {
            const std::string value(*entry);
            if (value.rfind("GTEST_", 0) != 0 && value.rfind("RTP_ENGINE_TP_TEST_", 0) != 0) {
                env_strings.push_back(value);
            }
        }
        env_strings.push_back("RTP_ENGINE_TP_TEST_RANK=" + std::to_string(rank));
        env_strings.push_back("RTP_ENGINE_TP_TEST_BASE=" + base);
        std::vector<char*> child_env;
        for (auto& value : env_strings) {
            child_env.push_back(value.data());
        }
        child_env.push_back(nullptr);
        std::vector<std::string> args = {"/proc/self/exe",
                                         "--gtest_filter=EngineTpStopChild.DISABLED_RunRank",
                                         "--gtest_also_run_disabled_tests",
                                         "--gtest_output="};
        std::vector<char*>       child_args;
        for (auto& arg : args) {
            child_args.push_back(arg.data());
        }
        child_args.push_back(nullptr);
        pid_t     pid    = -1;
        const int result = ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, child_args.data(), child_env.data());
        EXPECT_EQ(result, 0);
        if (result == 0) {
            children.push_back(pid);
        }
    }
    for (const auto pid : children) {
        int   status = 0;
        pid_t waited;
        do {
            waited = ::waitpid(pid, &status, 0);
        } while (waited < 0 && errno == EINTR);
        EXPECT_EQ(waited, pid);
        if (waited == pid) {
            EXPECT_TRUE(WIFEXITED(status)) << "TP stop child failed: " << status;
            if (WIFEXITED(status)) {
                EXPECT_EQ(WEXITSTATUS(status), 0);
            }
        }
    }
    ::unlink((base + "_0.sock").c_str());
    ::rmdir(temp_path.data());
}

TEST_P(EngineStopTest, StoppedSchedulerRejectsNewRequests) {
    auto engine = createMockEngine(CustomConfig{});
    EXPECT_FALSE(engine->isStopping());
    ASSERT_TRUE(engine->stop().ok());
    EXPECT_TRUE(engine->isStopping());
    auto query                             = std::make_shared<GenerateInput>();
    query->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
    query->generate_config                 = std::make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 2;
    auto stream                            = engine->enqueue(query);
    EXPECT_TRUE(stream->hasError());
    EXPECT_TRUE(engine->scheduler_->empty());
    auto [accepted, streams] = engine->enqueueMultiple({query, query});
    EXPECT_EQ(accepted, std::vector<bool>({false, false}));
    ASSERT_EQ(streams.size(), 2);
    EXPECT_TRUE(streams[0]->hasError());
    EXPECT_TRUE(streams[1]->hasError());
    EXPECT_TRUE(engine->scheduler_->empty());
}

TEST_P(EngineStopTest, ConcurrentStopCallsComplete) {
    auto engine = createMockEngine(CustomConfig{});
    auto first  = std::async(std::launch::async, [&] { return engine->stop(); });
    auto second = std::async(std::launch::async, [&] { return engine->stop(); });
    EXPECT_EQ(first.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(second.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(first.get().ok());
    EXPECT_TRUE(second.get().ok());
    EXPECT_EQ(engine->executor_, nullptr);
}

TEST_P(EngineStopTest, InvalidFixedDecodeBatchFailsBeforeDistributedStartup) {
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params                      = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    params.moe_config.moe_strategy            = "sm120_decode";
    params.hw_kernel_config.enable_cuda_graph = true;
    params.pd_sep_config.role_type            = RoleType::DECODE;
    params.parallelism_config.tp_size         = 1;
    params.parallelism_config.dp_size         = 8;
    params.parallelism_config.ep_size         = 8;
    params.parallelism_config.world_size      = 8;
    for (const char* value : {"1", "-1", "invalid", "4x"}) {
        SCOPED_TRACE(value);
        autil::EnvGuard fixed_batch("RTP_LLM_DECODE_FIXED_BS", value);
        EXPECT_THROW(std::make_shared<NormalEngine>(params, nullptr), std::invalid_argument);
    }
}

INSTANTIATE_TEST_SUITE_P(SerialAndParallel, EngineStopTest, ::testing::Values(0, 2));

}  // namespace
}  // namespace rtp_llm
