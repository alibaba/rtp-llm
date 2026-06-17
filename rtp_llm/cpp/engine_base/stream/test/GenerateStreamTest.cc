
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamBuilder {
public:
    GenerateStreamBuilder() {
        model_config_.max_seq_len = 2048;
    }

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    // 调用方可传入并保留 generate_config 的 shared_ptr：它与 stream 内部的 generate_input_->generate_config
    // 是同一对象，因此测试可直接通过该句柄改写 config（如 reuse_cache / timeout_ms），无需访问 protected 成员。
    GenerateStreamPtr createContextStream(std::vector<int> input_ids, std::shared_ptr<GenerateConfig> generate_config) {
        std::shared_ptr<GenerateInput> generate_input(new GenerateInput());
        ResourceContext                resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
    };

    GenerateStreamPtr createContextStream(std::vector<int> input_ids) {
        return createContextStream(std::move(input_ids), std::make_shared<GenerateConfig>());
    };

    GenerateStreamPtr createComplexContextStream(std::vector<int> input_ids) {
        autil::EnvGuard perf_scope("PERF_TEST", "1");

        auto cache_config  = init_config();
        auto cache_manager = std::make_shared<KVCacheManager>(cache_config);
        cache_manager->init();
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        resource_context.reuse_cache   = true;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 2;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        auto stream              = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);

        return stream;
    }

    GenerateStreamPtr createDecoderStream(std::vector<int> input_ids, std::vector<int> new_token_ids) {
        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        ResourceContext                 resource_context;
        generate_input->generate_config = generate_config;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        auto stream_ptr = std::make_shared<NormalGenerateStream>(
            generate_input, model_config_, runtime_config_, resource_context, nullptr);
        stream_ptr->setIsContextStream(false);
        auto complete_ids = stream_ptr->completeTokenIds();
        std::memcpy(complete_ids.data_ptr<int32_t>() + stream_ptr->seqLength(),
                    new_token_ids.data(),
                    new_token_ids.size() * sizeof(int));
        stream_ptr->setSeqLength(stream_ptr->seqLength() + new_token_ids.size());
        return stream_ptr;
    };

private:
    ModelConfig   model_config_;
    RuntimeConfig runtime_config_;
};

class GenerateStreamTest: public DeviceTestBase {
protected:
};

TEST_F(GenerateStreamTest, testConstruct) {
    auto builder = GenerateStreamBuilder();
    auto stream1 = builder.createContextStream({{1, 2, 3, 4, 5}, {}});
    auto stream2 = builder.createDecoderStream({1, 2, 3, 4, 5}, {1, 2, 3});
}

TEST_F(GenerateStreamTest, testGenerateStreamReuseCacheMethod) {
    auto builder         = GenerateStreamBuilder();
    auto generate_config = std::make_shared<GenerateConfig>();
    auto stream          = builder.createContextStream({1, 2, 3, 4, 5, 6}, generate_config);

    // default true
    ASSERT_TRUE(stream->reuseCache());

    // flip to false and verify（generate_config 与 stream 内部共享同一对象）
    generate_config->reuse_cache = false;
    ASSERT_FALSE(stream->reuseCache());

    // flip back to true and verify
    generate_config->reuse_cache = true;
    ASSERT_TRUE(stream->reuseCache());
}

// 回归测试：消费者阻塞在 nextOutput()（队列空、流未结束）时，reportError() 必须立即唤醒它，
// 而不是让它白等 SynchronizedQueue 的 1s 超时。覆盖 Error 事件唤醒输出消费者的路径。
TEST_F(GenerateStreamTest, testReportErrorWakesBlockedNextOutput) {
    auto builder = GenerateStreamBuilder();
    auto stream  = builder.createContextStream({1, 2, 3, 4, 5});

    std::atomic<bool>  returned{false};
    std::atomic<bool>  ok{true};
    std::atomic<int>   code{-1};
    std::promise<void> started_promise;
    auto               started_future = started_promise.get_future();

    const auto  start = std::chrono::steady_clock::now();
    std::thread consumer([&]() {
        started_promise.set_value();         // 通知主线程：消费者已启动，即将进入 nextOutput 阻塞
        auto result = stream->nextOutput();  // 队列空且流未结束 → 阻塞在 cv_ 等待
        ok          = result.ok();
        code        = static_cast<int>(result.status().code());
        returned    = true;
    });
    // RAII 守卫：无论后续断言是否提前返回或抛异常，都确保线程被 join，
    // 避免 joinable 线程析构触发 std::terminate。
    struct ThreadGuard {
        std::thread& t;
        ~ThreadGuard() {
            if (t.joinable()) {
                t.join();
            }
        }
    } guard{consumer};

    // 先确认消费者线程已启动，再留出时间让它进入 waitNotEmpty() 阻塞。
    started_future.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(returned.load());

    // 报错应立即唤醒阻塞的消费者。
    stream->reportError(ErrorCode::CANCELLED, "cancelled by test");
    consumer.join();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_TRUE(returned.load());
    EXPECT_FALSE(ok.load());
    EXPECT_EQ(code.load(), static_cast<int>(ErrorCode::CANCELLED));
    // 必须远早于 SynchronizedQueue 的 1s(1,000,000us) 超时被唤醒。
    EXPECT_LT(elapsed_ms, 800);
}

// 回归测试：超时路径应在 timeout_ms 到点附近返回 GENERATE_TIMEOUT，
// 而不会在设置超时错误后又白等输出队列 1s（P2）。
TEST_F(GenerateStreamTest, testNextOutputTimeoutDoesNotWaitExtraSecond) {
    auto builder = GenerateStreamBuilder();

    constexpr int64_t kTimeoutMs    = 200;
    auto              generate_config = std::make_shared<GenerateConfig>();
    generate_config->timeout_ms     = kTimeoutMs;
    auto stream                     = builder.createContextStream({1, 2, 3, 4, 5}, generate_config);
    stream->resetBeginTime(autil::TimeUtility::currentTimeInMicroSeconds());

    std::atomic<bool> returned{false};
    std::atomic<int>  code{-1};

    const auto  start = std::chrono::steady_clock::now();
    std::thread consumer([&]() {
        auto result = stream->nextOutput();  // 队列空、未结束 → 等到超时点自报 GENERATE_TIMEOUT
        code        = static_cast<int>(result.status().code());
        returned    = true;
    });
    struct ThreadGuard {
        std::thread& t;
        ~ThreadGuard() {
            if (t.joinable()) {
                t.join();
            }
        }
    } guard{consumer};

    consumer.join();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_TRUE(returned.load());
    EXPECT_EQ(code.load(), static_cast<int>(ErrorCode::GENERATE_TIMEOUT));
    // 大致等到超时点即返回；远小于"超时点 + 额外 1s 队列空等"。
    EXPECT_GE(elapsed_ms, kTimeoutMs - 50);
    EXPECT_LT(elapsed_ms, kTimeoutMs + 700);
}

}  // namespace rtp_llm
