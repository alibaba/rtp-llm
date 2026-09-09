#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {

class DeferredEngineStartTest: public DeviceTestBase {
protected:
    std::shared_ptr<NormalEngine> prepareEngine() {
        ModelConfig model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        auto params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        const auto vocab = model_config.vocab_size;
        NormalExecutor::test_model_factory = [vocab](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab);
        };
        auto engine = std::make_shared<NormalEngine>(params, nullptr, true);
        NormalExecutor::test_model_factory = nullptr;
        return engine;
    }

    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }
};

TEST_F(DeferredEngineStartTest, AbortedTemplateCanStopBeforeLoopStarts) {
    auto engine = prepareEngine();
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    EXPECT_FALSE(engine->running_.load());
    EXPECT_EQ(engine->loop_thread_, nullptr);
    EXPECT_TRUE(engine->stop().ok());
    // Destruction must also tolerate an initialized engine with no loop.
    engine.reset();
}

TEST_F(DeferredEngineStartTest, ReleaseStartsOneLoopAndGenerates) {
    auto engine = prepareEngine();
    ASSERT_FALSE(engine->running_.load());
    ASSERT_EQ(engine->loop_thread_, nullptr);
    std::shared_ptr<EngineBase> base = engine;
    ASSERT_TRUE(base->startLoop().ok());
    auto first_thread = engine->loop_thread_;
    ASSERT_NE(first_thread, nullptr);
    ASSERT_TRUE(base->startLoop().ok());
    EXPECT_EQ(engine->loop_thread_, first_thread);

    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 3;
    input->generate_config->is_streaming = false;
    auto stream = engine->enqueue(input);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    EXPECT_EQ(output.value().generate_outputs[0].aux_info.output_len, 3);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
}

}  // namespace rtp_llm
