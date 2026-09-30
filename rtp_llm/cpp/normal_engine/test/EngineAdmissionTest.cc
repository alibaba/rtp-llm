#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/engine_base/sleep/AdmissionGate.h"

namespace rtp_llm {
namespace {

class EngineAdmissionTest: public DeviceTestBase {};

TEST_F(EngineAdmissionTest, BindsSchedulerAdmissionBeforePublishingEngine) {
    for (const bool enabled : {false, true}) {
        for (const std::string mode : {"fifo", "ratio", "batch_decode"}) {
            SCOPED_TRACE(::testing::Message() << "sleep=" << enabled << " scheduler=" << mode);
            ModelConfig   model;
            RuntimeConfig runtime;
            KVCacheConfig cache;
            auto          params                    = createEngineInitParams(CustomConfig{}, model, runtime, cache);
            params.runtime_config.enable_sleep_mode = enabled;
            params.runtime_config.sleep_mode_level  = enabled ? 2 : 0;
            params.runtime_config.warm_up           = false;
            params.runtime_config.use_batch_decode_scheduler                    = mode == "batch_decode";
            params.runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode = mode == "ratio" ? "ratio" : "";
            params.pd_sep_config.role_type                                      = RoleType::PDFUSION;
            NormalExecutor::test_model_factory = [vocab = model.vocab_size](const GptModelInitParams&) {
                return std::make_unique<MockModel>(vocab);
            };
            struct ResetFactory {
                ~ResetFactory() {
                    NormalExecutor::test_model_factory = nullptr;
                }
            } reset_factory;

            // Use the real constructor and defer only the inference loop. A
            // manually bound controller would conceal missing startup wiring.
            NormalEngine engine(params, nullptr, true);
            auto&        controller = engine.sleepController();
            ASSERT_NE(controller.admission(), nullptr);
            EXPECT_EQ(controller.admission(), engine.getScheduler().admission());
            AdmissionGate gate(&controller, "engine-admission-test");
            EXPECT_TRUE(gate.check().ok());
            auto root = gate.acquire();
            ASSERT_TRUE(root.detail.admitted);
            ASSERT_TRUE(root.complete);
            EXPECT_EQ(controller.activeAdmissionCount(), 1);
            root.complete();
            EXPECT_EQ(controller.activeAdmissionCount(), 0);

            engine.requestTermination();
            EXPECT_FALSE(gate.acquire().detail.admitted);
            auto continuation = gate.acquireCacheTransfer();
            ASSERT_TRUE(continuation.detail.admitted);
            ASSERT_TRUE(continuation.complete);
            EXPECT_EQ(controller.activeAdmissionCount(), 1);
            continuation.complete();
            controller.admission()->sealContinuations();
            EXPECT_FALSE(gate.acquireCacheTransfer().detail.admitted);
            EXPECT_EQ(controller.activeAdmissionCount(), 0);
        }
    }
}

}  // namespace
}  // namespace rtp_llm
