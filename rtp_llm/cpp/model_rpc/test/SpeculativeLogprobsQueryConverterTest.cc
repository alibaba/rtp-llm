#include "gtest/gtest.h"
#include "google/protobuf/unknown_field_set.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

TEST(SpeculativeLogprobsQueryConverterTest, DedicatedFieldRestoresRequestedModeAfterWireRoundTrip) {
    for (const int mode : {2, 3}) {
        GenerateConfigPB sender;
        sender.set_max_new_tokens(1);
        sender.set_return_all_probs(false);
        sender.set_return_all_probs_mode(1);
        // Baseline headers do not know field77. Serialize before parsing so
        // a new header recognizes it normally, rather than reading unknowns.
        sender.GetReflection()->MutableUnknownFields(&sender)->AddVarint(77, mode);
        GenerateConfigPB received;
        ASSERT_TRUE(received.ParseFromString(sender.SerializeAsString()));
        const auto config = QueryConverter::transGenerateConfig(&received);
        EXPECT_EQ(config->return_all_probs, static_cast<ReturnAllProbsMode>(mode - 1));
        EXPECT_FALSE(config->force_disable_sp_run);
    }
}

TEST(SpeculativeLogprobsQueryConverterTest, OrdinaryAndForceDisabledRequestsKeepLegacyModeSemantics) {
    for (const bool disabled : {false, true}) {
        for (const int mode : {2, 3}) {
            GenerateConfigPB sender;
            sender.set_max_new_tokens(1);
            sender.set_return_all_probs(true);
            sender.set_return_all_probs_mode(mode);
            sender.set_force_disable_sp_run(disabled);
            GenerateConfigPB received;
            ASSERT_TRUE(received.ParseFromString(sender.SerializeAsString()));
            const auto config = QueryConverter::transGenerateConfig(&received);
            EXPECT_EQ(config->return_all_probs, static_cast<ReturnAllProbsMode>(mode - 1));
            EXPECT_EQ(config->force_disable_sp_run, disabled);
        }
    }
}

TEST(SpeculativeLogprobsQueryConverterTest, LegacyBoolFallbackAndExistingModeClampStayUnchanged) {
    struct Case {
        int                wire_mode;
        bool               legacy;
        ReturnAllProbsMode expected;
    };
    const Case cases[] = {
        {0, false, ReturnAllProbsMode::NONE},
        {0, true, ReturnAllProbsMode::DEFAULT},
        {1, true, ReturnAllProbsMode::NONE},
        {2, false, ReturnAllProbsMode::DEFAULT},
        {3, false, ReturnAllProbsMode::ORIGINAL},
        {4, true, ReturnAllProbsMode::NONE},
        {5, true, ReturnAllProbsMode::NONE},
    };
    for (const auto& test : cases) {
        GenerateConfigPB config;
        config.set_max_new_tokens(1);
        config.set_return_all_probs_mode(test.wire_mode);
        config.set_return_all_probs(test.legacy);
        EXPECT_EQ(QueryConverter::transGenerateConfig(&config)->return_all_probs, test.expected);
    }
}

}  // namespace rtp_llm
