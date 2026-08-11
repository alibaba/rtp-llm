#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/SpeculativeConfigValidator.h"

namespace rtp_llm {
namespace {

TEST(SpeculativeConfigValidatorTest, EffectiveInputVocabUsesPhysicalEmbeddingRows) {
    ModelConfig config;
    config.vocab_size = 128;
    EXPECT_EQ(effectiveInputVocabSize(config), 128);

    config.embedding_size = 127;
    EXPECT_EQ(effectiveInputVocabSize(config), 127);

    config.input_vocab_size = 120;
    EXPECT_EQ(effectiveInputVocabSize(config), 120);

    config.embedding_size = -1;
    EXPECT_THROW(effectiveInputVocabSize(config), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
