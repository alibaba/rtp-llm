#include "rtp_llm/cpp/embedding_engine/EmbeddingTokenRange.h"

int main() {
    const int32_t valid[] = {0, 2, 16};
    rtp_llm::validateEmbeddingTokenRange(valid, 3, 17);
    // Token type rows are independent of word-vocabulary size. Both
    // endpoints of the actual three-row checkpoint table are usable.
    rtp_llm::validateEmbeddingTokenRange(valid, 2, 3, "token type");
    for (int32_t invalid : {-1, 3}) {
        try {
            rtp_llm::validateEmbeddingTokenRange(&invalid, 1, 3, "token type");
            return 1;
        } catch (const std::invalid_argument& error) {
            if (std::string(error.what()).find("token type") == std::string::npos) {
                return 1;
            }
        }
    }
    for (int32_t invalid : {-1, 17, 31}) {
        bool rejected = false;
        try {
            // This covers supplied IDs and tokenizer-generated special IDs:
            // the host boundary checks the actual IDs, never token labels.
            rtp_llm::validateEmbeddingTokenRange(&invalid, 1, 17);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) {
            return 1;
        }
    }
    return 0;
}
