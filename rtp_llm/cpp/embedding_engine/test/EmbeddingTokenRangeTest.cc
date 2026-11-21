#include "rtp_llm/cpp/embedding_engine/EmbeddingTokenRange.h"

int main() {
    const int32_t valid[] = {0, 2, 16};
    rtp_llm::validateEmbeddingTokenRange(valid, 3, 17);
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
