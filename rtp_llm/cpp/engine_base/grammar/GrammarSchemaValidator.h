#pragma once

#include <string>

namespace rtp_llm {

struct GrammarKeyCpp;

enum class GrammarValidateStatus {
    Ok,
    InvalidSyntax,
    UnsupportedFeature,
    TooLarge,
};

struct GrammarValidateResult {
    GrammarValidateStatus status = GrammarValidateStatus::Ok;
    std::string           detail;
};

GrammarValidateResult validateGrammarKey(const GrammarKeyCpp& key);

}  // namespace rtp_llm
