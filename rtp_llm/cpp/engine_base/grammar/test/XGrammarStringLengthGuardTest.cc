#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

// A string length bound is the one JSON Schema keyword whose cost is not paid at compile time but on every
// decode step: xgrammar lowers it into a counted repetition, so the remaining-length counter joins the grammar
// state and every step inside the string lands in a state the adaptive token mask cache has never seen. The
// mask it recomputes away from the bounds is the same mask the unbounded field yields, so the whole
// vocabulary scan is waste, and it is charged on the engine loop thread once per stream per proposed token.
//
// The backend therefore strips the bound before compiling. These tests pin both halves of that claim: the
// pathology is real in the pinned xgrammar (RawXGrammar* below), and it can no longer be reached through the
// production compile path (Guarded* below). Absolute costs live in docs/dsv4/.

// The surface form a deepseek_xml structural tag drives, split into vocabulary pieces so the matcher can be
// walked with explicit token ids. Concatenated it spells the tag's two `begin` strings followed by the
// opening of the first parameter, which is where the raw-text body of a string field starts.
const std::vector<std::string> kTagPrefix = {
    "<",       "｜DSML｜", "tool_calls", ">\n",      "<",     "｜DSML｜", "invoke", " name",   "=\"",
    "emit_summary", "\">\n",   "<",          "｜DSML｜", "parameter", " name",   "=\"",    "action",
    "\"",      " string", "=\"",        "false",    "\">",
};

// The same position in a plain JSON schema: the opening quote of the first string property's value.
const std::vector<std::string> kJsonPrefix = {"{", "\"", "a", "\"", ":", "\""};

const char* kRawTextPiece = "x";

// Characters the matcher consumes inside the string body before the measurement. The cost of a fill climbs
// with the consumed length and then latches onto a plateau; the walk has to be long enough to reach it, or
// the measurement lands on the cheap ramp instead.
constexpr int kRawTextSteps = 256;

// Production scale matters: the cost of one pathological fill is linear in the vocabulary size, so a toy
// vocabulary would shrink the effect below the timer's resolution.
constexpr size_t kVocabSize = 4096;

// The bad case as the service received it: a deepseek_xml tool-call tag whose string parameters carry
// maxLength. Trimmed to the fields the walk needs; the reported request carried more.
const char* kTagTemplate = R"JSON({
  "type": "structural_tag",
  "format": {
    "type": "triggered_tags",
    "triggers": ["<｜DSML｜tool_calls>"],
    "tags": [{
      "type": "tag",
      "begin": "<｜DSML｜tool_calls>\n",
      "content": {
        "type": "tags_with_separator",
        "tags": [{
          "type": "tag",
          "begin": "<｜DSML｜invoke name=\"emit_summary\">\n",
          "content": {
            "type": "json_schema",
            "json_schema": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "action": {"type": "string", "minLength": 1@MAXLEN@},
                "errors": {"type": "string", "minLength": 1@MAXLEN@}
              },
              "required": ["action", "errors"]
            },
            "style": "deepseek_xml",
            "any_order": false
          },
          "end": "</｜DSML｜invoke>\n"
        }],
        "separator": "",
        "at_least_one": true,
        "stop_after_first": false
      },
      "end": "</｜DSML｜tool_calls>"
    }],
    "at_least_one": true,
    "stop_after_first": false
  }
})JSON";

const char* kJsonSchemaTemplate = R"JSON({
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "a": {"type": "string", "minLength": 1@MAXLEN@}
  },
  "required": ["a"]
})JSON";

std::string withMaxLength(const char* tmpl, bool with_max_length) {
    const std::string placeholder = "@MAXLEN@";
    const std::string value       = with_max_length ? ", \"maxLength\": 8000" : "";
    std::string       out         = tmpl;
    for (size_t pos = out.find(placeholder); pos != std::string::npos; pos = out.find(placeholder)) {
        out.replace(pos, placeholder.size(), value);
    }
    return out;
}

struct Vocabulary {
    std::vector<std::string>                 tokens;
    std::unordered_map<std::string, int32_t> index;

    std::vector<int32_t> drive(const std::vector<std::string>& prefix) const {
        std::vector<int32_t> out;
        for (const auto& piece : prefix) {
            out.push_back(index.at(piece));
        }
        out.insert(out.end(), kRawTextSteps, index.at(kRawTextPiece));
        return out;
    }

    int32_t words() const {
        return static_cast<int32_t>((tokens.size() + 31) / 32);
    }
};

const Vocabulary& vocabulary() {
    static const Vocabulary vocab = [] {
        Vocabulary                      out;
        std::unordered_set<std::string> seen;
        auto                            add = [&](const std::string& piece) {
            if (seen.insert(piece).second) {
                out.tokens.push_back(piece);
            }
        };

        add("<pad>");  // token 0, the stop token
        for (const auto& piece : kTagPrefix) {
            add(piece);
        }
        add(kRawTextPiece);
        for (int c = 1; c < 128; ++c) {
            add(std::string(1, static_cast<char>(c)));
        }
        for (int i = 0; out.tokens.size() < kVocabSize; ++i) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "f%04x", i);
            add(std::string("一") + buf);
        }

        for (int32_t i = 0; i < static_cast<int32_t>(out.tokens.size()); ++i) {
            out.index.emplace(out.tokens[i], i);
        }
        return out;
    }();
    return vocab;
}

xgrammar::TokenizerInfo tokenizerInfo() {
    const auto& vocab = vocabulary();
    return xgrammar::TokenizerInfo(
        vocab.tokens, xgrammar::VocabType::RAW, static_cast<int>(vocab.tokens.size()), std::vector<int32_t>{0});
}

XGrammarBackendOptions backendOptions() {
    XGrammarBackendOptions options;
    options.max_compiler_threads = 1;
    options.compile_timeout_ms   = 120000;
    return options;
}

DLTensor makeSingleRowBitmaskView(int32_t* data, int64_t* shape, int32_t words) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    shape[0]       = 1;
    shape[1]       = words;
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

struct FillCost {
    bool                 ok      = false;
    double               fill_us = 0.0;
    std::vector<int32_t> mask;
};

// Walks a matcher to the plateau inside the string body, then times the fills a decode loop would issue from
// there. `fill` and `accept` let the same measurement drive our matcher wrapper or a raw xgrammar matcher.
template <typename FillFn, typename AcceptFn>
FillCost measureFill(FillFn fill, AcceptFn accept, const std::vector<int32_t>& drive) {
    const auto&          vocab = vocabulary();
    std::vector<int32_t> bitmask(vocab.words(), 0);
    int64_t              shape[2];
    DLTensor             dl = makeSingleRowBitmaskView(bitmask.data(), shape, vocab.words());

    FillCost out;
    for (size_t i = 0; i < drive.size(); ++i) {
        if (!fill(&dl) || !accept(drive[i])) {
            ADD_FAILURE() << "walk rejected at step " << i;
            return out;
        }
    }

    // The plateau is flat and the ratios under test span orders of magnitude, so a few samples suffice. The
    // inner loop lifts the cheap variant's total above the clock's granularity; taking the minimum keeps a
    // scheduling hiccup from inflating either side.
    constexpr int kSamples    = 5;
    constexpr int kInnerFills = 64;
    double        best_us     = std::numeric_limits<double>::max();
    for (int s = 0; s < kSamples; ++s) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kInnerFills; ++i) {
            fill(&dl);
        }
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / kInnerFills;
        best_us = std::min(best_us, us);
    }
    out.ok      = true;
    out.fill_us = best_us;
    out.mask    = bitmask;
    return out;
}

// Compiles `key` through the production backend and measures a fill from the plateau. Everything the backend
// does to a spec before handing it to xgrammar is therefore in scope.
FillCost measureGuarded(const std::string& tokenizer_info_json, const GrammarKeyCpp& key, const std::vector<int32_t>& drive) {
    XGrammarBackendCpp backend(tokenizer_info_json, backendOptions());
    auto               result = backend.compileNow(key);
    if (result.compiled == nullptr) {
        ADD_FAILURE() << "compile failed: " << result.error_message;
        return {};
    }
    auto matcher = backend.createMatcher(result.compiled, /*require_reasoning=*/false, /*think_end_token_ids=*/std::nullopt);
    return measureFill([&](DLTensor* dl) { return matcher->fillBitmask(dl, 0); },
                       [&](int32_t token) { return matcher->acceptToken(token); },
                       drive);
}

// The same measurement with the spec handed straight to xgrammar, bypassing the backend's sanitizing. This is
// the control that keeps the guarded tests from passing for the wrong reason.
FillCost measureRaw(const std::string& tag_json, const std::vector<int32_t>& drive) {
    xgrammar::GrammarCompiler compiler(tokenizerInfo(), /*max_threads=*/1, /*cache_enabled=*/false);
    auto                      compiled = compiler.CompileStructuralTag(tag_json);
    xgrammar::GrammarMatcher  matcher(compiled);
    return measureFill([&](DLTensor* dl) { return matcher.FillNextTokenBitmask(dl, 0); },
                       [&](int32_t token) { return matcher.AcceptToken(token); },
                       drive);
}

// Timed with a fresh backend per sample, because a backend caches its verdict and would report the second
// compile as free.
double minCompileMs(const std::string& tokenizer_info_json, const GrammarKeyCpp& key) {
    constexpr int kSamples = 3;
    double        best_ms  = std::numeric_limits<double>::max();
    for (int s = 0; s < kSamples; ++s) {
        XGrammarBackendCpp backend(tokenizer_info_json, backendOptions());
        const auto         start  = std::chrono::steady_clock::now();
        auto               result = backend.compileNow(key);
        const double       ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (result.compiled == nullptr) {
            ADD_FAILURE() << "compile failed: " << result.error_message;
            return 0.0;
        }
        best_ms = std::min(best_ms, ms);
    }
    return best_ms;
}

// Ratio a guarded pair is allowed to differ by. Both variants compile to the same grammar, so the expected
// ratio is one; the slack only absorbs timer noise on a loaded machine.
constexpr double kMaxRatio = 5.0;
// Ratio the unguarded pathology clears by orders of magnitude. The bar sits far below the measured gap so a
// loaded machine cannot fail the control.
constexpr double kMinRatio = 20.0;

TEST(XGrammarStringLengthGuardTest, RawXGrammarRescansTheVocabularyForABoundedDeepSeekXmlString) {
    const auto drive         = vocabulary().drive(kTagPrefix);
    const auto constrained   = measureRaw(withMaxLength(kTagTemplate, /*with_max_length=*/true), drive);
    const auto unconstrained = measureRaw(withMaxLength(kTagTemplate, /*with_max_length=*/false), drive);
    ASSERT_TRUE(constrained.ok);
    ASSERT_TRUE(unconstrained.ok);

    // Nothing about the allowed set differs: the walk stops far from both bounds of the field, so maxLength
    // forbids no token that would otherwise be legal. The entire extra cost is waste.
    EXPECT_EQ(constrained.mask, unconstrained.mask);

    // A failure here means the pinned xgrammar no longer lowers length bounds into counted repetitions, i.e.
    // the upstream fix landed and the backend's strip can be reconsidered rather than that this test rotted.
    EXPECT_GT(constrained.fill_us, unconstrained.fill_us * kMinRatio)
        << "fill with maxLength " << constrained.fill_us << "us vs without " << unconstrained.fill_us << "us";
}

TEST(XGrammarStringLengthGuardTest, GuardedStructuralTagCompilesAsIfUnbounded) {
    const auto  tokenizer_info_json = tokenizerInfo().SerializeJSON();
    const auto  drive               = vocabulary().drive(kTagPrefix);
    const auto  bounded_key   = GrammarKeyCpp{"structural_tag", withMaxLength(kTagTemplate, true)};
    const auto  unbounded_key = GrammarKeyCpp{"structural_tag", withMaxLength(kTagTemplate, false)};

    const auto constrained   = measureGuarded(tokenizer_info_json, bounded_key, drive);
    const auto unconstrained = measureGuarded(tokenizer_info_json, unbounded_key, drive);
    ASSERT_TRUE(constrained.ok);
    ASSERT_TRUE(unconstrained.ok);

    EXPECT_EQ(constrained.mask, unconstrained.mask);
    EXPECT_LT(constrained.fill_us, unconstrained.fill_us * kMaxRatio)
        << "fill with maxLength " << constrained.fill_us << "us vs without " << unconstrained.fill_us << "us";

    // The state blow-up is charged at compile time too, which is what pushes such a request past its compile
    // budget and turns the first attempt into an outright rejection.
    const double bounded_ms   = minCompileMs(tokenizer_info_json, bounded_key);
    const double unbounded_ms = minCompileMs(tokenizer_info_json, unbounded_key);
    EXPECT_LT(bounded_ms, unbounded_ms * kMaxRatio)
        << "compile with maxLength " << bounded_ms << "ms vs without " << unbounded_ms << "ms";
}

TEST(XGrammarStringLengthGuardTest, GuardedJsonSchemaCompilesAsIfUnbounded) {
    const auto tokenizer_info_json = tokenizerInfo().SerializeJSON();
    const auto drive               = vocabulary().drive(kJsonPrefix);
    const auto bounded_key   = GrammarKeyCpp{"json", withMaxLength(kJsonSchemaTemplate, true)};
    const auto unbounded_key = GrammarKeyCpp{"json", withMaxLength(kJsonSchemaTemplate, false)};

    const auto constrained   = measureGuarded(tokenizer_info_json, bounded_key, drive);
    const auto unconstrained = measureGuarded(tokenizer_info_json, unbounded_key, drive);
    ASSERT_TRUE(constrained.ok);
    ASSERT_TRUE(unconstrained.ok);

    EXPECT_EQ(constrained.mask, unconstrained.mask);
    EXPECT_LT(constrained.fill_us, unconstrained.fill_us * kMaxRatio)
        << "fill with maxLength " << constrained.fill_us << "us vs without " << unconstrained.fill_us << "us";

    // In JSON style the field is quoted, so the vocabulary the bound blows up over is much smaller and the fill
    // gap alone is too narrow to hold this test up. Compile cost is where this path shows the blow-up.
    const double bounded_ms   = minCompileMs(tokenizer_info_json, bounded_key);
    const double unbounded_ms = minCompileMs(tokenizer_info_json, unbounded_key);
    EXPECT_LT(bounded_ms, unbounded_ms * kMaxRatio)
        << "compile with maxLength " << bounded_ms << "ms vs without " << unbounded_ms << "ms";
}

}  // namespace
}  // namespace rtp_llm
