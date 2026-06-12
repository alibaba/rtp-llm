#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

bool isGrammarBackendDisabled(const std::string& backend) {
    std::string s = backend;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s.empty() || s == "none";
}

std::vector<int32_t> collectStopTokenIds(py::object model) {
    py::object           mc = model.attr("model_config");
    py::object           st = mc.attr("special_tokens");
    std::vector<int32_t> ids;
    auto                 add_id = [&](int32_t id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    };

    py::object eos = st.attr("eos_token_id");
    if (py::isinstance<py::int_>(eos)) {
        add_id(eos.cast<int32_t>());
    } else if (py::isinstance<py::list>(eos) || py::isinstance<py::tuple>(eos)) {
        for (py::handle item : eos) {
            add_id(py::reinterpret_borrow<py::object>(item).cast<int32_t>());
        }
    }

    py::object stop_list = st.attr("stop_words_id_list");
    if (!stop_list.is_none()) {
        try {
            auto stop_seqs = stop_list.cast<std::vector<std::vector<int64_t>>>();
            for (const auto& seq : stop_seqs) {
                if (seq.size() == 1) {
                    add_id(static_cast<int32_t>(seq[0]));
                }
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("collectStopTokenIds: stop_words_id_list cast failed: %s", e.what());
        }
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

int64_t resolveThinkTokenId(const std::string& reasoning_parser, py::object tokenizer, const char* token_attr) {
    if (reasoning_parser.empty() || !token_attr) {
        return -1;
    }
    try {
        py::module reasoning_parser_mod =
            py::module::import("rtp_llm.openai.renderers.sglang_helpers.reasoning_parser");
        py::object parser_cls = reasoning_parser_mod.attr("ReasoningParser");
        py::object parser     = parser_cls(py::str(reasoning_parser));
        py::object detector   = parser.attr("detector");
        py::object token      = detector.attr(token_attr);
        // add_special_tokens=False: HF tokenizers default to True, which would
        // prepend BOS (and sometimes append EOS) and turn the single think
        // marker into a 2+ token sequence. The size==1 check below would then
        // silently fail and the gate would never resolve a real id, leaving
        // grammar disabled for any reasoning-mode request.
        py::object encoded =
            tokenizer.attr("encode")(token, py::arg("add_special_tokens") = false);
        if (py::len(encoded) != 1) {
            RTP_LLM_LOG_WARNING(
                "resolveThinkTokenId(%s): expected single id, got len=%zu", token_attr, py::len(encoded));
            return -1;
        }
        return encoded[py::int_(0)].cast<int64_t>();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("resolveThinkTokenId(%s) failed: %s", token_attr, e.what());
        return -1;
    }
}

}  // namespace

std::string buildXGrammarTokenizerInfoJson(
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::string&                              backend_tokenizer_str,
    int64_t                                         vocab_size,
    const std::vector<int32_t>&                     stop_token_ids) {
    static constexpr int64_t kMaxVocabSize = 1'000'000;
    if (vocab_size <= 0 || vocab_size > kMaxVocabSize) {
        throw std::invalid_argument("vocab_size must be in (0, " + std::to_string(kMaxVocabSize) + "], got "
                                    + std::to_string(vocab_size));
    }
    std::vector<std::string> encoded_vocab(static_cast<size_t>(vocab_size));
    std::vector<bool>        slot_filled(static_cast<size_t>(vocab_size), false);
    int64_t                  max_filled_id = -1;
    for (const auto& [tok, tid] : vocab) {
        if (tid < 0 || tid >= vocab_size) {
            throw std::invalid_argument("token id " + std::to_string(tid) + " is out of vocab range [0, "
                                        + std::to_string(vocab_size) + ")");
        }
        const auto idx = static_cast<size_t>(tid);
        encoded_vocab[idx] = tok;
        slot_filled[idx]   = true;
        max_filled_id      = std::max(max_filled_id, static_cast<int64_t>(tid));
    }
    if (max_filled_id < 0) {
        throw std::invalid_argument("buildXGrammarTokenizerInfoJson: vocab is empty");
    }

    // HF tokenizer.get_vocab() can be sparse vs vocab_size when the embedding
    // table is padded above the highest assigned id. If we hand xgrammar the
    // padded vocab as-is, the empty slots become zero-length tokens that the
    // DFA will accept as legal-but-empty matches and let it cross grammar
    // boundaries on what should be unreachable ids. Two malformations to
    // handle separately:
    //   * tail padding (ids past the highest real id): truncate the grammar
    //     vocab to max_filled_id+1. Sampling-time logits past this range are
    //     masked elsewhere — they cannot reach the grammar.
    //   * interior holes: fail fast. A hole inside the real id range is a
    //     vocab construction bug, not benign padding, and silently filling it
    //     with a placeholder would let those ids pass grammar checks.
    const int64_t effective_vocab_size = max_filled_id + 1;
    if (effective_vocab_size < vocab_size) {
        RTP_LLM_LOG_INFO(
            "buildXGrammarTokenizerInfoJson: truncating tail padding %lld..%lld; grammar vocab=%lld",
            static_cast<long long>(effective_vocab_size),
            static_cast<long long>(vocab_size - 1),
            static_cast<long long>(effective_vocab_size));
        encoded_vocab.resize(static_cast<size_t>(effective_vocab_size));
    }
    for (int64_t i = 0; i < effective_vocab_size; ++i) {
        if (!slot_filled[static_cast<size_t>(i)]) {
            throw std::invalid_argument(
                "buildXGrammarTokenizerInfoJson: interior vocab hole at id "
                + std::to_string(i) + " (max_filled_id=" + std::to_string(max_filled_id)
                + "); refusing to feed zero-length token to xgrammar");
        }
    }

    std::string meta = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);

    // Stop ids that fall inside the truncated tail padding [effective_vocab_size, vocab_size)
    // would be injected against a smaller declared vocab_size: xgrammar then silently drops
    // them from its internal stop_token_ids_, so grammar EOS-unmask paths (can_reach_end /
    // IsTerminated stop fast path) never recognise them. Filter explicitly and warn so the
    // operator notices the special_tokens / vocab mismatch instead of getting subtle
    // grammar-termination drift.
    std::vector<int32_t> filtered_stops;
    filtered_stops.reserve(stop_token_ids.size());
    int32_t dropped_count = 0;
    for (int32_t sid : stop_token_ids) {
        if (sid >= 0 && static_cast<int64_t>(sid) < effective_vocab_size) {
            filtered_stops.push_back(sid);
        } else {
            ++dropped_count;
        }
    }
    if (dropped_count > 0) {
        RTP_LLM_LOG_WARNING(
            "buildXGrammarTokenizerInfoJson: dropped %d stop_token_id(s) outside effective_vocab_size=%lld "
            "(declared vocab_size=%lld). xgrammar EOS-unmask will not recognise these ids.",
            dropped_count,
            static_cast<long long>(effective_vocab_size),
            static_cast<long long>(vocab_size));
    }
    std::string stops = "[";
    for (size_t i = 0; i < filtered_stops.size(); ++i) {
        if (i) stops += ",";
        stops += std::to_string(filtered_stops[i]);
    }
    stops += "]";

    const auto close_pos = meta.rfind('}');
    if (close_pos == std::string::npos) {
        throw std::runtime_error(
            "buildXGrammarTokenizerInfoJson: DetectMetadataFromHF returned a non-object metadata string: '"
            + meta + "'");
    }
    const auto first_content = meta.find_first_not_of(" \t\r\n", meta.find('{') + 1);
    const bool has_members   = first_content != std::string::npos && first_content < close_pos;
    std::string injected = std::string(has_members ? "," : "") + "\"vocab_size\":" + std::to_string(effective_vocab_size)
                           + ",\"stop_token_ids\":" + stops;
    meta.insert(close_pos, injected);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
}

void bootstrapGrammarConfigFromModel(py::object model, GrammarConfig& gc, ReasoningConfig& rc) {
    if (isGrammarBackendDisabled(gc.grammar_backend)) {
        gc.tokenizer_info_json = "";
        return;
    }

    std::vector<int32_t> stop_token_ids;
    int64_t              vocab_size = 0;
    try {
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        auto       vocab     = tokenizer.attr("get_vocab")().cast<std::unordered_map<std::string, int32_t>>();
        vocab_size           = model.attr("model_config").attr("vocab_size").cast<int64_t>();
        if (!vocab.empty()) {
            int32_t max_id = 0;
            for (const auto& [_, tid] : vocab) {
                max_id = std::max(max_id, tid);
            }
            vocab_size = std::max(vocab_size, static_cast<int64_t>(max_id) + 1);
        }
        stop_token_ids                = collectStopTokenIds(model);
        const std::string backend_str = tokenizer.attr("backend_tokenizer").attr("to_str")().cast<std::string>();
        gc.tokenizer_info_json        = buildXGrammarTokenizerInfoJson(vocab, backend_str, vocab_size, stop_token_ids);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("xgrammar bootstrap failed (%s); grammar disabled", e.what());
        gc.tokenizer_info_json = "";
        return;
    }

    if (!rc.reasoning_parser.empty()) {
        // Guard the attr chain itself: an attr() exception (model with no
        // tokenizer.tokenizer chain, import side effect on reasoning_parser)
        // would otherwise escape as pybind11::error_already_set after we've
        // already mutated gc.tokenizer_info_json. Leave think ids at -1 and
        // let reasoning admission degrade gracefully.
        try {
            py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
            rc.think_end_id      = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_end_token");
            rc.think_start_id    = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_start_token");
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("xgrammar reasoning bootstrap failed (%s); think ids left at -1", e.what());
            rc.think_end_id   = -1;
            rc.think_start_id = -1;
        }
    }

    RTP_LLM_LOG_INFO(
        "grammar bootstrap: vocab=%lld, json=%zuB, think_end_id=%lld, think_start_id=%lld, stop_tokens=%zu",
        static_cast<long long>(vocab_size),
        gc.tokenizer_info_json.size(),
        static_cast<long long>(rc.think_end_id),
        static_cast<long long>(rc.think_start_id),
        stop_token_ids.size());
}

}  // namespace rtp_llm
