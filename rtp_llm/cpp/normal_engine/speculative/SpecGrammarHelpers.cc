#include "rtp_llm/cpp/normal_engine/speculative/SpecGrammarHelpers.h"

#include <cstdlib>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace spec_grammar {

DLTensor makeBitmaskView(int32_t* data, int32_t batch_size, int32_t words) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    static thread_local int64_t shape[2];
    shape[0]       = batch_size;
    shape[1]       = words;
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

bool tokenAllowed(const int32_t* bitmask, int32_t tok, int32_t vocab_size) {
    if (tok < 0 || tok >= vocab_size) {
        return false;
    }
    const int32_t word = bitmask[tok / 32];
    return (static_cast<uint32_t>(word) & (1u << (tok % 32))) != 0u;
}

bool streamGroupsHaveAttachedGrammar(const StreamGroups& stream_groups) {
    for (const auto& s : stream_groups.allStreams()) {
        if (s->hasGrammarMatcher()) {
            return true;
        }
    }
    return false;
}

std::vector<ActiveGrammarStream> collectActiveGrammarStreams(
    const StreamGroups&  stream_groups,
    const torch::Tensor& chain_tensor,
    size_t               chain_skip_first,
    int32_t&             vocab_size_out) {
    std::vector<ActiveGrammarStream> active;
    const int64_t                    row_size = chain_tensor.size(1);
    const int*                       base_ptr = chain_tensor.data_ptr<int>();
    int                              stream_idx = 0;
    for (auto& stream : stream_groups.allStreams()) {
        RtpGrammarMatcher* m = stream->tryGetGrammarMatcher();
        if (m && !m->isTerminated() && !m->finished()) {
            ActiveGrammarStream a;
            a.batch_idx = stream_idx;
            a.matcher   = m;
            a.stream    = stream;
            a.chain_ptr = base_ptr + stream_idx * row_size + chain_skip_first;
            a.chain_len = static_cast<size_t>(row_size) - chain_skip_first;
            active.push_back(std::move(a));
            if (vocab_size_out < 0) {
                vocab_size_out = m->vocabSize();
            }
        }
        ++stream_idx;
    }
    return active;
}

bool streamGroupsHaveGrammar(const StreamGroups& stream_groups) {
    for (const auto& s : stream_groups.allStreams()) {
        const auto& cfg = s->generateConfig();
        if (cfg->json_schema.has_value() || cfg->regex.has_value() || cfg->ebnf.has_value()
            || cfg->structural_tag.has_value()) {
            return true;
        }
    }
    return false;
}

bool spAcceptTraceEnabled() {
    static const bool kEnabled = [] {
        const char* v = std::getenv("RTP_SP_ACCEPT_TRACE");
        return v != nullptr && std::string(v) == "1";
    }();
    return kEnabled;
}

std::string grammarKindTag(const GenerateStreamPtr& stream) {
    const auto& cfg = stream->generateConfig();
    if (cfg->json_schema.has_value()) {
        return "json_schema";
    }
    if (cfg->regex.has_value()) {
        return "regex";
    }
    if (cfg->ebnf.has_value()) {
        return "ebnf";
    }
    if (cfg->structural_tag.has_value()) {
        return "structural_tag";
    }
    return "none";
}

void logSpAcceptTrace(const StreamGroups&                          stream_groups,
                      const speculative::SpeculativeSamplerOutput& spec_output,
                      size_t                                       propose_step) {
    if (!spAcceptTraceEnabled()) {
        return;
    }
    size_t stream_idx = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        if (stream_idx >= spec_output.accept_len.size()) {
            break;
        }
        const int         accept_len = spec_output.accept_len[stream_idx];
        const std::string kind       = grammarKindTag(stream);
        RTP_LLM_LOG_INFO(
            "[sp_accept_trace] stream_id=%ld grammar=%s propose_step=%zu accept_len=%d",
            stream->streamId(),
            kind.c_str(),
            propose_step,
            accept_len);
        ++stream_idx;
    }
}

}  // namespace spec_grammar
}  // namespace rtp_llm
