#include "maga_transformer/cpp/speculative_engine/speculative_updater/SpeculativeUpdater.h"

namespace rtp_llm {

absl::Status SpeculativeUpdater::score_compact_kv_cache(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.score_compact_kv_cache) {
        return absl::OkStatus();
    }
    RETURN_IF_STATUS_ERROR(stream->releaseSequenceKVCache(stream->seqLength() + stream_output.propose_step + 1, stream_output.propose_step + 1 - stream_output.accepted_token_nums));
    return absl::OkStatus();
}


absl::Status SpeculativeUpdater::propose_compact_kv_cache(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.propose_compact_kv_cache) {
        return absl::OkStatus();
    }
    return absl::OkStatus();
}


absl::Status SpeculativeUpdater::save_score_last_state(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.save_score_last_state) {
        return absl::OkStatus();
    }
    // TODO(xyz): implement this
    return absl::OkStatus();
}

absl::Status SpeculativeUpdater::dispatch(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (stream->isChunkStream()) {
        return absl::OkStatus();
    }

    size_t num_accepted_tokens = stream_output.accepted_token_nums;
    const rtp_llm::BufferPtr& accepted_tokens = stream_output.accepted_tokens;
    const rtp_llm::BufferPtr& logits = stream_output.logits;
    const rtp_llm::BufferPtr& hidden_states = stream_output.hidden_states;
    const rtp_llm::BufferPtr& loss = stream_output.loss;
    const rtp_llm::BufferPtr& softmax_probs = stream_output.softmax_probs;
    stream->step();
    StreamUpdateInfo update_info{accepted_tokens, (int)num_accepted_tokens, logits, nullptr, softmax_probs, nullptr, nullptr, loss, hidden_states};
    stream->update(update_info);
    stream->setReuseLength(stream->seqLength() - 1);
    stream->setFallbackPrefixLength(stream->reuseLength());
    stream->setAccepedBounsToken(stream_output.acceped_bouns_token);
    stream->incSpEditSearchIndex(num_accepted_tokens - 1);
    stream->setSpEditRun(false);
    return absl::OkStatus();
}


}
