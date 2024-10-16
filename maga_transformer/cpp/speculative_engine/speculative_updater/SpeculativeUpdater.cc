#include "maga_transformer/cpp/speculative_engine/speculative_updater/SpeculativeUpdater.h"

namespace rtp_llm {

absl::Status SpeculativeUpdater::score_compact_kv_cache(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.score_compact_kv_cache) {
        return absl::OkStatus();
    }
    RETURN_IF_STATUS_ERROR(stream->releaseSequenceKVCache(stream->seqLength() + stream_output.propose_step + 1, stream_output.propose_step + 1 - stream_output.accepted_token_nums));
    return absl::OkStatus();
}


absl::Status SpeculativeUpdater::propose_compact_kv_cache(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.propose_compact_kv_cache) {
        return absl::OkStatus();
    }
    return absl::OkStatus();
}


absl::Status SpeculativeUpdater::save_score_last_state(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (!config_.save_score_last_state) {
        return absl::OkStatus();
    }
    // TODO(xyz): implement this
    return absl::OkStatus();
}

absl::Status SpeculativeUpdater::dispatch(const GenerateStreamPtr& stream, const SpeculativeSamplerStreamOutput& stream_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (stream->isChunkStream()) {
        return absl::OkStatus();
    }
    size_t num_accepted_tokens = stream_output.accepted_token_nums;
    const ft::BufferPtr& accepted_tokens = stream_output.accepted_tokens;
    const ft::BufferPtr& logits = stream_output.logits;
    const ft::BufferPtr& hidden_states = stream_output.hidden_states;
    const ft::BufferPtr& loss = stream_output.loss;
    stream->step();
    stream->update(accepted_tokens, num_accepted_tokens, logits, hidden_states, nullptr, nullptr, loss);
    stream->setReuseLength(stream->seqLength() - 1);
    stream->setFallbackPrefixLength(stream->reuseLength());
    stream->setAccepedBounsToken(stream_output.acceped_bouns_token);
    stream->incSpEditSearchIndex(num_accepted_tokens - 1);
    return absl::OkStatus();
}


}