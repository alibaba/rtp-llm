#pragma once

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace rtp_llm {

struct StreamGroups {
public:
    struct TokenCounts {
        int64_t context            = 0;
        int64_t context_with_cache = 0;
        int64_t generate           = 0;
        int64_t total              = 0;
    };
    using TokenCountsByPriority = std::map<int32_t, TokenCounts>;

    StreamGroups(const std::list<GenerateStreamPtr>& streams) {
        for (auto& stream : streams) {
            // A cache snapshot is published only for an established decode
            // stream. Avoid reading is_context_stream_ while the prior worker
            // redundantly commits the same decode state.
            const bool use_mtp_snapshot = stream->hasMtpCacheSnapshot();
            const bool is_context       = use_mtp_snapshot ? false : stream->isContextStream();
            // A speculative stream is admitted only with one fixed sequence.
            // currentBatchSize()/nextBatchSize() derive their answer from the
            // mutable output-token history, which belongs to the bookkeeping
            // worker while a device snapshot is active. Do not read that
            // history merely to rediscover the invariant batch size.
            if (use_mtp_snapshot) {
                RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                                        "MTP cache snapshots require one non-beam sequence per stream, stream=%ld",
                                        stream->streamId());
            }
            const auto cur_batch_size  = use_mtp_snapshot ? 1 : stream->currentBatchSize();
            const auto next_batch_size = use_mtp_snapshot ? 1 : stream->nextBatchSize();
            if (stream->isFakeStream()) {
                is_fake_stream_ = true;
            }
            if (is_context) {
                context_streams_.push_back(stream);
                total_context_batch_size_ += cur_batch_size;
                max_context_seq_len_ = std::max(max_context_seq_len_, (size_t)stream->contextLength());
                max_reuse_length_    = std::max(max_reuse_length_, (size_t)stream->reuseLength());
                cum_context_seq_len_ += (size_t)stream->contextLength();
                multimodal_features_len_ += stream->multimodalFeaturesLength();
                if (!has_multimodal_input_ && multimodal_features_len_ > 0) {
                    has_multimodal_input_ = true;
                }
            } else {
                decode_streams_.push_back(stream);
                total_decode_batch_size_ += cur_batch_size;
                // Decode does not consume multimodal feature tensors. Avoid
                // multimodalFeaturesLength(), which also derives a multiplier
                // from worker-owned output-token history.
                if (!use_mtp_snapshot && !has_multimodal_input_ && stream->multimodalFeaturesLength() > 0) {
                    has_multimodal_input_ = true;
                }
            }
            auto block_update_copy_num = stream->streamCacheResource().getKVBlockUpdateMapping().size();
            if (is_context) {
                context_block_update_copy_num_ += block_update_copy_num;
            } else {
                decode_block_update_copy_num_ += block_update_copy_num;
            }
            // A decode round always executes one input row per live sequence.
            // While an async MTP commit is pending, complete_token_ids is
            // worker-owned; do not read it merely to rediscover cur_batch_size.
            auto execute_token_size = use_mtp_snapshot ? static_cast<size_t>(cur_batch_size) :
                                                         static_cast<size_t>(stream->currentExecuteTokenSize());
            if (is_context) {
                auto reuse_length = stream->reuseLength();
                context_execute_token_size_ += execute_token_size;
                context_execute_token_size_with_cache_ += execute_token_size;
                if (reuse_length > 0) {
                    context_execute_token_size_with_cache_ += static_cast<size_t>(reuse_length) * cur_batch_size;
                }
            }
            model_execute_token_size_ += execute_token_size;
            total_sampler_batch_size_in_ +=
                !use_mtp_snapshot && stream->needTilingForSampling() ? next_batch_size : cur_batch_size;
            total_sampler_batch_size_out_ += next_batch_size;
            max_blocks_num_ = std::max(max_blocks_num_, stream->curBlocksNum());
            // cache_keys are context/CacheStore metadata and are not consumed
            // by a decode-only model input.  Avoid traversing them while the
            // speculative worker updates the corresponding token history.
            if (!use_mtp_snapshot && stream->hasCacheKeys()) {
                for (int32_t batch_id = 0; batch_id < cur_batch_size; ++batch_id) {
                    max_cache_keys_num_ = std::max(max_cache_keys_num_, stream->cacheKeys(batch_id).size());
                }
            }
            const int    snapshot_seq_len = stream->getMtpAsyncDeviceState().next_seq_len_upper_bound;
            const size_t seq_len          = use_mtp_snapshot && snapshot_seq_len > 0 ?
                                                static_cast<size_t>(snapshot_seq_len) :
                                                static_cast<size_t>(stream->seqLength());
            max_seq_len_                  = std::max(max_seq_len_, seq_len);
            total_score_batch_size_ += stream->scoreLen();
            adapter_names.push_back(stream->adapterName());
            gen_timeline_ |= use_mtp_snapshot ? stream->genTimelineAtSeqLength(static_cast<int>(seq_len)) :
                                                stream->genTimeline();
        }
    }

    size_t totalDecodeBatchSize() const {
        return total_decode_batch_size_;
    }
    size_t totalContextBatchSize() const {
        return total_context_batch_size_;
    }
    size_t totalModelBatchSize() const {
        return total_decode_batch_size_ + total_context_batch_size_;
    }
    size_t totalSamplerBatchSizeIn() const {
        return total_sampler_batch_size_in_;
    }
    size_t totalSamplerBatchSizeOut() const {
        return total_sampler_batch_size_out_;
    }
    size_t totalBlockUpdateCopyNum() const {
        return decode_block_update_copy_num_ + context_block_update_copy_num_;
    }
    size_t decodeBlockUpdateCopyNum() const {
        return decode_block_update_copy_num_;
    }
    size_t contextBlockUpdateCopyNum() const {
        return context_block_update_copy_num_;
    }
    size_t curBlocksNum() const {
        return max_blocks_num_;
    }
    size_t maxCacheKeysNum() const {
        return max_cache_keys_num_;
    }
    size_t modelExecuteTokenSize() const {
        return model_execute_token_size_;
    }
    size_t contextExecuteTokenSize() const {
        return context_execute_token_size_;
    }
    size_t contextExecuteTokenSizeWithCache() const {
        return context_execute_token_size_with_cache_;
    }

    TokenCountsByPriority tokenCountsByPriority() const {
        TokenCountsByPriority token_counts;
        for (const auto& stream : context_streams_) {
            auto& counts             = token_counts[stream->priority()];
            auto  execute_token_size = stream->currentExecuteTokenSize();
            counts.context += execute_token_size;
            counts.context_with_cache += execute_token_size;
            counts.total += execute_token_size;
            if (stream->reuseLength() > 0) {
                counts.context_with_cache += static_cast<int64_t>(stream->reuseLength()) * stream->currentBatchSize();
            }
        }
        for (const auto& stream : decode_streams_) {
            auto& counts = token_counts[stream->priority()];
            counts.generate += stream->currentBatchSize();
            counts.total += stream->currentExecuteTokenSize();
        }
        return token_counts;
    }
    size_t maxSeqLen() const {
        return max_seq_len_;
    }
    size_t maxContextSeqLen() const {
        return max_context_seq_len_;
    }
    size_t maxReuseLength() const {
        return max_reuse_length_;
    }
    size_t cumContextSeqLen() const {
        return cum_context_seq_len_;
    }
    size_t mmFeaturesLen() const {
        return multimodal_features_len_;
    }
    bool has_multimodal_input() const {
        return has_multimodal_input_;
    }
    size_t totalScoreBatchSize() const {
        return total_score_batch_size_;
    }

    bool empty() const {
        return context_streams_.empty() && decode_streams_.empty();
    }

    const std::list<GenerateStreamPtr>& contextStreams() const {
        return context_streams_;
    }

    const std::list<GenerateStreamPtr>& decodeStreams() const {
        return decode_streams_;
    }

    bool hasMMExtraInput() const {
        for (auto& stream : context_streams_) {
            if (stream->hasMultimodalExtraInput()) {
                return true;
            }
        }
        return false;
    }

    // NOTE: Aggregates by "max mode" across all streams in the batch
    // (NONE < DEFAULT < ORIGINAL). When the batch contains any ORIGINAL
    // stream, the sampler runs the ORIGINAL path for the entire batch and
    // DEFAULT streams will receive un-renormalized raw probabilities. This
    // is intentional: it avoids per-row branching inside the sampler kernel.
    // Callers that cannot tolerate this implicit degradation must ensure
    // streams with different modes are not scheduled into the same batch.
    //
    // CORRECTNESS RISK: bot-flagged P1 — when DEFAULT and ORIGINAL streams
    // coexist in a batch, the DEFAULT-requesting consumer silently gets raw
    // (un-softmaxed) probabilities. We emit a one-shot WARNING per process
    // below so this isn't completely silent; long-term the scheduler should
    // batch-partition by mode, or the sampler should branch per-row. See
    // PR #349 review for context.
    ReturnAllProbsMode needReturnAllProbs() const {
        // get the max return all probs mode from all streams
        ReturnAllProbsMode return_all_probs = ReturnAllProbsMode::NONE;
        bool               has_default      = false;
        bool               has_original     = false;
        for (auto& stream : context_streams_) {
            auto cur_return_all_probs = stream->getReturnAllProbs();
            if (cur_return_all_probs == ReturnAllProbsMode::DEFAULT) {
                has_default = true;
            } else if (cur_return_all_probs == ReturnAllProbsMode::ORIGINAL) {
                has_original = true;
            }
            if (cur_return_all_probs > return_all_probs) {
                return_all_probs = cur_return_all_probs;
            }
        }
        for (auto& stream : decode_streams_) {
            auto cur_return_all_probs = stream->getReturnAllProbs();
            if (cur_return_all_probs == ReturnAllProbsMode::DEFAULT) {
                has_default = true;
            } else if (cur_return_all_probs == ReturnAllProbsMode::ORIGINAL) {
                has_original = true;
            }
            if (cur_return_all_probs > return_all_probs) {
                return_all_probs = cur_return_all_probs;
            }
        }
        if (has_default && has_original) {
            // One-shot log per process: DEFAULT streams in this batch will
            // see un-renormalized raw probs because sampler runs ORIGINAL.
            static std::once_flag mixed_mode_warn_once;
            std::call_once(mixed_mode_warn_once, []() {
                RTP_LLM_LOG_WARNING("needReturnAllProbs: batch contains both DEFAULT and ORIGINAL "
                                    "return_all_probs streams; DEFAULT consumers will receive raw "
                                    "(un-renormalized) probabilities for this batch. The scheduler "
                                    "should partition by mode to avoid this silent degradation.");
            });
        }
        return return_all_probs;
    }

    bool needReturnCumLogProbs() const {
        // beam kernel need cum log probs input, tmp set true
        for (auto& stream : context_streams_) {
            if (stream->returnCumLogProbs() || stream->hasNumBeams()) {
                return true;
            }
        }
        for (auto& stream : decode_streams_) {
            if (stream->returnCumLogProbs() || stream->hasNumBeams()) {
                return true;
            }
        }
        return false;
    }

    std::list<GenerateStreamPtr> allStreams() const {
        std::list<GenerateStreamPtr> all_streams = decode_streams_;
        all_streams.splice(all_streams.end(), std::list<GenerateStreamPtr>(context_streams_));
        return all_streams;
    }

    int size() const {
        return context_streams_.size() + decode_streams_.size();
    }

    bool genTimeline() {
        return gen_timeline_;
    }

    bool isFakeStream() const {
        return is_fake_stream_;
    }

    void setFakeStream(bool is_fake_stream) {
        is_fake_stream_ = is_fake_stream;
    }

    std::string debugString() const {
        std::stringstream debug_string, context_stream_ids, decode_stream_ids;
        for (auto& stream : context_streams_) {
            context_stream_ids << stream->streamId() << ",";
        }
        for (auto& stream : decode_streams_) {
            decode_stream_ids << stream->streamId() << ",";
        }
        debug_string << "StreamGroups { "
                     << "context_stream_ids: " << context_stream_ids.str()
                     << ", decode_stream_ids: " << decode_stream_ids.str()
                     << ", total_decode_batch_size: " << total_decode_batch_size_
                     << ", total_context_batch_size: " << total_context_batch_size_
                     << ", total_model_batch_size: " << totalModelBatchSize()
                     << ", total_sampler_batch_size_in: " << total_sampler_batch_size_in_
                     << ", total_sampler_batch_size_out: " << total_sampler_batch_size_out_
                     << ", total_block_update_copy_num: " << totalBlockUpdateCopyNum()
                     << ", max_blocks_num_: " << max_blocks_num_ << ", max_cache_keys_num_: " << max_cache_keys_num_
                     << ", model_execute_token_size: " << model_execute_token_size_
                     << ", context_execute_token_size: " << context_execute_token_size_
                     << ", context_execute_token_size_with_cache: " << context_execute_token_size_with_cache_
                     << ", max_seq_len: " << max_seq_len_ << ", is_fake_stream: " << is_fake_stream_ << "}";
        return debug_string.str();
    }

    void updateStreams(const std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
        int stream_idx = 0;
        for (auto& stream : decode_streams_) {
            stream->specUpdate(spec_update_infos[stream_idx]);
            stream_idx++;
        }
        for (auto& stream : context_streams_) {
            stream->specUpdate(spec_update_infos[stream_idx]);
            stream_idx++;
        }
    }

private:
    std::list<GenerateStreamPtr> context_streams_;
    std::list<GenerateStreamPtr> decode_streams_;
    size_t                       total_sampler_batch_size_in_           = 0;
    size_t                       total_sampler_batch_size_out_          = 0;
    size_t                       total_decode_batch_size_               = 0;
    size_t                       total_context_batch_size_              = 0;
    size_t                       decode_block_update_copy_num_          = 0;
    size_t                       context_block_update_copy_num_         = 0;
    size_t                       max_blocks_num_                        = 0;
    size_t                       max_cache_keys_num_                    = 0;
    size_t                       model_execute_token_size_              = 0;
    size_t                       context_execute_token_size_            = 0;
    size_t                       context_execute_token_size_with_cache_ = 0;
    size_t                       max_seq_len_                           = 0;
    size_t                       max_context_seq_len_                   = 0;
    size_t                       max_reuse_length_                      = 0;
    size_t                       cum_context_seq_len_                   = 0;
    size_t                       multimodal_features_len_               = 0;
    size_t                       total_score_batch_size_                = 0;
    bool                         has_multimodal_input_                  = false;
    bool                         gen_timeline_                          = false;
    bool                         is_fake_stream_                        = false;
    std::list<std::string>       adapter_names;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;
}  // namespace rtp_llm
