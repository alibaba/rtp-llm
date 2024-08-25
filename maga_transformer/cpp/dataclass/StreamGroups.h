#pragma once

#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/core/Buffer.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace ft = fastertransformer;

namespace rtp_llm {

struct StreamGroups {
public:
    StreamGroups(const std::list<GenerateStreamPtr>& streams) {
        for (auto& stream : streams) {
            if (stream->isContextStream()) {
                context_streams_.push_back(stream);
                model_execute_token_size_ += stream->currentExecuteTokenSize();
                total_context_batch_size_ += stream->batchSize();
                total_sampler_batch_size_ += stream->tileNum();
                max_block_size_ = std::max(max_block_size_, stream->maxBlockSize());
                max_seq_len_    = std::max(max_seq_len_, (size_t)stream->seqLength());
                max_context_seq_len_ = std::max(max_context_seq_len_, (size_t)stream->contextLength());
                max_reuse_length_ = std::max(max_reuse_length_, (size_t)stream->reuseLength());
                cum_context_seq_len_ += (size_t)stream->contextLength();
                multimodal_features_len_ += stream->multimodalFeaturesLength();
                if (!has_multimodal_input_ && multimodal_features_len_ > 0) {
                    has_multimodal_input_ = true;
                }
                total_score_batch_size_   += stream->scoreLen();
                adapter_names.push_back(stream->adapterName());
            } else {
                decode_streams_.push_back(stream);
                model_execute_token_size_ += stream->currentExecuteTokenSize();
                total_sampler_batch_size_ += stream->tileNum();
                total_decode_batch_size_  += stream->batchSize();
                max_block_size_ = std::max(max_block_size_, stream->maxBlockSize());
                max_seq_len_    = std::max(max_seq_len_, (size_t)stream->seqLength());
                if (!has_multimodal_input_ && stream->multimodalFeaturesLength() > 0) {
                    has_multimodal_input_ = true;
                }
                total_score_batch_size_   += stream->scoreLen();
                adapter_names.push_back(stream->adapterName());
            }
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
    size_t totalSamplerBatchSize() const {
        return total_sampler_batch_size_;
    }
    size_t maxBlockSize() const {
        return max_block_size_;
    }
    size_t modelExecuteTokenSize() const {
        return model_execute_token_size_;
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

    std::list<GenerateStreamPtr> allStreams() const {
        std::list<GenerateStreamPtr> all_streams = decode_streams_;
        all_streams.splice(all_streams.end(), std::list<GenerateStreamPtr>(context_streams_));
        return all_streams;
    }

    int size() const {
        return context_streams_.size() + decode_streams_.size();
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
                     << ", total_sampler_batch_size: " << total_sampler_batch_size_
                     << ", max_block_size: " << max_block_size_
                     << ", model_execute_token_size: " << model_execute_token_size_ << ", max_seq_len: " << max_seq_len_
                     << "}";
        return debug_string.str();
    }

private:
    std::list<GenerateStreamPtr> context_streams_;
    std::list<GenerateStreamPtr> decode_streams_;
    size_t                       total_sampler_batch_size_ = 0;
    size_t                       total_decode_batch_size_  = 0;
    size_t                       total_context_batch_size_ = 0;
    size_t                       max_block_size_           = 0;
    size_t                       model_execute_token_size_ = 0;
    size_t                       max_seq_len_              = 0;
    size_t                       max_context_seq_len_      = 0;
    size_t                       max_reuse_length_         = 0;
    size_t                       cum_context_seq_len_      = 0;
    size_t                       multimodal_features_len_  = 0;
    size_t                       total_score_batch_size_   = 0;
    bool                         has_multimodal_input_     = false;
    std::list<std::string>       adapter_names;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;
}  // namespace rtp_llm
