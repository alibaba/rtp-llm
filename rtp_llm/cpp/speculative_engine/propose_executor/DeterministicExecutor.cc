#include <memory>
#include <vector>

#include "rtp_llm/cpp/speculative_engine/propose_executor/DeterministicExecutor.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeStream.h"

namespace rtp_llm {

absl::Status DeterministicExecutor::propose(const std::list<GenerateStreamPtr>& streams, bool skip_check) {
    for (auto& stream : streams) {
        if (!skip_check) {
            if (stream->finishedWithoutLock() || stream->stoppedWithoutLock()) {
                continue;
            }
        }

        if (!stream->containProposeStream()) {
            GenerateStreamPtr propose_stream = std::make_shared<ProposeStream>(*stream, 0);
            stream->setProposeStream(propose_stream);
        }
        ruleBasedTokenSelector(stream);
    }
    return absl::OkStatus();
}

void DeterministicExecutor::ruleBasedTokenSelector(const GenerateStreamPtr& stream) {
    auto& config               = stream->generateConfig();
    bool  use_sp_advice_prompt = config->sp_advice_prompt_token_ids.size() > 0;

    SpeculativeExecutorStreamOutputPtr stream_output = stream->getProposeStream()->getSPOutputBuffer();
    stream_output->propose_step                      = 0;
    stream_output->tokens                            = nullptr;
    stream_output->all_probs                         = nullptr;

    if (config->sp_edit) {
        SpEditTokenSelector(stream, stream_output, use_sp_advice_prompt);
    } else {
        PromptLookUpTokenSelector(stream, stream_output, use_sp_advice_prompt);
    }

    // backup prompt lookup
    if (stream_output->tokens == nullptr && use_sp_advice_prompt) {
        PromptLookUpTokenSelector(stream, stream_output, false);
    }
}

void DeterministicExecutor::SpEditTokenSelector(const GenerateStreamPtr&            stream,
                                                SpeculativeExecutorStreamOutputPtr& stream_output,
                                                bool                                use_sp_advice_prompt) {
    auto&  config           = stream->generateConfig();
    size_t advice_token_num = 0;
    int*   advice_token_ids = nullptr;
    if (use_sp_advice_prompt && config->sp_advice_prompt_token_ids.size() > 0) {
        advice_token_num = config->sp_advice_prompt_token_ids.size();
        advice_token_ids = config->sp_advice_prompt_token_ids.data();
    } else {
        advice_token_num = stream->seqLength();
        advice_token_ids = stream->completeTokenIds()->data<int>();
    }
    const auto&  sp_edit_search_index = stream->spEditSearchIndex();
    const size_t seq_len              = stream->seqLength();
    size_t max_new_tokens = std::min((size_t)config->max_new_tokens, stream->maxSeqLen() - stream->seqLength() - 1);
    if (stream->spEditFirstTime()) {
        size_t begin_propose_offset = seq_len - stream->inputLength();
        advice_token_num -= begin_propose_offset;
        size_t propose_len = std::min(max_new_tokens, std::min(propose_step_, advice_token_num - sp_edit_search_index));
        stream_output->tokens = std::make_shared<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_CPU,
                                                                  rtp_llm::DataType::TYPE_INT32,
                                                                  std::vector{1, propose_len},
                                                                  advice_token_ids + begin_propose_offset);
        stream->setSpEditFirstTime(false);
        stream->setSpEditRun(true);
        postProcess(stream, stream_output);
    } else if (min_token_match_len_ <= seq_len) {
        size_t begin_match_len = std::min(max_token_match_len_, seq_len);

        std::vector<int> max_latest_tokens = stream->getLatestTokens(begin_match_len);
        for (size_t match_len = begin_match_len; match_len >= min_token_match_len_; match_len--) {

            std::vector<int> latest_tokens(max_latest_tokens.end() - match_len, max_latest_tokens.end());
            size_t           begin_index = sp_edit_search_index < match_len ? 0 : sp_edit_search_index - match_len;
            for (size_t i = begin_index; i + match_len < advice_token_num - 1; i++) {
                size_t j = 0;
                for (; j < match_len; j++) {
                    if (latest_tokens[j] != advice_token_ids[i + j]) {
                        break;
                    }
                }
                if (j == match_len) {
                    size_t start_propose_index = i + match_len;
                    size_t propose_len =
                        std::min(max_new_tokens, std::min(propose_step_, advice_token_num - start_propose_index));
                    stream_output->tokens = std::make_shared<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_CPU,
                                                                              rtp_llm::DataType::TYPE_INT32,
                                                                              std::vector{1, propose_len},
                                                                              advice_token_ids + start_propose_index);
                    stream->setSpEditRun(true);
                    postProcess(stream, stream_output);
                    return;
                }
            }
        }
    }
}

void DeterministicExecutor::PromptLookUpTokenSelector(const GenerateStreamPtr&            stream,
                                                      SpeculativeExecutorStreamOutputPtr& stream_output,
                                                      bool                                use_sp_advice_prompt) {
    auto&  config           = stream->generateConfig();
    size_t advice_token_num = 0;
    int*   advice_token_ids = nullptr;
    if (use_sp_advice_prompt && config->sp_advice_prompt_token_ids.size() > 0) {
        advice_token_num = config->sp_advice_prompt_token_ids.size();
        advice_token_ids = config->sp_advice_prompt_token_ids.data();
    } else {
        advice_token_num = stream->seqLength();
        advice_token_ids = stream->completeTokenIds()->data<int>();
    }

    const size_t seq_len  = stream->seqLength();
    size_t max_new_tokens = std::min((size_t)config->max_new_tokens, stream->maxSeqLen() - stream->seqLength() - 1);

    if (min_token_match_len_ <= seq_len) {
        size_t           begin_match_len   = std::min(max_token_match_len_, seq_len);
        std::vector<int> max_latest_tokens = stream->getLatestTokens(begin_match_len);

        for (size_t match_len = begin_match_len; match_len >= min_token_match_len_; match_len--) {
            std::vector<int> latest_tokens(max_latest_tokens.end() - match_len, max_latest_tokens.end());
            for (size_t i = 0; i + match_len < advice_token_num - 1; i++) {
                size_t j = 0;
                for (; j < match_len; j++) {
                    if (latest_tokens[j] != advice_token_ids[i + j]) {
                        break;
                    }
                }
                if (j == match_len) {
                    size_t start_propose_index = i + match_len;
                    size_t propose_len =
                        std::min(max_new_tokens, std::min(propose_step_, advice_token_num - start_propose_index));
                    stream_output->tokens = std::make_shared<rtp_llm::Buffer>(rtp_llm::MemoryType::MEMORY_CPU,
                                                                              rtp_llm::DataType::TYPE_INT32,
                                                                              std::vector{1, propose_len},
                                                                              advice_token_ids + start_propose_index);
                    postProcess(stream, stream_output);
                    return;
                }
            }
        }
    }
}

void DeterministicExecutor::postProcess(const GenerateStreamPtr&            stream,
                                        SpeculativeExecutorStreamOutputPtr& stream_output) {
    RTP_LLM_LOG_DEBUG("RuleBasedTokenSelector select tokens %d num",
                      stream_output->tokens == nullptr ? 0 : stream_output->tokens->size());
    if (!stream_output->tokens) {
        return;
    }

    size_t propose_step         = stream_output->tokens->size();
    stream_output->propose_step = propose_step;
    size_t vocab_size           = stream->vocabSize();
    auto   all_probs            = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_FP32, {propose_step, vocab_size}, rtp_llm::AllocationType::HOST}, {""});
    device_->bufMemset(*all_probs, 0);
    for (size_t i = 0; i < propose_step; i++) {
        *(all_probs->view(i, 1).dataWithOffset<float>(*stream_output->tokens->dataWithOffset<int32_t>(i))) = 1.0;
    }
    stream_output->all_probs = device_->clone({*all_probs, rtp_llm::AllocationType::DEVICE, {"all_probs"}});
}

};  // namespace rtp_llm
