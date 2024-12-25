#include <memory>
#include <vector>

#include "maga_transformer/cpp/speculative_engine/propose_executor/DeterministicExecutor.h"
#include "ProposeOutput.h"

namespace rtp_llm {

absl::StatusOr<ProposeOutput> DeterministicExecutor::propose(const std::list<GenerateStreamPtr>& streams) {
    ProposeOutput output(streams.size());
    size_t        stream_index = 0;
    for (auto& stream : streams) {
        ruleBasedTokenSelector(stream, output.outputs[stream_index]);
        stream_index++;
    }
    return output;
}

void DeterministicExecutor::ruleBasedTokenSelector(const GenerateStreamPtr&            stream,
                                                   SpeculativeExecutorStreamOutputPtr& stream_output) {
    auto& config = stream->generateConfig();
    if (config->sp_input_lookup || config->sp_advice_prompt_token_ids.size() > 0) {
        if (config->sp_edit) {
            SpEditTokenSelector(stream, stream_output);
        } else {
            PromptLookUpTokenSelector(stream, stream_output);
        }
    }
}

void DeterministicExecutor::SpEditTokenSelector(const GenerateStreamPtr&            stream,
                                                SpeculativeExecutorStreamOutputPtr& stream_output) {
    auto&        config                  = stream->generateConfig();
    size_t advice_token_num = 0;
    int* advice_token_ids = nullptr;
    if (config->sp_input_lookup && config->sp_advice_prompt_token_ids.size() == 0) {
        advice_token_num = stream->seqLength();
        advice_token_ids = stream->completeTokenIds()->data<int>();
    } else {
        advice_token_num = config->sp_advice_prompt_token_ids.size();
        advice_token_ids = config->sp_advice_prompt_token_ids.data(); 
    }
    const auto&  sp_edit_search_index    = stream->spEditSearchIndex();
    const size_t output_token_len        = stream->outputTokenLen();
    size_t max_new_tokens = std::min((size_t)config->max_new_tokens, stream->maxSeqLen() - stream->seqLength() - 1);
    if (stream->spEditFirstTime()) {
        size_t propose_len    = std::min(max_new_tokens,
                                      std::min(propose_step_, advice_token_num - sp_edit_search_index));
        stream_output->tokens = std::make_shared<fastertransformer::Buffer>(fastertransformer::MemoryType::MEMORY_CPU,
                                                                            ft::DataType::TYPE_INT32,
                                                                            std::vector{1, propose_len},
                                                                            advice_token_ids);
        stream->setSpEditFirstTime(false);
        postProcess(stream, stream_output);
    } else if (min_token_match_len_ <= output_token_len) {
        size_t begin_match_len = std::min(max_token_match_len_, output_token_len);

        std::vector<int> max_latest_tokens = stream->getLatestTokens(begin_match_len);
        for (size_t match_len = begin_match_len; match_len >= min_token_match_len_; match_len--) {

            std::vector<int> latest_tokens(max_latest_tokens.end() - match_len, max_latest_tokens.end());
            for (size_t i = sp_edit_search_index - match_len; i + match_len < advice_token_num - 1; i++) {
                size_t j = 0;
                for (; j < match_len; j++) {
                    if (latest_tokens[j] != advice_token_ids[i + j]) {
                        break;
                    }
                }
                if (j == match_len) {
                    size_t start_propose_index = i + match_len;
                    size_t propose_len =
                        std::min(max_new_tokens,
                                 std::min(propose_step_, advice_token_num - start_propose_index));
                    stream_output->tokens = std::make_shared<fastertransformer::Buffer>(
                        fastertransformer::MemoryType::MEMORY_CPU,
                        ft::DataType::TYPE_INT32,
                        std::vector{1, propose_len},
                        advice_token_ids + start_propose_index);
                    postProcess(stream, stream_output);
                    return;
                }
            }
        }
    }
}

void DeterministicExecutor::PromptLookUpTokenSelector(const GenerateStreamPtr&            stream,
                                                      SpeculativeExecutorStreamOutputPtr& stream_output) {
    auto& config                  = stream->generateConfig();
    size_t advice_token_num = 0;
    int* advice_token_ids = nullptr;
    if (config->sp_input_lookup && config->sp_advice_prompt_token_ids.size() == 0) {
        advice_token_num = stream->seqLength();
        advice_token_ids = stream->completeTokenIds()->data<int>();
    } else {
        advice_token_num = config->sp_advice_prompt_token_ids.size();
        advice_token_ids = config->sp_advice_prompt_token_ids.data(); 
    }

    const size_t seq_len = stream->seqLength();
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
                        std::min(max_new_tokens,
                                 std::min(propose_step_, advice_token_num - start_propose_index));
                    stream->setSpEditSearchIndex(start_propose_index);
                    stream_output->tokens = std::make_shared<fastertransformer::Buffer>(
                        fastertransformer::MemoryType::MEMORY_CPU,
                        ft::DataType::TYPE_INT32,
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
    FT_LOG_DEBUG("RuleBasedTokenSelector select tokens %d num",
                 stream_output->tokens == nullptr ? 0 : stream_output->tokens->size());
    if (!stream_output->tokens) {
        return;
    }

    auto&  config               = stream->generateConfig();
    size_t propose_step         = stream_output->tokens->size();
    stream_output->propose_step = propose_step;

    if (!config->top1()) {
        const auto& all_probs = device_->allocateBuffer(
            {ft::DataType::TYPE_FP32, {propose_step, (size_t)stream->vocabSize()}, ft::AllocationType::HOST}, {""});
        device_->bufMemset(*all_probs, 0);
        for (size_t i = 0; i < propose_step; i++) {
            *all_probs->view(i, 0).dataWithOffset<float>(stream_output->tokens->data<int32_t>()[i]) = 1.0;
        }

        stream_output->all_probs = device_->clone({*all_probs, AllocationType::DEVICE, {"determinisitic_all_probs"}});
    }
}

};  // namespace rtp_llm
