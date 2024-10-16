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
    if (config->sp_advice_prompt_token_ids.size() > 0) {
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
    const auto&  advice_prompt_token_ids = config->sp_advice_prompt_token_ids;
    const auto&  sp_edit_search_index    = stream->spEditSearchIndex();
    const size_t output_token_len        = stream->outputTokenLen();
    size_t       max_match_len =
        std::min((size_t)config->max_new_tokens,
                 std::min(max_str_match_len_, advice_prompt_token_ids.size() - sp_edit_search_index));

    FT_LOG_INFO(
        "SpEditTokenSelector max_match_len %d, min_str_match_len_ %d, output_token_len %d, sp_edit_search_index %d",
        max_match_len,
        min_str_match_len_,
        output_token_len,
        sp_edit_search_index);
    if (stream->spEditFirstTime()) {
        stream_output->tokens = std::make_shared<fastertransformer::Buffer>(fastertransformer::MemoryType::MEMORY_CPU,
                                                                            ft::DataType::TYPE_INT32,
                                                                            std::vector{1, max_match_len},
                                                                            advice_prompt_token_ids.data());
        stream->setSpEditFirstTime(false);
    } else if (max_match_len >= min_str_match_len_ && min_str_match_len_ <= output_token_len) {

        std::vector<int> latest_tokens = stream->getLatestTokens(min_str_match_len_);
        for (size_t i = 0; i < latest_tokens.size(); i++) {
            std::cout << latest_tokens[i] << " ";
        }
        std::cout << std::endl;

        for (size_t i = sp_edit_search_index; i + min_str_match_len_ < advice_prompt_token_ids.size(); i++) {
            size_t j = 0;
            for (; j < min_str_match_len_; j++) {
                if (latest_tokens[j] != advice_prompt_token_ids[i + j]) {
                    break;
                }
            }
            if (j == min_str_match_len_) {
                size_t start_propose_index = i + min_str_match_len_;
                stream->setSpEditSearchIndex(start_propose_index);
                stream_output->tokens = std::make_shared<fastertransformer::Buffer>(
                    fastertransformer::MemoryType::MEMORY_CPU,
                    ft::DataType::TYPE_INT32,
                    std::vector{1, max_match_len - (start_propose_index - sp_edit_search_index)},
                    advice_prompt_token_ids.data() + start_propose_index);
                break;
            }
        }
    }

    postProcess(stream, stream_output);
}

void DeterministicExecutor::PromptLookUpTokenSelector(const GenerateStreamPtr&            stream,
                                                      SpeculativeExecutorStreamOutputPtr& stream_output) {
    // TODO(xyz): implement prompt lookup
    return;
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

        for (size_t i = 0; i < propose_step; i++) {
            *all_probs->view(0, i).dataWithOffset<float>(stream_output->tokens->data<int32_t>()[i]) = 1.0;
        }

        stream_output->all_probs = device_->clone({*all_probs, AllocationType::DEVICE, {"determinisitic_all_probs"}});
    }
}

};  // namespace rtp_llm
