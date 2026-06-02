#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"

#include <limits>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

RecommendationLogitsProcessor::RecommendationLogitsProcessor(std::vector<StreamRecommendationInfo> infos):
    infos_(std::move(infos)) {}

std::shared_ptr<RecommendationLogitsProcessor>
RecommendationLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input, int32_t num) {
    const auto& config = generate_input->generate_config;
    if (config->combo_token_size <= 0) {
        return nullptr;
    }

    // 过滤掉与 combo_token_size 不一致的 banned combo(保持鲁棒性)
    std::set<std::vector<int>> banned_combos;
    for (const auto& combo : config->banned_combo_token_ids) {
        if ((int32_t)combo.size() == config->combo_token_size) {
            banned_combos.insert(combo);
        }
    }

    const bool is_beam_search = config->hasNumBeams() || config->num_return_sequences > 1;
    // 若为空,think_done 初始为 true,Processor 行为等同历史版本(从首个 token 起累 combo)。
    const std::vector<int>& end_think_token_ids = config->end_think_token_ids;

    auto processor_ptr = std::make_shared<RecommendationLogitsProcessor>();
    for (int32_t i = 0; i < num; ++i) {
        StreamRecommendationInfo info(config->combo_token_size,
                                      generate_input->inputLength(),
                                      /*current_output_length=*/0,
                                      is_beam_search,
                                      banned_combos,
                                      end_think_token_ids);
        processor_ptr->infos_.push_back(std::move(info));
    }
    return processor_ptr;
}

void RecommendationLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    const size_t batch_size = finish_idx - start_idx;
    RTP_LLM_CHECK(batch_size == size());

    // 仅在位于 combo 最后一位时需要施加掩码
    bool need_process = false;
    for (const auto& info : infos_) {
        if (info.combo_token_size <= 0) {
            continue;
        }
        if (info.pos_in_combo == info.combo_token_size - 1 && !info.banned_combos.empty()) {
            need_process = true;
            break;
        }
    }
    if (!need_process) {
        return;
    }

    auto         logits     = inputs.logits.narrow(0, start_idx, batch_size);
    const size_t vocab_size = logits.size(1);

    // 只收集要屏蔽的 (row, col) 坐标,避开按 batch*vocab 分配 mask 张量与 H2D 拷贝的热路径开销。
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    for (size_t i = 0; i < batch_size; ++i) {
        auto& info = infos_[i];
        if (info.combo_token_size <= 0 || info.pos_in_combo != info.combo_token_size - 1) {
            continue;
        }
        const int combo_last_idx = info.combo_token_size - 1;
        // 对所有前 n-1 个 token 等于 current_prefix 的 banned combo,屏蔽其最后一位
        for (const auto& combo : info.banned_combos) {
            RTP_LLM_CHECK((int32_t)combo.size() == info.combo_token_size);
            bool prefix_match = true;
            for (int32_t k = 0; k < combo_last_idx; ++k) {
                if (combo[k] != info.current_prefix[k]) {
                    prefix_match = false;
                    break;
                }
            }
            if (!prefix_match) {
                continue;
            }
            const int banned_token = combo[combo_last_idx];
            if (banned_token >= 0 && (size_t)banned_token < vocab_size) {
                rows.push_back(static_cast<int64_t>(i));
                cols.push_back(static_cast<int64_t>(banned_token));
            }
        }
    }

    if (rows.empty()) {
        return;
    }

    // 直接在 logits 上按坐标批量写 -inf,按命中数量分摊 H2D 拷贝,体积 O(K) 远小于 O(B*V)。
    auto rows_t = torch::tensor(rows, torch::kLong).to(logits.device());
    auto cols_t = torch::tensor(cols, torch::kLong).to(logits.device());
    logits.index_put_({rows_t, cols_t}, -std::numeric_limits<float>::infinity());
}

void RecommendationLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamRecommendationInfo> new_infos;
    new_infos.reserve(src_batch_indices.size());
    for (const auto src_idx : src_batch_indices) {
        RTP_LLM_CHECK((size_t)src_idx < infos_.size());
        new_infos.push_back(infos_[src_idx].copy());
    }
    infos_ = std::move(new_infos);
}

void RecommendationLogitsProcessor::advanceOneToken(StreamRecommendationInfo& info, int token_id) {
    if (info.combo_token_size <= 0) {
        return;
    }

    // 未完成 think prelude 跳过时,本 token 只用于推进 DFA,不进 combo 前缀。
    // 遇到不匹配立即复位 match_pos,允许下一个 token 重新起算。
    if (!info.think_done) {
        if (info.end_think_match_pos < info.end_think_token_ids.size()
            && token_id == info.end_think_token_ids[info.end_think_match_pos]) {
            info.end_think_match_pos += 1;
            if (info.end_think_match_pos >= info.end_think_token_ids.size()) {
                info.think_done = true;
            }
        } else {
            info.end_think_match_pos = 0;
        }
        return;
    }

    if (info.pos_in_combo < info.combo_token_size - 1) {
        // combo 未结束:追加到前缀
        info.current_prefix.push_back(token_id);
        info.pos_in_combo += 1;
    } else {
        // 本次 token 即 combo 的最后一位:形成完整 combo 并加入去重集合
        std::vector<int> full_combo = info.current_prefix;
        full_combo.push_back(token_id);
        info.banned_combos.insert(std::move(full_combo));
        info.current_prefix.clear();
        info.pos_in_combo = 0;
    }
}

void RecommendationLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    // 明确 sampler 输出契约:仅接受 int32,避免 dtype 变动后 data_ptr<int>() 静默读错字节。
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < size(); ++i) {
        auto& info = infos_[i];
        if (info.combo_token_size <= 0) {
            continue;
        }

        const int64_t offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        const int64_t stride = new_tokens.size(1);
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            const int token_id = new_tokens.data_ptr<int>()[i * stride + j + offset];
            advanceOneToken(info, token_id);
        }

        info.current_output_length += num_new_tokens;
    }
}

}  // namespace rtp_llm
