#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// 遮蔽深度上界：防止 num_return_sequences 过大时采样退化为随机噪声。
// 实际场景推荐 N=2~4，设为 8 提供安全余量。
static constexpr int kMaxDivergeDepth = 8;

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
    // beam search 与跨序列去重互斥：updateMultiSeqStatus 会重排序列身份，破坏主序列不变量
    // combo_token_size == 1 时 diverge 与 ban 在同一步叠加，易导致采样退化，仅 combo_token_size >= 2 时启用
    const bool enable_cross_seq_ban = config->enable_cross_sequence_ban
                                      && !config->hasNumBeams()
                                      && config->combo_token_size >= 2;
    // 可观测性：当用户显式开启但被降级禁用时，输出 warning 帮助定位配置问题
    if (config->enable_cross_sequence_ban && !enable_cross_seq_ban) {
        if (config->hasNumBeams()) {
            RTP_LLM_LOG_WARNING("cross_sequence_ban disabled: incompatible with beam search");
        } else if (config->combo_token_size < 2) {
            RTP_LLM_LOG_WARNING("cross_sequence_ban disabled: combo_token_size must be >= 2, got %d",
                                config->combo_token_size);
        }
    }
    // 遮蔽深度上界校验：最大遮蔽 k = num-1，若超过 kMaxDivergeDepth 则警告采样可能退化
    if (enable_cross_seq_ban && num - 1 > kMaxDivergeDepth) {
        RTP_LLM_LOG_WARNING(
            "cross_sequence_ban: num_return_sequences=%d exceeds recommended max diverge depth %d, "
            "sampling quality may degrade for higher-indexed sequences",
            num, kMaxDivergeDepth);
    }
    const int32_t diverge_start_combo = std::max(0, config->cross_seq_diverge_start_combo);
    if (config->cross_seq_diverge_start_combo < 0) {
        RTP_LLM_LOG_WARNING("cross_seq_diverge_start_combo is negative (%d), clamped to 0",
                            config->cross_seq_diverge_start_combo);
    }
    // 若为空,think_done 初始为 true,Processor 行为等同历史版本(从首个 token 起累 combo)。
    const std::vector<int>& end_think_token_ids = config->end_think_token_ids;

    auto processor_ptr = std::make_shared<RecommendationLogitsProcessor>();
    for (int32_t i = 0; i < num; ++i) {
        StreamRecommendationInfo info(config->combo_token_size,
                                      generate_input->inputLength(),
                                      /*current_output_length=*/0,
                                      is_beam_search,
                                      banned_combos,
                                      end_think_token_ids,
                                      enable_cross_seq_ban,
                                      diverge_start_combo);
        processor_ptr->infos_.push_back(std::move(info));
    }
    return processor_ptr;
}

void RecommendationLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    const size_t batch_size = finish_idx - start_idx;
    RTP_LLM_CHECK(batch_size == size());

    // 判断是否需要进行 banned combo 屏蔽或 top-K 分叉遮蔽
    bool need_ban_process = false;
    bool need_diverge_process = false;
    for (size_t i = 0; i < batch_size; ++i) {
        const auto& info = infos_[i];
        if (info.combo_token_size <= 0) {
            continue;
        }
        if (info.pos_in_combo == info.combo_token_size - 1 && !info.banned_combos.empty()) {
            need_ban_process = true;
        }
        // top-K 分叉：非主序列(i>0) + 在 combo 起始位置 + 已达到分叉起始商品 + 开关开启
        if (i > 0 && info.enable_cross_sequence_ban
            && info.pos_in_combo == 0
            && info.completed_combo_count >= info.cross_seq_diverge_start_combo) {
            need_diverge_process = true;
        }
    }
    if (!need_ban_process && !need_diverge_process) {
        return;
    }

    auto         logits     = inputs.logits.narrow(0, start_idx, batch_size);
    const size_t vocab_size = logits.size(1);

    // --- top-K 分叉遮蔽：对非主序列在 combo 起始位置遮蔽前 i 个最大 logit ---
    // 批量化设计决策：对所有非主行做一次 batch topk(max_k)，而非逐行筛选后再 gather/topk/scatter。
    // 原因：1) 单次 kernel launch 比多次显著更快；2) N=2-4 时几乎所有非主行都符合条件，
    // 不符合条件的行其 topk 结果不会被使用（下方 for 循环中 skip），无副作用。
    if (need_diverge_process) {
        int max_k = 0;
        for (size_t i = 1; i < batch_size; ++i) {
            auto& info = infos_[i];
            if (info.combo_token_size <= 0 || !info.enable_cross_sequence_ban) continue;
            if (info.pos_in_combo != 0
                || info.completed_combo_count < info.cross_seq_diverge_start_combo) {
                continue;
            }
            int k = std::min({static_cast<int>(i), static_cast<int>(vocab_size) - 1, kMaxDivergeDepth});
            if (k > max_k) max_k = k;
        }
        if (max_k > 0) {
            // 仅对非主序列(row 1~N-1)做 topk，避免对 row 0 的无效计算
            auto non_primary_logits = logits.narrow(0, 1, batch_size - 1);
            auto topk_indices = non_primary_logits.topk(max_k, /*dim=*/1).indices();
            for (size_t i = 1; i < batch_size; ++i) {
                auto& info = infos_[i];
                if (info.combo_token_size <= 0 || !info.enable_cross_sequence_ban) continue;
                if (info.pos_in_combo != 0
                    || info.completed_combo_count < info.cross_seq_diverge_start_combo) {
                    continue;
                }
                // 防御：确保至少保留 1 个可选 token，同时不超过 kMaxDivergeDepth 避免采样退化
                const int k = std::min({static_cast<int>(i), static_cast<int>(vocab_size) - 1, kMaxDivergeDepth});
                if (k <= 0) continue;
                // 从预计算的 batch topk indices 中裁剪前 k 个位置进行遮蔽
                // topk_indices 行号 = i-1（因为 narrow 排除了 row 0）
                logits[i].index_put_({topk_indices[i - 1].slice(0, 0, k)},
                                     -std::numeric_limits<float>::infinity());
            }
        }
    }

    // --- banned combo 屏蔽（原有逻辑不变） ---
    if (!need_ban_process) {
        return;
    }

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

    // 安全检查：diverge 遮蔽 + banned combo 遮蔽叠加后，确保每行至少保留 1 个有效 token。
    // 若某行全部被 mask，恢复该行为均匀分布（安全降级），避免 sampler 采样 NaN/随机 token。
    for (size_t i = 0; i < batch_size; ++i) {
        auto row = logits[i];
        bool all_masked = (row.max().item<float>() == -std::numeric_limits<float>::infinity());
        if (all_masked) {
            row.fill_(0.0f);  // 均匀分布降级
            RTP_LLM_LOG_WARNING(
                "RecommendationLogitsProcessor: row %zu all logits masked after diverge+ban, "
                "falling back to uniform distribution", i);
        }
    }
}

void RecommendationLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    // 安全前置：避免空 infos_ 时访问 infos_[0] 导致未定义行为
    RTP_LLM_CHECK_WITH_INFO(!infos_.empty(),
        "updateMultiSeqStatus called on empty processor");
    // 业务不变量：updateMultiSeqStatus 用于 beam search 重排序列，与 cross-sequence ban 互斥
    RTP_LLM_CHECK_WITH_INFO(!infos_[0].enable_cross_sequence_ban,
        "updateMultiSeqStatus must not be called when cross_sequence_ban is enabled");
    std::vector<StreamRecommendationInfo> new_infos;
    new_infos.reserve(src_batch_indices.size());
    for (const auto src_idx : src_batch_indices) {
        RTP_LLM_CHECK((size_t)src_idx < infos_.size());
        new_infos.push_back(infos_[src_idx].copy());
    }
    infos_ = std::move(new_infos);
}

bool RecommendationLogitsProcessor::advanceOneToken(StreamRecommendationInfo& info, int token_id,
                                                    std::vector<std::vector<int>>* new_combos) {
    if (info.combo_token_size <= 0) {
        return false;
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
        return false;
    }

    if (info.pos_in_combo < info.combo_token_size - 1) {
        // combo 未结束:追加到前缀
        info.current_prefix.push_back(token_id);
        info.pos_in_combo += 1;
        return false;
    } else {
        // 本次 token 即 combo 的最后一位:形成完整 combo 并加入去重集合
        std::vector<int> full_combo = info.current_prefix;
        full_combo.push_back(token_id);
        if (new_combos) new_combos->push_back(full_combo);
        info.banned_combos.insert(std::move(full_combo));
        info.current_prefix.clear();
        info.pos_in_combo = 0;
        info.completed_combo_count += 1;
        return true;
    }
}

void RecommendationLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    // 明确 sampler 输出契约:仅接受 int32,避免 dtype 变动后 data_ptr<int>() 静默读错字节。
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    bool any_combo_completed = false;
    // 按序列分组收集新完成的 combo，广播时仅插入其他序列的 combo，避免自身重复插入
    std::vector<std::vector<std::vector<int>>> new_combos_per_seq(size());
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
            if (advanceOneToken(info, token_id, &new_combos_per_seq[i])) {
                any_combo_completed = true;
            }
        }

        info.current_output_length += num_new_tokens;
    }

    // 跨序列增量广播（非对称模式）：仅将其他序列本步新完成的 combo 插入非主序列，跳过自身已有的
    // 设计意图（primary-protected）：序列 0（主序列）仅保留自身产生的 banned_combos，不接收其他
    // 序列的 ban。补充序列接收其他序列的新增 combo，确保彼此不重复。
    if (any_combo_completed && size() > 1 && infos_[0].enable_cross_sequence_ban) {
        for (size_t i = 1; i < size(); ++i) {
            for (size_t j = 0; j < size(); ++j) {
                if (j == i) continue;  // 跳过自身，避免冗余 set::insert
                for (const auto& combo : new_combos_per_seq[j]) {
                    infos_[i].banned_combos.insert(combo);
                }
            }
        }
    }
}

}  // namespace rtp_llm
