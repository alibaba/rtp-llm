#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// 遮蔽深度上界：防止 num_return_sequences 过大时采样退化为随机噪声。
// 实际场景推荐 N=2~4，设为 8 提供安全余量。
static constexpr int kMaxDivergeDepth = 8;
// SYNC: 若修改此值，必须同步更新 Python generate_config.py::_MAX_DIVERGE_DEPTH
// 以及 test_generate_config_validators.py::test_max_diverge_depth_sync 中的硬编码期望值。
static_assert(kMaxDivergeDepth == 8,
    "SYNC: update Python _MAX_DIVERGE_DEPTH and test_max_diverge_depth_sync expected value");

// cross_seq_diverge_start_combo "过大" 告警阈值，Python 侧 generate_config.py 使用相同值。
static constexpr int kDivergeStartComboWarnThreshold = 100;
// SYNC: 若修改此值，必须同步更新 Python generate_config.py::_DIVERGE_START_COMBO_WARN_THRESHOLD
// 以及 test_generate_config_validators.py::test_diverge_start_combo_warn_threshold_sync 中的硬编码期望值。
static_assert(kDivergeStartComboWarnThreshold == 100,
    "SYNC: update Python _DIVERGE_START_COMBO_WARN_THRESHOLD and test expected value");

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

    const bool needs_token_offset = config->hasNumBeams() || config->num_return_sequences > 1;
    // beam search 与跨序列去重互斥：updateMultiSeqStatus 会重排序列身份，破坏主序列不变量
    // combo_token_size == 1 时 diverge 与 ban 在同一步叠加，易导致采样退化，仅 combo_token_size >= 2 时启用
    // C++ 自校验：num<=1 时 diverge 逻辑天然 no-op（i>0 分支不可达），但仍显式禁用以避免布尔开关处于「已置位但部分配置」状态
    // SYNC: 以下判定条件必须与 Python generate_config.py::_check_cross_seq_ban_compatibility
    // 中的判定逻辑保持一致（取反关系）。
    // ━━ 新增/修改启用条件时的 CHECKLIST ━━
    //   1. 同步修改另一侧的判定逻辑
    //   2. 更新双侧真值表测试：
    //      Python: TestCrossLanguageConstantSync::test_enable_conditions_sync
    //      C++:    RecommendationLogitsProcessorTest::testEnableConditionsTruthTable
    //   3. 确认新条件在双侧真值表中均有覆盖（正反例）
    //   未来演进：若条件进一步增多，应将真值表落为共享 JSON 数据文件（单一真源）。
    const bool enable_cross_seq_ban = config->enable_cross_sequence_ban
                                      && !config->hasNumBeams()
                                      && config->combo_token_size >= 2
                                      && num > 1;
    // 可观测性：当用户显式开启但被降级禁用时，一次性输出所有不兼容原因
    // 使用 INTERVAL_LOG 避免高 QPS 下同一配置重复告警形成日志风暴
    if (config->enable_cross_sequence_ban && !enable_cross_seq_ban) {
        std::string reasons;
        if (config->hasNumBeams()) {
            reasons += "incompatible with beam search; ";
        }
        if (config->combo_token_size < 2) {
            reasons += "combo_token_size must be >= 2 (got " + std::to_string(config->combo_token_size) + "); ";
        }
        if (num <= 1) {
            reasons += "num_return_sequences must be > 1 (got " + std::to_string(num) + "); ";
        }
        RTP_LLM_INTERVAL_LOG(300, WARN, "cross_sequence_ban disabled: %s", reasons.c_str());
    }
    // 遮蔽深度上界校验：最大遮蔽 k = num-1，若超过 kMaxDivergeDepth 则警告采样可能退化
    if (enable_cross_seq_ban && num - 1 > kMaxDivergeDepth) {
        RTP_LLM_INTERVAL_LOG(300, WARN,
            "cross_sequence_ban: num_return_sequences=%d exceeds recommended max diverge depth %d, "
            "sampling quality may degrade for higher-indexed sequences",
            num, kMaxDivergeDepth);
    }
    const int32_t diverge_start_combo = std::max(0, config->cross_seq_diverge_start_combo);
    if (config->cross_seq_diverge_start_combo < 0) {
        RTP_LLM_INTERVAL_LOG(300, WARN, "cross_seq_diverge_start_combo is negative (%d), clamped to 0",
                            config->cross_seq_diverge_start_combo);
    } else if (enable_cross_seq_ban && diverge_start_combo > kDivergeStartComboWarnThreshold) {
        RTP_LLM_INTERVAL_LOG(300, WARN,
            "cross_seq_diverge_start_combo=%d is very large, top-K diverge masking may never activate",
            diverge_start_combo);
    }
    // 若为空,think_done 初始为 true,Processor 行为等同历史版本(从首个 token 起累 combo)。
    const std::vector<int>& end_think_token_ids = config->end_think_token_ids;

    auto processor_ptr = std::make_shared<RecommendationLogitsProcessor>();
    for (int32_t i = 0; i < num; ++i) {
        StreamRecommendationInfo info(config->combo_token_size,
                                      generate_input->inputLength(),
                                      /*current_output_length=*/0,
                                      needs_token_offset,
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

    // 位置不变量（primary-index-0）：
    // cross-seq ban 的 primary-protected 语义依赖 infos_[0] 恒为主序列、且 size() 在整个
    // decode 生命周期内不变（无 stream 内子序列早停压缩或重排）。
    // 保证条件：updateMultiSeqStatus 已断言与 cross-seq ban 互斥，此处 batch_size==size()
    // 确认无压缩。若未来引入 intra-stream 子序列管理，需增加显式重排检测。

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
        // top-K 分叉：非主序列(i>0) + think完成 + 在 combo 起始位置 + 已达到分叉起始商品 + 开关开启
        if (i > 0 && info.enable_cross_sequence_ban && info.think_done
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
    RTP_LLM_CHECK_WITH_INFO(vocab_size > 0, "process called with vocab_size=0");

    // --- top-K 分叉遮蔽：对非主序列在 combo 起始位置遮蔽前 i 个最大 logit ---
    // 设计权衡（deliberate trade-off）：达到 diverge_start_combo 后，每个 combo 起点无条件遮蔽，
    // 不判断序列是否已与主序列分叉。原因：
    //   1) "已分叉" 难以定义 —— temperature/top_p 采样下 logits 不同不等于输出不同，
    //      反之 logits 相同也可能采出不同 token，判断开销高且不可靠；
    //   2) 遮蔽仅影响 combo 第一位（占生成 token 的 1/combo_token_size），且只剥夺 top-i
    //      （i 为行号，通常 1~3），对整体质量影响有限；
    //   3) 三层保护已约束风险：kMaxDivergeDepth 上界、diverge_start_combo 延迟启动、默认关闭。
    // 若未来观察到高索引序列质量明显下降，可引入基于 banned_combos 差异度的自适应退出。
    //
    // 可观测性演进方向（生产侧质量验证）：
    //   - 每序列平均被遮蔽 token 数（per-step masked_count / active_steps）
    //   - 分叉命中率（diverge 遮蔽后实际采样到非 top-i 的比例）
    //   - 高索引序列 combo 完成率 vs 主序列 combo 完成率
    //   上述指标可通过 kmonitor counter 暴露，为自适应退出策略提供数据支撑。
    //
    // 批量化设计决策：对所有非主行做一次 batch topk(max_k)，而非逐行筛选后再 gather/topk/scatter。
    // 原因：1) 单次 kernel launch 比多次显著更快；2) N=2-4 时几乎所有非主行都符合条件，
    // 不符合条件的行其 topk 结果不会被使用（下方 for 循环中 skip），无副作用。
    // trade-off 记录：对不需要 diverge 的行仍做了 topk，但避免了更复杂的条件筛选 + scatter 逻辑，
    // 在 N<=8 的场景下，多余 topk 计算的开销远小于额外 kernel launch 的开销。
    if (need_diverge_process) {
        int max_k = 0;
        for (size_t i = 1; i < batch_size; ++i) {
            auto& info = infos_[i];
            if (info.combo_token_size <= 0 || !info.enable_cross_sequence_ban || !info.think_done) continue;
            if (info.pos_in_combo != 0
                || info.completed_combo_count < info.cross_seq_diverge_start_combo) {
                continue;
            }
            int k = std::min({static_cast<int>(i), static_cast<int>(vocab_size) - 1, kMaxDivergeDepth});
            if (k > max_k) max_k = k;
        }
        if (max_k > 0) {
            // 防御：确保 topk 的 k 不超过 vocab_size（torch::topk 要求 k <= dim_size）
            max_k = std::min(max_k, static_cast<int>(vocab_size));
            // 仅对非主序列(row 1~N-1)做 topk，避免对 row 0 的无效计算
            auto non_primary_logits = logits.narrow(0, 1, batch_size - 1);
            auto topk_indices = std::get<1>(non_primary_logits.topk(max_k, /*dim=*/1));
            for (size_t i = 1; i < batch_size; ++i) {
                auto& info = infos_[i];
                if (info.combo_token_size <= 0 || !info.enable_cross_sequence_ban || !info.think_done) continue;
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

    // --- banned combo 屏蔽 ---
    if (need_ban_process) {
        // 只收集要屏蔽的 (row, col) 坐标,避开按 batch*vocab 分配 mask 张量与 H2D 拷贝的热路径开销。
        std::vector<int64_t> rows;
        std::vector<int64_t> cols;
        for (size_t i = 0; i < batch_size; ++i) {
            auto& info = infos_[i];
            if (info.combo_token_size <= 0 || info.pos_in_combo != info.combo_token_size - 1) {
                continue;
            }
            const int combo_last_idx = info.combo_token_size - 1;
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
        if (!rows.empty()) {
            auto rows_t = torch::tensor(rows, torch::kLong).to(logits.device());
            auto cols_t = torch::tensor(cols, torch::kLong).to(logits.device());
            logits.index_put_({rows_t, cols_t}, -std::numeric_limits<float>::infinity());
        }
    }
}

void RecommendationLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    RTP_LLM_CHECK_WITH_INFO(!infos_.empty(),
        "updateMultiSeqStatus called on empty processor");
    // 业务不变量：updateMultiSeqStatus 用于 beam search 重排序列，与 cross-sequence ban 互斥
    RTP_LLM_CHECK_WITH_INFO(
        std::none_of(infos_.begin(), infos_.end(),
                     [](const StreamRecommendationInfo& info) { return info.enable_cross_sequence_ban; }),
        "updateMultiSeqStatus must not be called when cross_sequence_ban is enabled");
    std::vector<StreamRecommendationInfo> new_infos;
    new_infos.reserve(src_batch_indices.size());
    for (const auto src_idx : src_batch_indices) {
        RTP_LLM_CHECK((size_t)src_idx < infos_.size());
        new_infos.push_back(infos_[src_idx]);
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
    // 位置不变量（primary-index-0）：广播逻辑 i>=1 接收、i==0 保护，依赖 infos_ 顺序
    // 在 decode 生命周期内不被重排。此不变量由 size()==new_tokens.size(0)(无压缩)
    // + updateMultiSeqStatus 与 cross-seq ban 互斥（无重排）共同保证。

    bool any_combo_completed = false;
    // 仅在跨序列 ban 开启时分配 combo 收集容器，避免未启用功能时的无效内存分配
    const bool need_broadcast = size() > 1 && infos_[0].enable_cross_sequence_ban;
    // 强不变量断言：同 Processor 内所有 infos 的 enable_cross_sequence_ban 必须一致
    // 仅在跨序列 ban 开启时执行 O(N) 扫描，避免未启用时在采样热路径上引入无谓开销
    if (need_broadcast) {
        RTP_LLM_CHECK_WITH_INFO(
            std::all_of(infos_.begin(), infos_.end(),
                        [&](const StreamRecommendationInfo& info) {
                            return info.enable_cross_sequence_ban == infos_[0].enable_cross_sequence_ban;
                        }),
            "updateStatus: enable_cross_sequence_ban flag inconsistent across infos");
    }
    std::vector<std::vector<std::vector<int>>> new_combos_per_seq;
    if (need_broadcast) {
        new_combos_per_seq.resize(size());
    }
    for (size_t i = 0; i < size(); ++i) {
        auto& info = infos_[i];
        if (info.combo_token_size <= 0) {
            continue;
        }

        const int64_t offset = info.needs_token_offset ? (info.current_output_length + info.input_length) : 0;

        if (!info.needs_token_offset) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        const int64_t stride = new_tokens.size(1);
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            const int token_id = new_tokens.data_ptr<int>()[i * stride + j + offset];
            if (advanceOneToken(info, token_id, need_broadcast ? &new_combos_per_seq[i] : nullptr)) {
                any_combo_completed = true;
            }
        }

        info.current_output_length += num_new_tokens;
    }

    // 跨序列增量广播（非对称模式）：将其他序列本步新完成的 combo 插入非主序列
    // 设计意图（primary-protected）：序列 0（主序列）仅保留自身产生的 banned_combos，不接收其他
    // 序列的 ban。补充序列接收所有其他序列的新增 combo，确保彼此不重复。
    // 复杂度：O((N-1) * N * new_combos_per_step)，生产场景 N=2~4、每步最多 1 个 combo 完成，
    // 实际开销可忽略。若未来 N 增大，可将序列 0 的 combo 批量插入后再处理交叉插入，或改用 shared 视图。
    if (any_combo_completed && need_broadcast) {
        for (size_t i = 1; i < size(); ++i) {
            for (size_t j = 0; j < size(); ++j) {
                if (j == i) continue;
                for (const auto& combo : new_combos_per_seq[j]) {
                    infos_[i].banned_combos.insert(combo);
                }
            }
        }
    }
}

}  // namespace rtp_llm
