#pragma once

#include <set>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

// 生成式推荐场景：每个商品由 combo_token_size 个连续 token 组成。
// 本 Processor 在每次生成 combo 的「最后一位」时施加掩码，实现两类约束：
//   1) 曝光过滤：用户通过 generate_config.banned_combo_token_ids 传入的禁止 combo。
//   2) 生成去重：每当模型输出一个完整 combo，自动将其加入 banned_combos，
//      从而避免后续再次生成相同商品。
struct StreamRecommendationInfo {
    int32_t combo_token_size      = 0;
    int32_t input_length          = 0;
    int32_t current_output_length = 0;
    bool    is_beam_search        = false;

    // 当前正在生成 combo 内的位置，取值 [0, combo_token_size-1]
    int32_t pos_in_combo = 0;
    // 当前 combo 已生成的前缀（长度 = pos_in_combo）
    std::vector<int> current_prefix;
    // 被禁止生成的 combo 集合；初始由 config 填充，每产生一个完整 combo 自动加入
    std::set<std::vector<int>> banned_combos;

    StreamRecommendationInfo() = default;
    StreamRecommendationInfo(int32_t                           combo_token_size,
                             int32_t                           input_length,
                             int32_t                           current_output_length,
                             bool                              is_beam_search,
                             const std::set<std::vector<int>>& banned_combos):
        combo_token_size(combo_token_size),
        input_length(input_length),
        current_output_length(current_output_length),
        is_beam_search(is_beam_search),
        banned_combos(banned_combos) {}

    StreamRecommendationInfo copy() const {
        StreamRecommendationInfo info;
        info.combo_token_size      = combo_token_size;
        info.input_length          = input_length;
        info.current_output_length = current_output_length;
        info.is_beam_search        = is_beam_search;
        info.pos_in_combo          = pos_in_combo;
        info.current_prefix        = current_prefix;
        info.banned_combos         = banned_combos;
        return info;
    }
};

class RecommendationLogitsProcessor: public BaseLogitsProcessor {
public:
    RecommendationLogitsProcessor() = default;
    explicit RecommendationLogitsProcessor(std::vector<StreamRecommendationInfo> infos);
    virtual ~RecommendationLogitsProcessor() {}

public:
    // 若 generate_config.combo_token_size <= 0 则返回 nullptr（未启用该功能）。
    static std::shared_ptr<RecommendationLogitsProcessor>
    fromGenerateInput(std::shared_ptr<GenerateInput> generate_input, int32_t num);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

public:
    size_t size() const {
        return infos_.size();
    }
    const std::vector<StreamRecommendationInfo>& infos() const {
        return infos_;
    }
    void insert(std::shared_ptr<RecommendationLogitsProcessor> others) {
        if (others != nullptr) {
            infos_.insert(infos_.end(), others->infos_.begin(), others->infos_.end());
        }
    }

private:
    // 将单个 token 推进到状态机；若形成完整 combo 则自动加入 banned_combos
    void advanceOneToken(StreamRecommendationInfo& info, int token_id);

private:
    std::vector<StreamRecommendationInfo> infos_;
};

using RecommendationLogitsProcessorPtr = std::shared_ptr<RecommendationLogitsProcessor>;

}  // namespace rtp_llm
