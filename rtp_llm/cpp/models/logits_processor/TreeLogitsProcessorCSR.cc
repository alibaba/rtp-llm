#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessorCSR.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

#include <algorithm>

using namespace std;

namespace rtp_llm {

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------

TreeLogitsProcessorCSR::TreeLogitsProcessorCSR(rtp_llm::DeviceBase* device)
    : BaseLogitsProcessor(device) {}

TreeLogitsProcessorCSR::TreeLogitsProcessorCSR(rtp_llm::DeviceBase*        device,
                                               std::vector<StreamTreeInfo> tree_infos)
    : BaseLogitsProcessor(device), tree_infos_(std::move(tree_infos)) {}

// ---------------------------------------------------------------------------
// process()
//
// 每步 decode 时调用。对每个 beam：
//   - state == 0（根节点）：合法 token 来自 start_mask（CPU 侧向量）
//   - 其他 state：合法 token 来自 packed_csr 中 state 对应行的 token 列
//
// 目前使用 CPU 侧 CSRIndex 枚举候选 token；
// GPU 侧 buffer（d_indptr / d_packed_csr_*）为未来 GPU kernel 升级预留。
// ---------------------------------------------------------------------------
void TreeLogitsProcessorCSR::process(const SamplerInputs& inputs,
                                     size_t               start_idx,
                                     size_t               finish_idx) {
    const size_t batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);

    auto         batch_logits = inputs.logits->slice(start_idx, batch_size);
    const size_t vocab_size   = batch_logits->shape()[1];

    bool                             need_process = false;
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }

        std::vector<size_t> candidate_token_ids;
        const int           state = info.current_state;

        if (state == 0) {
            // 根节点：直接查 start_mask（第0层合法token掩码）
            const auto& sm = info.csr_index.start_mask;
            for (int j = 0; j < static_cast<int>(vocab_size) && j < static_cast<int>(sm.size()); ++j) {
                if (sm[j]) {
                    candidate_token_ids.push_back(static_cast<size_t>(j));
                }
            }
        } else {
            // 非根节点：在 CSR 中查询 state 行，获取合法子 token
            const auto& indptr = info.csr_index.indptr;
            const auto& tokens = info.csr_index.packed_csr_tokens;
            RTP_LLM_CHECK(state + 1 < static_cast<int>(indptr.size()));
            int row_start = indptr[state];
            int row_end   = indptr[state + 1];
            for (int j = row_start; j < row_end; ++j) {
                int tok = tokens[j];
                // token == vocab_size 是填充行哨兵值，跳过
                if (tok < static_cast<int>(vocab_size)) {
                    candidate_token_ids.push_back(static_cast<size_t>(tok));
                }
            }
        }

        if (!candidate_token_ids.empty()) {
            need_process = true;
        }
        batch_candidate_token_ids[i] = std::move(candidate_token_ids);
    }

    if (!need_process) {
        return;
    }

    auto batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
}

// ---------------------------------------------------------------------------
// updateStatus()
//
// 采样完成后调用，推进每个 beam 在前缀树中的状态。
// 对每个 beam：在 CSR 中找到采样 token 对应的 next_state。
//
// new_tokens shape：[batch_size, num_new_tokens]
//   - 非 beam search：每行最后一个 token 即为采样结果
//   - beam search：提供完整序列历史，取位置 (input_length + current_output_length + j)
// ---------------------------------------------------------------------------
void TreeLogitsProcessorCSR::updateStatus(const rtp_llm::BufferPtr& new_tokens,
                                          int32_t                   num_new_tokens) {
    RTP_LLM_CHECK(new_tokens->dim() == 2);
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    const int max_col = static_cast<int>(new_tokens->shape()[1]);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }

        const auto& indptr = info.csr_index.indptr;
        const auto& tokens = info.csr_index.packed_csr_tokens;
        const auto& states = info.csr_index.packed_csr_states;

        for (int j = 0; j < num_new_tokens; ++j) {
            // 确定从 new_tokens 中读取采样 token 的列索引
            int col_idx = info.is_beam_search
                              ? (info.input_length + info.current_output_length + j)
                              : j;
            RTP_LLM_CHECK(col_idx < max_col);

            int sampled_token = *new_tokens->dataWithOffset<int32_t>(
                static_cast<int>(i) * max_col + col_idx);

            const int state = info.current_state;

            if (state == 0) {
                // 根节点：第0层节点 state_id = token_id + 1
                info.current_state = sampled_token + 1;
            } else {
                // 非根节点：在 CSR 行中找到采样 token，取对应的 next_state
                RTP_LLM_CHECK(state + 1 < static_cast<int>(indptr.size()));
                int row_start  = indptr[state];
                int row_end    = indptr[state + 1];
                int next_state = 0;  // 0 表示终止或未找到（视为重置到根节点）
                for (int k = row_start; k < row_end; ++k) {
                    if (tokens[k] == sampled_token) {
                        next_state = states[k];
                        break;
                    }
                }
                info.current_state = next_state;
            }
            info.current_output_length += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// updateMultiSeqStatus()
//
// beam search 选择完成后调用，按新的 beam 索引重排各 beam 的状态。
// src_batch_indices[i] = 第 i 个新 beam 对应旧 beam 的索引。
// GPU buffer 共享（只读），只需对 current_state 进行重排。
// ---------------------------------------------------------------------------
void TreeLogitsProcessorCSR::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    new_tree_infos.reserve(src_batch_indices.size());
    for (int src_idx : src_batch_indices) {
        RTP_LLM_CHECK(src_idx >= 0 && src_idx < static_cast<int>(tree_infos_.size()));
        new_tree_infos.push_back(tree_infos_[src_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

// ---------------------------------------------------------------------------
// fromGenerateInput()
//
// 1. 解析 ele_rq_ids -> 排序后的 sids<token_num> 数组
// 2. 在 CPU 上构建 CSRIndex<token_num>
// 3. 将 indptr / packed_csr 上传到 GPU（每次请求仅上传一次）
// 4. 创建 num 个 StreamTreeInfo（每个 beam / return-sequence 一个），
//    共享相同的 GPU buffer
// ---------------------------------------------------------------------------
std::shared_ptr<TreeLogitsProcessorCSR>
TreeLogitsProcessorCSR::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                          std::shared_ptr<GenerateInput> generate_input,
                                          int32_t                        num,
                                          int32_t                        vocab_size) {
    if (generate_input->generate_config->ele_rq_ids.empty()) {
        return nullptr;
    }

    // --- 1. 解析约束字符串 ---
    std::vector<sids<token_num>> origin_rq_ids =
        split_strings<token_num>(generate_input->generate_config->ele_rq_ids);
    std::sort(origin_rq_ids.begin(), origin_rq_ids.end());

    // --- 2. 在 CPU 上构建 CSR 索引 ---
    CSRIndex<token_num> csr_index;
    bool success = build_csr_from_fresh_data<token_num>(origin_rq_ids, csr_index, vocab_size);
    if (!success) {
        return nullptr;
    }

    // --- 3. 将 CSR 数组上传到 GPU（每次请求一次） ---
    // indptr：int32 向量 -> CPU buffer -> clone 到 DEVICE
    rtp_llm::BufferPtr d_indptr;
    {
        auto cpu_buf = rtp_llm::vector2Buffer(csr_index.indptr);
        d_indptr     = device->clone({*cpu_buf, rtp_llm::AllocationType::DEVICE});
    }

    rtp_llm::BufferPtr d_packed_csr_tokens;
    {
        auto cpu_buf        = rtp_llm::vector2Buffer(csr_index.packed_csr_tokens);
        d_packed_csr_tokens = device->clone({*cpu_buf, rtp_llm::AllocationType::DEVICE});
    }

    rtp_llm::BufferPtr d_packed_csr_states;
    {
        auto cpu_buf        = rtp_llm::vector2Buffer(csr_index.packed_csr_states);
        d_packed_csr_states = device->clone({*cpu_buf, rtp_llm::AllocationType::DEVICE});
    }

    // --- 4. 为每个 beam 创建 StreamTreeInfo ---
    bool is_multi_seq = generate_input->generate_config->hasNumBeams()
                        || generate_input->generate_config->num_return_sequences > 1;

    auto processor_ptr =
        std::make_shared<TreeLogitsProcessorCSR>(rtp_llm::DeviceFactory::getDefaultDevice());

    for (int32_t i = 0; i < num; ++i) {
        StreamTreeInfo tree_info(
            /*in_tree_mode=*/true,
            /*input_length=*/generate_input->inputLength(),
            /*current_output_length=*/0,
            /*is_beam_search=*/is_multi_seq,
            /*current_state=*/0,
            /*csr_index=*/csr_index,                          // CPU 只读副本
            /*d_indptr=*/d_indptr,                            // 共享 GPU buffer
            /*d_packed_csr_tokens=*/d_packed_csr_tokens,
            /*d_packed_csr_states=*/d_packed_csr_states);

        std::vector<StreamTreeInfo> single_vec = {std::move(tree_info)};
        auto single_processor =
            std::make_shared<TreeLogitsProcessorCSR>(device, std::move(single_vec));
        processor_ptr->insert(single_processor);
    }

    return processor_ptr;
}

}  // namespace rtp_llm
