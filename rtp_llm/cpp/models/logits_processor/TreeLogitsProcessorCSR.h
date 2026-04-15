#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/csr_utils.h"
#include "rtp_llm/cpp/core/Buffer.h"

// ---------------------------------------------------------------------------
// StreamTreeInfo：基于 CSR 前缀树的限制性解码中，每个 beam 的约束状态。
//
// CSRIndex 在 fromGenerateInput() 中于 CPU 上构建一次，随后上传到 GPU。
// 同一请求的所有 beam 共享以下只读 GPU buffer（通过 shared_ptr 引用计数）：
//   d_indptr / d_packed_csr_tokens / d_packed_csr_states / d_start_mask
//
// 每个 beam 各自持有私有的 GPU buffer：
//   d_current_state : int32 [1]，存储当前在前缀树中的状态ID
//
// CPU 侧的 csr_index 保留用于 fallback 和调试。
// ---------------------------------------------------------------------------
struct StreamTreeInfo {
    bool    in_tree_mode          = false;
    int32_t input_length          = 0;
    int32_t current_output_length = 0;
    bool    is_beam_search        = false;
    int32_t current_state         = 0;  // CPU 侧当前状态（与 d_current_state 保持同步）

    // CPU 侧 CSRIndex（各 beam 通过拷贝共享，初始化后只读）
    CSRIndex<token_num> csr_index;

    // ---------------------------------------------------------------------------
    // 只读共享 GPU buffer（同一请求所有 beam 共享，通过 shared_ptr 引用计数）
    // ---------------------------------------------------------------------------
    rtp_llm::BufferPtr d_indptr;              // int32 [num_states + 2]
    rtp_llm::BufferPtr d_packed_csr_tokens;   // int32 [num_transitions + vocab_size]
    rtp_llm::BufferPtr d_packed_csr_states;   // int32 [num_transitions + vocab_size]
    rtp_llm::BufferPtr d_start_mask;          // uint8 [vocab_size]，根节点合法 token 掩码

    // ---------------------------------------------------------------------------
    // 每个 beam 私有的 GPU buffer
    // ---------------------------------------------------------------------------
    rtp_llm::BufferPtr d_current_state;  // int32 [1]，当前前缀树状态ID

    StreamTreeInfo() = default;

    StreamTreeInfo(bool                in_tree_mode_,
                   int32_t             input_length_,
                   int32_t             current_output_length_,
                   bool                is_beam_search_,
                   int32_t             current_state_,
                   CSRIndex<token_num> csr_index_,
                   rtp_llm::BufferPtr  d_indptr_,
                   rtp_llm::BufferPtr  d_packed_csr_tokens_,
                   rtp_llm::BufferPtr  d_packed_csr_states_,
                   rtp_llm::BufferPtr  d_start_mask_,
                   rtp_llm::BufferPtr  d_current_state_):
        in_tree_mode(in_tree_mode_),
        input_length(input_length_),
        current_output_length(current_output_length_),
        is_beam_search(is_beam_search_),
        current_state(current_state_),
        csr_index(std::move(csr_index_)),
        d_indptr(std::move(d_indptr_)),
        d_packed_csr_tokens(std::move(d_packed_csr_tokens_)),
        d_packed_csr_states(std::move(d_packed_csr_states_)),
        d_start_mask(std::move(d_start_mask_)),
        d_current_state(std::move(d_current_state_)) {}

    // 拷贝语义：
    //   - 只读共享 GPU buffer 用 shared_ptr 共享（引用计数 +1，零拷贝）
    //   - d_current_state 是每个 beam 私有的，需要深拷贝（由调用方负责分配并初始化）
    //   - current_state（CPU 侧镜像）和 current_output_length 各 beam 独立
    StreamTreeInfo copy() const {
        StreamTreeInfo info;
        info.in_tree_mode          = in_tree_mode;
        info.input_length          = input_length;
        info.current_output_length = current_output_length;
        info.is_beam_search        = is_beam_search;
        info.current_state         = current_state;        // 每个 beam 从相同起始状态出发
        info.csr_index             = csr_index;            // CPU 只读副本，共享即可
        info.d_indptr              = d_indptr;             // 共享只读 GPU buffer
        info.d_packed_csr_tokens   = d_packed_csr_tokens;
        info.d_packed_csr_states   = d_packed_csr_states;
        info.d_start_mask          = d_start_mask;
        // 注意：d_current_state 不在这里拷贝，
        // 必须由 updateMultiSeqStatus 在深拷贝后单独分配和初始化。
        info.d_current_state       = nullptr;
        return info;
    }
};

namespace rtp_llm {

// ---------------------------------------------------------------------------
// TreeLogitsProcessorCSR：基于预建 CSR 前缀树的限制性解码处理器。
//
// 每次请求的生命周期：
//   1. fromGenerateInput()    – 解析 ele_rq_ids，在 CPU 构建 CSRIndex，
//                               上传到 GPU，为每个 beam 创建 StreamTreeInfo，
//                               并分配持久化 GPU buffer。
//   2. process()              – 每步 decode：直接用持久化 d_states_batch 调用
//                               invokeCsrBuildMask，在 GPU 上生成 mask，再 maskLogits。
//                               无需每步 H2D 传输状态。
//   3. updateStatus()         – 采样后用 invokeCsrGatherTokens 在 GPU 上 gather token，
//                               再用 invokeCsrUpdateStates 原地更新 d_states_batch，
//                               最后一次 D2H 同步 CPU 镜像。
//   4. updateMultiSeqStatus() – beam search 选择后，按新的 beam 索引重排状态；
//                               重建 d_states_batch 并从旧 beam CPU 镜像初始化。
// ---------------------------------------------------------------------------
class TreeLogitsProcessorCSR: public BaseLogitsProcessor {
public:
    explicit TreeLogitsProcessorCSR(rtp_llm::DeviceBase* device);
    TreeLogitsProcessorCSR(rtp_llm::DeviceBase*        device,
                           std::vector<StreamTreeInfo> tree_infos);
    virtual ~TreeLogitsProcessorCSR() = default;

    // 工厂方法：在 GenerateStream 构造时调用一次。
    // num        = init_batch_size（beam search 时为1，多路独立采样时为 num_return_sequences）
    // vocab_size = 模型词表大小
    static std::shared_ptr<TreeLogitsProcessorCSR>
    fromGenerateInput(rtp_llm::DeviceBase*           device,
                      std::shared_ptr<GenerateInput> generate_input,
                      int32_t                        num,
                      int32_t                        vocab_size);

    // BaseLogitsProcessor 接口实现
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) override;

    size_t size() const { return tree_infos_.size(); }

    void insert(std::shared_ptr<TreeLogitsProcessorCSR> others) {
        if (others != nullptr) {
            tree_infos_.insert(
                tree_infos_.end(), others->tree_infos_.begin(), others->tree_infos_.end());
        }
    }

private:
    std::vector<StreamTreeInfo> tree_infos_;

    // ---------------------------------------------------------------------------
    // 持久化 GPU buffer（processor 级，跨 decode 步复用，避免每步重新分配/传输）
    // ---------------------------------------------------------------------------
    // d_states_batch：[batch_size] int32，各 beam 当前前缀树状态的 GPU 副本。
    // process() 直接读取，updateStatus() 原地更新，updateMultiSeqStatus() 重建。
    rtp_llm::BufferPtr d_states_batch_;

    // d_sampled_tokens：[batch_size] int32，每步 gather 出的采样 token，复用分配。
    rtp_llm::BufferPtr d_sampled_tokens_;

    // d_col_offsets：[batch_size] int32，每步各 beam 对应的 new_tokens 列索引，复用分配。
    rtp_llm::BufferPtr d_col_offsets_;
};

typedef std::shared_ptr<TreeLogitsProcessorCSR> TreeLogitsProcessorCSRPtr;

}  // namespace rtp_llm
