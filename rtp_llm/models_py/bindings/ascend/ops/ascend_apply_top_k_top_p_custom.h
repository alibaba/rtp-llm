#pragma once
#if USING_ASCEND

#include <torch/all.h>

#include "op_api_common.h"        // EXEC_NPU_CMD 宏（内含 torch_npu 全部头文件）
#include "aclnn_apply_top_k_top_p_custom.h"      // ACLNN 接口声明

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace ascend {

/**
 * @brief 对输入张量执行 top-k/top-p 过滤。
 *
 * 函数名命名规范：以小驼峰命名，体现算子功能语义，如 applyTopKTopP / moeGatingTopK / causalConv1d。
 * 应与 Step 4.1 中复制的 aclnn_<op_name>.h 中声明的 C 函数名（去掉 aclnn 前缀）对应。
 *
 * @param input   输入张量（logits），形状 [batch_size, vocab_size]，在 NPU 上
 * @param param1  top_p 阈值张量，形状 [batch_size] 或 nullopt
 * @param param2  top_k 值张量，形状 [batch_size] 或 nullopt，INT32 类型
 * @param output  输出张量，形状与 input 一致（低于阈值的 logits 被置为 -inf），在 NPU 上
 *
 * 调用链：EXEC_NPU_CMD 宏
 *   → dlopen("libopapi.so")
 *   → aclnnApplyTopKTopPCustomGetWorkspaceSize()
 *   → aclnnApplyTopKTopPCustom()
 */
inline void applyTopKTopP(    // ← 按算子功能命名，如 applyTopKTopP / moeGatingTopK 等
    const at::Tensor& input,
    const c10::optional<at::Tensor>& param1,
    const c10::optional<at::Tensor>& param2,
    at::Tensor& output)
{
    // ---- 参数校验 ----
    TORCH_CHECK(input.dim() == 2,
                "input must be 2D [batch_size, feature_dim]");
    TORCH_CHECK(param1.has_value() || param2.has_value(),
                "at least one of param1/param2 must be provided");

    // ---- 调用 ACLNN 自定义算子 ----
    // EXEC_NPU_CMD 内部自动完成：
    //   1. dlopen libopapi.so 查找函数地址
    //   2. 调用 GetWorkspaceSize 计算 workspace
    //   3. 分配 workspace
    //   4. 调用执行函数提交到 NPU stream
    // 注意：多态分发使用 c10::optional<at::Tensor>，
    //       nullptr 表示该参数不参与计算
    if (param1.has_value() && param2.has_value()) {
        EXEC_NPU_CMD(aclnnApplyTopKTopPCustom, input, param1.value(), param2.value(), output);
    } else if (param1.has_value()) {
        EXEC_NPU_CMD(aclnnApplyTopKTopPCustom, input, param1.value(), c10::nullopt, output);
    } else {
        EXEC_NPU_CMD(aclnnApplyTopKTopPCustom, input, c10::nullopt, param2.value(), output);
    }
}

}  // namespace ascend
}  // namespace rtp_llm

#endif  // USING_ASCEND