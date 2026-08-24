package org.flexlb.state;

/**
 * 端点世代三元组：端点身份 + 世代 + 批次。
 *
 * <p>裁决矩阵的第一优先级屏障（REJECT_GENERATION）：观察事件的世代三元组与条目所属世代
 * 不匹配时整报拒绝，防止跨世代脏写。</p>
 *
 * @param endpointId 端点 ID
 * @param generation 端点世代号（重启/重置递增）
 * @param batchId    批次 ID（同一世代内批次隔离；无批次语义时约定为 -1）
 */
public record GenerationTriple(int endpointId, long generation, long batchId) {
}
