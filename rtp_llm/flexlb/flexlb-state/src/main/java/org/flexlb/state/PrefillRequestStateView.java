package org.flexlb.state;

import java.util.List;

/**
 * P 侧请求状态只读视图（快照语义：构造时拍照，不暴露内部可变对象）。
 *
 * @param requestId        请求 ID
 * @param createdAtMs      创建时间（创建时刻固定不可续命的 TTL 基准）
 * @param phaseOrdinal     当前相位格高度（= PrefillPhase.ordinal）
 * @param phaseName        当前相位名（如 "P_RUNNING"）
 * @param batchId          所属批次（-1 = 散请求）
 * @param pendingCancel    正交取消意图标记
 * @param binding          世代绑定（发送前可重绑，DISPATCHED 后不可变）
 * @param kvTokensReported 引擎上报 KV（引擎上报观察侧，KV 残留感知基础；0 = unknown，不更新）
 * @param lastSeenRound    最近被引擎上报观察到的轮次
 * @param engineOwned      引擎已见（引擎上报观察）
 * @param dispatchedAtMs   派发时刻（派发流水线侧）
 * @param lastVersion      最近接受的引擎上报序号
 * @param trace            相位进入历史（人类可读，最旧→最新；含终态标记）
 */
public record PrefillRequestStateView(
        long requestId,
        long createdAtMs,
        int phaseOrdinal,
        String phaseName,
        long batchId,
        boolean pendingCancel,
        GenerationTriple binding,
        long kvTokensReported,
        long lastSeenRound,
        boolean engineOwned,
        long dispatchedAtMs,
        long lastVersion,
        List<String> trace) {

    public PrefillRequestStateView {
        trace = List.copyOf(trace);
    }
}
