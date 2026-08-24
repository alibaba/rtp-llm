package org.flexlb.state;

import java.util.List;

/**
 * D 侧请求状态只读视图（快照语义：构造时拍照，不暴露内部可变对象）。
 *
 * @param requestId          请求 ID
 * @param createdAtMs        创建时间
 * @param phaseOrdinal       当前相位格高度（= DecodePhase.ordinal）
 * @param phaseName          当前相位名（如 "D_LOADING"）
 * @param pendingCancel      正交取消意图标记
 * @param binding            世代绑定
 * @param reservedKv         D① 影子预占当前占用（KV_ALLOCATED 确认后清 0）
 * @param reservedExpectedKv 预约时声明的期望 KV（历史记录，保留）
 * @param kvTokensReported   D② 引擎事实 KV（KV_ALLOCATED 起接管；0 = unknown，E1）
 * @param lastSeenRound      最近被引擎上报观察到的轮次
 * @param engineOwned        引擎已见
 * @param lastVersion        最近接受的引擎上报序号
 * @param trace              相位进入历史（人类可读；含终态标记）
 */
public record DecodeRequestStateView(
        long requestId,
        long createdAtMs,
        int phaseOrdinal,
        String phaseName,
        boolean pendingCancel,
        GenerationTriple binding,
        long reservedKv,
        long reservedExpectedKv,
        long kvTokensReported,
        long lastSeenRound,
        boolean engineOwned,
        long lastVersion,
        List<String> trace) {

    public DecodeRequestStateView {
        trace = List.copyOf(trace);
    }
}
