package org.flexlb.state;

import java.util.List;
import java.util.Optional;

/**
 * per-request 诊断故事线（只读视图）：{@link StateLedger#traceOf(long)} 的产物。
 *
 * <p>组合四路事实：P 侧活跃条目 + D 侧活跃条目 + 两侧墓碑终态。活跃条目给出
 * 当前相位/世代绑定/批次/时间戳与 trace 环内容；墓碑给出终态与终局时刻的
 * trace 环快照（保留期内可查——条目移除后故事线不丢失）。</p>
 *
 * @param prefillActive    P 侧活跃条目视图（排队/派发/引擎执行中；无则 empty）
 * @param decodeActive     D 侧活跃条目视图（预占/派发/加载/执行中；无则 empty）
 * @param prefillTombstone P 侧墓碑终态（已终局且保留期内；无则 empty）
 * @param decodeTombstone  D 侧墓碑终态（已终局且保留期内；无则 empty）
 */
public record LedgerTraceView(
        Optional<PrefillRequestStateView> prefillActive,
        Optional<DecodeRequestStateView> decodeActive,
        Optional<TombstoneView> prefillTombstone,
        Optional<TombstoneView> decodeTombstone) {

    /**
     * 墓碑终态视图（条目已移除后的最终事实）。
     *
     * @param requestId   请求 ID
     * @param state       终态
     * @param reason      终局受控原因（墓碑 reason 字符串解析；不可解析时兜底 SUCCEEDED）
     * @param terminalAtMs 终局时刻（epoch 毫秒）
     * @param entryTrace  终局时刻的条目 trace 环快照（人类可读相位历史，最旧→最新）
     */
    public record TombstoneView(long requestId, TerminalState state, TerminalReason reason,
                                long terminalAtMs, List<String> entryTrace) {

        public TombstoneView {
            entryTrace = List.copyOf(entryTrace);
        }
    }
}
