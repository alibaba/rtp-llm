package org.flexlb.state;

import java.util.Objects;

/**
 * 终局结果：状态 + 受控原因 + 受控补充信息。
 *
 * <p>reason 为受控枚举（O2 语义），<b>必填</b>——禁止用自由文本 detail 滥用替代原因分类；
 * detail 仅承载受控补充信息（如错误码、触发阈值快照），可为空字符串。</p>
 *
 * @param state  终态（含回边态 PREEMPTED）
 * @param reason 受控终态原因，不可为 null
 * @param detail 受控补充信息（可为空串，禁止承载自由文本语义）
 */
public record TerminalOutcome(TerminalState state, TerminalReason reason, String detail) {

    public TerminalOutcome {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(reason, "reason 必填：终局必须携带受控原因，禁止自由文本滥用");
        detail = detail == null ? "" : detail;
    }
}
