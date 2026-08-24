package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * 共享内核契约：TerminalState 吸收态体系（设计 §8）与 TerminalOutcome reason 必填约束。
 */
class TerminalStateTest {

    /** §8：前四个终态吸收，PREEMPTED 回边非吸收。 */
    @Test
    void absorbingSemantics() {
        assertTrue(TerminalState.COMPLETED.isAbsorbing());
        assertTrue(TerminalState.CANCELLED.isAbsorbing());
        assertTrue(TerminalState.SLO_TIMEOUT.isAbsorbing());
        assertTrue(TerminalState.FAILED.isAbsorbing());
        assertFalse(TerminalState.PREEMPTED.isAbsorbing(), "PREEMPTED 是回边态（可重试回已决策）");
    }

    /** TerminalOutcome：reason 必填（O2 受控原因，禁止自由文本滥用），detail 可空。 */
    @Test
    void outcomeRequiresControlledReason() {
        TerminalOutcome outcome = new TerminalOutcome(
                TerminalState.COMPLETED, TerminalReason.SUCCEEDED, null);
        assertEquals(TerminalReason.SUCCEEDED, outcome.reason());
        assertEquals("", outcome.detail(), "null detail 规范化为空串");

        assertThrows(NullPointerException.class,
                () -> new TerminalOutcome(TerminalState.COMPLETED, null, "free text"));
        assertThrows(NullPointerException.class,
                () -> new TerminalOutcome(null, TerminalReason.SUCCEEDED, ""));
    }

    /** 终态与终态原因的配对健全性：每个 TerminalReason 都有对应可表达的语义出口。 */
    @Test
    void reasonValueDomain() {
        assertNotNull(TerminalReason.valueOf("PREEMPTED"));
        assertNotNull(CleanupReason.valueOf("FENCE_HOLD"));
        assertNotNull(SettleReason.valueOf("CAUSAL_CLOSURE"));
        assertNotNull(TransitionReason.valueOf("ENGINE_OBSERVATION"));
    }
}
