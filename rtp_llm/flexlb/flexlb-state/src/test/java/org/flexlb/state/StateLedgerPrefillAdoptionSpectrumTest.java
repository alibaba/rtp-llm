package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * P 侧收养相位 × 后续命运全谱系（3 收养相位 × 8 命运 = 24 组合）：
 * 每个组合独立 requestId 走完「收养 → 后续观察序列变体 → 终局」，
 * 收敛后增量账与全量重算对账零偏差——收养入账按收养相位参数化记账，
 * 任意后续观察（同相位刷新 / 推进 / 迟到低位 / 终局后迟到重放）与
 * 入账口径在任意交错下恒配平，端点簿相位人口不漂移。
 *
 * <p>八个命运（后续观察序列变体）：本地取消直接终局 / finished 越级
 * 直接终局 / 同相位刷新后终局 / 推进执行位后终局 / 多连观察混合后终局 /
 * 迟到低位观察后终局 / 终局后迟到 running 与 finished 双吸收（墓碑优先）/
 * 终局后本地取消迟到让位（墓碑吸收迟到取消）。</p>
 */
class StateLedgerPrefillAdoptionSpectrumTest {

    /** 三个收养相位（引擎观察相位 → P 侧格映射的完整值域）。 */
    private static final EnginePhase[] ADOPT_PHASES = {
            EnginePhase.RECEIVED, EnginePhase.KV_ALLOCATED, EnginePhase.RUNNING};

    /** 观察时刻基准：真实墙钟（墓碑过期判定用真实时钟，假时间戳会瞬间“过期”）。 */
    private static final long NOW = System.currentTimeMillis();

    @Test
    void adoptionPhaseByFateMatrixKeepsBooksExact() {
        StateLedger ledger = new StateLedger();
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, 5_000L);
        TerminalOutcome completed = new TerminalOutcome(TerminalState.COMPLETED,
                TerminalReason.SUCCEEDED, "");
        TerminalOutcome cancelled = new TerminalOutcome(TerminalState.CANCELLED,
                TerminalReason.CANCELLED_IMPLICIT, "");

        for (int a = 0; a < ADOPT_PHASES.length; a++) {
            EnginePhase adoptPhase = ADOPT_PHASES[a];
            for (int fate = 0; fate < 8; fate++) {
                long id = 1_000L + a * 8L + fate;
                long kv = 100L + id;
                long seq = id * 100L;
                // 收养（version=1；收养相位由引擎观察相位映射）
                ledger.observeAdopting(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                        TestEndpoints.running(id, StateRole.PREFILL, adoptPhase, -1L, kv, 1L)));
                long v = 1L;
                switch (fate) {
                    case 0 -> {
                        // 命运 0：无后续观察——本地取消直接终局（出账按收养相位现态）
                        assertTrue(ledger.prefill().settle(id, cancelled, SettleReason.LOCAL_CANCEL),
                                "组合[收养=" + adoptPhase + ",命运=0] 本地取消应终局胜出");
                    }
                    case 1 -> {
                        // 命运 1：无后续观察——引擎 finished 直接越级终局
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 2 -> {
                        // 命运 2：同相位重复观察（版本与 KV 刷新，相位不动账）→ finished
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, adoptPhase, -1L, kv + 16L, ++v)));
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 3 -> {
                        // 命运 3：推进到执行位（桶迁移账以收养相位为 from）→ finished
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, kv + 32L, ++v)));
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 4 -> {
                        // 命运 4：同相位刷新 → 越级推进 → 同相位刷新 → finished（多连观察混合）
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, adoptPhase, -1L, kv + 48L, ++v)));
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, kv + 64L, ++v)));
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, kv + 80L, ++v)));
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 5 -> {
                        // 命运 5：迟到低位观察（不推进；新鲜版本仍刷新引擎观察）→ finished
                        ledger.observe(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RECEIVED, -1L, kv + 96L, ++v)));
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 6 -> {
                        // 命运 6：终局后迟到 running 与迟到 finished 双吸收（墓碑优先，不复活）
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                        ledger.observeAdopting(TestEndpoints.runningOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, kv + 112L, ++v)));
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                    }
                    case 7 -> {
                        // 命运 7：finished 终局后本地取消迟到让位（对已终局条目 no-op → 墓碑吸收迟到取消）
                        ledger.observe(TestEndpoints.finishedOnly(pEp, ++seq, NOW + seq,
                                TestEndpoints.finished(id, StateRole.PREFILL, 0, NOW + seq, ++v)));
                        assertTrue(!ledger.prefill().settle(id, cancelled, SettleReason.LOCAL_CANCEL),
                                "组合[收养=" + adoptPhase + ",命运=7] 终局后本地取消必让位");
                    }
                    default -> throw new AssertionError("unreachable fate: " + fate);
                }
                assertTrue(ledger.prefill().get(id).isEmpty(),
                        "组合[收养=" + adoptPhase + ",命运=" + fate + "] 应收敛终局");
            }
        }

        // 主断言：24 组合全部收敛——增量账 vs 全量重算（全局 + 端点级）零偏差
        assertTrue(ledger.auditAndDrift().clean(),
                () -> "收养谱系 drift:\n" + ledger.auditAndDrift());

        // 全局账归零（活跃与 engineOwned）
        ledger.prefill().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight(), "24 组合全部终局");
        assertEquals(0L, ledger.prefill().snapshot().engineOwned());

        // 端点簿全字段归零（收养全在同一端点桶）
        PrefillEndpointCounters book = ledger.prefill().endpointCounters(1);
        assertEquals(0, book.activeTotal());
        assertEquals(0, book.engineOwnedCount());
        assertEquals(0L, book.estimatedWaitMs());
        for (int p = 0; p < book.phaseCounts().size(); p++) {
            assertEquals(0L, book.phaseCounts().get(p), "phase[" + p + "] 归零");
        }

        // 墓碑守恒：24 个 requestId 恰 24 条墓碑（每 id 恰一；迟到事件不增殖）
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(24L, s.prefillTombstones());
        // 迟到事件全部被墓碑吸收：命运 6 双吸收（迟到 running + 迟到 finished）× 3 相位 = 6
        assertEquals(6L, s.lateEventsAbsorbed());
        // 命运 7 迟到本地取消吸收 × 3 相位 = 3
        assertEquals(3L, s.lateCancelsAbsorbed());
    }
}
