package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.flexlb.state.internal.LedgerJanitor;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 重启重建与引擎收养：清空两侧账后按序重放全量历史——
 * 不认识的 running 条目按引擎收养入账（batchId=-1、engineOwned=true），
 * 历史 finished 收尾对应条目，跨侧规则在重放中同样生效，重建后账面可对账。
 */
class StateLedgerRebuildTest {

    @Test
    void rebuildAdoptsEngineOwnedEntriesAndKeepsCounters() {
        StateLedger ledger = new StateLedger();
        // 模拟重启前的引擎观察历史（世代取任意大值，验证 observeGeneration 防归零）
        long pGen = 900_000L;
        long dGen = 901_000L;
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);

        List<EngineObservation> history = List.of(
                // round 1：P 侧两条 running（100 装载中、101 执行中）
                TestEndpoints.observation(pEp, 1L, 1_000L,
                        List.of(
                                TestEndpoints.running(100L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L),
                                TestEndpoints.running(101L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 1024L, 2L)),
                        List.of()),
                // round 1：D 侧两条 running（100 传输中、200 执行中）
                TestEndpoints.observation(dEp, 1L, 1_010L,
                        List.of(
                                TestEndpoints.running(100L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 3L),
                                TestEndpoints.running(200L, StateRole.DECODE, EnginePhase.RUNNING, -1L, 8192L, 4L)),
                        List.of()),
                // round 2：P 侧 101 完成；D 侧 200 完成
                TestEndpoints.observation(pEp, 2L, 1_020L,
                        List.of(TestEndpoints.running(100L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 512L, 5L)),
                        List.of(TestEndpoints.finished(101L, StateRole.PREFILL, 0, 1_020L, 6L))),
                TestEndpoints.observation(dEp, 2L, 1_030L,
                        List.of(TestEndpoints.running(100L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 7L)),
                        List.of(TestEndpoints.finished(200L, StateRole.DECODE, 0, 1_030L, 8L))));

        ledger.rebuild(history);

        // 引擎收养语义：100 被收养（batchId=-1、engineOwned=true、binding=观察端点世代、kvTokens 入账）
        PrefillRequestStateView pv = ledger.prefill().get(100L).orElseThrow();
        assertTrue(pv.engineOwned());
        assertEquals(-1L, pv.batchId()); // 引擎收养：不认识的 running 条目按散请求收养
        assertEquals(new GenerationTriple(1, pGen, -1L), pv.binding());
        assertEquals(512L, pv.kvTokensReported());
        // D KV_ALLOCATED 重放触发跨侧收缩：P 100 从 P_WAITING_LOADED(7) → PREFILL_DONE(9)
        assertEquals(9, pv.phaseOrdinal());

        // 101：历史 finished 收尾（COMPLETED 终局移除 + 墓碑）
        assertTrue(ledger.prefill().get(101L).isEmpty());

        // D 侧收养：100 D_LOADING（引擎事实接管、无预占历史）；200 finished 移除
        DecodeRequestStateView dv = ledger.decode().get(100L).orElseThrow();
        assertEquals(2, dv.phaseOrdinal());
        assertTrue(dv.engineOwned());
        assertEquals(0L, dv.reservedKv());     // 收养无预占
        assertEquals(0L, dv.reservedExpectedKv());
        assertEquals(512L, dv.kvTokensReported());
        assertTrue(ledger.decode().get(200L).isEmpty());

        // 两侧计数（refresh 后精确）
        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        PrefillCounterSnapshot ps = ledger.prefill().snapshot();
        assertEquals(1L, ps.inflight());
        assertEquals(1L, ps.phaseCounts().get(9)); // PREFILL_DONE
        assertEquals(1L, ps.engineOwned());
        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(1L, ds.activeTotal());
        assertEquals(1L, ds.phaseCounts().get(2)); // D_LOADING
        assertEquals(0L, ds.reservedKvTotal());    // 收养条目无预占
        assertEquals(512L, ds.kvTokensReportedTotal());
        assertEquals(1L, ds.confirmed());

        // 墓碑：P 101 + D 200
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(1L, s.prefillTombstones());
        assertEquals(1L, s.decodeTombstones());
        // 重建不产生 unknown/跨代拒绝（历史按序重放）
        assertEquals(0L, s.unknownRunningEvents());
        assertEquals(0L, s.crossGenerationRejects());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());

        // 重建后继续正常观察（同代报文仍被接受——observeGeneration 防归零）
        ledger.observe(TestEndpoints.runningOnly(dEp, 3L, 2_000L,
                TestEndpoints.running(100L, StateRole.DECODE, EnginePhase.RUNNING, -1L, 512L, 9L)));
        assertEquals(3, ledger.decode().get(100L).orElseThrow().phaseOrdinal());
    }

    /** rebuild 乱序旧代报文：更高代已登记后，旧代整报拒绝（重建以最后状态为准）。 */
    @Test
    void rebuildRejectsOutOfOrderOldGenerationReports() {
        StateLedger ledger = new StateLedger();
        long g1 = 100L;
        long g2 = 200L;
        TestEndpoints.Endpoint ep1 = TestEndpoints.ep(1L, StateRole.PREFILL, g1);
        TestEndpoints.Endpoint ep2 = TestEndpoints.ep(1L, StateRole.PREFILL, g2);

        List<EngineObservation> history = List.of(
                // 新代先出现（条目 1 收养），随后旧代报文迟到
                TestEndpoints.observation(ep2, 1L, 1_000L,
                        List.of(TestEndpoints.running(1L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 0L, 1L)),
                        List.of()),
                TestEndpoints.observation(ep1, 2L, 1_010L,
                        List.of(TestEndpoints.running(1L, StateRole.PREFILL, EnginePhase.RECEIVED, -1L, 0L, 2L)),
                        List.of()));

        ledger.rebuild(history);
        // 旧代整报拒绝：条目 1 保持新代观察的 P_RUNNING(8)
        assertEquals(8, ledger.prefill().get(1L).orElseThrow().phaseOrdinal());
        assertEquals(1L, ledger.snapshot().crossGenerationRejects());
    }

    /**
     * observeAdopting（非清空式收养）：master 重启后每端点首报的增量收养——
     * 既有账目不清空（重启窗口内本地开账的新请求不受影响，与 rebuild 的关键
     * 差别），未知 running 收养、未知 finished 静默跳过；收养条目经后续正常
     * observe 接续生命周期，首报 tick 不触发 janitor 缺席扫描。
     */
    @Test
    void observeAdoptingAdoptsUnknownRunningWithoutClearingExistingEntries() {
        StateLedger ledger = new StateLedger();
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, 5_000L);
        LedgerJanitor janitor = ledger.createJanitor(
                new LedgerJanitorConfig(3, 60_000L, 120_000L, 100));

        // 重启后本地先开账的新请求（master 重启窗口内新 submit）
        assertEquals(RegisterResult.OK, ledger.prefill().register(10L, -1L));
        ledger.prefill().onQueued(10L);

        // 首报收养：未知 running 20 收养；未知 finished 99（重启前已终局）静默跳过
        ledger.observeAdopting(TestEndpoints.observation(pEp, 1L, 1_000L,
                List.of(TestEndpoints.running(20L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 256L, 1L)),
                List.of(TestEndpoints.finished(99L, StateRole.PREFILL, 0, 1_000L, 1L))));

        // 既有条目未被清空（与 rebuild 的关键差别）
        assertTrue(ledger.prefill().get(10L).isPresent(), "observeAdopting 不得清空既有账目");

        // 未知 running 收养入账（engineOwned、batchId=-1、binding=观察端点世代）
        PrefillRequestStateView adopted = ledger.prefill().get(20L).orElseThrow();
        assertTrue(adopted.engineOwned());
        assertEquals(-1L, adopted.batchId());
        assertEquals(new GenerationTriple(1, 5_000L, -1L), adopted.binding());
        assertEquals(256L, adopted.kvTokensReported());

        // 未知 finished 静默跳过（收养基线不含历史终局——不计 unknown、无墓碑）
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(0L, s.unknownFinishedEvents());
        assertEquals(0L, s.prefillTombstones());

        // 首报 tick 不触发 janitor 缺席扫描（round 基线未建立，缺席判定从下一完整 tick 起算）
        assertEquals(0L, janitor.stats().roundEndTicks());
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, 1_100L,
                TestEndpoints.running(20L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 512L, 2L)));
        assertEquals(1L, janitor.stats().roundEndTicks(), "后续正常 observe 才推缺席扫描");

        // 收养条目经后续正常 observe 接续生命周期（running 观察刷新 → finished 终局）
        assertEquals(512L, ledger.prefill().get(20L).orElseThrow().kvTokensReported());
        ledger.observe(TestEndpoints.finishedOnly(pEp, 3L, 1_200L,
                TestEndpoints.finished(20L, StateRole.PREFILL, 0, 1_200L, 3L)));
        assertTrue(ledger.prefill().get(20L).isEmpty(), "收养条目 finished 后正常终局");
        assertEquals(TerminalState.COMPLETED,
                ledger.terminalOutcomeOf(20L, StateRole.PREFILL).orElseThrow().state());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }
}
