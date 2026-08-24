package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 跨侧规则（StateLedger.observe 独占实现）核心测试：
 * <ol>
 *   <li>C7/L4：D KV_ALLOCATED 同 tick 收缩 P 条目到 PREFILL_DONE（P 释放点 = D 确认点）。</li>
 *   <li>F1：D finished(success) 因果闭包收缩 P 条目。</li>
 *   <li>cancel 双清：settle(CANCELLED) 同 tick 清两侧账（各自计数独立减）。</li>
 *   <li>C1 临界点：RECEIVED 期间 D① 预占保持（双记）+ KV_ALLOCATED 后预占撤/引擎值接管。</li>
 * </ol>
 */
class StateLedgerCrossSideTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    private static TerminalOutcome cancelOutcome() {
        return new TerminalOutcome(TerminalState.CANCELLED, TerminalReason.CANCELLED_ACK, "");
    }

    /** 构造 P 条目并推进到目标引擎观察相位的辅助。 */
    private static PrefillRequestStateView prefillAt(StateLedger ledger, TestEndpoints.Endpoint pEp,
                                                     GenerationTriple pBinding, long id,
                                                     EnginePhase targetPhase, long version) {
        ledger.prefill().register(id, 77L);
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, 77L);
        assertTrue(ledger.prefill().onDispatched(id, pBinding));
        if (targetPhase != null) {
            ledger.observe(TestEndpoints.runningOnly(pEp, version, 1_000L + version,
                    TestEndpoints.running(id, StateRole.PREFILL, targetPhase, 77L, 0L, version)));
        }
        return ledger.prefill().get(id).orElseThrow();
    }

    // ---- 规则 1：C7/L4 D 确认点即 P 释放点（同 tick 收缩）----

    @Test
    void dKvAllocatedShrinksWaitingLoadedPrefillSameTick() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 100L;

        // P 条目到 P_WAITING_LOADED(7)；D 条目到 DISPATCHED(1)
        prefillAt(ledger, pEp, pBinding, id, EnginePhase.KV_ALLOCATED, 1L);
        ledger.decode().reserve(id, 512L, 4096L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));

        // D KV_ALLOCATED：同 tick 收缩 P（7 → 9 闭包：补记 P8/P9）
        ledger.observe(TestEndpoints.runningOnly(dEp, 5L, 2_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 4096L, 5L)));

        PrefillRequestStateView pv = ledger.prefill().get(id).orElseThrow();
        assertEquals(9, pv.phaseOrdinal()); // PREFILL_DONE——P 账释放无中间窗口
        assertTrue(pv.trace().stream().anyMatch(s -> s.contains("→P9")), () -> "trace=" + pv.trace());

        // P 计数：PREFILL_DONE 人口 1
        ledger.prefill().refreshSnapshot();
        assertEquals(1L, ledger.prefill().snapshot().phaseCounts().get(9));
        assertEquals(1L, ledger.prefill().snapshot().inflight());

        // D 计数：C1 撤预占 + 引擎接管 + confirmed
        ledger.decode().refreshSnapshot();
        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(0L, ds.reservedKvTotal());
        assertEquals(4096L, ds.kvTokensReportedTotal());
        assertEquals(1L, ds.confirmed());
        assertEquals(1L, ds.activeTotal());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    @Test
    void dKvAllocatedShrinksReceivedPrefillSameTick() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 101L;

        // P 条目仅到 P_RECEIVED(5)：D 确认时闭包推进 5 → 9
        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RECEIVED, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));

        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 2_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 2L)));
        assertEquals(9, ledger.prefill().get(id).orElseThrow().phaseOrdinal());
        assertTrue(ledger.auditAndDrift().clean());
    }

    /** P_RUNNING 不收缩（边算边传重叠窗口）——crossSide KV_TRANSFERRING 推导（S9）。 */
    @Test
    void pRunningNotShrunkWhileTransferOverlap() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 102L;

        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));

        // D KV_ALLOCATED：P 条目在 P_RUNNING(8) → 不收缩（超出 [P_RECEIVED..P_WAITING_LOADED]）
        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 2_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 2L)));
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal());
        assertEquals(1, ledger.crossSide().kvTransferringCount());
        assertEquals(java.util.List.of(id), ledger.crossSide().kvTransferringRequestIds());

        // D 进入 D_RUNNING：传输仍在进行（P_RUNNING ∧ D 已报）
        ledger.observe(TestEndpoints.runningOnly(dEp, 3L, 2_010L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 512L, 3L)));
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal());
        assertEquals(1, ledger.crossSide().kvTransferringCount());

        // D finished(success)：F1 收缩 P → 条目两侧移除 → 传输重叠清零
        ledger.observe(TestEndpoints.finishedOnly(dEp, 4L, 2_020L,
                TestEndpoints.finished(id, StateRole.DECODE, 0, 2_020L, 4L)));
        assertTrue(ledger.prefill().get(id).isEmpty());
        assertTrue(ledger.decode().get(id).isEmpty());
        assertEquals(0, ledger.crossSide().kvTransferringCount());
    }

    // ---- 规则 2：F1 因果闭包 ----

    @Test
    void dFinishedSuccessCausallyClosesPrefill() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 103L;

        // P 条目 P_RUNNING（故意不 settle——等待 D 完成闭包）
        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        ledger.decode().onDispatched(id, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 2_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 512L, 2L)));

        // D finished(success) → 同 tick 收缩 P
        ledger.observe(TestEndpoints.finishedOnly(dEp, 3L, 2_020L,
                TestEndpoints.finished(id, StateRole.DECODE, 0, 2_020L, 3L)));
        assertTrue(ledger.decode().get(id).isEmpty());
        assertTrue(ledger.prefill().get(id).isEmpty());
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(1L, s.prefillTombstones());
        assertEquals(1L, s.decodeTombstones());
        assertTrue(ledger.auditAndDrift().clean());
    }

    /** D 失败不闭包收缩 P（P 侧由自身证据通道收尾）。 */
    @Test
    void dFinishedFailureLeavesPrefillOpen() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 104L;

        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        ledger.decode().onDispatched(id, dBinding);
        ledger.observe(TestEndpoints.finishedOnly(dEp, 2L, 2_000L,
                TestEndpoints.finished(id, StateRole.DECODE, 7, 2_000L, 2L)));

        assertTrue(ledger.decode().get(id).isEmpty()); // D FAILED 移除
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal()); // P 仍在
    }

    // ---- 规则 3：C1 临界点（超卖窗口修正）----

    @Test
    void c1DoubleBookWindowThenKvTakeover() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 200L;

        // reserve：D① 影子预占入账
        ledger.decode().reserve(id, 512L, 4096L, dBinding);
        ledger.decode().refreshSnapshot();
        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(4096L, ds.reservedKvTotal());
        assertEquals(4096L, ds.expectedKvTotal());
        assertEquals(0L, ds.kvTokensReportedTotal());

        // DISPATCHED 相位收到 RECEIVED（同相位无推进）：双记——预占保持 + 引擎已见
        assertTrue(ledger.decode().onDispatched(id, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RECEIVED, -1L, 0L, 1L)));
        DecodeRequestStateView dv = ledger.decode().get(id).orElseThrow();
        assertTrue(dv.engineOwned()); // 引擎已见（B 道事实）
        assertEquals(1, dv.phaseOrdinal()); // DISPATCHED 不动
        ledger.decode().refreshSnapshot();
        ds = ledger.decode().snapshot();
        assertEquals(4096L, ds.reservedKvTotal()); // 预占保持——修正超卖窗口的核心语义
        assertEquals(4096L, ds.expectedKvTotal());
        assertEquals(1L, ds.activeTotal());

        // KV_ALLOCATED：撤 D① 预占 + D② 引擎事实接管 + confirmed
        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 1_010L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 4096L, 2L)));
        dv = ledger.decode().get(id).orElseThrow();
        assertEquals(2, dv.phaseOrdinal());
        assertEquals(0L, dv.reservedKv()); // 预占撤出
        assertEquals(4096L, dv.reservedExpectedKv()); // 历史记录保留
        assertEquals(4096L, dv.kvTokensReported()); // 引擎值接管
        ledger.decode().refreshSnapshot();
        ds = ledger.decode().snapshot();
        assertEquals(0L, ds.reservedKvTotal());
        assertEquals(4096L, ds.expectedKvTotal());
        assertEquals(4096L, ds.kvTokensReportedTotal());
        assertEquals(1L, ds.confirmed());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /** E1：kvTokens=0（unknown）不更新引擎事实账。 */
    @Test
    void kvTokensZeroMeansUnknownAndKeepsZero() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 201L;

        ledger.decode().reserve(id, 64L, 512L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 0L, 1L)));
        DecodeRequestStateView dv = ledger.decode().get(id).orElseThrow();
        assertEquals(2, dv.phaseOrdinal());
        assertEquals(0L, dv.kvTokensReported()); // unknown 不更新
        ledger.decode().refreshSnapshot();
        assertEquals(0L, ledger.decode().snapshot().kvTokensReportedTotal());
    }

    // ---- 规则 4：cancel 双清（两侧计数独立归零）----

    @Test
    void prefillCancelSettleClearsBothSidesSameTick() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 300L;

        // 两侧都有活账：P P_RUNNING、D D_LOADING（已接管 KV）
        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 2_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 2L)));
        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        assertEquals(1L, ledger.prefill().snapshot().inflight());
        assertEquals(1L, ledger.decode().snapshot().activeTotal());
        assertEquals(512L, ledger.decode().snapshot().kvTokensReportedTotal());
        assertEquals(1L, ledger.decode().snapshot().confirmed());

        // P 侧 settle(CANCELLED)：同 tick 双清两侧
        assertTrue(ledger.prefill().settle(id, cancelOutcome(), SettleReason.LOCAL_CANCEL));
        assertTrue(ledger.prefill().get(id).isEmpty());
        assertTrue(ledger.decode().get(id).isEmpty());

        // 各自计数独立归零
        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        PrefillCounterSnapshot ps = ledger.prefill().snapshot();
        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(0L, ps.inflight());
        assertEquals(0L, ps.engineOwned());
        assertEquals(0L, ds.activeTotal());
        assertEquals(0L, ds.reservedKvTotal());
        assertEquals(0L, ds.kvTokensReportedTotal());
        assertEquals(0L, ds.confirmed());
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(1L, s.prefillTombstones());
        assertEquals(1L, s.decodeTombstones());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /** 对称：D 侧 settle(CANCELLED) 双清 P 侧。 */
    @Test
    void decodeCancelSettleClearsPrefillSide() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 301L;

        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);
        assertTrue(ledger.decode().onDispatched(id, dBinding));

        assertTrue(ledger.decode().settle(id, cancelOutcome(), SettleReason.LOCAL_CANCEL));
        assertTrue(ledger.decode().get(id).isEmpty());
        assertTrue(ledger.prefill().get(id).isEmpty());
        ledger.prefill().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight());
    }

    /** 非 CANCELLED 终局不触发双清（P settle COMPLETED 不影响 D 条目）。 */
    @Test
    void nonCancelSettleDoesNotCrossClear() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 302L;

        prefillAt(ledger, pEp, pBinding, id, EnginePhase.RUNNING, 1L);
        ledger.decode().reserve(id, 64L, 512L, dBinding);

        assertTrue(ledger.prefill().settle(id,
                new TerminalOutcome(TerminalState.COMPLETED, TerminalReason.SUCCEEDED, ""),
                SettleReason.ENGINE_FINISHED));
        assertTrue(ledger.prefill().get(id).isEmpty());
        // D 条目独立存活
        assertTrue(ledger.decode().get(id).isPresent());
        assertEquals(0, ledger.decode().get(id).orElseThrow().phaseOrdinal());
    }

    // ---- 批次影子视图（B6）----

    @Test
    void batchShadowViewMaxMinPhase() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 55L);
        // 批次 55：三条成员，相位分别为 QUEUED / P_WAITING_LOADED / P_RUNNING
        for (long id : new long[]{10L, 11L, 12L}) {
            ledger.prefill().register(id, 55L);
            ledger.prefill().onQueued(id);
            ledger.prefill().onDispatching(id, 55L);
            assertTrue(ledger.prefill().onDispatched(id, pBinding));
        }
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(11L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, 55L, 0L, 1L)));
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, 1_010L,
                TestEndpoints.running(12L, StateRole.PREFILL, EnginePhase.RUNNING, 55L, 0L, 2L)));

        BatchShadowView batch = ledger.prefill().batchView(55L);
        assertEquals(3, batch.members().size());
        assertEquals(8, batch.maxPhaseOrdinal()); // P_RUNNING
        assertEquals(4, batch.minPhaseOrdinal()); // DISPATCHED（最弱成员）
        assertTrue(batch.anyRunning());

        // 空批次
        BatchShadowView empty = ledger.prefill().batchView(999L);
        assertTrue(empty.isEmpty());
        assertEquals(-1, empty.maxPhaseOrdinal());
        assertEquals(-1, empty.minPhaseOrdinal());
    }
}
