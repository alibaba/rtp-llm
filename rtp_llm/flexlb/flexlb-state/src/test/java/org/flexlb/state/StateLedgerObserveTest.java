package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.flexlb.state.internal.FenceRegistry;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * StateLedger.observe 主链路：正常全生命周期（P/D 双侧）、乱序注入
 * （迟到中间态 DROP_LATE / 版本回退 DROP_DUP / 跨代 REJECT）、
 * 墓碑判重与迟到吸收、世代 failover 语义、fence→release 驱逐断言集成。
 */
class StateLedgerObserveTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    /** 终局取消 outcome（CANCELLED_ACK 通道）。 */
    private static TerminalOutcome cancelOutcome() {
        return new TerminalOutcome(TerminalState.CANCELLED, TerminalReason.CANCELLED_ACK, "");
    }

    // ---- 1. 正常全生命周期 ----

    @Test
    void fullLifecycleHappyPath() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, 77L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 100L;
        long t = 1_000L;

        // register → INIT(0)
        assertEquals(RegisterResult.OK, ledger.prefill().register(id, 77L));
        assertEquals(0, ledger.prefill().get(id).orElseThrow().phaseOrdinal());
        assertEquals("INIT", ledger.prefill().get(id).orElseThrow().phaseName());
        assertEquals(77L, ledger.prefill().get(id).orElseThrow().batchId());

        // queued → QUEUED(2)
        ledger.prefill().onQueued(id);
        assertEquals(2, ledger.prefill().get(id).orElseThrow().phaseOrdinal());

        // dispatching → DISPATCHING(3)
        ledger.prefill().onDispatching(id, 77L);
        assertEquals(3, ledger.prefill().get(id).orElseThrow().phaseOrdinal());

        // dispatched → DISPATCHED(4)，世代绑定生效
        assertTrue(ledger.prefill().onDispatched(id, pBinding));
        PrefillRequestStateView v = ledger.prefill().get(id).orElseThrow();
        assertEquals(4, v.phaseOrdinal());
        assertEquals(pBinding, v.binding());
        assertTrue(v.dispatchedAtMs() > 0);

        // DISPATCHED 后重派发拒绝且绑定不可变（setBindingOnce 语义）
        assertFalse(ledger.prefill().onDispatched(id, new GenerationTriple(9, 9, 9)));
        assertEquals(pBinding, ledger.prefill().get(id).orElseThrow().binding());

        // 引擎观察三相位：RECEIVED → P_RECEIVED(5)
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, t,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RECEIVED, 77L, 0L, 1L)));
        v = ledger.prefill().get(id).orElseThrow();
        assertEquals(5, v.phaseOrdinal());
        assertTrue(v.engineOwned());
        assertEquals(1L, v.lastVersion());

        // KV_ALLOCATED → P_WAITING_LOADED(7)，kvTokens 入引擎上报观察账
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, t + 10,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, 77L, 128L, 2L)));
        v = ledger.prefill().get(id).orElseThrow();
        assertEquals(7, v.phaseOrdinal());
        assertEquals(128L, v.kvTokensReported());
        assertEquals(2L, v.lastSeenRound());

        // RUNNING → P_RUNNING(8)
        ledger.observe(TestEndpoints.runningOnly(pEp, 3L, t + 20,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, 77L, 128L, 3L)));
        v = ledger.prefill().get(id).orElseThrow();
        assertEquals(8, v.phaseOrdinal());
        // trace 覆盖沿途相位（越级闭包补记）：至少可见 P8 进入记录
        assertTrue(v.trace().stream().anyMatch(s -> s.contains("→P8")), "trace=" + v.trace());
        // 快照：P_RUNNING 人口 1、engineOwned 1
        ledger.prefill().refreshSnapshot();
        PrefillCounterSnapshot ps = ledger.prefill().snapshot();
        assertEquals(1L, ps.phaseCounts().get(8));
        assertEquals(1L, ps.engineOwned());
        assertEquals(1L, ps.inflight());

        // P finished(success) → COMPLETED：移除 + 墓碑，P 账归零
        ledger.observe(TestEndpoints.finishedOnly(pEp, 4L, t + 40,
                TestEndpoints.finished(id, StateRole.PREFILL, 0, t + 40, 4L)));
        assertTrue(ledger.prefill().get(id).isEmpty());
        ledger.prefill().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight());
        assertEquals(1L, ledger.snapshot().prefillTombstones());

        // D reserve → RESERVED(0)，影子预占双轨入账
        assertEquals(ReserveResult.OK, ledger.decode().reserve(id, 512L, 4096L, dBinding));
        DecodeRequestStateView dv = ledger.decode().get(id).orElseThrow();
        assertEquals(0, dv.phaseOrdinal());
        assertEquals(4096L, dv.reservedKv());
        assertEquals(4096L, dv.reservedExpectedKv());
        assertEquals(dBinding, dv.binding());
        ledger.decode().refreshSnapshot();
        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(1L, ds.activeTotal());
        assertEquals(4096L, ds.reservedKvTotal());
        assertEquals(4096L, ds.expectedKvTotal());

        // D dispatched → DISPATCHED(1)
        assertTrue(ledger.decode().onDispatched(id, dBinding));
        assertEquals(1, ledger.decode().get(id).orElseThrow().phaseOrdinal());

        // D KV_ALLOCATED → D_LOADING(2)：撤预占 + 引擎事实接管 + confirmed
        ledger.observe(TestEndpoints.runningOnly(dEp, 5L, t + 60,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 4096L, 5L)));
        assertEquals(2, ledger.decode().get(id).orElseThrow().phaseOrdinal());
        ledger.decode().refreshSnapshot();
        ds = ledger.decode().snapshot();
        assertEquals(0L, ds.reservedKvTotal());
        assertEquals(4096L, ds.kvTokensReportedTotal());
        assertEquals(1L, ds.confirmed());

        // D RUNNING → D_RUNNING(3)
        ledger.observe(TestEndpoints.runningOnly(dEp, 6L, t + 80,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 4096L, 6L)));
        assertEquals(3, ledger.decode().get(id).orElseThrow().phaseOrdinal());

        // D finished(success) → COMPLETED：移除 + 墓碑
        ledger.observe(TestEndpoints.finishedOnly(dEp, 7L, t + 100,
                TestEndpoints.finished(id, StateRole.DECODE, 0, t + 100, 7L)));
        assertTrue(ledger.decode().get(id).isEmpty());
        ledger.decode().refreshSnapshot();
        assertEquals(0L, ledger.decode().snapshot().activeTotal());
        assertEquals(0L, ledger.decode().snapshot().kvTokensReportedTotal());
        assertEquals(1L, ledger.snapshot().decodeTombstones());

        // 全链路后账面干净：对账无 drift，无 unknown/迟到
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(0L, s.unknownRunningEvents());
        assertEquals(0L, s.unknownFinishedEvents());
        assertEquals(0L, s.lateEventsAbsorbed());
    }

    /** 正交取消意图标记：只标记不改相位，终局仍走 settle。 */
    @Test
    void markPendingCancelIsOrthogonalIntent() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        assertEquals(RegisterResult.OK, ledger.prefill().register(1L, -1L));
        ledger.prefill().onQueued(1L);
        ledger.prefill().markPendingCancel(1L);
        PrefillRequestStateView v = ledger.prefill().get(1L).orElseThrow();
        assertTrue(v.pendingCancel());
        assertEquals(2, v.phaseOrdinal()); // 相位不受取消意图影响
        // 不存在的条目 no-op
        ledger.prefill().markPendingCancel(999L);
        assertTrue(ledger.prefill().get(999L).isEmpty());
        assertNotNull(pGen);
    }

    // ---- 3. 乱序注入 ----

    /** 迟到中间态：D_RUNNING 后到达的 RECEIVED 观察 → DROP_LATE 计数，相位不动。 */
    @Test
    void lateIntermediateObservationDropped() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        long id = 7L;
        ledger.decode().reserve(id, 100L, 1024L, dBinding);
        ledger.decode().onDispatched(id, dBinding);
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 1024L, 1L)));
        ledger.observe(TestEndpoints.runningOnly(dEp, 2L, 1_010L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, 1024L, 2L)));
        assertEquals(3, ledger.decode().get(id).orElseThrow().phaseOrdinal());

        // 迟到 RECEIVED（映射 DISPATCHED(1) < D_RUNNING(3)）→ DROP_LATE
        ledger.observe(TestEndpoints.runningOnly(dEp, 3L, 1_020L,
                TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RECEIVED, -1L, 0L, 3L)));
        assertEquals(3, ledger.decode().get(id).orElseThrow().phaseOrdinal());
        assertEquals(1L, ledger.snapshot().verdictCounts().get(PhaseVerdict.DROP_LATE));
        // 相位丢弃但观察账新鲜（版本更高）：engineOwned 已是 true，round 推进
        DecodeRequestStateView v = ledger.decode().get(id).orElseThrow();
        assertEquals(3L, v.lastSeenRound());
    }

    /** 版本回退：eventVersion < lastVersion → DROP_DUP，相位与观察账都不动。 */
    @Test
    void staleVersionObservationDroppedAsDup() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);
        long id = 8L;
        ledger.prefill().register(id, -1L);
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, -1L);
        ledger.prefill().onDispatched(id, pBinding);
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 64L, 5L)));
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal());

        // 版本回退（v3 < v5）→ DROP_DUP；版本屏障先于相位比较
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, 1_010L,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RECEIVED, -1L, 999L, 3L)));
        PrefillRequestStateView v = ledger.prefill().get(id).orElseThrow();
        assertEquals(8, v.phaseOrdinal());
        assertEquals(64L, v.kvTokensReported()); // 陈旧数据不覆盖
        assertEquals(1L, v.lastSeenRound());
        assertEquals(1L, ledger.snapshot().verdictCounts().get(PhaseVerdict.DROP_DUP));
    }

    /** 条目级跨代 REJECT：failover 后旧代 binding 条目拒绝新代事件。 */
    @Test
    void crossGenerationEventRejectedAtEntryLevel() {
        StateLedger ledger = new StateLedger();
        long g1 = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint ep1 = TestEndpoints.ep(1L, StateRole.PREFILL, g1);
        GenerationTriple binding1 = new GenerationTriple(1, g1, -1L);
        long id = 9L;
        ledger.prefill().register(id, -1L);
        ledger.prefill().onQueued(id);
        ledger.prefill().onDispatching(id, -1L);
        ledger.prefill().onDispatched(id, binding1);
        ledger.observe(TestEndpoints.runningOnly(ep1, 1L, 1_000L,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 32L, 1L)));
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal());

        // failover 换代 → 新代报文到达旧代条目
        long g2 = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint ep2 = TestEndpoints.ep(1L, StateRole.PREFILL, g2);
        ledger.observe(TestEndpoints.runningOnly(ep2, 2L, 1_010L,
                TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 32L, 2L)));
        // 条目 binding 是 g1，报文 g2 → REJECT_GENERATION（相位不动）
        assertEquals(8, ledger.prefill().get(id).orElseThrow().phaseOrdinal());
        assertEquals(1L, ledger.snapshot().verdictCounts().get(PhaseVerdict.REJECT_GENERATION));
        assertEquals(1L, ledger.snapshot().crossGenerationRejects());
    }

    /** 整报级旧代拒绝：报文世代 ≠ 端点当前登记代 → 整报丢弃（未到条目仲裁）。 */
    @Test
    void wholeOldGenerationReportRejected() {
        StateLedger ledger = new StateLedger();
        long g1 = ledger.newGeneration(P_EP0);
        ledger.newGeneration(P_EP0); // g2 已登记，g1 成为旧代
        TestEndpoints.Endpoint oldEp = TestEndpoints.ep(1L, StateRole.PREFILL, g1);
        ledger.observe(TestEndpoints.runningOnly(oldEp, 1L, 1_000L,
                TestEndpoints.running(42L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 0L, 1L)));
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(1L, s.crossGenerationRejects());
        // 未到条目仲裁：无 accept 计数、无 unknown 计数（verdictCounts 为全 key 枚举 map，零值语义）
        assertEquals(0L, s.verdictCounts().get(PhaseVerdict.ACCEPT_ADVANCE));
        assertEquals(0L, s.verdictCounts().get(PhaseVerdict.REJECT_GENERATION));
        assertEquals(0L, s.unknownRunningEvents());
    }

    /** failover 语义：旧代条目事件 REJECT，但新代 binding 的新条目正常工作。 */
    @Test
    void failoverAllowsNewGenerationEntriesWhileOldRejected() {
        StateLedger ledger = new StateLedger();
        long g1 = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint ep1 = TestEndpoints.ep(1L, StateRole.PREFILL, g1);
        // 旧代条目 A
        GenerationTriple bindingA = new GenerationTriple(1, g1, -1L);
        ledger.prefill().register(1L, -1L);
        ledger.prefill().onQueued(1L);
        ledger.prefill().onDispatching(1L, -1L);
        ledger.prefill().onDispatched(1L, bindingA);

        // failover → 新代
        long g2 = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint ep2 = TestEndpoints.ep(1L, StateRole.PREFILL, g2);
        // 新代条目 B（新代 binding）正常开户
        GenerationTriple bindingB = new GenerationTriple(1, g2, -1L);
        ledger.prefill().register(2L, -1L);
        ledger.prefill().onQueued(2L);
        ledger.prefill().onDispatching(2L, -1L);
        assertTrue(ledger.prefill().onDispatched(2L, bindingB));

        // 新代报文：B 接受推进（P_RUNNING）
        ledger.observe(TestEndpoints.runningOnly(ep2, 1L, 1_000L,
                TestEndpoints.running(2L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 16L, 1L)));
        assertEquals(8, ledger.prefill().get(2L).orElseThrow().phaseOrdinal());

        // 同报文对 A（旧代 binding）：REJECT_GENERATION
        ledger.observe(TestEndpoints.runningOnly(ep2, 2L, 1_010L,
                TestEndpoints.running(1L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 16L, 2L)));
        assertEquals(4, ledger.prefill().get(1L).orElseThrow().phaseOrdinal()); // 不动
        assertEquals(1L, ledger.snapshot().verdictCounts().get(PhaseVerdict.REJECT_GENERATION));
    }

    // ---- 4. 墓碑（ledger 集成部分；组件级见 internal.TombstoneStoreTest）----

    /** 判重窗口内重复登记 → DUPLICATE_TOMBSTONE；迟到 finished/cancel 被吸收计数。 */
    @Test
    void tombstoneRejectsReRegisterAndAbsorbsLateEvents() {
        StateLedger ledger = new StateLedger();
        assertEquals(RegisterResult.OK, ledger.prefill().register(5L, -1L));
        assertTrue(ledger.prefill().settle(5L,
                new TerminalOutcome(TerminalState.COMPLETED, TerminalReason.SUCCEEDED, ""),
                SettleReason.ENGINE_FINISHED));

        // 存活→重复登记拒绝
        assertEquals(RegisterResult.DUPLICATE_TOMBSTONE, ledger.prefill().register(5L, -1L));

        // 迟到 finished 被墓碑吸收（事件入口路径）
        long pGen = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        ledger.observe(TestEndpoints.finishedOnly(pEp, 1L, 1_000L,
                TestEndpoints.finished(5L, StateRole.PREFILL, 0, 1_000L, 9L)));
        assertEquals(1L, ledger.snapshot().lateEventsAbsorbed());

        // 迟到 cancel 被墓碑吸收（settle 入口路径）
        assertFalse(ledger.prefill().settle(5L, cancelOutcome(), SettleReason.LOCAL_CANCEL));
        assertEquals(1L, ledger.snapshot().lateCancelsAbsorbed());
    }

    /** 墓碑过期后允许重新登记（retention=0 → 立即过期；janitor evict 清库存）。 */
    @Test
    void expiredTombstoneAllowsReRegister() {
        StateLedger ledger = new StateLedger(new StateLedgerConfig(0L, 300_000L, 64));
        assertEquals(RegisterResult.OK, ledger.prefill().register(6L, -1L));
        assertTrue(ledger.prefill().settle(6L, cancelOutcome(), SettleReason.LOCAL_CANCEL));
        assertEquals(1L, ledger.snapshot().prefillTombstones()); // 存量仍在（未清）

        // retention=0：窗口已过，判重放行
        assertEquals(RegisterResult.OK, ledger.prefill().register(6L, -1L));

        // janitor 维护 tick 清空墓碑库存
        ledger.createJanitor(LedgerJanitorConfig.defaults()).runMaintenanceTick();
        assertEquals(0L, ledger.snapshot().prefillTombstones());
    }

    // ---- 6. fence 驱逐断言（ledger 集成：fenced 条目 release 拒绝）----

    @Test
    void fencedDecodeEntryCannotBeReleased() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        assertEquals(ReserveResult.OK, ledger.decode().reserve(11L, 64L, 256L, dBinding));

        // fence 登记后驱逐断言拒绝（fence 防线）
        ledger.fences().fence("cancel-flow", 11L, FenceRegistry.FenceType.CANCEL);
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> ledger.decode().release(11L));
        assertTrue(ex.getMessage().contains("11"), ex.getMessage());

        // fence 过期或解除后可正常释放；释放不进墓碑 → 可重新 reserve
        ledger.fences().unfence(11L);
        assertTrue(ledger.decode().release(11L));
        assertTrue(ledger.decode().get(11L).isEmpty());
        assertEquals(0L, ledger.snapshot().decodeTombstones());
        assertEquals(ReserveResult.OK, ledger.decode().reserve(11L, 64L, 256L, dBinding));
    }

    /** 未知条目（非 rebuild 路径、非墓碑）的引擎事件计入 unknown 观测。 */
    @Test
    void unknownEventsCountedOutsideRebuild() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(77L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 0L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 2L, 1_010L,
                TestEndpoints.finished(77L, StateRole.PREFILL, 0, 1_010L, 2L)));
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(1L, s.unknownRunningEvents());
        assertEquals(1L, s.unknownFinishedEvents());
    }

    /** D 侧重复 reserve 判重。 */
    @Test
    void decodeReserveDuplicateRejected() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);
        assertEquals(ReserveResult.OK, ledger.decode().reserve(3L, 10L, 100L, dBinding));
        assertEquals(ReserveResult.DUPLICATE_ALIVE, ledger.decode().reserve(3L, 10L, 100L, dBinding));
        assertTrue(ledger.decode().settle(3L, cancelOutcome(), SettleReason.LOCAL_CANCEL));
        assertEquals(ReserveResult.DUPLICATE_TOMBSTONE, ledger.decode().reserve(3L, 10L, 100L, dBinding));
        List.of(dGen); // 引用避免 unused 提示
    }
}
