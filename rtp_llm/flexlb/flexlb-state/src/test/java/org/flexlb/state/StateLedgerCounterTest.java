package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 派生计数器：零锁快照发布语义（volatile 滞后 + refresh 精确）、
 * 快照与全量重算一致性、auditAndDrift 无 drift。
 * （单写者 ArchUnit 强制见 ArchitectureDependencyTest。）
 */
class StateLedgerCounterTest {

    private static final TestEndpoints.Endpoint P_EP0 = TestEndpoints.ep(1L, StateRole.PREFILL, 0L);
    private static final TestEndpoints.Endpoint D_EP0 = TestEndpoints.ep(2L, StateRole.DECODE, 0L);

    /** 快照发布间隔语义：interval=64 时少量转换后 published 滞后，refresh 后精确。 */
    @Test
    void publishedSnapshotIsLazyUntilRefresh() {
        StateLedger ledger = new StateLedger(); // defaults: interval 64
        assertEquals(RegisterResult.OK, ledger.prefill().register(1L, -1L));
        ledger.prefill().onQueued(1L);

        // 1 次转换 < 64：published 快照仍是初始零值（零锁弱一致读）
        assertEquals(0L, ledger.prefill().snapshot().inflight());

        // 显式 refresh 后精确
        ledger.prefill().refreshSnapshot();
        PrefillCounterSnapshot ps = ledger.prefill().snapshot();
        assertEquals(1L, ps.inflight());
        assertEquals(1L, ps.phaseCounts().get(2)); // QUEUED
        assertTrue(ledger.auditAndDrift().clean());
    }

    /** interval=1：每次转换都发布——快照始终与账同步。 */
    @Test
    void everyTransitionPublishesWhenIntervalIsOne() {
        StateLedger ledger = new StateLedger(new StateLedgerConfig(60_000L, 300_000L, 1));
        assertEquals(RegisterResult.OK, ledger.prefill().register(1L, -1L));
        ledger.prefill().onQueued(1L);
        assertEquals(1L, ledger.prefill().snapshot().phaseCounts().get(2));
        ledger.prefill().onDispatching(1L, -1L);
        assertEquals(1L, ledger.prefill().snapshot().phaseCounts().get(3));
        assertEquals(0L, ledger.prefill().snapshot().phaseCounts().get(2));
    }

    /** 混合转换后：已发布快照与全量重算逐项一致（P/D 两侧）。 */
    @Test
    void snapshotMatchesFullRecountAfterMixedTransitions() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // 3 个 P 条目：DISPATCHED（无引擎观察）/ P_WAITING_LOADED / P_RUNNING（engineOwned 2 条）
        for (long id : new long[]{1L, 2L, 3L}) {
            assertEquals(RegisterResult.OK, ledger.prefill().register(id, -1L));
            ledger.prefill().onQueued(id);
            ledger.prefill().onDispatching(id, -1L);
            assertTrue(ledger.prefill().onDispatched(id, pBinding));
        }
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(2L, StateRole.PREFILL, EnginePhase.KV_ALLOCATED, -1L, 100L, 1L)));
        ledger.observe(TestEndpoints.runningOnly(pEp, 2L, 1_010L,
                TestEndpoints.running(3L, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 200L, 2L)));

        // 2 个 D 条目：RESERVED（预占 512）/ D_LOADING（预占 0、引擎 1024、confirmed）
        assertEquals(ReserveResult.OK, ledger.decode().reserve(2L, 128L, 512L, dBinding));
        assertEquals(ReserveResult.OK, ledger.decode().reserve(3L, 128L, 1024L, dBinding));
        assertTrue(ledger.decode().onDispatched(3L, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 3L, 1_020L,
                TestEndpoints.running(3L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 1024L, 3L)));

        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        PrefillCounterSnapshot ps = ledger.prefill().snapshot();
        assertEquals(3L, ps.inflight());
        assertEquals(1L, ps.phaseCounts().get(4));  // DISPATCHED（无引擎观察）
        assertEquals(1L, ps.phaseCounts().get(7));  // P_WAITING_LOADED
        assertEquals(1L, ps.phaseCounts().get(8));  // P_RUNNING
        assertEquals(2L, ps.engineOwned());
        assertEquals(0L, ps.dispatching());

        DecodeCounterSnapshot ds = ledger.decode().snapshot();
        assertEquals(2L, ds.activeTotal());
        assertEquals(1L, ds.phaseCounts().get(0)); // RESERVED
        assertEquals(1L, ds.phaseCounts().get(2)); // D_LOADING
        assertEquals(512L, ds.reservedKvTotal());  // 仅 RESERVED 条目仍占
        assertEquals(1536L, ds.expectedKvTotal()); // 512 + 1024 历史合计
        assertEquals(1024L, ds.kvTokensReportedTotal());
        assertEquals(1L, ds.confirmed());

        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /** 全生命周期收敛后对账仍干净（终局归位无残留）。 */
    @Test
    void auditAndDriftCleanAfterFullLifecycle() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(1L, StateRole.PREFILL, pGen);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        for (long id = 1L; id <= 5L; id++) {
            assertEquals(RegisterResult.OK, ledger.prefill().register(id, -1L));
            ledger.prefill().onQueued(id);
            ledger.prefill().onDispatching(id, -1L);
            assertTrue(ledger.prefill().onDispatched(id, pBinding));
            ledger.observe(TestEndpoints.runningOnly(pEp, id, 1_000L + id,
                    TestEndpoints.running(id, StateRole.PREFILL, EnginePhase.RUNNING, -1L, 64L, id)));
            assertEquals(ReserveResult.OK, ledger.decode().reserve(id, 64L, 256L, dBinding));
            assertTrue(ledger.decode().onDispatched(id, dBinding));
            ledger.observe(TestEndpoints.runningOnly(dEp, id + 10, 2_000L + id,
                    TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 256L, id)));
            ledger.observe(TestEndpoints.finishedOnly(dEp, id + 20, 3_000L + id,
                    TestEndpoints.finished(id, StateRole.DECODE, 0, 3_000L + id, id)));
        }
        ledger.prefill().refreshSnapshot();
        ledger.decode().refreshSnapshot();
        assertEquals(0L, ledger.prefill().snapshot().inflight());
        assertEquals(0L, ledger.decode().snapshot().activeTotal());
        // 计数守恒：终态数 == 墓碑数
        LedgerSnapshot s = ledger.snapshot();
        assertEquals(5L, s.prefillTombstones());
        assertEquals(5L, s.decodeTombstones());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /**
     * per-EP 派生计数（读取换权阶段 G4 调度读数数据源）：D 侧按端点聚合活跃条目，
     * 未确认双轨口径（unconfirmed = phase < D_LOADING 的影子预占）；KV_ALLOCATED
     * 确认后逐条撤出 unconfirmed、引擎事实 KV 接管；终局后计数归零。
     */
    @Test
    void decodeEndpointCountersTrackUnconfirmedDualTrackPerEndpoint() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(D_EP0);
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // 51：RESERVED（seqLen=100, expectedKv=200——影子预占在账）
        // 52：DISPATCHED → D_LOADING 并 KV 确认（预占撤出、引擎事实 KV=1024 接管）
        assertEquals(ReserveResult.OK, ledger.decode().reserve(51L, 100L, 200L, dBinding));
        assertEquals(ReserveResult.OK, ledger.decode().reserve(52L, 300L, 400L, dBinding));
        assertTrue(ledger.decode().onDispatched(52L, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(52L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 1024L, 1L)));

        DecodeEndpointCounters c = ledger.decode().endpointCounters(2);
        assertEquals(2, c.activeTotal(), "端点名下两活跃条目");
        assertEquals(1, c.unconfirmedCount(), "仅 RESERVED 条目计 unconfirmed（52 已 D_LOADING）");
        assertEquals(200L, c.unconfirmedExpectedKv(), "unconfirmed Σ reservedKv（仅 51）");
        assertEquals(100L, c.unconfirmedSeqKv(), "unconfirmed Σ seqLen（仅 51）");
        assertEquals(1024L, c.kvTokensReportedTotal(), "引擎事实 KV 合计（52 确认接管）");
        assertEquals(1, c.engineOwnedCount(), "52 引擎已观察");
        assertEquals(1L, c.phaseCounts().get(0)); // 51 RESERVED
        assertEquals(1L, c.phaseCounts().get(2)); // 52 D_LOADING

        // 异端点隔离：不存在条目的端点返回全零视图
        assertEquals(0, ledger.decode().endpointCounters(999).activeTotal());

        // 终局后归零
        ledger.observe(TestEndpoints.finishedOnly(dEp, 2L, 1_100L,
                TestEndpoints.finished(51L, StateRole.DECODE, 0, 1_100L, 2L)));
        ledger.observe(TestEndpoints.finishedOnly(dEp, 3L, 1_200L,
                TestEndpoints.finished(52L, StateRole.DECODE, 0, 1_200L, 3L)));
        assertEquals(0, ledger.decode().endpointCounters(2).activeTotal(), "终局后 per-EP 计数归零");
    }

    /** P 侧 per-EP 派生计数：仅已派发（onDispatched 绑定世代）条目进入端点索引。 */
    @Test
    void prefillEndpointCountersOnlyCoverDispatchedEntries() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(P_EP0);
        GenerationTriple pBinding = new GenerationTriple(1, pGen, -1L);

        assertEquals(RegisterResult.OK, ledger.prefill().register(61L, -1L));
        ledger.prefill().onQueued(61L);
        assertEquals(0, ledger.prefill().endpointCounters(1).activeTotal(),
                "排队窗口条目（UNBOUND）不在端点索引");

        ledger.prefill().onDispatching(61L, -1L);
        assertTrue(ledger.prefill().onDispatched(61L, pBinding));

        PrefillEndpointCounters pc = ledger.prefill().endpointCounters(1);
        assertEquals(1, pc.activeTotal(), "派发后进入端点索引");
        assertEquals(1L, pc.phaseCounts().get(4)); // DISPATCHED
        assertEquals(0, pc.engineOwnedCount());
    }
}
