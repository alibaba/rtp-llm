package org.flexlb.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateRole;
import org.junit.jupiter.api.Test;

/**
 * 端点级派生计数簿一致性：增量账（状态转换 CAS 胜者临界区内 O(1) 增减）
 * vs 独立全量重算——读取换权阶段调度读数的 O(1) 数据源不许漂移。
 *
 * <p>主断言复用 auditAndDrift 的端点级对账（桶增量账 vs 按已绑定活跃
 * 条目全量重算，不静默修正）；并发用例另用测试内独立重算做交叉验证
 * （期望值在测试内按请求规则直接累加，不经过被测代码的聚合路径）。
 * 覆盖三类场景：并发混合生命周期（含派发前重绑 / 主动释放 / 确认临界 /
 * 引擎事实 KV 连续增量）、重绑桶间全账迁移、主动释放归位。</p>
 */
class StateLedgerPerEndpointCounterTest {

    /**
     * 并发混合生命周期：多端点并发跑"预约（部分派发前重绑）→ 确认临界 →
     * 引擎事实 KV 连续增量 → 终局 / 主动释放 / 留活跃"全谱系后，
     * 桶增量账与全量重算一致（auditAndDrift 端点级对账零偏差），
     * 且端点读数与测试内独立重算逐字段相等；并发读者持续做 O(1) 快照读
     * 无异常、无负值。
     */
    @Test
    void concurrentMixedLifecyclesKeepPerEndpointBooksConsistent() throws Exception {
        StateLedgerConfig cfg = new StateLedgerConfig(60_000L, 300_000L, 8);
        StateLedger ledger = new StateLedger(cfg);
        int[] pEpIds = {1, 3, 5};
        int[] dEpIds = {2, 4, 6};
        long[] pGens = new long[3];
        long[] dGens = new long[3];
        for (int k = 0; k < 3; k++) {
            pGens[k] = ledger.newGeneration(TestEndpoints.ep(pEpIds[k], StateRole.PREFILL, 0L));
            dGens[k] = ledger.newGeneration(TestEndpoints.ep(dEpIds[k], StateRole.DECODE, 0L));
        }
        TestEndpoints.Endpoint[] pEps = {
                TestEndpoints.ep(1L, StateRole.PREFILL, pGens[0]),
                TestEndpoints.ep(3L, StateRole.PREFILL, pGens[1]),
                TestEndpoints.ep(5L, StateRole.PREFILL, pGens[2])};
        TestEndpoints.Endpoint[] dEps = {
                TestEndpoints.ep(2L, StateRole.DECODE, dGens[0]),
                TestEndpoints.ep(4L, StateRole.DECODE, dGens[1]),
                TestEndpoints.ep(6L, StateRole.DECODE, dGens[2])};

        int n = 120;
        int threads = 8;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>(2 * n);
        // 并发读者（模拟调度热路径的 O(1) 快照读）：无异常、无非负性破坏
        CountDownLatch stopReader = new CountDownLatch(1);
        Future<?> reader = pool.submit(() -> {
            while (stopReader.getCount() > 0) {
                for (int k = 0; k < 3; k++) {
                    DecodeEndpointCounters dc = ledger.decode().endpointCounters(dEpIds[k]);
                    assertTrue(dc.activeTotal() >= 0);
                    assertTrue(dc.unconfirmedCount() >= 0);
                    assertTrue(dc.unconfirmedExpectedKv() >= 0);
                    assertTrue(dc.unconfirmedSeqKv() >= 0);
                    assertTrue(dc.kvTokensReportedTotal() >= 0);
                    PrefillEndpointCounters pc = ledger.prefill().endpointCounters(pEpIds[k]);
                    assertTrue(pc.activeTotal() >= 0);
                    assertTrue(pc.engineOwnedCount() >= 0);
                    assertTrue(pc.estimatedWaitMs() >= 0);
                }
            }
            return null;
        });

        for (int i = 0; i < n; i++) {
            final int idx = i;
            // D 侧请求 [5000, 5000+n)：初始端点槽 = idx%3；idx%3==0 的请求在派发时
            // 重绑到相邻槽（预约后、派发前的换端点窗口）；结局按 idx%3 三分——
            // 0 终局、1 主动释放、2 留活跃（引擎事实 KV 再涨一次）
            futures.add(pool.submit(() -> {
                start.await();
                int initialSlot = idx % 3;
                int finalSlot = (idx % 3 == 0) ? 1 : initialSlot;
                long id = 5_000L + idx;
                long seqLen = 64L + (idx % 4) * 16L;
                long expectedKv = 256L + (idx % 4) * 64L;
                long kv1 = 1_024L + (idx % 4) * 100L;
                long kv2 = kv1 + 256L;
                long kv3 = kv2 + 128L;
                GenerationTriple initialBinding = new GenerationTriple(dEpIds[initialSlot], dGens[initialSlot], -1L);
                GenerationTriple finalBinding = new GenerationTriple(dEpIds[finalSlot], dGens[finalSlot], -1L);
                TestEndpoints.Endpoint dEp = dEps[finalSlot];
                assertEquals(ReserveResult.OK, ledger.decode().reserve(id, seqLen, expectedKv, initialBinding));
                assertTrue(ledger.decode().onDispatched(id, finalBinding));
                ledger.observe(TestEndpoints.runningOnly(dEp, id, 1_000L + id,
                        TestEndpoints.running(id, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, kv1, 1L)));
                ledger.observe(TestEndpoints.runningOnly(dEp, id, 2_000L + id,
                        TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, kv2, 2L)));
                int outcome = idx % 3;
                if (outcome == 0) {
                    ledger.observe(TestEndpoints.finishedOnly(dEp, id, 3_000L + id,
                            TestEndpoints.finished(id, StateRole.DECODE, 0, 3_000L + id, 3L)));
                } else if (outcome == 1) {
                    assertTrue(ledger.decode().release(id));
                } else {
                    ledger.observe(TestEndpoints.runningOnly(dEp, id, 4_000L + id,
                            TestEndpoints.running(id, StateRole.DECODE, EnginePhase.RUNNING, -1L, kv3, 3L)));
                }
                return null;
            }));
            // P 侧请求 [1000, 1000+n)：派发绑定后引擎观察；槽 0 终局、槽 1/2 留活跃
            // （槽 2 观察停在 KV 装载位，槽 1 推进到执行位）
            futures.add(pool.submit(() -> {
                start.await();
                int slot = idx % 3;
                long id = 1_000L + idx;
                long batch = 700L + idx;
                long shareMs = 40L + (idx % 5) * 20L; // 分摊批次预测耗时（40~120 五档）
                GenerationTriple pBinding = new GenerationTriple(pEpIds[slot], pGens[slot], batch);
                TestEndpoints.Endpoint pEp = pEps[slot];
                assertEquals(RegisterResult.OK, ledger.prefill().register(id, batch));
                ledger.prefill().onQueued(id);
                ledger.prefill().onDispatching(id, batch);
                // 生产时序：派发流水线在 dispatch 绑定前记录分摊预测耗时
                ledger.prefill().notePredictedBatchMs(id, shareMs);
                assertTrue(ledger.prefill().onDispatched(id, pBinding));
                EnginePhase observed = (slot == 2) ? EnginePhase.KV_ALLOCATED : EnginePhase.RUNNING;
                ledger.observe(TestEndpoints.runningOnly(pEp, id, 1_000L + id,
                        TestEndpoints.running(id, StateRole.PREFILL, observed, batch, 128L, 1L)));
                if (slot == 0) {
                    ledger.observe(TestEndpoints.finishedOnly(pEp, id, 2_000L + id,
                            TestEndpoints.finished(id, StateRole.PREFILL, 0, 2_000L + id, 2L)));
                }
                return null;
            }));
        }
        start.countDown();
        for (Future<?> f : futures) {
            f.get(60L, TimeUnit.SECONDS);
        }
        stopReader.countDown();
        reader.get(10L, TimeUnit.SECONDS);
        pool.shutdown();
        assertTrue(pool.awaitTermination(10L, TimeUnit.SECONDS));

        // 主断言：增量账 vs 全量重算（含端点级对账）零偏差
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());

        // 测试内独立重算交叉验证（期望值按请求规则直接累加）
        // D 侧留活跃 = idx%3==2（全部落在槽 2 / 端点 6，确认后停执行位）
        int activePerSlot = n / 3;
        long expectedKvTotal = 0L;
        long expectedUnconfirmedSeq = 0L;
        long expectedUnconfirmedKv = 0L;
        for (int i = 0; i < n; i++) {
            if (i % 3 == 2) {
                expectedKvTotal += 1_024L + (i % 4) * 100L + 256L + 128L;
            }
        }
        // 留活跃条目全部过确认临界——未确认双轨为零
        assertEquals(0L, expectedUnconfirmedSeq);
        assertEquals(0L, expectedUnconfirmedKv);

        DecodeEndpointCounters dSlot2 = ledger.decode().endpointCounters(6);
        assertEquals(activePerSlot, dSlot2.activeTotal(), "留活跃条目全在重绑目标端点");
        assertEquals(0, dSlot2.unconfirmedCount(), "全部过确认临界——未确认清零");
        assertEquals(0L, dSlot2.unconfirmedExpectedKv());
        assertEquals(0L, dSlot2.unconfirmedSeqKv());
        assertEquals(activePerSlot, dSlot2.engineOwnedCount());
        assertEquals(expectedKvTotal, dSlot2.kvTokensReportedTotal(), "引擎事实 KV 三连增量合计");
        assertEquals(activePerSlot, dSlot2.phaseCounts().get(3)); // D_RUNNING
        // 重绑源端点（槽 0 全部迁走）与终局/释放清空的槽 1 均归零
        assertEquals(0, ledger.decode().endpointCounters(2).activeTotal());
        assertEquals(0, ledger.decode().endpointCounters(4).activeTotal());
        // 端点合计与全局账交叉一致
        int dSum = ledger.decode().endpointCounters(2).activeTotal()
                + ledger.decode().endpointCounters(4).activeTotal()
                + dSlot2.activeTotal();
        assertEquals(n / 3, dSum);
        ledger.decode().refreshSnapshot();
        assertEquals(n / 3, ledger.decode().snapshot().activeTotal());

        // P 侧：槽 0 终局归零；槽 1 留活跃在执行位（引擎已观察）；
        // 槽 2 留活跃停在 KV 装载位（引擎已观察）
        assertEquals(0, ledger.prefill().endpointCounters(1).activeTotal());
        assertEquals(0L, ledger.prefill().endpointCounters(1).estimatedWaitMs(), "终局桶预测耗时随条目移除归零");
        long expectedWaitSlot1 = 0L;
        long expectedWaitSlot2 = 0L;
        for (int i = 0; i < n; i++) {
            long shareMs = 40L + (i % 5) * 20L;
            if (i % 3 == 1) {
                expectedWaitSlot1 += shareMs;
            } else if (i % 3 == 2) {
                expectedWaitSlot2 += shareMs;
            }
        }
        PrefillEndpointCounters pSlot1 = ledger.prefill().endpointCounters(3);
        assertEquals(activePerSlot, pSlot1.activeTotal());
        assertEquals(activePerSlot, pSlot1.engineOwnedCount());
        assertEquals(activePerSlot, pSlot1.phaseCounts().get(8)); // P_RUNNING
        assertEquals(expectedWaitSlot1, pSlot1.estimatedWaitMs(), "留活跃条目分摊预测耗时——测试内独立重算");
        PrefillEndpointCounters pSlot2 = ledger.prefill().endpointCounters(5);
        assertEquals(activePerSlot, pSlot2.activeTotal());
        assertEquals(activePerSlot, pSlot2.engineOwnedCount());
        assertEquals(activePerSlot, pSlot2.phaseCounts().get(7)); // P_WAITING_LOADED
        assertEquals(expectedWaitSlot2, pSlot2.estimatedWaitMs(), "留活跃条目分摊预测耗时——测试内独立重算");
        ledger.prefill().refreshSnapshot();
        assertEquals(2 * n / 3, ledger.prefill().snapshot().inflight());
    }

    /**
     * 派发前重绑桶间全账迁移（单线程精确断言）：预约入源桶（未确认双轨
     * 在账）→ 派发换绑目标端点（源桶清零、全账随条目迁移）→ 确认临界
     * 在目标桶撤未确认、引擎事实 KV 接管 → 终局归位。
     */
    @Test
    void dispatchRebindTransfersFullAccountBetweenBuckets() {
        StateLedger ledger = new StateLedger();
        long dGen1 = ledger.newGeneration(TestEndpoints.ep(2L, StateRole.DECODE, 0L));
        long dGen2 = ledger.newGeneration(TestEndpoints.ep(4L, StateRole.DECODE, 0L));
        TestEndpoints.Endpoint dEp2 = TestEndpoints.ep(4L, StateRole.DECODE, dGen2);
        GenerationTriple ep1Binding = new GenerationTriple(2, dGen1, -1L);
        GenerationTriple ep2Binding = new GenerationTriple(4, dGen2, -1L);

        assertEquals(ReserveResult.OK, ledger.decode().reserve(11L, 100L, 200L, ep1Binding));
        DecodeEndpointCounters c1 = ledger.decode().endpointCounters(2);
        assertEquals(1, c1.activeTotal());
        assertEquals(1, c1.unconfirmedCount());
        assertEquals(200L, c1.unconfirmedExpectedKv());
        assertEquals(100L, c1.unconfirmedSeqKv());
        assertEquals(1L, c1.phaseCounts().get(0)); // RESERVED

        // 派发前重绑：源桶清零、全账迁入目标桶（同一临界区，无瞬态丢失）
        assertTrue(ledger.decode().onDispatched(11L, ep2Binding));
        DecodeEndpointCounters source = ledger.decode().endpointCounters(2);
        assertEquals(0, source.activeTotal(), "源桶活跃清零");
        assertEquals(0L, source.unconfirmedExpectedKv(), "源桶未确认账随迁出清零");
        DecodeEndpointCounters moved = ledger.decode().endpointCounters(4);
        assertEquals(1, moved.activeTotal());
        assertEquals(1, moved.unconfirmedCount(), "派发位未过确认临界——未确认账随条目迁移");
        assertEquals(200L, moved.unconfirmedExpectedKv());
        assertEquals(100L, moved.unconfirmedSeqKv());
        assertEquals(1L, moved.phaseCounts().get(1)); // DISPATCHED

        // 确认临界在目标桶生效：撤未确认三项、引擎事实 KV 接管
        ledger.observe(TestEndpoints.runningOnly(dEp2, 1L, 1_000L,
                TestEndpoints.running(11L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 1_024L, 1L)));
        DecodeEndpointCounters confirmed = ledger.decode().endpointCounters(4);
        assertEquals(0, confirmed.unconfirmedCount());
        assertEquals(0L, confirmed.unconfirmedExpectedKv());
        assertEquals(0L, confirmed.unconfirmedSeqKv());
        assertEquals(1_024L, confirmed.kvTokensReportedTotal());

        // 终局归位：活跃与引擎事实 KV 双清零
        ledger.observe(TestEndpoints.finishedOnly(dEp2, 2L, 1_100L,
                TestEndpoints.finished(11L, StateRole.DECODE, 0, 1_100L, 2L)));
        DecodeEndpointCounters settled = ledger.decode().endpointCounters(4);
        assertEquals(0, settled.activeTotal());
        assertEquals(0L, settled.kvTokensReportedTotal());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /**
     * 主动释放归位（单线程精确断言）：确认前释放——影子预占双轨随条目
     * 回退；确认后释放——引擎事实 KV 随条目回退。释放不进墓碑。
     */
    @Test
    void releaseRetiresPerEndpointAccountAtCurrentPhase() {
        StateLedger ledger = new StateLedger();
        long dGen = ledger.newGeneration(TestEndpoints.ep(2L, StateRole.DECODE, 0L));
        TestEndpoints.Endpoint dEp = TestEndpoints.ep(2L, StateRole.DECODE, dGen);
        GenerationTriple dBinding = new GenerationTriple(2, dGen, -1L);

        // 确认前（派发位）释放：未确认三项回退
        assertEquals(ReserveResult.OK, ledger.decode().reserve(21L, 128L, 256L, dBinding));
        assertTrue(ledger.decode().onDispatched(21L, dBinding));
        assertEquals(1, ledger.decode().endpointCounters(2).unconfirmedCount());
        assertTrue(ledger.decode().release(21L));
        DecodeEndpointCounters afterEarly = ledger.decode().endpointCounters(2);
        assertEquals(0, afterEarly.activeTotal());
        assertEquals(0, afterEarly.unconfirmedCount());
        assertEquals(0L, afterEarly.unconfirmedExpectedKv());
        assertEquals(0L, afterEarly.unconfirmedSeqKv());

        // 确认后（加载位）释放：引擎事实 KV 回退
        assertEquals(ReserveResult.OK, ledger.decode().reserve(22L, 64L, 128L, dBinding));
        assertTrue(ledger.decode().onDispatched(22L, dBinding));
        ledger.observe(TestEndpoints.runningOnly(dEp, 1L, 1_000L,
                TestEndpoints.running(22L, StateRole.DECODE, EnginePhase.KV_ALLOCATED, -1L, 512L, 1L)));
        assertEquals(512L, ledger.decode().endpointCounters(2).kvTokensReportedTotal());
        assertTrue(ledger.decode().release(22L));
        DecodeEndpointCounters afterLate = ledger.decode().endpointCounters(2);
        assertEquals(0, afterLate.activeTotal());
        assertEquals(0, afterLate.unconfirmedCount(), "确认后释放走确认账回退（由对账覆盖）");
        assertEquals(0L, afterLate.kvTokensReportedTotal());

        // 释放不进墓碑（非终局）
        assertEquals(0L, ledger.snapshot().decodeTombstones());
        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }

    /**
     * 批次预测耗时入账（单线程精确断言）：dispatch 绑定前记录的分摊值随
     * 条目入桶；DISPATCHED 后迟到写入不再生效（计数簿入账/出账对称性
     * 优先，与对账重算口径一致）；未记预测的条目计 0、负值钳位为 0；
     * 终局归位时随条目清零。
     */
    @Test
    void predictedBatchMsEntersBucketOnBindAndRetiresOnSettlement() {
        StateLedger ledger = new StateLedger();
        long pGen = ledger.newGeneration(TestEndpoints.ep(9L, StateRole.PREFILL, 0L));
        TestEndpoints.Endpoint pEp = TestEndpoints.ep(9L, StateRole.PREFILL, pGen);
        GenerationTriple binding = new GenerationTriple(9, pGen, 500L);

        // 记录预测值（派发流水线窗口）→ dispatch 首绑入桶：分摊值精确入账
        assertEquals(RegisterResult.OK, ledger.prefill().register(31L, 500L));
        ledger.prefill().onQueued(31L);
        ledger.prefill().onDispatching(31L, 500L);
        ledger.prefill().notePredictedBatchMs(31L, 80L);
        assertTrue(ledger.prefill().onDispatched(31L, binding));
        // 未记预测的条目：只占活跃位，预测账计 0
        assertEquals(RegisterResult.OK, ledger.prefill().register(32L, 500L));
        ledger.prefill().onQueued(32L);
        ledger.prefill().onDispatching(32L, 500L);
        assertTrue(ledger.prefill().onDispatched(32L, binding));
        // 负值防御：钳位为 0 入账
        assertEquals(RegisterResult.OK, ledger.prefill().register(33L, 500L));
        ledger.prefill().onQueued(33L);
        ledger.prefill().onDispatching(33L, 500L);
        ledger.prefill().notePredictedBatchMs(33L, -5L);
        assertTrue(ledger.prefill().onDispatched(33L, binding));

        PrefillEndpointCounters bound = ledger.prefill().endpointCounters(9);
        assertEquals(3, bound.activeTotal());
        assertEquals(80L, bound.estimatedWaitMs());

        // DISPATCHED 后迟到写入：不再生效
        ledger.prefill().notePredictedBatchMs(31L, 999L);
        assertEquals(80L, ledger.prefill().endpointCounters(9).estimatedWaitMs());

        // 终局归位：预测账随条目移除清零（引擎观察先确认归属）
        ledger.observe(TestEndpoints.runningOnly(pEp, 1L, 1_000L,
                TestEndpoints.running(31L, StateRole.PREFILL, EnginePhase.RUNNING, 500L, 128L, 1L)));
        ledger.observe(TestEndpoints.finishedOnly(pEp, 2L, 1_100L,
                TestEndpoints.finished(31L, StateRole.PREFILL, 0, 1_100L, 2L)));
        PrefillEndpointCounters settled = ledger.prefill().endpointCounters(9);
        assertEquals(2, settled.activeTotal());
        assertEquals(0L, settled.estimatedWaitMs(), "终局条目预测耗时随移除归零");

        assertTrue(ledger.auditAndDrift().clean(), () -> ledger.auditAndDrift().toString());
    }
}
