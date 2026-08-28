package org.flexlb.mockengine;

import org.flexlb.balance.scheduler.LedgerReconciliationHarness;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * M1 三方对拍长稳浸泡（plan 第 6 节阶段 1 验收：“三方对拍 24h diff=0”
 * 的远端可行窗口版）。默认跳过——只有显式
 * {@code -Dflexlb.soak=true} 才启用（否则会拖慢常规回归），时长由
 * {@code -Dflexlb.soak.minutes}（默认 30）控制：
 *
 * <pre>
 * ./mvnw -P'opensource,!internal' -pl flexlb-mock-engine -am test \
 *   -Dtest=LedgerReconciliationSoakTest -Dsurefire.failIfNoSpecifiedTests=false \
 *   -Dflexlb.soak=true -Dflexlb.soak.minutes=30
 * </pre>
 *
 * <p>持续 priority 流量（批次提交 + 批内全终态再续批），影子对拍
 * loop 5ms 周期 + REAL 确认窗 3 周期（15ms 持续分裂才计 REAL）。断言：</p>
 *
 * <ol>
 *   <li>浸泡全程零确认 REAL diff（结构性分裂跨周期持续即上报）；</li>
 *   <li>浸泡结束静默收敛后手工对拍一次，REAL / 撕裂候选 / TRANSIENT
 *       三者均为空。</li>
 * </ol>
 *
 * <p>撕裂候选（单快照窗）只计数不断言：它们必须全部在确认窗内收敛，
 * 结束时的空报告是收敛性的最终证据。24h 全量浸泡按同配方放大
 * minutes 即可，需跨会话/后台安排。</p>
 */
class LedgerReconciliationSoakTest {

    private static final int SOAK_PORT = 63700;
    private static final int BATCH_SIZE = 60;
    private static final int[] PRIORITIES = {70, 50, 30};

    @Test
    void sustainedPriorityTrafficReconcilesCleanForTheSoakWindow()
            throws Exception {
        Assumptions.assumeTrue(
                Boolean.getBoolean("flexlb.soak"),
                "soak test is opt-in: -Dflexlb.soak=true");
        long soakMinutes = Long.getLong("flexlb.soak.minutes", 30L);
        long deadline = System.nanoTime()
                + TimeUnit.MINUTES.toNanos(soakMinutes);

        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                SOAK_PORT, 2, 2, "5", 1.0, false, true)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getCapacity()
                    .setMaxWaitingRequestsPerPrefillWorker(1024);
            // Mock-engine requests routinely finish inside one pump
            // sampling gap, so the decode acceptance fact is never
            // observed and the admission permit falls back to the 30 s
            // delivered-not-accepted timeout.  Sustained batch traffic
            // (~200 req/s) would exhaust the production default of 200
            // permits within a second; this soak targets ledger
            // reconciliation, not the admission capacity gate, so raise
            // the ceiling for the soak window only.
            h.config.queueScheduler().getLifecycle()
                    .setMaxDeliveredNotAcceptedRequestsGlobal(100_000);
            h.prefillSelector = ctx -> (int) (ctx.getRequestId() % 2);
            h.startAutoPump(10);

            List<LedgerReconciliationHarness.LedgerDiff> realDiffs =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> pendingDiffs =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> transientDiffs =
                    new CopyOnWriteArrayList<>();
            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> {
                        realDiffs.addAll(report.realDiffs());
                        pendingDiffs.addAll(report.pendingRealDiffs());
                        transientDiffs.addAll(report.transientDiffs());
                    }, 3);
            reconciler.startShadowLoop(5);

            // 预热一笔压热链路（不计入浸泡流量）。
            h.scheduler.submit(h.context(1999, 50)).get(10, TimeUnit.SECONDS);

            AtomicLong submitted = new AtomicLong();
            long rid = 2000;
            int batches = 0;
            while (System.nanoTime() < deadline) {
                List<CompletableFuture<Response>> futures =
                        new ArrayList<>(BATCH_SIZE);
                for (int i = 0; i < BATCH_SIZE; i++) {
                    futures.add(h.scheduler.submit(h.context(
                            rid++, PRIORITIES[i % PRIORITIES.length])));
                }
                submitted.addAndGet(BATCH_SIZE);
                AutoTpmE2EHarness.await(
                        () -> futures.stream()
                                .allMatch(CompletableFuture::isDone),
                        60_000,
                        "soak batch must fully terminate");
                for (CompletableFuture<Response> future : futures) {
                    Response response = future.get(1, TimeUnit.SECONDS);
                    assertTrue(response.isSuccess(),
                            "soak traffic must fully succeed, got "
                                    + response.getCode() + ": "
                                    + response.getErrorMessage());
                }
                batches++;
            }

            // 静默收敛：pump 尾部 settle + 事件投影排空。
            Thread.sleep(500);
            LedgerReconciliationHarness.ReconciliationReport finalReport =
                    reconciler.reconcileOnce();

            assertTrue(realDiffs.isEmpty(),
                    () -> "confirmed REAL diffs during soak ("
                            + realDiffs.size() + "): " + realDiffs);
            assertTrue(finalReport.realDiffs().isEmpty(),
                    () -> "quiesced REAL diffs: " + finalReport.realDiffs());
            assertTrue(finalReport.pendingRealDiffs().isEmpty(),
                    () -> "quiesced pending REAL candidates: "
                            + finalReport.pendingRealDiffs());
            assertTrue(finalReport.transientDiffs().isEmpty(),
                    () -> "quiesced TRANSIENT diffs: "
                            + finalReport.transientDiffs());

            System.out.printf(
                    "[m1-soak] %d minutes, %d batches / %d requests,"
                            + " shadow loop 5ms + confirm window 3:"
                            + " %d transient window diffs, %d single-snapshot"
                            + " tear candidates (all converged), 0 confirmed"
                            + " real diffs, quiesced report clean (slots=%d,"
                            + " decode=%d, prefill=%d)%n",
                    soakMinutes, batches, submitted.get(),
                    transientDiffs.size(), pendingDiffs.size(),
                    finalReport.slotCount(),
                    finalReport.decodeEndpointCount(),
                    finalReport.prefillRegistryCount());

            reconciler.close();
        }
    }
}
