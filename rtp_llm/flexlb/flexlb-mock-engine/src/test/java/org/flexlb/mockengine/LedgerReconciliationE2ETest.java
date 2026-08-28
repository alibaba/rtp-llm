package org.flexlb.mockengine;

import org.flexlb.balance.scheduler.LedgerReconciliationHarness;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * M1 三方对拍 E2E（plan 第 6 节阶段 1 验收）：requestSlots 账本 ↔
 * DecodeEndpoint 八层账本 ↔ PrefillWorkRegistry 队列账本，在真实 mock
 * engine 流量下周期比对，断言：
 *
 * <ol>
 *   <li>流量全程（影子 loop 10ms 连续采样 + REAL 确认窗 3 周期）零
 *       确认 REAL diff —— 任何结构性分裂（预占与投影并存/双缺、token
 *       不匹配、数值镜像不匹配、反向投影、orphan）跨周期持续出现
 *       才会被确认上报；单快照撕裂候选（新请求跨捕获顺序窗、引擎
 *       已退投影而 slot 终态事实在途窗）在 1-2 周期内收敛，不计入；</li>
 *   <li>静默收敛后（全部请求终态 + pump 尾部 settle + 事件投影排空）
 *       手工对拍一次，确认 REAL、撕裂候选与 TRANSIENT 均为空。</li>
 * </ol>
 *
 * <p>场景覆盖：基线 FIFO（全开关关闭，与旧逻辑一致）与 auto-tpm 优先级
 * 队列两形态；对拍与调度策略正交，两场景均应 diff=0。抢占/取消路径的
 * 对拍由 preemption 专项 E2E 与 KvAllocatedSameTickAtomicityTest 覆盖，
 * 本类不注入故障。</p>
 */
class LedgerReconciliationE2ETest {

    private static final int BASELINE_PORT = 63600;
    private static final int PRIORITY_PORT = 63650;
    private static final int TOTAL_REQUESTS = 60;
    private static final int[] PRIORITIES = {70, 50, 30};

    @Test
    @Timeout(120)
    void baselineFifoTrafficReconcilesCleanAcrossThreeLedgers() throws Exception {
        runThreeWayReconciliationScenario(BASELINE_PORT, false);
    }

    @Test
    @Timeout(120)
    void priorityQueueTrafficReconcilesCleanAcrossThreeLedgers() throws Exception {
        runThreeWayReconciliationScenario(PRIORITY_PORT, true);
    }

    private void runThreeWayReconciliationScenario(
            int basePort, boolean autoTpm) throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                basePort, 2, 2, "5", 1.0, false, autoTpm)) {
            h.fixedWindowDecision().setMaxCollectionWaitMs(5);
            h.fixedWindowDecision().setMaxRequests(2);
            h.config.queueScheduler().getCapacity()
                    .setMaxWaitingRequestsPerPrefillWorker(1024);
            // 轮转 prefill 路由：两个 prefill ledger 都有账可比。
            h.prefillSelector = ctx -> (int) (ctx.getRequestId() % 2);
            h.startAutoPump(10);

            List<LedgerReconciliationHarness.LedgerDiff> realDuringRun =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> transientDuringRun =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> pendingDuringRun =
                    new CopyOnWriteArrayList<>();
            // REAL 确认窗 3：10ms 周期下需连续 ~30ms 复现才计 REAL；
            // 单快照撕裂窗（毫秒级）只进 pending 统计，不 fail。
            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> {
                        realDuringRun.addAll(report.realDiffs());
                        pendingDuringRun.addAll(report.pendingRealDiffs());
                        transientDuringRun.addAll(report.transientDiffs());
                    }, 3);
            reconciler.startShadowLoop(10);

            // 预热一笔压热链路（不计入断言流量）。
            h.scheduler.submit(h.context(1999, 50)).get(10, TimeUnit.SECONDS);
            AutoTpmE2EHarness.await(
                    () -> !h.engineArrivalOrder.isEmpty(), 5_000,
                    "warm-up request must reach the engine");

            List<CompletableFuture<Response>> futures = new ArrayList<>();
            long rid = 2000;
            for (int i = 0; i < TOTAL_REQUESTS; i++) {
                futures.add(h.scheduler.submit(
                        h.context(rid++, PRIORITIES[i % PRIORITIES.length])));
            }

            AutoTpmE2EHarness.await(
                    () -> futures.stream().allMatch(CompletableFuture::isDone),
                    60_000,
                    "all " + TOTAL_REQUESTS + " requests must reach a"
                            + " terminal state");
            for (CompletableFuture<Response> future : futures) {
                Response response = future.get(1, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(),
                        "reconciled traffic must fully succeed, got "
                                + response.getCode() + ": "
                                + response.getErrorMessage());
            }

            // 静默收敛：pump 尾部 settle（10ms 周期）+ 事件投影排空。
            Thread.sleep(500);
            LedgerReconciliationHarness.ReconciliationReport finalReport =
                    reconciler.reconcileOnce();
            assertTrue(finalReport.realDiffs().isEmpty(),
                    () -> "quiesced REAL diffs: " + finalReport.realDiffs());
            assertTrue(finalReport.pendingRealDiffs().isEmpty(),
                    () -> "quiesced pending REAL candidates must converge"
                            + " to empty: " + finalReport.pendingRealDiffs());
            assertTrue(finalReport.transientDiffs().isEmpty(),
                    () -> "quiesced TRANSIENT diffs must converge to empty: "
                            + finalReport.transientDiffs());

            // 流量全程影子采样：零确认 REAL diff（撕裂候选只统计不断言，
            // 它们必须全部在确认窗内收敛——静默后的空报告已验证这一点）。
            assertTrue(realDuringRun.isEmpty(),
                    () -> "REAL diffs during traffic (" + realDuringRun.size()
                            + "): " + realDuringRun);

            System.out.printf(
                    "[m1-reconcile] autoTpm=%s: %d requests, shadow samples"
                            + " saw %d transient window diffs (all engine-ahead)"
                            + " and %d single-snapshot tear candidates (all"
                            + " converged within the confirm window),"
                            + " 0 confirmed real diffs, quiesced report clean"
                            + " (slots=%d, decode=%d, prefill=%d)%n",
                    autoTpm, TOTAL_REQUESTS, transientDuringRun.size(),
                    pendingDuringRun.size(),
                    finalReport.slotCount(),
                    finalReport.decodeEndpointCount(),
                    finalReport.prefillRegistryCount());

            reconciler.close();
        }
    }
}
