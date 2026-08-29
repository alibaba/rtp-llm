package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.LedgerReconciliationHarness;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * M2 T7 S1 incoming-shadow window empirical E2E（裁决 2 实证）：
 * engine-owned 优先级驱逐在真实 cancel 协议往返期间产生 L4b attempt
 * 影子窗口 —— {@code beginPriorityPreemptionPinned} 在同一 admission tick
 * 内登记 attempt 集与 incoming 的 layer-1 数值影子（预留 KV 数值承载），
 * slot 侧 publication 绑定发生在协议成功之后。本场景挂三方对拍
 * shadow loop 实测：
 *
 * <ol>
 *   <li>窗口被对拍观测：TRANSIENT 清单出现 "priority protocol in
 *       flight" 分类的 {@code DECODE_INFLIGHT_AHEAD_OF_SLOT}（数值冗余
 *       承载假设的直接证据——L1 影子在 slot 绑定前已持有数值）；</li>
 *   <li>新预埋规则 {@code INFLIGHT_PROW_CROSSCHECK} 全程零确认 REAL
 *       （incoming 豁免生效，规则不误报协议窗口）；</li>
 *   <li>协议正确性：victim 8429（引擎 cancel 归因）+ 高优在释放确认后
 *       拿到容量；</li>
 *   <li>静默收敛：终态排空后 REAL / 撕裂候选 / TRANSIENT 三清单全空
 *       （窗口收敛性——attempt 影子随协议 settle 退场）。</li>
 * </ol>
 *
 * <p>对比注记：master-queued victim 的本地驱逐（A2 型）不产生本窗口——
 * {@code evictLocalReservationsAndReserveIncomingPinned} 在同一
 * admission tick 内原子完成 victim 移除与 incoming 预留，attempt 集不
 * 参与；该路径的对拍覆盖由 PreemptionPhasesE2ETest 与
 * DecodePendingQueueHardGateTest 承担。</p>
 */
class LedgerReconciliationPreemptionE2ETest {

    private static final int BASE_PORT = 63700;

    @Test
    @Timeout(120)
    void engineOwnedPreemptionReconcilesCleanWithIncomingShadowClassified() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT, 1, 1, "50", 10_000.0, true)) {
            h.allowPreemption(VictimStage.DECODE_RESERVED, VictimStage.DECODE_ENGINE_OWNED);
            h.config.getRouter().getRoles().getDecode().getAvailability()
                    .setMaxEngineRequests(1L);
            // 窗口策略同 A3：初始单请求窗让 victim 立即派发，
            // victim canonical 后开大窗放后续流量。
            h.fixedWindowDecision().setMaxCollectionWaitMs(10_000);
            h.fixedWindowDecision().setMaxRequests(1);
            h.config.priorityOrdering().getPreemption().getEngineCancellation()
                    .setCompletionTimeoutMs(3_000);

            DecodeEndpoint decodeEp = h.decodeEndpoint(0);
            JavaMockEngineCluster.FastRpcService decodeEngine = h.decodeEngines.get(0);

            List<LedgerReconciliationHarness.LedgerDiff> realDuringRun =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> transientDuringRun =
                    new CopyOnWriteArrayList<>();
            List<LedgerReconciliationHarness.LedgerDiff> pendingDuringRun =
                    new CopyOnWriteArrayList<>();
            // REAL 确认窗 3：10ms 周期下需连续 ~30ms 复现才计 REAL；
            // 协议窗口（秒级）内的 incoming-shadow TRANSIENT 持续出现，
            // 但 transient 规则本身不进确认窗。
            LedgerReconciliationHarness reconciler =
                    h.scheduler.attachLedgerReconciliation(report -> {
                        realDuringRun.addAll(report.realDiffs());
                        pendingDuringRun.addAll(report.pendingRealDiffs());
                        transientDuringRun.addAll(report.transientDiffs());
                    }, 3);
            reconciler.startShadowLoop(10);

            try (AutoCloseable ignored = h.holdBatchAck(301)) {
                CompletableFuture<Response> low = h.scheduler.submit(h.context(301, 30));
                AutoTpmE2EHarness.await(
                        () -> decodeEngine.getRunningCount() >= 1, 2_000,
                        "victim running on decode mock");
                h.pumpDecodeOnce(0); // mock v1 equals the discovered fixture cursor
                h.pumpDecodeOnce(0); // mock v2 publishes the canonical RUNNING owner
                assertEquals(1, decodeEp.getRunningLayerCount());
                // victim canonical 后开大窗：后续高优流量不再被收集窗卡住。
                h.fixedWindowDecision().setMaxRequests(100);

                // 高优提交放到后台线程：commit 会同步等待 cancel 释放确认。
                AtomicReference<CompletableFuture<Response>> highRef =
                        new AtomicReference<>();
                Thread submitter = new Thread(() ->
                        highRef.set(h.scheduler.submit(h.context(302, 70))),
                        "t7-s1-high-submitter");
                submitter.start();

                // 协议窗口在途：attempt 集已登记、incoming 影子已建立、
                // cancel 意图必须先到达引擎。
                AutoTpmE2EHarness.await(
                        () -> decodeEngine.getCancelledCount() >= 1, 3_000,
                        "cancel must reach the mock engine");
                assertFalse(low.isDone(),
                        "victim must NOT get its terminal before the engine"
                                + " confirms the release (iron rule 4)");

                // 确定性窗口采样：cancel 意图已到引擎而释放确认未回——
                // incoming 影子确定在途，此刻对拍必须把 302 分类为协议
                // transient（裁决 2 数值冗余承载假设的直接观测），且
                // 预埋的 cross-check 规则不得把它报成 REAL。
                LedgerReconciliationHarness.ReconciliationReport inFlight =
                        reconciler.reconcileOnce();
                assertTrue(inFlight.realDiffs().isEmpty(),
                        () -> "protocol-window REAL diffs: "
                                + inFlight.realDiffs());
                assertEquals(1, inFlight.transientDiffs().size(),
                        () -> "protocol-window transients: "
                                + inFlight.transientDiffs());
                LedgerReconciliationHarness.LedgerDiff windowDiff =
                        inFlight.transientDiffs().get(0);
                assertEquals(302L, windowDiff.requestId());
                assertEquals(
                        LedgerReconciliationHarness.Rule
                                .DECODE_INFLIGHT_AHEAD_OF_SLOT,
                        windowDiff.rule());
                assertTrue(windowDiff.detail()
                                .contains("priority protocol in flight"),
                        "the window must classify as the incoming-shadow"
                                + " protocol round trip: "
                                + windowDiff.detail());

                // 泵回真实 WorkerStatus：CANCELLED completion → 释放确认 + 8429 归因。
                h.pumpPrefillOnce(0);
                submitter.join(5_000);
                CompletableFuture<Response> high = highRef.get();
                assertTrue(high != null && !high.isDone(),
                        "high request waits in the batcher (window held open)");

                Response victim = low.get(5, TimeUnit.SECONDS);
                assertFalse(victim.isSuccess());
                assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                        victim.getCode(),
                        "engine-owned victim cancelled via engine must be 8429: "
                                + victim.getErrorMessage());
                assertTrue(decodeEp.layeredAdmissionView().reserved()
                                .containsKey(302L),
                        "incoming may take the freed capacity only after"
                                + " confirmed release");
                assertEquals(1, decodeEp.getInflightCount());
            }

            // 静默收敛：pump 尾部 settle + 事件投影排空（高优仍活跃
            // reserved —— L1 影子与 slot 绑定一致的干净活跃态）。
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

            // 流量全程零确认 REAL —— INFLIGHT_PROW_CROSSCHECK 的 incoming
            // 豁免在真实协议窗口下不误报。
            assertTrue(realDuringRun.isEmpty(),
                    () -> "REAL diffs during traffic (" + realDuringRun.size()
                            + "): " + realDuringRun);

            // 裁决 2 实证核心：协议窗口内的确定性采样已把 incoming
            // 影子观测为分类 transient（上方 inFlight 断言）；shadow
            // loop 的流量采样是补充观测 —— 统计打印不强制（采样是否
            // 撞上窗口有相位运气成分）。
            long incomingShadowSamples = transientDuringRun.stream()
                    .filter(diff -> diff.detail()
                            .contains("priority protocol in flight"))
                    .count();
            long prowCrosscheckSamples = transientDuringRun.stream()
                    .filter(diff -> diff.rule()
                            == LedgerReconciliationHarness.Rule
                                    .INFLIGHT_PROW_CROSSCHECK)
                    .count();
            assertTrue(prowCrosscheckSamples == 0,
                    "the cross-check must stay exempt for the protocol"
                            + " incoming shadow: " + transientDuringRun);

            System.out.printf(
                    "[t7-s1-preempt] engine-owned preemption: shadow samples"
                            + " saw %d transient window diffs (%d classified"
                            + " as incoming-shadow protocol windows) and %d"
                            + " single-snapshot tear candidates (all converged"
                            + " within the confirm window), 0 confirmed real"
                            + " diffs, quiesced report clean (slots=%d,"
                            + " decode=%d, prefill=%d)%n",
                    transientDuringRun.size(), incomingShadowSamples,
                    pendingDuringRun.size(),
                    finalReport.slotCount(),
                    finalReport.decodeEndpointCount(),
                    finalReport.prefillRegistryCount());

            reconciler.close();
        }
    }
}
