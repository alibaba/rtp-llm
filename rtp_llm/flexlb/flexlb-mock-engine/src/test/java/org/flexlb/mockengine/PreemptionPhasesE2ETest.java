package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/**
 * Task35 场景 A：分阶段抢占在真实调度器栈 + 进程内 mock 引擎上的 E2E 验证。
 *
 * <ul>
 *   <li>A1 队列驱逐 — victim 8400 + "yielded"，高优入队成功；</li>
 *   <li>A2 decode reserved 驱逐 — victim 8400，影子账目正确移交；</li>
 *   <li>A3 accepted 让位 — 真实 MockEngineCancelChannel，victim 8429，
 *       cancel→确认→派发顺序 + cancel 超时不泄漏（铁律4）；</li>
 *   <li>A5 同优不抢占 — 同优请求满载时绝不驱逐同优 victim。</li>
 * </ul>
 */
class PreemptionPhasesE2ETest {

    private static final int BASE_PORT = 62700;

    // ==================== A1 队列驱逐 ====================

    @Test
    @Timeout(30)
    void a1_queue_full_evicts_lowest_priority_victim_with_8400_yielded() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT, 1, 1, "50", 1.0, false)) {
            h.config.setAutoTpmEnabled(true);
            h.config.setAutoTpmPrefillQueueEvictEnabled(true);
            h.config.setFlexlbBatchQueueMaxSize(2);
            // 大窗口停住派发：队列状态稳定可断言
            h.config.setFlexlbBatchWindowMs(10_000);
            h.config.setFlexlbBatchSizeMax(100);

            CompletableFuture<Response> low1 = h.scheduler.submit(h.context(101, 30));
            CompletableFuture<Response> low2 = h.scheduler.submit(h.context(102, 40));
            assertEquals(2, h.prefillEndpoint(0).getBatcher().queueSize());
            assertFalse(low1.isDone());
            assertFalse(low2.isDone());

            CompletableFuture<Response> high = h.scheduler.submit(h.context(103, 70));

            // victim = 队列内最低优 P30：8400 + yielded 消息
            Response victim = low1.get(2, TimeUnit.SECONDS);
            assertFalse(victim.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), victim.getCode());
            assertNotNull(victim.getErrorMessage());
            assertTrue(victim.getErrorMessage().contains("yielded"),
                    "queue victim must carry the yielded attribution: " + victim.getErrorMessage());
            assertTrue(victim.getErrorMessage().contains("103"),
                    "yielded message must name the incoming request: " + victim.getErrorMessage());

            // 高优与 P40 留在队列，未被驱逐
            assertFalse(high.isDone());
            assertFalse(low2.isDone());
            assertEquals(2, h.prefillEndpoint(0).getBatcher().queueSize());
        }
    }

    // ==================== A2 decode reserved 驱逐 ====================

    @Test
    @Timeout(30)
    void a2_decode_reserved_eviction_victim_8400_and_shadow_accounting_transfers() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT + 10, 1, 1, "50", 1.0, false)) {
            h.config.setAutoTpmEnabled(true);
            h.config.setAutoTpmDecodeReservedEvictEnabled(true);
            h.config.setDecodeConcurrencyLimit(1);
            h.config.setFlexlbBatchWindowMs(10_000);
            h.config.setFlexlbBatchSizeMax(100);

            DecodeEndpoint decodeEp = h.decodeEndpoint(0);
            h.setDecodeKvCapacity(0, 128, 256);
            CompletableFuture<Response> low = h.scheduler.submit(h.context(201, 30));
            assertFalse(low.isDone());
            assertTrue(decodeEp.reservedView().containsKey(201L));
            // victim 仍由 Master 排队持有，因此走本地 queued eviction，无需 Engine Cancel。
            long hardKvBefore = decodeEp.inflightHardKvReserved();
            assertTrue(hardKvBefore > 0);

            CompletableFuture<Response> high = h.scheduler.submit(h.context(202, 70));

            Response victim = low.get(2, TimeUnit.SECONDS);
            assertFalse(victim.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), victim.getCode(),
                    "reserved victim terminal must be 8400 (never 8429): " + victim.getErrorMessage());

            // 账目正确：victim 影子预留释放，高优恰好占据一份
            assertFalse(high.isDone(), "high-priority request should sit in the queue after eviction");
            assertFalse(decodeEp.reservedView().containsKey(201L));
            assertTrue(decodeEp.reservedView().containsKey(202L));
            assertEquals(1, decodeEp.getInflightCount());
            assertEquals(hardKvBefore, decodeEp.inflightHardKvReserved(),
                    "hard KV must transfer 1:1 from victim to incoming");
        }
    }

    // ==================== A3 accepted 让位（真实 cancel 通道） ====================

    @Test
    @Timeout(30)
    void a3_accepted_eviction_cancels_via_real_channel_victim_8429_in_order() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT + 20, 1, 1, "50", 10_000.0, true)) {
            h.config.setAutoTpmEnabled(true);
            // decode 驱逐入口由 reserved-evict 总开关把门（Phase 4 gate）
            h.config.setAutoTpmDecodeReservedEvictEnabled(true);
            h.config.setAutoTpmDecodeAcceptedEvictEnabled(true);
            h.config.setDecodeConcurrencyLimit(1);
            h.config.setFlexlbBatchWindowMs(10_000);
            h.config.setFlexlbBatchSizeMax(1);
            h.config.setAutoTpmCancelCompletionTimeoutMs(3_000);

            DecodeEndpoint decodeEp = h.decodeEndpoint(0);
            JavaMockEngineCluster.FastRpcService prefillEngine = h.prefillEngines.get(0);
            JavaMockEngineCluster.FastRpcService decodeEngine = h.decodeEngines.get(0);

            CompletableFuture<Void> lowAckGate = h.holdNextBatchAck();

            // victim P30 真实派发；只延迟 ACK，不阻塞 Prefill -> Decode 执行
            CompletableFuture<Response> low = h.scheduler.submit(h.context(301, 30));
            AutoTpmE2EHarness.await(() -> {
                var state = h.scheduler.getRequestState(301, 0);
                return state != null && state.batchId() > 0;
            }, 2_000, "victim must acquire a real Prefill batch generation");
            AutoTpmE2EHarness.await(() -> decodeEngine.getRunningCount() >= 1, 2_000,
                    "victim running on decode mock");

            AutoTpmE2EHarness.await(() -> {
                h.pumpOnce();
                return decodeEp.isConfirmedTracked(301L);
            }, 2_000, "WorkerStatus must confirm the real Decode owner");
            assertTrue(decodeEp.isConfirmedTracked(301L));
            assertEquals(1, decodeEp.getRunningLayerCount());
            assertFalse(low.isDone());

            // Hold subsequent traffic in the batcher while high priority admission
            // waits for the real Cancel completion.
            h.config.setFlexlbBatchSizeMax(100);

            // 高优提交放到后台线程：commit 会同步等待 cancel 释放确认
            CompletableFuture<Response> high;
            java.util.concurrent.atomic.AtomicReference<CompletableFuture<Response>> highRef =
                    new java.util.concurrent.atomic.AtomicReference<>();
            Thread submitter = new Thread(() ->
                    highRef.set(h.scheduler.submit(h.context(302, 70))), "a3-high-submitter");
            submitter.start();

            // cancel 意图必须先到达引擎（顺序断言第 1 段：cancel 先于确认）
            AutoTpmE2EHarness.await(() -> decodeEngine.getCancelledCount() >= 1, 3_000,
                    "cancel must reach the mock engine");
            assertFalse(low.isDone(),
                    "victim must NOT get its terminal before the engine confirms the release (iron rule 4)");

            // 泵回真实 WorkerStatus，直到 Prefill typed-CANCELED 结算 victim。
            AutoTpmE2EHarness.await(() -> {
                h.pumpOnce();
                return low.isDone();
            }, 3_000, "typed Prefill cancel must settle the victim");

            submitter.join(5_000);
            high = highRef.get();
            assertNotNull(high);

            Response victim = low.get(2, TimeUnit.SECONDS);
            assertFalse(victim.isSuccess());
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), victim.getCode(),
                    "accepted victim cancelled via engine must be 8429: " + victim.getErrorMessage());
            assertTrue(victim.getErrorMessage().contains("302"));

            // 顺序断言第 2 段：确认后高优才拿到容量（reserve 成功、进入队列待派发）
            assertTrue(decodeEp.reservedView().containsKey(302L),
                    "incoming may take the freed capacity only after confirmed release");
            assertFalse(decodeEp.isConfirmedTracked(301L));
            assertFalse(high.isDone(), "high request waits in the batcher (window held open)");
            assertEquals(1, decodeEp.getInflightCount());

            // Late successful ACK belongs to an already-terminal lifecycle and is a no-op.
            lowAckGate.complete(null);

            // 引擎侧无泄漏
            assertEquals(0, decodeEngine.getRunningCount());
            assertEquals(1, decodeEngine.getCancelledCount());
            assertEquals(0, prefillEngine.getDownstreamOwnershipCount());
            assertEquals(0, decodeEngine.getUpstreamOwnershipCount());
        }
    }

    @Test
    @Timeout(30)
    void a3_cancel_timeout_fails_incoming_without_dispatch_and_without_leak() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT + 30, 1, 1, "50", 10_000.0, true)) {
            h.config.setAutoTpmEnabled(true);
            h.config.setAutoTpmDecodeReservedEvictEnabled(true);
            h.config.setAutoTpmDecodeAcceptedEvictEnabled(true);
            h.config.setDecodeConcurrencyLimit(1);
            h.config.setFlexlbBatchWindowMs(10_000);
            h.config.setFlexlbBatchSizeMax(1);
            // 短等待窗口 + 不泵 → 引擎释放永远得不到确认 → 超时
            h.config.setAutoTpmCancelCompletionTimeoutMs(100);

            DecodeEndpoint decodeEp = h.decodeEndpoint(0);
            JavaMockEngineCluster.FastRpcService prefillEngine = h.prefillEngines.get(0);
            JavaMockEngineCluster.FastRpcService decodeEngine = h.decodeEngines.get(0);

            CompletableFuture<Void> lowAckGate = h.holdNextBatchAck();
            CompletableFuture<Response> low = h.scheduler.submit(h.context(311, 30));
            AutoTpmE2EHarness.await(() -> {
                var state = h.scheduler.getRequestState(311, 0);
                return state != null && state.batchId() > 0;
            }, 2_000, "victim must acquire a real Prefill batch generation");
            AutoTpmE2EHarness.await(() -> decodeEngine.getRunningCount() >= 1, 2_000,
                    "victim running on decode mock");
            AutoTpmE2EHarness.await(() -> {
                h.pumpOnce();
                return decodeEp.isConfirmedTracked(311L);
            }, 2_000, "WorkerStatus must confirm the real Decode owner");
            assertTrue(decodeEp.isConfirmedTracked(311L));
            assertEquals(1, decodeEp.getRunningLayerCount());

            h.config.setFlexlbBatchSizeMax(100);

            CompletableFuture<Response> high = h.scheduler.submit(h.context(312, 70));

            // 铁律4：cancel 超时绝不乐观派发 —— incoming 明确失败
            Response highResp = high.get(5, TimeUnit.SECONDS);
            assertFalse(highResp.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), highResp.getCode());
            assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    highResp.getAdmissionRejectReason());
            assertTrue(highResp.getErrorMessage().contains("cancel_completion_unknown"),
                    "timeout must be explicit: " + highResp.getErrorMessage());
            assertFalse(decodeEp.reservedView().containsKey(312L),
                    "incoming must NOT take capacity on cancel timeout");

            // victim 保持 CANCEL_REQUESTED，等 WorkerStatus 迟到确认 → 8429 late confirm
            assertFalse(low.isDone());
            AutoTpmE2EHarness.await(() -> {
                h.pumpOnce();
                return low.isDone();
            }, 3_000, "late typed Prefill cancel must settle the victim");
            Response victim = low.get(2, TimeUnit.SECONDS);
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(), victim.getCode(),
                    "late CANCELLED confirm still attributes 8429");
            lowAckGate.complete(null);

            // 无泄漏：确认层清空、引擎无 running、调度器 inflight 只剩尚未派发的项
            assertFalse(decodeEp.isConfirmedTracked(311L));
            assertEquals(0, decodeEngine.getRunningCount());
            assertEquals(0, decodeEp.getInflightCount());
            assertEquals(0L, decodeEp.inflightHardKvReserved());
            assertEquals(0, prefillEngine.getDownstreamOwnershipCount());
            assertEquals(0, decodeEngine.getUpstreamOwnershipCount());
        }
    }

    // ==================== A5 同优不抢占 ====================

    @Test
    @Timeout(30)
    void a5_equal_priority_never_preempts_queue_or_reserved_victims() throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(BASE_PORT + 50, 1, 1, "50", 1.0, false)) {
            h.config.setAutoTpmEnabled(true);
            h.config.setAutoTpmPrefillQueueEvictEnabled(true);
            h.config.setAutoTpmDecodeReservedEvictEnabled(true);
            h.config.setDecodeConcurrencyLimit(1);
            h.config.setFlexlbBatchQueueMaxSize(1);
            h.config.setFlexlbBatchWindowMs(10_000);
            h.config.setFlexlbBatchSizeMax(100);

            DecodeEndpoint decodeEp = h.decodeEndpoint(0);
            h.setDecodeKvCapacity(0, 128, 256);
            // P50 占据 decode 唯一槽位 + prefill 唯一队列位
            CompletableFuture<Response> holder = h.scheduler.submit(h.context(501, 50));
            assertFalse(holder.isDone());
            assertTrue(decodeEp.reservedView().containsKey(501L));

            // 同优新请求：decode 槽位满 → 不驱逐同优 → 明确失败
            CompletableFuture<Response> equal = h.scheduler.submit(h.context(502, 50));
            Response equalResp = equal.get(2, TimeUnit.SECONDS);
            assertFalse(equalResp.isSuccess());
            assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(),
                    equalResp.getCode(),
                    "same-priority capacity blocker must use typed 429");
            assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                    equalResp.getAdmissionRejectReason());

            // victim 完全不受影响
            assertFalse(holder.isDone());
            assertTrue(decodeEp.reservedView().containsKey(501L));
            assertEquals(1, h.prefillEndpoint(0).getBatcher().queueSize());
            verify(h.priorityReporter, never()).reportVictim(anyInt(), anyInt(),
                    anyString(), anyString());
        }
    }

}
