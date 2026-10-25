package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RequestSchedulerV2Test {

    private final FlexlbConfig config = SchedulingTestConfig.newConfig();
    private final QueueCapacity queueCapacity = new QueueCapacity();
    private RequestSchedulerV2 scheduler;

    @AfterEach
    void closeScheduler() {
        if (scheduler != null) {
            scheduler.close();
        }
    }

    @Test
    void submittedRequestWaitsInMainQueueUntilCancelled() {
        scheduler = scheduler();

        CompletableFuture<Response> future = scheduler.submit(context("1", 50, 60_000L));

        assertFalse(future.isDone());
        assertEquals(1, scheduler.queuedRequestCount());
        assertEquals(1L, queueCapacity.globalQueueAndEngineSubQueueCount());
        assertEquals(RequestState.Phase.QUEUED, scheduler.getRequestState("1", 0L).state());

        RequestState cancelled = scheduler.cancelRequest("1", 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCELLED, cancelled.state());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), future.join().getCode());
        assertEquals(0, scheduler.trackedRequestCount());
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void fullQueueRejectsNewRequest() {
        config.getScheduler().setMaxQueuedRequests(1);
        scheduler = scheduler();

        scheduler.submit(context("1", 50, 60_000L));
        Response rejected = scheduler.submit(context("2", 50, 60_000L)).join();

        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), rejected.getCode());
        assertEquals(1, scheduler.trackedRequestCount());
    }

    @Test
    void highPriorityRequestPushesOutNewestLowPriorityRequestWhenQueueIsFull() {
        SchedulingTestConfig.usePriorityQueue(config);
        config.priorityOrdering().setHighPriorityThreshold(60);
        config.getScheduler().setMaxQueuedRequests(2);
        scheduler = scheduler();

        CompletableFuture<Response> older = scheduler.submit(context("1", 10, 60_000L));
        CompletableFuture<Response> newer = scheduler.submit(context("2", 10, 60_000L));
        Response lowRejected = scheduler.submit(context("3", 50, 60_000L)).join();
        CompletableFuture<Response> high = scheduler.submit(context("4", 80, 60_000L));

        assertEquals(StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED.getErrorCode(), lowRejected.getCode());
        assertFalse(older.isDone());
        assertEquals(StrategyErrorType.QUEUE_FULL_LOW_PRIORITY_REJECTED.getErrorCode(), newer.join().getCode());
        assertFalse(high.isDone());
        assertEquals(2L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void requestIsCancelledAtDeadline() throws Exception {
        scheduler = scheduler();

        Response response = scheduler.submit(context("1", 50, 50L)).get(5, TimeUnit.SECONDS);

        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(0, scheduler.trackedRequestCount());
        // 超时由计时线程完成 future，清理在它的回调里，get 返回时清理可能还没结束
        awaitNoTrackedRequest();
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void expiredRequestIsRejectedBeforeTracking() {
        scheduler = scheduler();

        Response response = scheduler.submit(context("1", 50, -1L)).join();

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(0, scheduler.trackedRequestCount());
    }

    @Test
    void requestForUndeployedRoleIsRejected() {
        scheduler = scheduler();
        BalanceContext context = context("1", 50, 60_000L);
        context.setRequestedRoles(Set.of(RoleType.VIT));

        Response response = scheduler.submit(context).join();

        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertNull(scheduler.getRequestState("1", 0L));
    }

    @Test
    void closeFailsWaitingRequestsAndRejectsNewOnes() {
        scheduler = scheduler();
        CompletableFuture<Response> waiting = scheduler.submit(context("1", 50, 60_000L));

        scheduler.close();
        Response rejected = scheduler.submit(context("2", 50, 60_000L)).join();

        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), waiting.join().getCode());
        assertEquals(RequestSchedulerV2.SHUTTING_DOWN_MESSAGE, rejected.getErrorMessage());
        assertEquals(0L, queueCapacity.globalQueueAndEngineSubQueueCount());
    }

    /**
     * Prefill 机器数为 0，主队列名额上限也是 0，请求会一直留在主队列。
     */
    private RequestSchedulerV2 scheduler() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        ModelMetaConfig modelMetaConfig = mock(ModelMetaConfig.class);
        when(modelMetaConfig.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        return new RequestSchedulerV2(configService, queueCapacity, mock(EndpointRegistry.class),
                new PlacementAvailability(), mock(DefaultRouter.class), mock(CostBasedPrefillStrategy.class),
                modelMetaConfig);
    }

    private BalanceContext context(String requestId, int priority, long expiresInMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, System.currentTimeMillis() + expiresInMs));
        return context;
    }
}

    private void awaitNoTrackedRequest() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 5_000L;
        while (scheduler.trackedRequestCount() > 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(5L);
        }
    }
