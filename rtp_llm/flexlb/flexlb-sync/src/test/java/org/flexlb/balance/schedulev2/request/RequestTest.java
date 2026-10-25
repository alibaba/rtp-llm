package org.flexlb.balance.schedulev2.request;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DeliveryClaimKind;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestTest {

    @Test
    void returnMarkerAndDispatchCompeteThroughTheSameAtomicState() throws Exception {
        var executor = java.util.concurrent.Executors.newFixedThreadPool(2);
        try {
            for (int i = 0; i < 100; i++) {
                Request request = request("race-" + i);
                request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
                var start = new java.util.concurrent.CountDownLatch(1);
                var mark = executor.submit(() -> {
                    start.await();
                    return request.markPrefillReturn(80);
                });
                var dispatch = executor.submit(() -> {
                    start.await();
                    return request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
                });
                start.countDown();
                boolean marked = mark.get(5, java.util.concurrent.TimeUnit.SECONDS);
                boolean dispatched = dispatch.get(5, java.util.concurrent.TimeUnit.SECONDS);
                assertTrue(marked != dispatched);
                if (marked) {
                    assertFalse(request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING));
                    assertTrue(request.moveTo(Stage.PREFILL_WAITING, Stage.MAIN_WAITING));
                    assertTrue(request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
                    assertTrue(request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING));
                } else {
                    assertEquals(Stage.DECODE_WAITING, request.stage());
                }
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void moveToAdvancesOnlyFromTheExpectedStage() {
        Request request = request("1");

        assertTrue(request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
        assertFalse(request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
        assertEquals(Stage.PREFILL_WAITING, request.stage());
        assertThrows(IllegalArgumentException.class, () -> request.moveTo(Stage.PREFILL_WAITING, Stage.PUBLISHED));
    }

    @Test
    void failCompletesTheFutureOnce() {
        Request request = request("2");

        assertTrue(request.fail(StrategyErrorType.DISPATCH_FAILED, "first"));
        assertFalse(request.fail(StrategyErrorType.QUEUE_FULL, "second"));
        assertFalse(request.cancel(CancelReason.CLIENT_CANCELLED));

        Response response = request.future().join();
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(Stage.FAILED, request.stage());
        assertEquals(RequestState.Phase.FAILED, request.toRequestState().state());
        assertEquals("first", request.toRequestState().detail());
    }

    @Test
    void deadlineCancelUsesRequestPriorityInEveryWaitingStage() {
        for (Stage stage : new Stage[]{Stage.MAIN_WAITING, Stage.PREFILL_WAITING, Stage.DECODE_WAITING,
                Stage.DECODE_RESERVED}) {
            for (int priority : new int[]{59, 60, 61}) {
                Request request = request("timeout-" + stage + "-" + priority, priority, 60);
                if (stage != Stage.MAIN_WAITING) {
                    request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING);
                }
                if (stage == Stage.DECODE_WAITING || stage == Stage.DECODE_RESERVED) {
                    request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING);
                }
                if (stage == Stage.DECODE_RESERVED) {
                    request.moveTo(Stage.DECODE_WAITING, Stage.DECODE_RESERVED);
                }
                assertTrue(request.cancel(CancelReason.DEADLINE_EXCEEDED));
                Response response = request.future().join();
                assertEquals(priority < 60 ? 8433 : 8434, response.getCode());
                assertEquals("request deadline exceeded", response.getErrorMessage());
                assertEquals(RequestState.Phase.TIMED_OUT, request.toRequestState().state());
                assertFalse(request.cancel(CancelReason.CLIENT_CANCELLED));
            }
        }
    }

    @Test
    void publishedRequestCannotFailOrCancel() {
        Request request = request("4");
        assertTrue(request.moveTo(Stage.MAIN_WAITING, Stage.PREFILL_WAITING));
        assertTrue(request.moveTo(Stage.PREFILL_WAITING, Stage.DECODE_WAITING));
        assertTrue(request.moveTo(Stage.DECODE_WAITING, Stage.DECODE_RESERVED));
        assertTrue(request.moveTo(Stage.DECODE_RESERVED, Stage.PUBLISHED));

        assertFalse(request.fail(StrategyErrorType.DISPATCH_FAILED, "late"));
        assertFalse(request.cancel(CancelReason.CLIENT_CANCELLED));

        RequestState state = request.toRequestState();
        assertEquals(RequestState.Phase.ACKNOWLEDGED, state.state());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION, state.deliveryClaimKind());
        assertFalse(request.future().isDone());
    }

    static Request request(String requestId) {
        return request(requestId, 50, 50);
    }

    static Request request(String requestId, int priority, int threshold) {
        org.flexlb.dao.loadbalance.Request body = new org.flexlb.dao.loadbalance.Request();
        body.setRequestId(requestId);
        body.setSeqLen(16L);
        body.setPriority(priority);
        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().getOrdering().setHighPriorityThreshold(threshold);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(body);
        return new Request(context, new CompletableFuture<>(), System.currentTimeMillis());
    }
}
