package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

/**
 * Distinguishes waiting for frontend responses from releasing resources or archiving their request contexts.
 */
class RequestDrainTest {
    @Test
    void responseCompletionWakesDrainWithoutReleasingOrArchivingTheRequest() throws Exception {
        var requests = new RequestRepository();
        var owner = mock(AbstractRequestScheduler.class);
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 1L);
        var response = new RequestFuture((kind, result, failure) -> false);
        assertEquals(
                RequestRepository.RegistrationResult.REGISTERED,
                requests.register(
                        context,
                        owner,
                        response,
                        60_000L,
                        org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED));
        assertThrows(IllegalStateException.class, () -> requests.awaitResponses(Long.MAX_VALUE));
        requests.stopAccepting();
        assertFalse(requests.awaitResponses(System.nanoTime()));
        var drained = new CompletableFuture<Boolean>();
        Thread waiter =
                Thread.ofPlatform()
                        .start(
                                () ->
                                        drained.complete(
                                                requests.awaitResponses(
                                                        System.nanoTime()
                                                                + TimeUnit.SECONDS.toNanos(5))));
        try {
            response.completeOwned(Response.error(StrategyErrorType.DISPATCH_FAILED));
            assertTrue(drained.get(2, TimeUnit.SECONDS));
            assertTrue(requests.isCurrent(context));
            verifyNoInteractions(owner);
        } finally {
            waiter.interrupt();
            waiter.join(2000);
        }
    }

    @Test
    void failedAndCancelledResponsesAreDrainedButUnfinishedResponsesAreNot() {
        var requests = new RequestRepository();
        var owner = mock(AbstractRequestScheduler.class);
        var failed = new RequestFuture((kind, result, failure) -> false);
        var cancelled = new RequestFuture((kind, result, failure) -> false);
        var pending = new RequestFuture((kind, result, failure) -> false);
        var responses = java.util.List.of(failed, cancelled, pending);
        for (int i = 0; i < responses.size(); i++) {
            var context =
                    RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), i + 1L);
            assertEquals(
                    RequestRepository.RegistrationResult.REGISTERED,
                    requests.register(
                            context,
                            owner,
                            responses.get(i),
                            60_000L,
                            org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED));
        }
        requests.stopAccepting();
        failed.completeExceptionallyOwned(new IllegalStateException("test failure"));
        cancelled.cancelOwned();
        assertFalse(requests.awaitResponses(System.nanoTime()));
        pending.cancelOwned();
        assertTrue(requests.awaitResponses(System.nanoTime()));
        assertEquals(3, requests.liveRequestCount());
        verifyNoInteractions(owner);
    }

    @Test
    void interruptedDrainPreservesInterruptAndLeavesTheRequestForSettlement() {
        var requests = new RequestRepository();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 1L);
        var response = new RequestFuture((kind, result, failure) -> false);
        requests.register(
                context,
                mock(AbstractRequestScheduler.class),
                response,
                60_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        requests.stopAccepting();
        Thread.currentThread().interrupt();
        try {
            assertFalse(requests.awaitResponses(Long.MAX_VALUE));
            assertTrue(Thread.currentThread().isInterrupted());
            assertFalse(response.isDone());
            assertTrue(requests.isCurrent(context));
        } finally {
            Thread.interrupted();
        }
    }
}
