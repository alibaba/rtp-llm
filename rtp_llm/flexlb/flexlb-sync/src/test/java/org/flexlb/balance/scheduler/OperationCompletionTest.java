package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.VictimResolution;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Checks that completion observers share exact results without acquiring authority to cancel or complete them.
 */
class OperationCompletionTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryCompletionIsRetainedBeforeAndAfterSubscription(boolean completeFirst)
            throws Exception {
        var member = new Delivery.Member(mock(RouteResult.class), null);
        if (completeFirst) member.complete();
        var observed = member.completion().toCompletableFuture();
        assertEquals(completeFirst, observed.isDone());
        member.complete();
        observed.get(2, TimeUnit.SECONDS);
        member.complete();
        assertTrue(member.completion().toCompletableFuture().isDone());
    }

    @Test
    void observersCannotCompleteOrCancelCanonicalDelivery() throws Exception {
        var member = new Delivery.Member(mock(RouteResult.class), null);
        var forged = member.completion().toCompletableFuture();
        assertTrue(forged.complete(null));
        assertFalse(member.completion().toCompletableFuture().isDone());
        var cancelled = member.completion().toCompletableFuture();
        assertTrue(cancelled.cancel(false));
        var observer = member.completion().toCompletableFuture();
        assertFalse(observer.isDone());
        member.complete();
        observer.get(2, TimeUnit.SECONDS);
    }

    @Test
    void racingSubscriptionsShareTheExactPreemptionResult() throws Exception {
        var operation =
                new PreemptionRegistration(
                        mock(RequestContext.class),
                        1L,
                        mock(DecodeResources.RequestReservation.class),
                        new CancelTarget("127.0.0.1", 9000));
        var result =
                new VictimResolution(String.valueOf(1L), VictimResolution.Outcome.DELIVERY_RESUMED);
        var callbacks = new AtomicInteger();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newVirtualThreadPerTaskExecutor()) {
            var observed =
                    new ArrayList<
                            java.util.concurrent.Future<CompletableFuture<VictimResolution>>>();
            for (int i = 0; i < 16; i++) {
                observed.add(
                        workers.submit(
                                () -> {
                                    assertTrue(start.await(2, TimeUnit.SECONDS));
                                    return operation
                                            .completion()
                                            .thenApply(
                                                    value -> {
                                                        callbacks.incrementAndGet();
                                                        return value;
                                                    })
                                            .toCompletableFuture();
                                }));
            }
            start.countDown();
            operation.complete(result);
            for (var future : observed)
                assertSame(result, future.get(2, TimeUnit.SECONDS).get(2, TimeUnit.SECONDS));
        }
        operation.complete(
                new VictimResolution(String.valueOf(1L), VictimResolution.Outcome.REQUEST_END));
        assertSame(result, operation.result());
        assertEquals(16, callbacks.get());
        var forged = operation.completion().toCompletableFuture();
        forged.obtrudeValue(
                new VictimResolution(String.valueOf(2L), VictimResolution.Outcome.REQUEST_END));
        assertSame(result, operation.completion().toCompletableFuture().get(2, TimeUnit.SECONDS));
    }
}
