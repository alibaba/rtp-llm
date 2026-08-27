package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Canonical request-generation ownership tests, independent of the facade. */
class RequestLifecycleCoordinatorTest {

    private FlexlbConfig config;
    private RequestLifecycleCoordinator lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestLifecycleCoordinator(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(EngineCancelChannel.class));
    }

    @AfterEach
    void tearDown() {
        if (lifecycle.closeAdmissionAndAwaitMutations()) {
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
        }
    }

    @Test
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        BalanceContext context = context(101L);
        CompletableFuture<Response> canonical = lifecycle.register(context, 8);

        CompletableFuture<Response> duplicate = lifecycle.register(context(101L), 8);

        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicate.join().getCode());
        assertSame(canonical, lifecycle.requestSlot(101L).future());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void globalOutstandingPermitIsAtomicAndReusableAfterLocalTerminal() {
        CompletableFuture<Response> first = lifecycle.register(context(201L), 1);
        CompletableFuture<Response> rejected = lifecycle.register(context(202L), 1);

        assertFalse(first.isDone());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                rejected.join().getCode());

        RequestLifecycleSnapshot cancellation = lifecycle.cancelRequest(
                201L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(cancellation);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                first.join().getCode());

        CompletableFuture<Response> admittedAgain =
                lifecycle.register(context(203L), 1);
        assertFalse(admittedAgain.isDone(),
                "the exact terminal must release its one outstanding permit");
    }

    @Test
    void admissionMutationDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L), 4);
        RequestLifecycleCoordinator.AdmissionScope scope =
                lifecycle.beginAdmission(301L, future);
        assertNotNull(scope);

        RequestLifecycleSnapshot requested = lifecycle.cancelRequest(
                301L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                lifecycle.getRequestState(301L, 0L).state());
    }

    @Test
    void exactMutationCanBindPostPnrResourcesAfterCancellationWasDeferred() {
        CompletableFuture<Response> future = lifecycle.register(context(302L), 4);
        AdmissionMutation mutation = lifecycle.claimAdmissionMutation(302L, future);
        assertNotNull(mutation);
        assertTrue(mutation.seal(),
                "the exact open mutation must linearize before its PNR");

        lifecycle.cancelRequest(302L, 0L, CancelReason.DEADLINE_EXCEEDED);
        AtomicInteger releases = new AtomicInteger();

        assertTrue(lifecycle.bindAdmissionResources(
                302L, future, mutation, releases::incrementAndGet, 100L));
        assertFalse(future.isDone());
        assertEquals(0, releases.get());

        mutation.close();

        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                future.join().getCode());
        assertEquals(1, releases.get(),
                "the canonical timeout cleanup must release the exact permit once");
    }

    @Test
    void cancellationBeforeSealPreventsDestructiveAdmissionCommit() {
        CompletableFuture<Response> future = lifecycle.register(context(303L), 4);
        AdmissionMutation mutation = lifecycle.claimAdmissionMutation(303L, future);
        assertNotNull(mutation);

        lifecycle.cancelRequest(303L, 0L, CancelReason.DEADLINE_EXCEEDED);

        assertFalse(mutation.seal(),
                "a deadline that wins first must prevent the destructive PNR");
        mutation.close();
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void shutdownGateWaitsForTheExactAdmissionMutationAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                lifecycle.register(context(401L), 4);
        RequestLifecycleCoordinator.AdmissionScope held =
                lifecycle.beginAdmission(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(lifecycle::closeAdmissionAndAwaitMutations);
            awaitCondition(lifecycle::isShuttingDown);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    lifecycle.register(context(402L), 4);
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.close();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = lifecycle.register(context(501L), 4);

        assertNull(lifecycle.cancelRequest(
                999L, 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancelRequest(
                501L, 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestLifecycleSnapshot exact = lifecycle.cancelRequest(
                501L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    private BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private static void awaitCondition(java.util.function.BooleanSupplier condition)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }
}
