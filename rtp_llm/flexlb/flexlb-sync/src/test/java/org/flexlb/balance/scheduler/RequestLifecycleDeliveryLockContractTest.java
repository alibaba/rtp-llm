package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bind;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.bindRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Executable documentation for the RequestSlot delivery lock boundary. */
class RequestLifecycleDeliveryLockContractTest {

    private FlexlbConfig config;
    private RequestRegistry lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestRegistry(
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
    void itemPublicationDoesNotOwnTheExactSlotMonitor()
            throws Exception {
        Registered registered = registerItem(101L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        AdmissionMutation admission =
                lifecycle.claimAdmissionMutation(
                        registered.item().requestId(), registered.future());
        assertNotNull(admission);
        CountDownLatch actionEntered = new CountDownLatch(1);
        CountDownLatch releaseAction = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (slot) {
                contenderEntered.countDown();
            }
        }, "commit-inflight-slot-contender");

        try {
            Future<Boolean> committed = owner.submit(() ->
                    lifecycle.commitItemForPublication(
                            registered.item(),
                            false,
                            () -> {
                                actionEntered.countDown();
                                assertFalse(Thread.holdsLock(slot));
                                await(releaseAction);
                                return true;
                            }));

            assertTrue(actionEntered.await(5, TimeUnit.SECONDS));
            contender.start();
            assertTrue(contenderStarted.await(5, TimeUnit.SECONDS));
            assertTrue(contenderEntered.await(5, TimeUnit.SECONDS),
                    "slot operations must proceed during endpoint publication");
            synchronized (slot) {
                assertSame(registered.item(), slot.activeItem());
            }
            assertTrue(lifecycle.prepareIfOwned(
                            registered.item(), () -> Boolean.TRUE)
                    .orElseThrow(),
                    "queue publication must never expose an unready slot");

            releaseAction.countDown();
            assertTrue(committed.get(5, TimeUnit.SECONDS));
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
        } finally {
            releaseAction.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
            admission.close();
        }
    }

    @Test
    void itemPublicationRequiresAnExactAdmissionMutation() {
        Registered registered = registerItem(106L);
        boolean[] publicationCalled = new boolean[1];

        assertFalse(lifecycle.commitItemForPublication(
                registered.item(), false, () -> {
                    publicationCalled[0] = true;
                    return true;
                }));
        assertFalse(publicationCalled[0]);
    }

    @Test
    void declinedPublicationClearsTheProvisionalSlotBinding() {
        Registered registered = registerItem(111L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertFalse(lifecycle.commitItemForPublication(
                    registered.item(), true, () -> false));
            synchronized (slot) {
                assertNull(slot.activeItem());
                assertFalse(slot.wasPriorityAdmission());
            }
        }
    }

    @Test
    void throwingPublicationPreservesTheFailureAndClearsTheBinding() {
        Registered registered = registerItem(121L);
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        IllegalStateException expected =
                new IllegalStateException("queue publication failed");

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            IllegalStateException actual = assertThrows(
                    IllegalStateException.class,
                    () -> lifecycle.commitItemForPublication(
                            registered.item(), false, () -> {
                                throw expected;
                            }));
            assertSame(expected, actual);
            synchronized (slot) {
                assertNull(slot.activeItem());
            }
        }
    }

    @Test
    void slotIsDeliveryReadyBeforeQueuePublicationCanBecomeVisible() {
        Registered registered = registerItem(131L);
        boolean[] preparedBeforeResolution = new boolean[1];

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), false, () -> {
                        preparedBeforeResolution[0] = lifecycle.prepareIfOwned(
                                registered.item(), () -> Boolean.TRUE)
                                .isPresent();
                        return true;
                    }));
            assertTrue(preparedBeforeResolution[0],
                    "a queue-visible item must never race an unready slot");
            assertTrue(lifecycle.prepareIfOwned(
                            registered.item(), () -> Boolean.TRUE)
                    .orElseThrow());
        }
    }

    @Test
    void cancellationEntersTheSlotButWaitsForPublicationResolution()
            throws Exception {
        Registered registered = registerItem(141L);
        CountDownLatch publicationEntered = new CountDownLatch(1);
        CountDownLatch releasePublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        lifecycle.commitItemForPublication(
                                registered.item(), false, () -> {
                                    publicationEntered.countDown();
                                    await(releasePublication);
                                    return false;
                                }));
                assertTrue(publicationEntered.await(5, TimeUnit.SECONDS));

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancelRequest(
                                registered.item().requestId(),
                                0L,
                                CancelReason.CLIENT_CANCELLED));
                RequestState pending =
                        cancellation.get(5, TimeUnit.SECONDS);
                assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                        pending.state());
                assertFalse(publication.isDone(),
                        "cancellation must not resolve endpoint publication");

                releasePublication.countDown();
                assertFalse(publication.get(5, TimeUnit.SECONDS));
            } finally {
                releasePublication.countDown();
                operations.shutdownNow();
            }
        }

        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(
                        registered.item().requestId(), 0L).state());
    }

    @Test
    void cancellationAfterQueuePointOfNoReturnRemovesTheExactItemOnce()
            throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(151L, prefill);
        CountDownLatch queuePublished = new CountDownLatch(1);
        CountDownLatch returnFromPublication = new CountDownLatch(1);
        ExecutorService operations = Executors.newFixedThreadPool(2);

        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            try {
                Future<Boolean> publication = operations.submit(() ->
                        lifecycle.commitItemForPublication(
                                registered.item(), false, () -> {
                                    queuePublished.countDown();
                                    await(returnFromPublication);
                                    return true;
                                }));
                assertTrue(queuePublished.await(5, TimeUnit.SECONDS));

                Future<RequestState> cancellation =
                        operations.submit(() -> lifecycle.cancelRequest(
                                registered.item().requestId(),
                                0L,
                                CancelReason.CLIENT_CANCELLED));
                assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                        cancellation.get(5, TimeUnit.SECONDS).state());

                returnFromPublication.countDown();
                assertTrue(publication.get(5, TimeUnit.SECONDS));
            } finally {
                returnFromPublication.countDown();
                operations.shutdownNow();
            }
        }

        assertFalse(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        verify(prefill).removeQueued(
                eq(registered.item()), anyString());
    }

    @Test
    void deliveryClaimKeepsEndpointHandoffAndSlotClaimAtomic()
            throws Exception {
        Registered registered = registerItem(201L);
        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), false, () -> true));
        }
        RequestSlot slot = lifecycle.requestSlot(registered.item().requestId());
        CountDownLatch transferEntered = new CountDownLatch(1);
        CountDownLatch releaseTransfer = new CountDownLatch(1);
        CountDownLatch contenderStarted = new CountDownLatch(1);
        CountDownLatch contenderEntered = new CountDownLatch(1);
        ExecutorService owner = Executors.newSingleThreadExecutor();
        Thread contender = new Thread(() -> {
            contenderStarted.countDown();
            synchronized (slot) {
                contenderEntered.countDown();
            }
        }, "try-commit-slot-contender");

        try {
            Future<RequestRegistry.DeliveryClaim> committed = owner.submit(() ->
                    lifecycle.tryClaimRouteDelivery(
                            registered.item(),
                            () -> {
                                assertTrue(Thread.holdsLock(slot));
                                transferEntered.countDown();
                                await(releaseTransfer);
                                return true;
                            }));

            assertTrue(transferEntered.await(5, TimeUnit.SECONDS));
            contender.start();
            assertTrue(contenderStarted.await(5, TimeUnit.SECONDS));
            awaitCondition(() -> contender.getState() == Thread.State.BLOCKED
                    || contenderEntered.getCount() == 0L);

            assertEquals(Thread.State.BLOCKED, contender.getState());
            assertEquals(1L, contenderEntered.getCount(),
                    "another slot operation must not enter during endpoint transfer");

            releaseTransfer.countDown();
            RequestRegistry.DeliveryClaim claim = committed.get(5, TimeUnit.SECONDS);
            assertNotNull(claim);
            contender.join(TimeUnit.SECONDS.toMillis(5));
            assertFalse(contender.isAlive());
            assertEquals(0L, contenderEntered.getCount());

            lifecycle.complete(
                    claim, DeliveryResult.delivered());
            assertTrue(registered.future().get(5, TimeUnit.SECONDS).isSuccess());
        } finally {
            releaseTransfer.countDown();
            owner.shutdownNow();
            contender.join(TimeUnit.SECONDS.toMillis(5));
        }
    }

    @Test
    void failedEndpointTransferLeavesTheSlotUnclaimed() {
        Registered rejected = registerItem(202L);
        bind(lifecycle, rejected);

        assertThrows(IllegalStateException.class, () -> lifecycle.tryClaimRouteDelivery(
                rejected.item(),
                () -> false));
        assertQueuedWithoutClaim(rejected.item().requestId());

        Registered failed = registerItem(203L);
        bind(lifecycle, failed);
        IllegalStateException expected = new IllegalStateException(
                "synthetic endpoint failure");
        assertSame(expected, assertThrows(IllegalStateException.class,
                () -> lifecycle.tryClaimRouteDelivery(
                        failed.item(),
                        () -> {
                            throw expected;
                        })));
        assertQueuedWithoutClaim(failed.item().requestId());
    }

    @Test
    void batchClaimIsCompleteWhenTryCommitReturns() {
        Registered registered = registerItem(204L);
        bind(lifecycle, registered);

        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(),
                701L,
                () -> true);

        assertNotNull(claim);
        RequestState snapshot = lifecycle.getRequestState(
                registered.item().requestId(), 0L);
        assertEquals(RequestState.Phase.DISPATCHING, snapshot.state());
        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE,
                snapshot.deliveryClaimKind());
        assertEquals(701L, snapshot.batchId());
        assertTrue(lifecycle.requestSlot(registered.item().requestId())
                .getBatchEnqueueStartedAtMs() > 0L);
    }

    @Test
    void schedulingDeadlineCannotCancelAfterBatchDeliveryPointOfNoReturn() {
        Registered registered = registerItem(206L);
        bind(lifecycle, registered);

        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(
                registered.item(),
                702L,
                () -> true);
        assertNotNull(claim);

        RequestState afterDeadline = lifecycle.cancelRequest(
                registered.item().requestId(),
                0L,
                CancelReason.DEADLINE_EXCEEDED);
        assertEquals(RequestState.Phase.DISPATCHING, afterDeadline.state(),
                "the committed delivery claim must own the deadline race");

        lifecycle.complete(
                claim, DeliveryResult.delivered());
        assertTrue(registered.future().join().isSuccess());
    }

    @Test
    void acknowledgedDeliveryArmsDecodeAcceptanceDeadline()
            throws Exception {
        Registered registered = registerItem(
                205L, null, mock(DecodeEndpoint.class));
        bindRoute(lifecycle, registered, 1, 10L);

        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(
                registered.item(),
                () -> true);
        assertNotNull(claim);
        lifecycle.complete(
                claim, DeliveryResult.delivered());

        assertTrue(registered.future().join().isSuccess());
        assertEquals(RequestState.Phase.ACKNOWLEDGED,
                lifecycle.getRequestState(205L, 0L).state());
        awaitCondition(() -> lifecycle.decodeAcceptanceCount() == 0);
    }

    private void assertQueuedWithoutClaim(long requestId) {
        RequestState snapshot = lifecycle.getRequestState(
                requestId, 0L);
        assertEquals(RequestState.Phase.QUEUED, snapshot.state());
        assertEquals(DeliveryClaimKind.NONE, snapshot.deliveryClaimKind());
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, null);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefillEndpoint) {
        return registerItem(requestId, prefillEndpoint, null);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefillEndpoint,
            DecodeEndpoint decodeEndpoint) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 8);
        ScheduledRequest item = new ScheduledRequest(
                context,
                future,
                new Response(),
                null,
                null,
                prefillEndpoint,
                decodeEndpoint,
                null,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }
}
