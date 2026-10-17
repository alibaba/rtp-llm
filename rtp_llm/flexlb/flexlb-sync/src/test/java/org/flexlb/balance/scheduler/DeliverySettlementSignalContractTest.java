package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.*;

/** Transport proof and sender exit cannot stand in for actual local P/D settlement. */
@Timeout(15)
class DeliverySettlementSignalContractTest {
    private enum Side { PREFILL, DECODE }

    @ParameterizedTest
    @EnumSource(Side.class)
    void asyncBatchSignalAndCleanupDedupWaitForBothPhysicalReleases(Side side) throws Exception {
        try (var fixture = new Fixture(side, false)) {
            fixture.cancelWithRemoteProof();
            assertTrue(fixture.releaseEntered.await(2, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(() -> fixture.pendingWork() == 1);
            assertEquals(ContextResult.Phase.ENDING, fixture.context.snapshot().phase());
            assertTrue(fixture.scheduler.requests.isCurrent(fixture.context));
            assertEquals(1, fixture.decode.routingView().totalLoad());
            assertEquals(side == Side.PREFILL ? 1L : 0L,
                    fixture.prefill.ownership.progress().outstandingRequests());
            assertFalse(fixture.settled.isDone(), "a release still owns real resources");
            assertEquals(DeliveryCleanup.Result.REMOTE_CLEANED, fixture.session.get(2, TimeUnit.SECONDS),
                    "remote stop completes independently of local settlement");
            assertSame(fixture.session, fixture.cleanupSession(), "retain the exact cleanup registration");
            assertEquals(1, fixture.cleanupRegistry().size());

            fixture.allowRelease.countDown();
            fixture.assertPhysicallySettledAndArchived();
            assertEquals(1, fixture.cancelCalls.get());
        }
    }

    @Test
    void persistentDecodeReleaseFailureKeepsSignalAndCleanupRegistrationUntilRealRetry() throws Exception {
        try (var fixture = new Fixture(Side.DECODE, true)) {
            fixture.cancelWithRemoteProof();
            RequestProtocolTestSupport.awaitCondition(() -> fixture.releaseAttempts.get() == 3
                    && fixture.pendingWork() == 0);
            var close = fixture.context.snapshot().close();
            assertTrue(close.prefillSettled());
            assertFalse(close.decodeSettled());
            assertEquals(0L, fixture.prefill.ownership.progress().outstandingRequests());
            assertEquals(1, fixture.decode.routingView().totalLoad());
            assertTrue(fixture.scheduler.requests.isCurrent(fixture.context));
            assertFalse(fixture.settled.isDone(), "failed release is still an obligation");
            assertEquals(DeliveryCleanup.Result.REMOTE_CLEANED, fixture.session.get(2, TimeUnit.SECONDS));
            assertSame(fixture.session, fixture.cleanupSession());
            assertEquals(1, fixture.cleanupRegistry().size());
            assertSame(fixture.releaseFailure, ReflectionTestUtils.getField(fixture.runtime, "failure"));

            fixture.failRelease.set(false);
            fixture.allowRelease.countDown();
            fixture.scheduler.requestCleanup(fixture.context);
            fixture.assertPhysicallySettledAndArchived();
            assertEquals(4, fixture.releaseAttempts.get());
            assertEquals(1, fixture.cancelCalls.get(), "local retry needs no new remote proof session");
        }
    }

    private static final class Fixture implements AutoCloseable {
        private static final long ID = 981_001L;
        private static final long BATCH = 981L;
        final AbstractRequestScheduler scheduler;
        final SchedulerRuntime runtime;
        final DeliveryDispatcher dispatcher;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final RequestContext context;
        final RequestRoute route;
        final DeliveryClaim claim;
        final EngineGrpcClient channel = mock(EngineGrpcClient.class);
        final CompletableFuture<EngineGrpcClient.CancelReply> proof = new CompletableFuture<>();
        final CompletableFuture<Void> settled;
        final AtomicInteger cancelCalls = new AtomicInteger();
        final AtomicInteger releaseAttempts = new AtomicInteger();
        final CountDownLatch releaseEntered = new CountDownLatch(1);
        final CountDownLatch allowRelease = new CountDownLatch(1);
        final AtomicBoolean failRelease;
        final IllegalStateException releaseFailure = new IllegalStateException("real Decode release paused by failure");
        final boolean expectsFailure;
        CompletableFuture<DeliveryCleanup.Result> session;

        Fixture(Side side, boolean persistentFailure) {
            expectsFailure = persistentFailure;
            failRelease = new AtomicBoolean(persistentFailure);
            var config = SchedulingTestConfig.batchConfig();
            config.getRequestLifecycle().getRequest().setTimeoutMs(TimeUnit.HOURS.toMillis(1));
            var service = SchedulingTestConfig.configService(config);
            scheduler = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                    mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
            runtime = scheduler.runtime;
            dispatcher = SchedulerTestSupport.dispatcher(runtime);
            when(channel.cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == ID), anyLong())).thenAnswer(call -> {
                cancelCalls.incrementAndGet();
                return proof;
            });
            ReflectionTestUtils.setField(runtime, "engineClient", channel);
            prefill = spy(EndpointTestSupport.unstartedPrefill(config,
                    WorkerStatus.createDiscovered(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null),
                    dispatcher, scheduler));
            decode = spy(EndpointTestSupport.decode(
                    WorkerStatus.createDiscovered(RoleType.DECODE, null, "127.0.0.2", 8080, 8081, null),
                    scheduler.requests));
            var status = new WorkerStatusResponse();
            status.setRole(RoleType.DECODE);
            status.setStatusVersion(1L);
            status.setAlive(true);
            status.setRunningTaskInfo(Map.of());
            status.setFinishedTaskInfo(Map.of());
            status.setTotalKvCacheTokens(10_000L);
            status.setAvailableKvCacheTokens(10_000L);
            var worker = decode.getStatus();
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection;
            worker.lock.lock();
            try { projection = decode.applyPreparedStatus(worker, worker.prepareNewStatus(
                    org.flexlb.dao.master.WorkerStatusTestSupport.freezeStatusResponse(worker, status))); }
            finally { worker.lock.unlock(); }
            decode.publishStatus(projection);
            DecodeResources.ReservationHandle reservation;
            try (var pin = decode.tryPinGeneration()) {
                reservation = decode.tryReserveQueuedRequest(pin, ID, 10L, 10L, 50, null).handle();
            }
            assertNotNull(reservation);
            context = RequestProtocolTestSupport.context(config, ID);
            var future = RequestProtocolTestSupport.register(scheduler, context);
            route = SchedulingTestConfig.createRoute(context, null, null, null, prefill, decode,
                    reservation, System.currentTimeMillis());
            var state = prefill.ownership;
            try (var admission = scheduler.claimAdmissionHandle(ID, future)) {
                assertNotNull(admission);
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        scheduler.commitRoute(route, RequestProtocolTestSupport.publication(() ->
                                state.updateQueue(PrefillState.QueueChange.ENQUEUE, route,
                                        Long.MAX_VALUE, System.currentTimeMillis()) == route)));
                admission.finish();
            }
            var prepared = prefill.reserveBatch(route, BATCH, 2);
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, prepared.status());
            state.lock.lock();
            PrefillState.CommittedHandoff committed;
            try { committed = prefill.commitLocked(new PrefillState.CommitPlan(PrefillState.CommitKind.BATCH, List.of(route), List.of(),
                    prepared.reservation(), new long[]{100L}, null, System.currentTimeMillis())); }
            finally { state.lock.unlock(); }
            assertNotNull(committed);
            var dispatch = decode.acquireDispatchPermit(reservation, new DecodeResources.AdmissionCapacity(0, 100L));
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, dispatch.status());
            claim = scheduler.claimDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, BATCH, dispatch.handle());
            assertNotNull(claim);
            assertTrue(scheduler.tryStartSend(claim));
            DeliveryDispatcher.closeCommitted(List.of(new DeliveryDispatcher.Member(route, dispatch.handle())), committed);
            settled = runtime.signals().delivery(claim).toCompletableFuture();
            if (side == Side.PREFILL) {
                doAnswer(call -> { awaitRelease(); return call.callRealMethod(); }).when(prefill).release(null, route);
            } else {
                doAnswer(call -> { awaitRelease(); return call.callRealMethod(); }).when(decode)
                        .release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            }
        }

        private void awaitRelease() throws InterruptedException {
            assertFalse(Thread.holdsLock(context));
            assertFalse(prefill.ownership.lock.isHeldByCurrentThread());
            releaseAttempts.incrementAndGet();
            releaseEntered.countDown();
            if (failRelease.get()) { throw releaseFailure; }
            assertTrue(allowRelease.await(5, TimeUnit.SECONDS));
        }

        void cancelWithRemoteProof() {
            scheduler.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim,
                    DeliveryResult.uncertain(new IllegalStateException("transport response lost")), System.currentTimeMillis()));
            scheduler.cancel(ID, BATCH, CancelReason.CLIENT_CANCELLED);
            session = cleanupSession();
            assertFalse(settled.isDone());
            proof.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
        }

        CompletableFuture<DeliveryCleanup.Result> cleanupSession() { return dispatcher.stop(claim); }

        Map<?, ?> cleanupRegistry() {
            return (Map<?, ?>) ReflectionTestUtils.getField(
                    ReflectionTestUtils.getField(dispatcher, "cleanup"), "sessions");
        }

        long pendingWork() {
            Object lifecycle = ReflectionTestUtils.getField(runtime, "workLifecycle");
            synchronized (lifecycle) { return (Long) ReflectionTestUtils.getField(runtime, "pendingWork"); }
        }

        void assertPhysicallySettledAndArchived() throws Exception {
            settled.get(2, TimeUnit.SECONDS);
            session.get(2, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(runtime);
            assertEquals(0L, prefill.ownership.progress().outstandingRequests());
            assertEquals(0, prefill.ownership.progress().batchSlots());
            assertEquals(0, decode.routingView().totalLoad());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertTrue(context.snapshot().close().prefillSettled());
            assertTrue(context.snapshot().close().decodeSettled());
            assertEquals(ContextResult.Phase.CLOSED, context.snapshot().phase());
            assertEquals(0, scheduler.requests.liveRequestCount());
            assertTrue(cleanupRegistry().isEmpty());
            assertFalse(route.future().get(2, TimeUnit.SECONDS).isSuccess());
        }

        @Override public void close() {
            failRelease.set(false);
            allowRelease.countDown();
            if (context.snapshot().phase() == ContextResult.Phase.ENDING) { scheduler.requestCleanup(context); }
            try {
                if (expectsFailure) { assertThrows(IllegalStateException.class, runtime::shutdown); }
                else { runtime.shutdown(); }
            } finally { prefill.close(); decode.close(); }
        }
    }
}
