package org.flexlb.mock.grpc;

import org.flexlb.balance.schedule.CancelReason;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.cache.VitWorkerCacheManager;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises Decode capacity loss through real mock-engine status RPCs and production routing.
 * Queued requests must recover or finish once on cancellation, duplicate submission and expiry.
 */
@Timeout(20)
class DecodeCapacityFailureSchedulingTest extends FlexLBMockTestBase {
    @Override
    protected RequestRouteSelector createRouter() {
        ModelMetaConfig meta = mock(ModelMetaConfig.class);
        when(meta.requiredRoles()).thenReturn(List.of(RoleType.DECODE, RoleType.PREFILL));
        when(meta.getServiceRoutes()).thenReturn(List.of());
        return new RequestRouteSelector(
                endpointRegistry,
                new CostBasedPrefillStrategy(
                        configService,
                        endpointRegistry,
                        mock(CacheAwareService.class),
                        mock(CacheMetricsReporter.class)),
                new DecodeStrategy(
                        configService,
                        new org.flexlb.balance.schedule.SchedulerConfiguration()
                                .decodeAdmissionCapacity(configService),
                        endpointRegistry),
                new VitStrategy(mock(VitWorkerCacheManager.class), endpointRegistry),
                new EncoderStrategy(configService, endpointRegistry),
                meta);
    }

    @Test
    void exhaustedDecodeQueuesAndRecoveryReturnsOneRoute() throws Exception {
        syncDecodeCapacity(0);
        var future = submitRequest(92001L);
        awaitBlocked();
        assertFalse(future.isDone());
        assertTrue(getDecodeEndpoint().resourceSnapshot().requests().isEmpty());
        syncDecodeCapacity(1_000_000L);
        var response = future.get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertEquals(2, response.getServerStatus().size());
        assertTrue(
                response.getServerStatus().stream()
                        .anyMatch(status -> status.getHttpPort() == decodeHttpPort));
        assertTrue(
                response.getServerStatus().stream()
                        .anyMatch(status -> status.getHttpPort() == prefillHttpPort));
        assertEquals(1, getDecodeEndpoint().resourceSnapshot().requests().size());
        scheduler.cancel("92001", CancelReason.CLIENT_CANCELLED);
        awaitClean();
    }

    @Test
    void cancelledWaiterCannotBeRevivedByCapacityRecovery() throws Exception {
        syncDecodeCapacity(0);
        var future = submitRequest(92002L);
        awaitBlocked();
        scheduler.cancel("92002", CancelReason.CLIENT_CANCELLED);
        var cancelled = future.get(5, TimeUnit.SECONDS);
        assertFalse(cancelled.isSuccess());
        awaitClean();
        syncDecodeCapacity(1_000_000L);
        assertTrue(submitRequest(92003L).get(5, TimeUnit.SECONDS).isSuccess());
        assertSame(cancelled, future.join());
        assertFalse(getDecodeEndpoint().resourceSnapshot().requests().containsKey("92002"));
        scheduler.cancel("92003", CancelReason.CLIENT_CANCELLED);
        awaitClean();
    }

    @Test
    void duplicateIdDoesNotReplaceBlockedRequest() throws Exception {
        syncDecodeCapacity(0);
        var original = submitRequest(92004L);
        awaitBlocked();
        var duplicate = submitRequest(92004L).get(5, TimeUnit.SECONDS);
        assertFalse(duplicate.isSuccess());
        assertEquals(8406L, duplicate.getCode());
        assertFalse(original.isDone());
        assertEquals(1, requestRegistry().liveRequestCount());
        syncDecodeCapacity(1_000_000L);
        assertTrue(original.get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, getDecodeEndpoint().resourceSnapshot().requests().size());
        scheduler.cancel("92004", CancelReason.CLIENT_CANCELLED);
        awaitClean();
    }

    @Test
    void expiredWaiterReleasesQueueBeforeRecovery() throws Exception {
        syncDecodeCapacity(0);
        var context = createRequestContext(92005L);
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.of(
                        0, null, System.currentTimeMillis() + 2000L, config.defaultPriority()));
        var future = scheduler.submit(context);
        awaitBlocked();
        var expired = future.get(5, TimeUnit.SECONDS);
        assertFalse(expired.isSuccess());
        awaitClean();
        syncDecodeCapacity(1_000_000L);
        assertTrue(submitRequest(92006L).get(5, TimeUnit.SECONDS).isSuccess());
        assertSame(expired, future.join());
        assertFalse(getDecodeEndpoint().resourceSnapshot().requests().containsKey("92005"));
        scheduler.cancel("92006", CancelReason.CLIENT_CANCELLED);
        awaitClean();
    }

    private void syncDecodeCapacity(long available) throws Exception {
        mockDecodeWorker.setBehavior(
                MockWorkerBehavior.builder().availableKvCache(available).build());
        var status = getDecodeEndpoint().getStatus();
        var pb =
                grpcClient
                        .getWorkerStatusAsync(
                                decodeIp,
                                decodeGrpcPort,
                                EngineRpcService.StatusVersionPB.newBuilder()
                                        .setLatestCacheVersion(
                                                status.appliedStatusCursor().statusVersion())
                                        .build(),
                                5000L)
                        .get(5, TimeUnit.SECONDS);
        assertEquals(available, pb.getAvailableKvCache());
        applyWorkerStatusObservation(
                status,
                WorkerStatusRpcClient.convertToStatusObservation(
                        status,
                        pb.toBuilder()
                                .setRole("DECODE")
                                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                                .build()));
    }

    private void awaitBlocked() throws Exception {
        awaitCondition(() -> schedulerRuntime.scheduler().blockedRequestCount() == 1);
    }

    private void awaitCondition(java.util.function.BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) return;
            Thread.sleep(10L);
        }
        assertTrue(condition.getAsBoolean(), "scheduler did not converge within five seconds");
    }

    private void awaitClean() throws Exception {
        awaitCondition(
                () ->
                        requestRegistry().liveRequestCount() == 0
                                && schedulerRuntime.scheduler().queuedRequestCount() == 0
                                && schedulerRuntime.scheduler().blockedRequestCount() == 0
                                && getDecodeEndpoint().resourceSnapshot().requests().isEmpty());
        assertEquals(0, requestRegistry().liveRequestCount());
        assertEquals(0, schedulerRuntime.scheduler().queuedRequestCount());
        assertEquals(0, schedulerRuntime.scheduler().blockedRequestCount());
        assertTrue(getDecodeEndpoint().resourceSnapshot().requests().isEmpty());
        assertEquals(0, mockPrefillWorker.getEnqueueCount());
        assertEquals(0, mockDecodeWorker.getEnqueueCount());
    }
}
