package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.priority.ReleaseTracker;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class EndpointRegistryLifecycleTest {

    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry registry;
    private ExecutorService executor;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        ReleaseTracker.global().reset();
        ConfigService configService = mock(ConfigService.class);
        config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchFixedWaitMs(10_000L);
        org.mockito.Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        scheduler = mock(FlexlbBatchScheduler.class);
        registry = new EndpointRegistry(
                configService, () -> scheduler, mock(BatchSchedulerReporter.class));
        executor = Executors.newCachedThreadPool();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        registry.close();
        executor.shutdownNow();
        executor.awaitTermination(2, TimeUnit.SECONDS);
        ReleaseTracker.global().reset();
    }

    @Test
    void validated_publication_calibrates_then_promotes_probing_generation() {
        WorkerStatus status = probingStatus(RoleType.DECODE, 8100);
        WorkerStatusResponse response = response(RoleType.DECODE, 1L);
        response.setAvailableKvCacheTokens(1234L);

        WorkerEndpoint endpoint = registry.publishValidatedEndpoint(
                RoleType.DECODE, status.getIpPort(), status, response);

        assertNotNull(endpoint);
        assertTrue(status.isReady());
        assertTrue(endpoint.isReady());
        assertSame(endpoint, registry.getDecode(status.getIpPort()));
        assertEquals(1234L, ((DecodeEndpoint) endpoint).realKvAvailable());
        assertSame(endpoint, registry.publishValidatedEndpoint(
                RoleType.DECODE, status.getIpPort(), status, response));
    }

    @Test
    void retirement_unpublishes_first_and_blocks_replacement_until_barrier_completes()
            throws Exception {
        WorkerStatus oldStatus = readyStatus(RoleType.VIT, 8200);
        WorkerEndpoint oldEndpoint = registry.ensureEndpoint(
                RoleType.VIT, oldStatus.getIpPort(), oldStatus);
        WorkerStatus replacement = readyStatus(RoleType.VIT, 8200);
        CountDownLatch schedulerEntered = new CountDownLatch(1);
        CountDownLatch allowSettlement = new CountDownLatch(1);
        doAnswer(invocation -> {
            schedulerEntered.countDown();
            assertTrue(allowSettlement.await(2, TimeUnit.SECONDS));
            return 0;
        }).when(scheduler).retireEndpoint(any(), any(), anyList());

        CompletableFuture<Boolean> retirement = CompletableFuture.supplyAsync(
                () -> registry.retire(RoleType.VIT, oldStatus.getIpPort(), oldStatus,
                        EndpointRetireCause.HEALTH_CHECK_FAILED), executor);

        assertTrue(schedulerEntered.await(2, TimeUnit.SECONDS));
        assertNull(registry.get(RoleType.VIT, oldStatus.getIpPort()));
        assertEquals(EndpointLifecycleState.RETIRING, oldEndpoint.getLifecycleState());
        assertNull(registry.ensureEndpoint(RoleType.VIT, replacement.getIpPort(), replacement));

        allowSettlement.countDown();
        assertTrue(retirement.get(2, TimeUnit.SECONDS));
        assertEquals(EndpointLifecycleState.CLOSED, oldEndpoint.getLifecycleState());

        WorkerEndpoint newEndpoint = registry.ensureEndpoint(
                RoleType.VIT, replacement.getIpPort(), replacement);
        assertNotNull(newEndpoint);
        assertTrue(newEndpoint.getEndpointId().generation()
                > oldEndpoint.getEndpointId().generation());
        assertFalse(oldEndpoint.tryOnWorkerStatusUpdate(oldStatus, response(RoleType.VIT, 2L)));
        assertSame(newEndpoint, registry.get(RoleType.VIT, replacement.getIpPort()));
    }

    @Test
    void retirement_waits_for_operation_lease_and_runs_settlement_once() throws Exception {
        WorkerStatus status = readyStatus(RoleType.VIT, 8300);
        WorkerEndpoint endpoint = registry.ensureEndpoint(RoleType.VIT, status.getIpPort(), status);
        EndpointOperationLease lease = EndpointOperationLease.acquire(List.of(endpoint)).orElseThrow();

        CompletableFuture<Boolean> first = CompletableFuture.supplyAsync(
                () -> registry.retire(RoleType.VIT, status.getIpPort(), status,
                        EndpointRetireCause.STATUS_EXPIRED), executor);

        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (endpoint.getLifecycleState() == EndpointLifecycleState.READY
                && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(EndpointLifecycleState.RETIRING, endpoint.getLifecycleState());
        verify(scheduler, never()).retireEndpoint(any(), any(), anyList());
        assertFalse(registry.retire(RoleType.VIT, status.getIpPort(), status,
                EndpointRetireCause.MANUAL));

        lease.close();
        assertTrue(first.get(2, TimeUnit.SECONDS));
        verify(scheduler).retireEndpoint(endpoint, EndpointRetireCause.STATUS_EXPIRED, List.of());
    }

    @Test
    void retirement_clears_prefill_and_decode_local_accounting() {
        WorkerStatus prefillStatus = readyStatus(RoleType.PREFILL, 8400);
        PrefillEndpoint prefill = (PrefillEndpoint) registry.ensureEndpoint(
                RoleType.PREFILL, prefillStatus.getIpPort(), prefillStatus);
        prefill.commitBatch(11L, 100L, List.of());
        BatchItem queued = batchItem(102L, prefill);
        assertTrue(prefill.getBatcher().tryOffer(queued));
        assertEquals(1, prefill.getBatcher().queueSize());

        WorkerStatus decodeStatus = readyStatus(RoleType.DECODE, 8500);
        DecodeEndpoint decode = (DecodeEndpoint) registry.ensureEndpoint(
                RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus);
        decode.reserve(101L, 100L, 200L);
        decode.markQueuedPhase(101L);

        assertTrue(registry.retire(RoleType.PREFILL, prefillStatus.getIpPort(), prefillStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertTrue(registry.retire(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));

        assertEquals(0, prefill.getInflightBatchCount());
        assertEquals(0, prefill.realPendingCount());
        verify(scheduler).retireEndpoint(eq(prefill),
                eq(EndpointRetireCause.HEALTH_CHECK_FAILED),
                org.mockito.ArgumentMatchers.argThat(items -> items.contains(queued)));
        assertEquals(0, decode.getInflightCount());
        assertEquals(0, decode.getTotalLoad());
        assertEquals(0, decode.inflightHardKvReserved());
    }

    @Test
    void failed_scheduler_settlement_stays_fenced_then_reconciles_and_releases_replacement()
            throws Exception {
        WorkerStatus oldStatus = readyStatus(RoleType.VIT, 8600);
        WorkerEndpoint oldEndpoint = registry.ensureEndpoint(
                RoleType.VIT, oldStatus.getIpPort(), oldStatus);
        WorkerStatus replacement = readyStatus(RoleType.VIT, 8600);
        AtomicInteger attempts = new AtomicInteger();
        doAnswer(invocation -> {
            if (attempts.getAndIncrement() == 0) {
                throw new IllegalStateException("injected scheduler settlement failure");
            }
            return 0;
        }).when(scheduler).retireEndpoint(any(), any(), anyList());

        assertFalse(registry.retire(RoleType.VIT, oldStatus.getIpPort(), oldStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertEquals(EndpointLifecycleState.RETIRING, oldEndpoint.getLifecycleState());
        assertFalse(registry.getRetirementBarrier(RoleType.VIT, oldStatus.getIpPort()).isDone());
        assertNull(registry.ensureEndpoint(RoleType.VIT, replacement.getIpPort(), replacement));

        assertEquals(1, registry.reconcilePendingRetirements());
        assertTrue(registry.getRetirementBarrier(RoleType.VIT, oldStatus.getIpPort()).isDone());
        WorkerEndpoint recovered = registry.ensureEndpoint(
                RoleType.VIT, replacement.getIpPort(), replacement);
        assertNotNull(recovered);
        assertTrue(recovered.getEndpointId().generation() > oldEndpoint.getEndpointId().generation());
        assertEquals(2, attempts.get());
    }

    @Test
    void decode_retirement_fails_release_waiters_and_drops_ip_port_release_cache() throws Exception {
        WorkerStatus status = readyStatus(RoleType.DECODE, 8700);
        registry.ensureEndpoint(RoleType.DECODE, status.getIpPort(), status);
        ReleaseTracker tracker = ReleaseTracker.global();
        tracker.onWorkerStatus(new ReleaseTracker.ReleaseObservation(
                status.getIpPort(), 0L, 1L, 951L, true, 1L, 0));
        assertTrue(tracker.awaitReleased(status.getIpPort(), 951L, 1_000L).isDone());
        CompletableFuture<ReleaseTracker.ReleaseObservation> waiter =
                tracker.awaitReleased(status.getIpPort(), 952L, 5_000L);
        assertFalse(waiter.isDone());

        assertTrue(registry.retire(RoleType.DECODE, status.getIpPort(), status,
                EndpointRetireCause.HEALTH_CHECK_FAILED));

        assertThrows(ExecutionException.class, () -> waiter.get(1, TimeUnit.SECONDS));
        assertFalse(tracker.awaitReleased(status.getIpPort(), 951L, 1_000L).isDone());
    }

    private static WorkerStatus probingStatus(RoleType role, int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        return status;
    }

    private static WorkerStatus readyStatus(RoleType role, int port) {
        WorkerStatus status = probingStatus(role, port);
        status.tryMarkReady();
        return status;
    }

    private static WorkerStatusResponse response(RoleType role, long version) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        return response;
    }

    private static BatchItem batchItem(long requestId, PrefillEndpoint prefill) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return new BatchItem(context, new CompletableFuture<>(), null,
                null, null, prefill, null, System.currentTimeMillis());
    }
}
