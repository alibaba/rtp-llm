package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcCacheStatusCheckRunnerTest {

    private final WorkerStatusRpcClient workerStatusRpcClient = Mockito.mock(WorkerStatusRpcClient.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    private final CacheAwareService localKvCacheAwareManager = Mockito.mock(CacheAwareService.class);

    private final DynamicCacheIntervalService cacheIntervalService =
            Mockito.mock(DynamicCacheIntervalService.class);

    @Test
    void decodeCacheResponseUpdatesCapacityWithoutBuildingDetailedIndex() {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        EngineRpcService.CacheStatusPB response = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(workerStatusRpcClient.getCacheStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE)))
                .thenReturn(CompletableFuture.completedFuture(response));

        new GrpcCacheStatusCheckRunner(
                "test-model",
                status, status.tryBeginCachePoll(), directory(status),
                engineHealthReporter, workerStatusRpcClient, localKvCacheAwareManager, cacheIntervalService,
                20, new LongAdder(), 50L, true, Runnable::run).run();

        assertEquals(1000, status.getCacheStatus().getAvailableKvCache());
        assertEquals(2000, status.getCacheStatus().getTotalKvCache());
        verifyNoInteractions(localKvCacheAwareManager);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void decodeMetadataPollingDoesNotSlowStatusCapacityUpdates(boolean rejectDuePoll) {
        var config = Mockito.mock(org.flexlb.config.ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        var directory = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.DECODE, "", "127.0.0.1", 8080, 8081, "test-site");
        directory.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        var cycle = new LongAdder();
        var tick = new AtomicLong();
        when(workerStatusRpcClient.getCacheStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE)))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(1).setAvailableKvCache(1000).setTotalKvCache(2000).setBlockSize(128).build()));
        when(workerStatusRpcClient.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE)))
                .thenAnswer(call -> CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE).setAlive(true)
                        .setStatusVersion(tick.get()).setAvailableKvCache(1000 - tick.get()).setTotalKvCache(2000).build()));
        var addresses = Mockito.mock(WorkerAddressService.class);
        when(addresses.getEngineWorkerList("test-model", RoleType.DECODE))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        var executor = Mockito.mock(ExecutorService.class);
        when(executor.submit(any(Runnable.class))).thenAnswer(call -> {
            if (rejectDuePoll && tick.get() == 50 && call.getArgument(0) instanceof GrpcCacheStatusCheckRunner) {
                throw new java.util.concurrent.RejectedExecutionException("full");
            }
            call.getArgument(0, Runnable.class).run();
            return CompletableFuture.completedFuture(null);
        });
        Mockito.doAnswer(call -> {
            call.getArgument(0, Runnable.class).run();
            return null;
        }).when(executor).execute(any(Runnable.class));
        var runner = new EngineSyncRunner("test-model", directory, addresses, executor,
                engineHealthReporter, workerStatusRpcClient, RoleType.DECODE, localKvCacheAwareManager,
                cacheIntervalService, 5000L, cycle, 20L, false, 10_000_000L);
        for (int i = 7; i <= 106; i++) {
            tick.set(i);
            cycle.reset();
            cycle.add(i);
            runner.run();
            assertEquals(1000L - i, status.committedEngineObservation().availableKvCacheTokens());
            assertEquals(128, status.getCacheStatus().getBlockSize(), "new generations fetch metadata immediately");
        }
        verify(executor, times(rejectDuePoll ? 104 : 103)).submit(any(Runnable.class));
        var lease = status.tryBeginCachePoll();
        assertNotNull(lease, "skipped submissions release their exact poll lease");
        lease.close();
        verify(workerStatusRpcClient, times(100)).getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE));
        verify(workerStatusRpcClient, times(3)).getCacheStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE));
        verifyNoInteractions(localKvCacheAwareManager);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value = RoleType.class, names = {"PREFILL", "PDFUSION", "DECODE"})
    void duePollSurvivesSkippedTickAndExecutorDelay(RoleType role) {
        WorkerStatus status = RunnerTestSupport.discovered(role, null, "127.0.0.1", 8080, 8081, "test-site");
        var directory = RunnerTestSupport.endpointRegistry(Mockito.mock(org.flexlb.config.ConfigService.class));
        directory.currentOrDiscover(role, status.getIpPort(), () -> status);
        status.lock.lock();
        try { status.publishCacheStatus(org.flexlb.dao.master.CacheStatus.builder().version(1).blockSize(128).build()); }
        finally { status.lock.unlock(); }
        Mockito.lenient().when(cacheIntervalService.getCurrentIntervalMs()).thenReturn(1000L);
        var cycle = new LongAdder();
        cycle.add(49);
        var lease = status.tryBeginCachePoll();
        var skipped = new GrpcCacheStatusCheckRunner("test-model", status, lease, directory,
                engineHealthReporter, workerStatusRpcClient, localKvCacheAwareManager, cacheIntervalService,
                5000L, cycle, 20L, false, Runnable::run);
        assertNull(skipped.whenDue());
        lease.close();
        cycle.add(2); // Cycle 50 was coalesced while discovery was busy.
        var response = new CompletableFuture<EngineRpcService.CacheStatusPB>();
        when(workerStatusRpcClient.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(role)))
                .thenReturn(response);
        var due = new GrpcCacheStatusCheckRunner("test-model", status, status.tryBeginCachePoll(), directory,
                engineHealthReporter, workerStatusRpcClient, localKvCacheAwareManager, cacheIntervalService,
                5000L, cycle, 20L, false, Runnable::run);
        assertNotNull(due.whenDue());
        assertNull(status.tryBeginCachePoll(), "queued polls retain exclusive ownership");
        cycle.increment(); // The executor starts the accepted task one tick later.
        due.run();
        verify(workerStatusRpcClient).getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(role));
        response.completeExceptionally(new IllegalStateException("transport unavailable"));
        var retryLease = status.tryBeginCachePoll();
        assertNotNull(retryLease, "failed asynchronous polls release ownership");
        retryLease.close();
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
            "PREFILL,false", "PDFUSION,false", "DECODE,false",
            "PREFILL,true", "PDFUSION,true", "DECODE,true"})
    void delayedPollAccountsForThePeriodInWhichRpcActuallyStarts(RoleType role, boolean failRpc) {
        WorkerStatus status = RunnerTestSupport.discovered(role, null, "127.0.0.1", 8080, 8081, "test-site");
        var directory = directory(status);
        status.lock.lock();
        try { status.publishCacheStatus(org.flexlb.dao.master.CacheStatus.builder().version(1).blockSize(128).build()); }
        finally { status.lock.unlock(); }
        Mockito.lenient().when(cacheIntervalService.getCurrentIntervalMs()).thenReturn(1000L);
        Mockito.lenient().when(localKvCacheAwareManager.updateEngineBlockCache(status))
                .thenReturn(WorkerCacheUpdateResult.builder().success(true).build());
        var response = new CompletableFuture<EngineRpcService.CacheStatusPB>();
        when(workerStatusRpcClient.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(role)))
                .thenReturn(response);
        var cycle = new LongAdder();
        java.util.function.Function<WorkerStatus.PollLease, GrpcCacheStatusCheckRunner> poll = lease ->
                new GrpcCacheStatusCheckRunner("test-model", status, lease, directory,
                        engineHealthReporter, workerStatusRpcClient, localKvCacheAwareManager, cacheIntervalService,
                        5000L, cycle, 20L, false, Runnable::run);
        cycle.add(49);
        poll.apply(status.tryBeginCachePoll()).run();
        verifyNoInteractions(workerStatusRpcClient);
        cycle.increment();
        var delayed = poll.apply(status.tryBeginCachePoll());
        assertNotNull(delayed.whenDue());
        assertNull(status.tryBeginCachePoll(), "a queued poll retains its lease");
        cycle.add(50); // The executor starts the task in period 100, not period 50.
        delayed.run();
        if (failRpc) { response.completeExceptionally(new IllegalStateException("unavailable")); }
        else { response.complete(EngineRpcService.CacheStatusPB.newBuilder().setVersion(2).setBlockSize(128).build()); }
        verify(workerStatusRpcClient, times(1)).getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(role));
        cycle.increment();
        try (var lease = status.tryBeginCachePoll()) {
            assertNotNull(lease);
            assertNull(poll.apply(lease).whenDue(), "the RPC at cycle 100 already covered this period");
        }
        cycle.add(49);
        var next = poll.apply(status.tryBeginCachePoll());
        assertNotNull(next.whenDue(), "the next regular period must still be polled");
        next.run();
        verify(workerStatusRpcClient, times(2)).getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(role));
    }

    @Test
    void decodeInitializationRetriesOnNextTickAfterFailure() {
        WorkerStatus status = RunnerTestSupport.discovered(RoleType.DECODE, null, "127.0.0.1", 8080, 8081, "test-site");
        var directory = RunnerTestSupport.endpointRegistry(Mockito.mock(org.flexlb.config.ConfigService.class));
        directory.currentOrDiscover(RoleType.DECODE, status.getIpPort(), () -> status);
        when(workerStatusRpcClient.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE)))
                .thenReturn(CompletableFuture.failedFuture(new IllegalStateException("unavailable")))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(1).setBlockSize(128).build()));
        var cycle = new LongAdder();
        cycle.add(7);
        for (int attempt = 0; attempt < 2; attempt++) {
            var poll = new GrpcCacheStatusCheckRunner("test-model", status, status.tryBeginCachePoll(), directory,
                    engineHealthReporter, workerStatusRpcClient, localKvCacheAwareManager, cacheIntervalService,
                    5000L, cycle, 20L, false, Runnable::run);
            assertNotNull(poll.whenDue());
            poll.run();
            cycle.increment();
        }
        assertEquals(128, status.getCacheStatus().getBlockSize());
        verify(workerStatusRpcClient, times(2)).getCacheStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.DECODE));
    }

    @Test
    void testGrpcCacheStatusCheckRunner() {
        // Arrange
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";

        WorkerStatus workerStatus = workerStatus();
        EndpointRegistry directory = directory(workerStatus);

        EngineRpcService.CacheStatusPB cacheStatusPB = EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(1)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build();
        when(workerStatusRpcClient.getCacheStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), eq(RoleType.PREFILL))).thenReturn(CompletableFuture.completedFuture(cacheStatusPB));
        when(localKvCacheAwareManager.updateEngineBlockCache(workerStatus))
                .thenReturn(WorkerCacheUpdateResult.builder()
                        .success(true)
                        .build());

        // Act
        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                modelName,  workerStatus,
                workerStatus.tryBeginCachePoll(),
                directory,
                engineHealthReporter, workerStatusRpcClient,
                localKvCacheAwareManager, cacheIntervalService,
                20, new LongAdder(), 50L, true, Runnable::run);
        runner.run();

        // Give some time for async execution
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Assert
        verify(workerStatusRpcClient).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), eq(-1L), eq(20L), eq(RoleType.PREFILL));
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());
    }

    @Test
    void failedCacheIndexUpdateRetriesTheSameWorkerVersion() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus workerStatus = workerStatus();
        EndpointRegistry directory = Mockito.spy(directory(workerStatus));
        EngineRpcService.CacheStatusPB firstResponse =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(1)
                        .setBlockSize(128)
                        .build();
        EngineRpcService.CacheStatusPB secondResponse =
                EngineRpcService.CacheStatusPB.newBuilder()
                        .setVersion(2)
                        .setBlockSize(128)
                        .build();
        when(workerStatusRpcClient.getCacheStatusAsync(
                anyString(), anyInt(), anyLong(),
                anyLong(), eq(RoleType.PREFILL)))
                .thenReturn(
                        CompletableFuture.completedFuture(firstResponse),
                        CompletableFuture.completedFuture(secondResponse),
                        CompletableFuture.completedFuture(secondResponse));
        when(localKvCacheAwareManager.updateEngineBlockCache(workerStatus))
                .thenReturn(
                        WorkerCacheUpdateResult.builder()
                                .success(true)
                                .build(),
                        WorkerCacheUpdateResult.builder()
                                .success(false)
                                .errorMessage("index failed")
                                .build(),
                        WorkerCacheUpdateResult.builder()
                                .success(true)
                                .build());

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());
        verify(directory, times(1)).publishCacheAvailability(workerStatus);

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(1L, workerStatus.cacheIndexSnapshot().indexedVersion());
        verify(directory, times(1)).publishCacheAvailability(workerStatus);
        assertEquals(2L, workerStatus.getCacheStatus().getVersion());

        runCachePoll(ipPort, workerStatus, directory, false);
        assertEquals(2L, workerStatus.cacheIndexSnapshot().indexedVersion());
        verify(directory, times(2)).publishCacheAvailability(workerStatus);
        verify(workerStatusRpcClient).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081),
                eq(-1L), eq(20L), eq(RoleType.PREFILL));
        verify(workerStatusRpcClient, times(2)).getCacheStatusAsync(
                eq("127.0.0.1"), eq(8081),
                eq(1L), eq(20L), eq(RoleType.PREFILL));
    }

    @Test
    void staleGenerationCallbackCannotPublishAddressCache() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus oldStatus = workerStatus();
        var config = Mockito.mock(org.flexlb.config.ConfigService.class);
        EndpointRegistry directory = RunnerTestSupport.endpointRegistry(config);
        WorkerStatus replacement = workerStatus();
        directory.currentOrDiscover(RoleType.PREFILL, ipPort, () -> replacement);
        CompletableFuture<EngineRpcService.CacheStatusPB> response =
                new CompletableFuture<>();
        when(workerStatusRpcClient.getCacheStatusAsync(
                anyString(), anyInt(), anyLong(),
                anyLong(), eq(RoleType.PREFILL))).thenReturn(response);

        GrpcCacheStatusCheckRunner runner = new GrpcCacheStatusCheckRunner(
                "test-model",
                oldStatus, oldStatus.tryBeginCachePoll(), directory,
                engineHealthReporter, workerStatusRpcClient,
                localKvCacheAwareManager, cacheIntervalService,
                20, new LongAdder(), 50L, true, Runnable::run);
        runner.run();

        response.complete(EngineRpcService.CacheStatusPB.newBuilder()
                .setVersion(7)
                .setAvailableKvCache(1000)
                .setTotalKvCache(2000)
                .setBlockSize(128)
                .build());

        verify(localKvCacheAwareManager, never())
                .updateEngineBlockCache(oldStatus);
    }

    private void runCachePoll(
            String ipPort,
            WorkerStatus workerStatus,
            EndpointRegistry directory,
            boolean debug) {
        new GrpcCacheStatusCheckRunner(
                "test-model",
                workerStatus, workerStatus.tryBeginCachePoll(), directory,
                engineHealthReporter, workerStatusRpcClient,
                localKvCacheAwareManager, cacheIntervalService,
                20, new LongAdder(), 50L, debug, Runnable::run).run();
    }

    private static WorkerStatus workerStatus() {
        return RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
    }

    private static EndpointRegistry directory(WorkerStatus status) {
        var config = Mockito.mock(org.flexlb.config.ConfigService.class);
        EndpointRegistry directory = RunnerTestSupport.endpointRegistry(config);
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
        return directory;
    }

}
