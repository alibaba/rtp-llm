package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.sync.schedule.ExpirationCleaner;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EngineSyncRunnerTest {

    private final String modelName = "test-model";

    @Test
    void healthAndCacheTasksUseDifferentExecutorsAndCacheRejectionReleasesLease() {
        ExecutorService health = Mockito.mock(ExecutorService.class);
        ExecutorService cache = Mockito.mock(ExecutorService.class);
        WorkerDirectory directory = new WorkerDirectory(endpointRegistry);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        when(cache.submit(any(Runnable.class))).thenThrow(new RejectedExecutionException("cache full"));
        EngineSyncRunner runner = new EngineSyncRunner(modelName, directory, workerAddressService,
                health, cache, engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                localKvCacheAwareManager, cacheIntervalService, syncRequestTimeoutMs, syncCount.longValue(),
                syncEngineStatusInterval, false);
        runner.run();
        verify(health).submit(any(GrpcWorkerStatusRunner.class));
        verify(health, never()).submit(any(GrpcCacheStatusCheckRunner.class));
        verify(cache).submit(any(GrpcCacheStatusCheckRunner.class));
        WorkerStatus status = directory.statusSnapshot(RoleType.PREFILL).get("127.0.0.1:8080");
        WorkerStatus.PollLease lease = status.tryBeginCachePoll(1L);
        org.junit.jupiter.api.Assertions.assertNotNull(lease, "cache rejection must allow retry");
        lease.close();
    }

    @Mock
    private EndpointRegistry endpointRegistry;

    private WorkerDirectory workerDirectory;

    @Mock
    private WorkerAddressService workerAddressService;

    @Mock
    private ExecutorService statusCheckExecutor;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private EngineGrpcService engineGrpcService;

    private final RoleType roleType = RoleType.PREFILL;

    @Mock
    private CacheAwareService localKvCacheAwareManager;

    @Mock
    private DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs = 5000L;

    @Mock
    private LongAdder syncCount;

    private final long syncEngineStatusInterval = 20L;

    private EngineSyncRunner engineSyncRunner;

    @BeforeEach
    void setUp() {
        workerDirectory = new WorkerDirectory(endpointRegistry);
        engineSyncRunner = new EngineSyncRunner(
                modelName,
                workerDirectory,
                workerAddressService,
                statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount.longValue(),
                syncEngineStatusInterval,
                false
        );
    }

    @Test
    void should_not_submit_tasks_when_no_workers_exist() {
        // Execute
        engineSyncRunner.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_handle_null_worker_status_gracefully() {
        EngineSyncRunner runnerWithEmptyDirectory = new EngineSyncRunner(
                modelName,
                new WorkerDirectory(endpointRegistry),
                workerAddressService,
                statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount.longValue(),
                syncEngineStatusInterval,
                false
        );

        // Execute
        runnerWithEmptyDirectory.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void discoveredWorkerRemainsUnknownUntilItsFirstStatusResponse() {
        String ipPort = "127.0.0.1:8080";
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerDirectory, workerAddressService, statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.VIT,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount.longValue(),
                syncEngineStatusInterval, false);

        runner.run();

        WorkerStatus discovered = workerDirectory.statusSnapshot(RoleType.VIT).get(ipPort);
        assertEquals(WorkerStatus.StatusContact.UNKNOWN, discovered.pollHealth().contact());
        assertTrue(discovered.isActiveGeneration());
    }

    @Test
    void queuedRunnerKeepsDispatchCycleAndCacheCadence() {
        LongAdder cycles = new LongAdder();
        cycles.add(7L);
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerDirectory, workerAddressService, statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, roleType, localKvCacheAwareManager,
                cacheIntervalService, syncRequestTimeoutMs, cycles.longValue(),
                syncEngineStatusInterval, false);
        cycles.increment();
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        List<Runnable> submitted = new ArrayList<>();
        when(statusCheckExecutor.submit(any(Runnable.class))).thenAnswer(invocation -> {
            submitted.add(invocation.getArgument(0));
            return CompletableFuture.completedFuture(null);
        });

        runner.run();

        WorkerStatus status = workerDirectory.statusSnapshot(roleType).get("127.0.0.1:8080");
        var snapshot = status.trace().snapshot();
        assertEquals(7L, snapshot.cachePoll().syncCycleId());
        when(cacheIntervalService.getCurrentIntervalMs()).thenReturn(40L);
        submitted.get(1).run();
        assertEquals("POLL_SKIPPED:CACHE_INTERVAL", status.trace().snapshot().cachePoll().outcome());
        verify(engineGrpcService, never()).getCacheStatusAsync(
                any(), Mockito.anyInt(), any(), Mockito.anyLong(), Mockito.anyLong(), any());
    }

    @Test
    void executorRejectionReturnsBothExactPollLeases() {
        String ipPort = "127.0.0.1:8080";
        when(workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        when(statusCheckExecutor.submit(any(Runnable.class)))
                .thenThrow(new RejectedExecutionException("full"));
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerDirectory, workerAddressService,
                statusCheckExecutor, statusCheckExecutor, engineHealthReporter, engineGrpcService,
                RoleType.PREFILL, localKvCacheAwareManager,
                cacheIntervalService, syncRequestTimeoutMs, syncCount.longValue(),
                syncEngineStatusInterval, false);

        runner.run();

        WorkerStatus status = workerDirectory.statusSnapshot(
                RoleType.PREFILL).get(ipPort);
        WorkerStatus.PollLease statusLease = status.tryBeginStatusPoll(1L);
        WorkerStatus.PollLease cacheLease = status.tryBeginCachePoll(1L);
        assertNotNull(statusLease);
        assertNotNull(cacheLease);
        statusLease.close();
        cacheLease.close();
    }

    @Test
    void should_remove_status_and_endpoint_when_service_discovery_is_empty() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site");
        discover(directory, status);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ipPort, status);
        WorkerStatus.PollLease inFlightPoll = status.tryBeginStatusPoll(1L);
        assertNotNull(inFlightPoll);
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of());

        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, directory, workerAddressService, statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount.longValue(),
                syncEngineStatusInterval, false);
        runner.run();

        assertFalse(status.isActiveGeneration());
        assertFalse(directory.statusSnapshot(RoleType.PREFILL)
                .containsKey(ipPort));
        assertNull(registry.get(RoleType.PREFILL, ipPort));
        inFlightPoll.close();
        registry.close();
    }

    @Test
    void discoveryFailureKeepsThePublishedGenerationAndContinuesPollingIt() {
        ConfigService config = Mockito.mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        try {
            WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
            WorkerStatus status = RunnerTestSupport.discovered(
                    RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "site");
            discover(directory, status);
            WorkerEndpoint endpoint = RunnerTestSupport.publishEndpoint(
                    registry, RoleType.PREFILL, status.getIpPort(), status);
            when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                    .thenThrow(new IllegalStateException("VIP unavailable"));
            EngineSyncRunner runner = new EngineSyncRunner(
                    modelName, directory, workerAddressService,
                    statusCheckExecutor, statusCheckExecutor,
                    engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                    localKvCacheAwareManager, cacheIntervalService, syncRequestTimeoutMs,
                    syncCount.longValue(), syncEngineStatusInterval, false);

            runner.run();

            assertTrue(status.isActiveGeneration());
            assertTrue(directory.statusSnapshot(RoleType.PREFILL).get(status.getIpPort()) == status);
            assertTrue(registry.get(RoleType.PREFILL, status.getIpPort()) == endpoint);
            verify(statusCheckExecutor).submit(any(GrpcWorkerStatusRunner.class));
            verify(statusCheckExecutor).submit(any(GrpcCacheStatusCheckRunner.class));
            verify(localKvCacheAwareManager, never()).removeEngineBlockCache(any(), any());
        } finally {
            registry.close();
        }
    }

    @Test
    void confirmedRemovalSurvivesBusyWorkerLockAndLaterDiscoveryFailure() throws Exception {
        ConfigService config = Mockito.mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(config);
        try (ExecutorService executor = Executors.newSingleThreadExecutor()) {
            WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
            WorkerStatus status = RunnerTestSupport.discovered(
                    RoleType.PREFILL, "group", "127.0.0.1", 8080, 8081, "site");
            discover(directory, status);
            RunnerTestSupport.publishEndpoint(registry, RoleType.PREFILL, status.getIpPort(), status);
            when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                    .thenReturn(List.of()).thenThrow(new IllegalStateException("VIP unavailable"));
            EngineSyncRunner runner = new EngineSyncRunner(
                    modelName, directory, workerAddressService,
                    statusCheckExecutor, statusCheckExecutor,
                    engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                    localKvCacheAwareManager, cacheIntervalService, syncRequestTimeoutMs,
                    syncCount.longValue(), syncEngineStatusInterval, false);
            status.lock.lock();
            try {
                executor.submit(runner).get(5, TimeUnit.SECONDS);
                assertEquals(WorkerStatus.RetirementReason.DISCOVERY_REMOVED, status.retirementReason());
                assertFalse(status.isAvailableForScheduling());
                assertTrue(directory.isCurrentStatus(RoleType.PREFILL, status.getIpPort(), status));

                executor.submit(runner).get(5, TimeUnit.SECONDS);
                assertEquals(WorkerStatus.RetirementReason.DISCOVERY_REMOVED, status.retirementReason());
                assertFalse(status.isAvailableForScheduling());
                verify(statusCheckExecutor, never()).submit(any(Runnable.class));
            } finally {
                status.lock.unlock();
            }

            new ExpirationCleaner(localKvCacheAwareManager, directory).retireUnavailableWorkers();

            assertFalse(directory.isCurrentStatus(RoleType.PREFILL, status.getIpPort(), status));
            assertNull(registry.get(RoleType.PREFILL, status.getIpPort()));
            verify(localKvCacheAwareManager).removeEngineBlockCache(status, status.trace());
        } finally {
            registry.close();
        }
    }

    @Test
    void discoveryPublishesOnlyInactiveStatusUntilFirstResponseCommits() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        // Discovery alone never publishes an endpoint, so the registry never
        // resolves the load-balance config in this scenario. Keep the stub
        // lenient so strict stubbing does not flag it as unnecessary.
        Mockito.lenient().when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName,
                directory,
                workerAddressService,
                statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount.longValue(),
                syncEngineStatusInterval,
                false
        );
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(new WorkerHost("127.0.0.1", 61000)));

        try {
            runner.run();

            WorkerStatus discovered = directory.statusSnapshot(RoleType.PREFILL)
                    .get("127.0.0.1:61000");
            assertEquals(RoleType.PREFILL, discovered.getRole());
            assertFalse(discovered.pollHealth().reportedAlive(),
                    "service discovery alone must not make a worker routable");
            assertNull(registry.get(RoleType.PREFILL, "127.0.0.1:61000"),
                    "an endpoint is published only by a committed status response");
            verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
        } finally {
            registry.close();
        }
    }

    @Test
    void groupChangeReplacesGenerationEvenWhileOldStatusRpcIsHung() {
        assertTopologyReplacement("group-a", "group-b");
    }

    @Test
    void assigningPreviouslyUnscopedWorkerReplacesGeneration() {
        assertTopologyReplacement(null, "group-b");
    }

    @Test
    void should_not_publish_running_load_variance_from_one_observation() {
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        WorkerEndpoint firstEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerStatus first = varianceStatus("127.0.0.1", 61001, 10.0);
        WorkerStatus second = varianceStatus("127.0.0.2", 61002, 20.0);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        discover(directory, first);
        discover(directory, second);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(
                        WorkerHost.of(first.getIp(), first.getPort()),
                        WorkerHost.of(second.getIp(), second.getPort())));
        when(registry.get(RoleType.PREFILL, first.getIpPort(), first))
                .thenReturn(firstEndpoint);
        when(registry.get(RoleType.PREFILL, second.getIpPort(), second))
                .thenReturn(secondEndpoint);
        when(firstEndpoint.getLoadMetric()).thenReturn(OptionalLong.of(10L));
        when(secondEndpoint.getLoadMetric()).thenReturn(OptionalLong.empty());

        EngineSyncRunner runner = varianceRunner(directory);

        runner.run();

        verify(engineHealthReporter).reportStepLatencyVariance(
                modelName, RoleType.PREFILL.toString(), 50.0);
        verify(engineHealthReporter, never()).reportRunningLoadVariance(
                any(), any(), Mockito.anyDouble());
    }

    @Test
    void should_publish_running_load_variance_from_two_observations() {
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        WorkerEndpoint firstEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerStatus first = varianceStatus("127.0.0.1", 61001, 10.0);
        WorkerStatus second = varianceStatus("127.0.0.2", 61002, 20.0);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        discover(directory, first);
        discover(directory, second);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(
                        WorkerHost.of(first.getIp(), first.getPort()),
                        WorkerHost.of(second.getIp(), second.getPort())));
        when(registry.get(RoleType.PREFILL, first.getIpPort(), first))
                .thenReturn(firstEndpoint);
        when(registry.get(RoleType.PREFILL, second.getIpPort(), second))
                .thenReturn(secondEndpoint);
        when(firstEndpoint.getLoadMetric()).thenReturn(OptionalLong.of(10L));
        when(secondEndpoint.getLoadMetric()).thenReturn(OptionalLong.of(30L));

        EngineSyncRunner runner = varianceRunner(directory);

        runner.run();

        verify(engineHealthReporter).reportRunningLoadVariance(
                modelName, RoleType.PREFILL.toString(), 200.0);
    }

    private EngineSyncRunner varianceRunner(WorkerDirectory directory) {
        return new EngineSyncRunner(
                modelName, directory, workerAddressService,
                statusCheckExecutor, statusCheckExecutor, engineHealthReporter, engineGrpcService,
                RoleType.PREFILL, localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount.longValue(), syncEngineStatusInterval,
                false);
    }

    private void assertTopologyReplacement(
            String oldGroup, String newGroup) {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry, Runnable::run);
        String ipPort = "127.0.0.1:61000";
        WorkerStatus oldStatus = RunnerTestSupport.discovered(
                RoleType.PREFILL, oldGroup, "127.0.0.1",
                61000, 61001, "site-a");
        discover(directory, oldStatus);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, ipPort, oldStatus);
        WorkerStatus.PollLease hungStatusPoll = oldStatus.tryBeginStatusPoll(1L);
        assertNotNull(hungStatusPoll);
        WorkerHost replacement = new WorkerHost(
                "127.0.0.1",
                61000,
                61001,
                61005,
                "site-b",
                newGroup);
        when(workerAddressService.getEngineWorkerList(
                modelName, RoleType.PREFILL))
                .thenReturn(List.of(replacement));

        EngineSyncRunner runner = new EngineSyncRunner(
                modelName,
                directory,
                workerAddressService,
                statusCheckExecutor, statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount.longValue(),
                syncEngineStatusInterval,
                false);
        try {
            // First discovery pass retires the old topology generation: it
            // detaches the endpoint, marks the status RETIRING, and removes the
            // business identity from the map. The replacement generation is
            // deliberately deferred until real endpoint retirement has removed
            // the RETIRING holder, so no routable identity exists yet.
            runner.run();

            assertFalse(directory.statusSnapshot(RoleType.PREFILL)
                            .containsKey(ipPort),
                    "old generation is retired and removed; replacement is deferred");
            assertNull(registry.get(RoleType.PREFILL, ipPort),
                    "old endpoint is detached immediately and replacement is unpublished");
            // A subsequent discovery pass publishes the replacement generation
            // for the same address under the new topology group.
            runner.run();

            WorkerStatus current = directory.statusSnapshot(
                    RoleType.PREFILL).get(ipPort);
            assertTrue(current != oldStatus);
            assertTrue(current.getGenerationId()
                    > oldStatus.getGenerationId());
            assertEquals(newGroup, current.getGroup());
            assertFalse(current.pollHealth().reportedAlive(),
                    "replacement is not routable before its first status commit");
            assertNull(registry.get(RoleType.PREFILL, ipPort),
                    "replacement endpoint is unpublished before its first status commit");
        } finally {
            hungStatusPoll.close();
            registry.close();
        }
    }

    private static WorkerStatus varianceStatus(
            String ip, int port, double stepLatencyMs) {
        WorkerStatus status = RunnerTestSupport.discovered(
                RoleType.PREFILL, "", ip, port, port + 1, "");
        RunnerTestSupport.publish(status, RunnerTestSupport.response(
                status, true, 1L, 0L, 0L,
                stepLatencyMs, Map.of()));
        assertNotNull(status.tryBeginStatusPoll(1L));
        assertNotNull(status.tryBeginCachePoll(1L));
        return status;
    }

    private static void discover(
            WorkerDirectory directory, WorkerStatus status) {
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
    }
}
