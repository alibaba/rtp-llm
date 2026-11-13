package org.flexlb.worker.status;

import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.worker.cache.DynamicCacheIntervalService;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;

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

/**
 * Checks discovery creates and replaces worker generations and submits status polling safely.
 * Malformed addresses and rejected tasks must not strand poll leases or other workers.
 */
@ExtendWith(MockitoExtension.class)
class EngineSyncRunnerTest {
    private final String modelName = "test-model";

    private EndpointRegistry endpointRegistry;
    @Mock private WorkerAddressService workerAddressService;
    @Mock private ExecutorService statusCheckExecutor;
    @Mock private ServiceMetricsReporter serviceMetricsReporter;
    @Mock private WorkerStatusRpcClient workerStatusRpcClient;

    private final RoleType roleType = RoleType.PREFILL;

    @Mock private CacheAwareService localKvCacheAwareManager;
    @Mock private DynamicCacheIntervalService cacheIntervalService;

    private final long syncRequestTimeoutMs = 5000L;
    private final long syncCount = 0L;
    private final long syncEngineStatusInterval = 20L;

    private Runnable engineSyncRunner;

    @BeforeEach
    void setUp() {
        endpointRegistry =
                RunnerTestSupport.endpointRegistry(
                        org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                org.flexlb.balance.schedule.SchedulingTestConfig.newConfig()));
        engineSyncRunner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        endpointRegistry,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        roleType,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);
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
        Runnable runnerWithEmptyDirectory =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        endpointRegistry,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        roleType,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);

        // Execute
        runnerWithEmptyDirectory.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_initialize_poll_health_at_discovery_time() {
        String ipPort = "127.0.0.1:8080@0";
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(DiscoveredHost.of("127.0.0.1", 8080)));
        Runnable runner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        endpointRegistry,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        RoleType.VIT,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);

        runner.run();

        assertTrue(
                endpointRegistry
                                .statusSnapshot(RoleType.VIT)
                                .get(ipPort)
                                .pollHealth()
                                .lastSuccessfulPollUs()
                        > 0);
    }

    @Test
    void malformedDiscoveryAddressDoesNotStarveLaterWorkers() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(
                        List.of(DiscoveredHost.of("", 8080), DiscoveredHost.of("127.0.0.1", 8080)));

        engineSyncRunner.run();

        assertNotNull(endpointRegistry.statusSnapshot(roleType).get("127.0.0.1:8080@0"));
        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }

    @Test
    void executorRejectionReturnsBothExactPollLeases() {
        String ipPort = "127.0.0.1:8080@0";
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(DiscoveredHost.of("127.0.0.1", 8080)));
        when(statusCheckExecutor.submit(any(Runnable.class)))
                .thenThrow(new RejectedExecutionException("full"));
        Runnable runner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        endpointRegistry,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        RoleType.PREFILL,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);

        runner.run();

        WorkerStatus status = endpointRegistry.statusSnapshot(RoleType.PREFILL).get(ipPort);
        WorkerStatus.PollLease statusLease = status.tryBeginStatusPoll();
        WorkerStatus.PollLease cacheLease = status.tryBeginCachePoll();
        assertNotNull(statusLease);
        assertNotNull(cacheLease);
        statusLease.close();
        cacheLease.close();
    }

    @Test
    void should_remove_status_and_endpoint_when_service_discovery_is_empty() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.schedule.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        EndpointRegistry directory = registry;
        String ipPort = "127.0.0.1:8080@0";
        WorkerStatus status =
                Mockito.spy(
                        RunnerTestSupport.discovered(
                                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, "test-site"));
        discover(directory, status);
        RunnerTestSupport.publishEndpoint(registry, RoleType.PREFILL, ipPort, status);
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of());

        Runnable runner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        directory,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        RoleType.PREFILL,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);
        runner.run();

        assertFalse(status.isActiveGeneration());
        RunnerTestSupport.awaitCondition(
                () -> !directory.statusSnapshot(RoleType.PREFILL).containsKey(ipPort));
        assertNull(registry.getEndpoint(RoleType.PREFILL, ipPort));
        registry.close();
    }

    @Test
    void discoveryPublishesOnlyInactiveStatusUntilFirstResponseCommits() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        // Discovery alone never publishes an endpoint, so the registry never
        // resolves the load-balance config in this scenario. Keep the stub
        // lenient so strict stubbing does not flag it as unnecessary.
        Mockito.lenient()
                .when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.schedule.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        EndpointRegistry directory = registry;
        Runnable runner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        directory,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        RoleType.PREFILL,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(new DiscoveredHost("127.0.0.1", 61000)));

        try {
            runner.run();

            WorkerStatus discovered =
                    directory.statusSnapshot(RoleType.PREFILL).get("127.0.0.1:61000@0");
            assertEquals(RoleType.PREFILL, discovered.getRole());
            assertFalse(
                    discovered.pollHealth().reportedAlive(),
                    "service discovery alone must not make a worker routable");
            assertNull(
                    registry.getEndpoint(RoleType.PREFILL, "127.0.0.1:61000@0"),
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
    void failedDiscoveryTelemetryStillReplacesTopologyGeneration() {
        Mockito.doThrow(new IllegalStateException("monitor unavailable"))
                .when(serviceMetricsReporter)
                .reportServiceDiscoveryResult(modelName, 1, RoleType.PREFILL.toString());

        assertTopologyReplacement("group-a", "group-b");

        verify(serviceMetricsReporter, times(2))
                .reportServiceDiscoveryResult(modelName, 1, RoleType.PREFILL.toString());
        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }

    @Test
    void should_not_publish_running_load_variance_from_one_observation() {
        EndpointRegistry registry =
                Mockito.spy(
                        RunnerTestSupport.endpointRegistry(
                                org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                        org.flexlb.balance.schedule.SchedulingTestConfig
                                                .newConfig())));
        WorkerEndpoint firstEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerStatus first = varianceStatus("127.0.0.1", 61001, 10.0);
        WorkerStatus second = varianceStatus("127.0.0.2", 61002, 20.0);
        when(firstEndpoint.getStatus()).thenReturn(first);
        when(secondEndpoint.getStatus()).thenReturn(second);
        EndpointRegistry directory = registry;
        discover(directory, first);
        discover(directory, second);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(
                        List.of(
                                DiscoveredHost.of(first.getIp(), first.getPort()),
                                DiscoveredHost.of(second.getIp(), second.getPort())));
        Mockito.doReturn(firstEndpoint)
                .when(registry)
                .getEndpoint(RoleType.PREFILL, first.getLogicalIpPort());
        Mockito.doReturn(secondEndpoint)
                .when(registry)
                .getEndpoint(RoleType.PREFILL, second.getLogicalIpPort());
        when(firstEndpoint.tryGetLoadMetric()).thenReturn(OptionalLong.of(10L));
        when(secondEndpoint.tryGetLoadMetric()).thenReturn(OptionalLong.empty());

        Runnable runner = varianceRunner(directory);

        runner.run();

        verify(serviceMetricsReporter)
                .reportStepLatencyVariance(modelName, RoleType.PREFILL.toString(), 50.0);
        verify(serviceMetricsReporter, never())
                .reportRunningLoadVariance(any(), any(), Mockito.anyDouble());
    }

    @Test
    void should_publish_running_load_variance_from_two_observations() {
        EndpointRegistry registry =
                Mockito.spy(
                        RunnerTestSupport.endpointRegistry(
                                org.flexlb.balance.schedule.SchedulingTestConfig.configService(
                                        org.flexlb.balance.schedule.SchedulingTestConfig
                                                .newConfig())));
        WorkerEndpoint firstEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerEndpoint secondEndpoint = Mockito.mock(WorkerEndpoint.class);
        WorkerStatus first = varianceStatus("127.0.0.1", 61001, 10.0);
        WorkerStatus second = varianceStatus("127.0.0.2", 61002, 20.0);
        when(firstEndpoint.getStatus()).thenReturn(first);
        when(secondEndpoint.getStatus()).thenReturn(second);
        EndpointRegistry directory = registry;
        discover(directory, first);
        discover(directory, second);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(
                        List.of(
                                DiscoveredHost.of(first.getIp(), first.getPort()),
                                DiscoveredHost.of(second.getIp(), second.getPort())));
        Mockito.doReturn(firstEndpoint)
                .when(registry)
                .getEndpoint(RoleType.PREFILL, first.getLogicalIpPort());
        Mockito.doReturn(secondEndpoint)
                .when(registry)
                .getEndpoint(RoleType.PREFILL, second.getLogicalIpPort());
        when(firstEndpoint.tryGetLoadMetric()).thenReturn(OptionalLong.of(10L));
        when(secondEndpoint.tryGetLoadMetric()).thenReturn(OptionalLong.of(30L));

        Runnable runner = varianceRunner(directory);

        runner.run();

        verify(serviceMetricsReporter)
                .reportRunningLoadVariance(modelName, RoleType.PREFILL.toString(), 200.0);
    }

    private Runnable varianceRunner(EndpointRegistry directory) {
        return org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                modelName,
                directory,
                workerAddressService,
                statusCheckExecutor,
                serviceMetricsReporter,
                workerStatusRpcClient,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false,
                statusCheckExecutor,
                statusCheckExecutor);
    }

    private void assertTopologyReplacement(String oldGroup, String newGroup) {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(org.flexlb.balance.schedule.SchedulingTestConfig.newConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        EndpointRegistry directory = registry;
        String ipPort = "127.0.0.1:61000@0";
        WorkerStatus oldStatus =
                RunnerTestSupport.discovered(
                        RoleType.PREFILL, oldGroup, "127.0.0.1", 61000, 61001, "site-a");
        assertNotNull(oldStatus.tryBeginStatusPoll());
        discover(directory, oldStatus);
        RunnerTestSupport.publishEndpoint(registry, RoleType.PREFILL, ipPort, oldStatus);
        DiscoveredHost replacement =
                new DiscoveredHost("127.0.0.1", 61000, 61001, 61005, "site-b", newGroup);
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(replacement));

        Runnable runner =
                org.flexlb.worker.status.WorkerSyncTestSupport.runner(
                        modelName,
                        directory,
                        workerAddressService,
                        statusCheckExecutor,
                        serviceMetricsReporter,
                        workerStatusRpcClient,
                        RoleType.PREFILL,
                        localKvCacheAwareManager,
                        cacheIntervalService,
                        syncRequestTimeoutMs,
                        syncCount,
                        syncEngineStatusInterval,
                        false,
                        statusCheckExecutor,
                        statusCheckExecutor);
        try {
            // First discovery pass retires the old topology generation: it
            // detaches the endpoint, marks the status RETIRING, and removes the
            // business identity from the map. The replacement generation is
            // deliberately deferred until real endpoint cleanup has removed
            // the RETIRING holder, so no routable identity exists yet.
            runner.run();

            RunnerTestSupport.awaitCondition(
                    () -> !directory.statusSnapshot(RoleType.PREFILL).containsKey(ipPort));
            assertNull(
                    registry.getEndpoint(RoleType.PREFILL, ipPort),
                    "old endpoint is detached immediately and replacement is unpublished");
            // A subsequent discovery pass publishes the replacement generation
            // for the same address under the new topology group.
            runner.run();

            WorkerStatus current = directory.statusSnapshot(RoleType.PREFILL).get(ipPort);
            assertTrue(current != oldStatus);
            assertTrue(current.getGenerationId() > oldStatus.getGenerationId());
            assertEquals(newGroup, current.getGroup());
            assertFalse(
                    current.pollHealth().reportedAlive(),
                    "replacement is not routable before its first status commit");
            assertNull(
                    registry.getEndpoint(RoleType.PREFILL, ipPort),
                    "replacement endpoint is unpublished before its first status commit");
        } finally {
            registry.close();
        }
    }

    private static void discover(EndpointRegistry directory, WorkerStatus status) {
        directory.currentOrDiscover(status.getRole(), status.getLogicalIpPort(), () -> status);
    }

    private static WorkerStatus varianceStatus(String ip, int port, double stepLatencyMs) {
        WorkerStatus status =
                RunnerTestSupport.discovered(RoleType.PREFILL, "", ip, port, port + 1, "");
        RunnerTestSupport.publish(
                status,
                RunnerTestSupport.response(status, true, 1L, 0L, 0L, stepLatencyMs, Map.of()));
        assertNotNull(status.tryBeginStatusPoll());
        assertNotNull(status.tryBeginCachePoll());
        return status;
    }
}
