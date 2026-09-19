package org.flexlb.sync.runner;

import io.grpc.Status;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
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
    private static final long STATUS_STALE_AFTER_US = 10_000_000L;

    private EngineSyncRunner engineSyncRunner;

    @BeforeEach
    void setUp() {
        workerDirectory = new WorkerDirectory(endpointRegistry);
        engineSyncRunner = new EngineSyncRunner(
                modelName,
                workerDirectory,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false,
                STATUS_STALE_AFTER_US
        );
    }

    @Test
    void should_apply_configured_vit_policy_to_scheduled_status_polls() {
        FlexlbConfig config = ConfigService.parse("""
                {"workerRegistry":{"health":{
                    "statusRpcTimeoutMs":20,
                    "vitStatusRpcTimeoutMs":1000,
                    "retainVitAliveOnTimeout":false,
                    "statusStaleAfterMs":2000
                }}}
                """);
        ConfigService service = Mockito.mock(ConfigService.class);
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(service);
        WorkerDirectory directory = new WorkerDirectory(registry);
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(host));
        var response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(1L).setAlive(true).build();
        when(engineGrpcService.getWorkerStatusAsync(
                Mockito.anyString(), Mockito.anyInt(), Mockito.anyLong(),
                Mockito.anyLong(), any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response),
                        CompletableFuture.failedFuture(
                                Status.DEADLINE_EXCEEDED.asRuntimeException()));
        Mockito.doAnswer(call -> {
            call.getArgument(0, Runnable.class).run();
            return CompletableFuture.completedFuture(null);
        }).when(statusCheckExecutor).submit(any(Runnable.class));
        Mockito.doAnswer(call -> {
            call.getArgument(0, Runnable.class).run();
            return null;
        }).when(statusCheckExecutor).execute(any(Runnable.class));
        var health = config.getWorkerRegistry().getHealth();
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, directory, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.VIT,
                localKvCacheAwareManager, cacheIntervalService,
                health.getStatusRpcTimeoutMs(), syncCount, syncEngineStatusInterval,
                false, health.getStatusStaleAfterMs() * 1000L,
                health.getVitStatusRpcTimeoutMs(), health.isRetainVitAliveOnTimeout());
        try {
            runner.run();
            WorkerStatus status = directory.statusSnapshot(RoleType.VIT).get(host.getIpPort());
            assertNotNull(registry.get(RoleType.VIT, host.getIpPort()));
            runner.run();
            runner.run();
            assertTrue(status.isActiveGeneration());
            runner.run();

            assertFalse(status.isActiveGeneration());
            assertNull(registry.get(RoleType.VIT, host.getIpPort()));
            assertFalse(directory.isCurrentStatus(RoleType.VIT, host.getIpPort(), status));
            verify(engineGrpcService, times(4)).getWorkerStatusAsync(
                    Mockito.eq("127.0.0.1"), Mockito.eq(8081), Mockito.anyLong(),
                    Mockito.eq(1000L), Mockito.eq(RoleType.VIT));
            verify(statusCheckExecutor, times(4)).submit(any(Runnable.class));
        } finally {
            registry.close();
        }
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
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false,
                STATUS_STALE_AFTER_US
        );

        // Execute
        runnerWithEmptyDirectory.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_start_new_worker_expiration_window_at_discovery_time() {
        String ipPort = "127.0.0.1:8080";
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerDirectory, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.VIT,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, false, STATUS_STALE_AFTER_US);

        runner.run();

        assertTrue(workerDirectory.statusSnapshot(RoleType.VIT).get(ipPort)
                .pollHealth().lastSuccessfulPollUs() > 0);
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
                statusCheckExecutor, engineHealthReporter, engineGrpcService,
                RoleType.PREFILL, localKvCacheAwareManager,
                cacheIntervalService, syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, false, STATUS_STALE_AFTER_US);

        runner.run();

        WorkerStatus status = workerDirectory.statusSnapshot(
                RoleType.PREFILL).get(ipPort);
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
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry);
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = Mockito.spy(RunnerTestSupport.discovered(
                RoleType.PREFILL, null, "127.0.0.1",
                8080, 8081, "test-site"));
        when(status.pollHealth()).thenReturn(new WorkerStatus.PollHealth(
                System.nanoTime() / 1_000 - 2_000_000L,
                20_000L, 0L, true));
        discover(directory, status);
        RunnerTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ipPort, status);
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of());

        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, directory, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, false,
                1_000_000L);
        runner.run();

        assertFalse(status.isActiveGeneration());
        assertFalse(directory.statusSnapshot(RoleType.PREFILL)
                .containsKey(ipPort));
        assertNull(registry.get(RoleType.PREFILL, ipPort));
        registry.close();
    }

    @Test
    void discoveryPublishesOnlyInactiveStatusUntilFirstResponseCommits() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        // Discovery alone never publishes an endpoint, so the registry never
        // resolves the load-balance config in this scenario. Keep the stub
        // lenient so strict stubbing does not flag it as unnecessary.
        Mockito.lenient().when(configService.loadBalanceConfig())
                .thenReturn(new FlexlbConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry);
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName,
                directory,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false,
                STATUS_STALE_AFTER_US
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
        WorkerDirectory directory = new WorkerDirectory(registry);
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
        WorkerDirectory directory = new WorkerDirectory(registry);
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

    @Test
    void should_submit_only_worker_status_check_for_vit() {
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(host));

        createRunner(RoleType.VIT).run();

        ArgumentCaptor<Runnable> taskCaptor = ArgumentCaptor.forClass(Runnable.class);
        verify(statusCheckExecutor).submit(taskCaptor.capture());
        assertTrue(taskCaptor.getValue() instanceof GrpcWorkerStatusRunner);
        WorkerStatus workerStatus = workerDirectory.statusSnapshot(RoleType.VIT)
                .get(host.getIpPort());
        WorkerStatus.PollLease cacheLease = workerStatus.tryBeginCachePoll();
        assertNotNull(cacheLease);
        cacheLease.close();
    }

    @Test
    void should_submit_worker_and_cache_status_checks_for_prefill() {
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(host));

        createRunner(RoleType.PREFILL).run();

        ArgumentCaptor<Runnable> taskCaptor = ArgumentCaptor.forClass(Runnable.class);
        verify(statusCheckExecutor, times(2)).submit(taskCaptor.capture());
        assertTrue(taskCaptor.getAllValues().stream()
                .anyMatch(GrpcWorkerStatusRunner.class::isInstance));
        assertTrue(taskCaptor.getAllValues().stream()
                .anyMatch(GrpcCacheStatusCheckRunner.class::isInstance));
    }

    private EngineSyncRunner createRunner(RoleType actualRoleType) {
        return new EngineSyncRunner(
                modelName, workerDirectory, workerAddressService,
                statusCheckExecutor, engineHealthReporter, engineGrpcService,
                actualRoleType, localKvCacheAwareManager, cacheIntervalService,
                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                false, STATUS_STALE_AFTER_US);
    }

    private EngineSyncRunner varianceRunner(WorkerDirectory directory) {
        return new EngineSyncRunner(
                modelName, directory, workerAddressService,
                statusCheckExecutor, engineHealthReporter, engineGrpcService,
                RoleType.PREFILL, localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs, syncCount, syncEngineStatusInterval,
                false, STATUS_STALE_AFTER_US);
    }

    private void assertTopologyReplacement(
            String oldGroup, String newGroup) {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig())
                .thenReturn(new FlexlbConfig());
        EndpointRegistry registry = RunnerTestSupport.endpointRegistry(configService);
        WorkerDirectory directory = new WorkerDirectory(registry);
        String ipPort = "127.0.0.1:61000";
        WorkerStatus oldStatus = RunnerTestSupport.discovered(
                RoleType.PREFILL, oldGroup, "127.0.0.1",
                61000, 61001, "site-a");
        assertNotNull(oldStatus.tryBeginStatusPoll());
        discover(directory, oldStatus);
        RunnerTestSupport.publishEndpoint(registry,
                RoleType.PREFILL, ipPort, oldStatus);
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
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                cacheIntervalService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false, STATUS_STALE_AFTER_US);
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
        assertNotNull(status.tryBeginStatusPoll());
        assertNotNull(status.tryBeginCachePoll());
        return status;
    }

    private static void discover(
            WorkerDirectory directory, WorkerStatus status) {
        directory.currentOrDiscover(
                status.getRole(), status.getIpPort(), () -> status);
    }
}
