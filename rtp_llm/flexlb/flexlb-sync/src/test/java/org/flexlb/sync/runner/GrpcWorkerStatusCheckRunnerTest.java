package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerLifecycleState;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link GrpcWorkerStatusRunner}.
 *
 * <p>Key API changes since original implementation:
 * <ul>
 *   <li>Proto field {@code is_waiting} replaced by {@code TaskPhase phase}</li>
 *   <li>{@code WorkerStatus.runningTaskList} replaces old {@code waitingTaskList + localTaskMap}</li>
 *   <li>Constructor requires {@code FlexlbBatchScheduler + EndpointRegistry} (nullable)</li>
 *   <li>Task list refresh only occurs when status version advances (not on equal version)</li>
 * </ul>
 */
class GrpcWorkerStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    @Test
    void should_callGrpcServiceAndVerifyInteraction_when_runnerExecutes() {
        // Arrange
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";
        String group = "test-group";

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setAvailableConcurrency(10)
                .setRunningQueryLen(5)
                .setWaitingQueryLen(3)
                .setStepLatencyMs(100)
                .setIterateCount(20)
                .setDpSize(2)
                .setTpSize(4)
                .setStatusVersion(100)
                .setAlive(true)
                .build();

        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(workerStatusPB));

        // Act — pass null for FlexlbBatchScheduler and EndpointRegistry (not needed in unit test)
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, Map.of(ipPort, workerStatus),
                engineHealthReporter, engineGrpcService, 20L, null, null, Runnable::run);
        runner.run();

        // Assert — gRPC port is derived from HTTP port 8080 → 8081
        verify(engineGrpcService).getWorkerStatusAsync("127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void should_not_update_task_list_when_status_version_is_unchanged() {
        // When the gRPC response version equals the local version, the status update
        // is skipped — including the runningTaskList refresh. This avoids unnecessary
        // state churn when the engine hasn't changed.
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";
        String group = "test-group";

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.getStatusVersion().set(100L);

        // Use TaskPhasePB instead of the removed is_waiting field
        EngineRpcService.TaskInfoPB taskInfo = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setInputLength(100)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(100L)
                .setAlive(true)
                .addRunningTaskInfo(taskInfo)
                .build();

        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(workerStatusPB));

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, Map.of(ipPort, workerStatus),
                engineHealthReporter, engineGrpcService, 20L, null, null, Runnable::run);
        runner.run();

        // Version not advanced → runningTaskList should NOT be populated from response
        assertNull(workerStatus.getRunningTaskList(),
                "runningTaskList should not be updated when status version is unchanged");
    }

    @Test
    void should_ignore_status_callback_from_expired_generation() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(8080);
        WorkerStatus current = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, current);
        EndpointRegistry registry = registry();
        WorkerEndpoint currentEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, current);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(100L)
                .setAlive(true)
                .build();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                expired, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);
        runner.run();

        assertSame(currentEndpoint, registry.get(RoleType.VIT, ipPort));
        assertSame(current, currentEndpoint.getStatus());
        assertEquals(-1L, expired.getStatusVersion().get());
        registry.close();
    }

    @Test
    void should_remove_endpoint_after_consecutive_grpc_failures() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, status);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                status, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();
        runner.run();
        runner.run();

        assertFalse(status.isAlive());
        assertNull(registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void should_restore_endpoint_with_a_new_probing_generation_after_retirement() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus retired = status(8080);
        retired.getStatusVersion().set(100L);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, retired);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, retired);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner oldRunner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                retired, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        oldRunner.run();
        oldRunner.run();
        oldRunner.run();
        assertEquals(WorkerLifecycleState.CLOSED, retired.getLifecycleState());
        assertFalse(statuses.containsKey(ipPort));

        WorkerStatus recoveredGeneration = new WorkerStatus();
        recoveredGeneration.setRole(RoleType.VIT);
        recoveredGeneration.setIp("127.0.0.1");
        recoveredGeneration.setPort(8080);
        recoveredGeneration.setGrpcPort(8081);
        statuses.put(ipPort, recoveredGeneration);
        EngineRpcService.WorkerStatusPB recovered = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(100L)
                .setAlive(true)
                .build();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(recovered));
        GrpcWorkerStatusRunner newRunner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                recoveredGeneration, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        newRunner.run();

        assertNotSame(retired, recoveredGeneration);
        assertTrue(recoveredGeneration.isAlive());
        assertEquals(WorkerLifecycleState.READY, recoveredGeneration.getLifecycleState());
        assertSame(recoveredGeneration, registry.get(RoleType.VIT, ipPort).getStatus());
        registry.close();
    }

    @Test
    void failure_failure_success_failure_resets_counter_and_keeps_endpoint() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, status);
        EndpointRegistry registry = registry();
        WorkerEndpoint endpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, status);
        CompletableFuture<EngineRpcService.WorkerStatusPB> failed =
                CompletableFuture.failedFuture(new RuntimeException("unavailable"));
        EngineRpcService.WorkerStatusPB success = liveStatus(RoleType.VIT, 100L);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(failed, failed, CompletableFuture.completedFuture(success), failed);
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                status, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();
        runner.run();
        assertEquals(2L, status.getConsecutiveFailures().get());
        runner.run();
        assertEquals(0L, status.getConsecutiveFailures().get());
        runner.run();

        assertEquals(1L, status.getConsecutiveFailures().get());
        assertEquals(WorkerLifecycleState.READY, status.getLifecycleState());
        assertSame(endpoint, registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void failed_probe_does_not_refresh_heartbeat_or_recreate_generation() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, probing);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        EndpointRegistry registry = registry();
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                probing, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();
        runner.run();
        runner.run();

        assertSame(probing, statuses.get(ipPort));
        assertEquals(WorkerLifecycleState.PROBING, probing.getLifecycleState());
        assertEquals(3L, probing.getConsecutiveFailures().get());
        assertEquals(-1L, probing.getStatusLastUpdateTime().get());
        assertNull(registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void delayed_failure_from_old_generation_cannot_touch_replacement() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus old = status(8080);
        WorkerStatus replacement = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, old);
        EndpointRegistry registry = registry();
        CompletableFuture<EngineRpcService.WorkerStatusPB> delayed = new CompletableFuture<>();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(delayed);
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                old, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);
        runner.run();

        statuses.put(ipPort, replacement);
        WorkerEndpoint replacementEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, replacement);
        delayed.completeExceptionally(new RuntimeException("late failure"));

        assertEquals(0L, old.getConsecutiveFailures().get());
        assertEquals(WorkerLifecycleState.READY, replacement.getLifecycleState());
        assertSame(replacementEndpoint, registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void valid_status_is_the_only_path_that_publishes_a_probing_generation() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, probing);
        EndpointRegistry registry = registry();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(liveStatus(RoleType.VIT, 100L)));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                probing, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        assertNull(registry.get(RoleType.VIT, ipPort));
        runner.run();

        assertEquals(WorkerLifecycleState.READY, probing.getLifecycleState());
        assertTrue(probing.getStatusLastUpdateTime().get() > 0L);
        assertSame(probing, registry.get(RoleType.VIT, ipPort).getStatus());
        registry.close();
    }

    @Test
    void telemetry_failure_cannot_block_valid_generation_publication() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, probing);
        EndpointRegistry registry = registry();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(liveStatus(RoleType.VIT, 100L)));
        Mockito.doThrow(new RuntimeException("metrics unavailable"))
                .when(engineHealthReporter)
                .reportStatusCheckRemoteInfo(anyString(), anyString(), anyLong());
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                probing, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();

        assertEquals(WorkerLifecycleState.READY, probing.getLifecycleState());
        assertSame(probing, registry.get(RoleType.VIT, ipPort).getStatus());
        registry.close();
    }

    @Test
    void telemetry_failure_cannot_suppress_health_failure_accounting() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, probing);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        Mockito.doThrow(new RuntimeException("metrics unavailable"))
                .when(engineHealthReporter)
                .reportStatusCheckerFail(anyString(), Mockito.any(), Mockito.any());
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                probing, statuses, engineHealthReporter, engineGrpcService,
                20L, null, null, Runnable::run);

        runner.run();

        assertEquals(1L, probing.getConsecutiveFailures().get());
        assertEquals(-1L, probing.getStatusLastUpdateTime().get());
        assertEquals(WorkerLifecycleState.PROBING, probing.getLifecycleState());
    }

    @Test
    void retried_first_publication_reconciles_equal_version_snapshot_and_commits_cursor() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus probing = probingStatus(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, probing);
        EndpointRegistry registry = Mockito.mock(EndpointRegistry.class);
        WorkerEndpoint endpoint = Mockito.mock(WorkerEndpoint.class);
        FlexlbBatchScheduler scheduler = Mockito.mock(FlexlbBatchScheduler.class);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(100L)
                .setLatestFinishedVersion(7L)
                .setAlive(true)
                .build();
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
        when(registry.publishValidatedEndpoint(
                Mockito.eq(RoleType.VIT), Mockito.eq(ipPort), Mockito.same(probing), Mockito.any()))
                .thenReturn(null)
                .thenAnswer(ignored -> {
                    probing.tryMarkReady();
                    return endpoint;
                });
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                probing, statuses, engineHealthReporter, engineGrpcService,
                20L, scheduler, registry, Runnable::run);

        runner.run();
        assertEquals(100L, probing.getStatusVersion().get());
        assertEquals(-1L, probing.getLatestFinishedTaskVersion().get());
        assertEquals(WorkerLifecycleState.PROBING, probing.getLifecycleState());

        runner.run();

        assertEquals(WorkerLifecycleState.READY, probing.getLifecycleState());
        assertEquals(7L, probing.getLatestFinishedTaskVersion().get());
        verify(scheduler, Mockito.times(1)).onWorkerStatusUpdate(Mockito.any());
    }

    private static EndpointRegistry registry() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
    }

    private static WorkerStatus status(int port) {
        WorkerStatus status = probingStatus(port);
        status.setAlive(true);
        return status;
    }

    private static WorkerStatus probingStatus(int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.VIT);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        return status;
    }

    private static EngineRpcService.WorkerStatusPB liveStatus(RoleType role, long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(role == RoleType.VIT
                        ? EngineRpcService.RoleTypePB.ROLE_TYPE_VIT
                        : EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(version)
                .setAlive(true)
                .build();
    }
}
