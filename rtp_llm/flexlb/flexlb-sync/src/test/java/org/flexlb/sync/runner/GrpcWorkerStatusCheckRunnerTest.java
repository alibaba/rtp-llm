package org.flexlb.sync.runner;

import io.grpc.Status;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.sync.schedule.ExpirationCleaner;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Original poll contracts exercised through the current generation and lease API. */
class GrpcWorkerStatusCheckRunnerTest {
    private static final String ADDRESS = "127.0.0.1:8080";
    private final EngineGrpcService grpc = mock(EngineGrpcService.class);
    private final EngineHealthReporter reporter = mock(EngineHealthReporter.class);
    private final CacheAwareService cache = mock(CacheAwareService.class);
    private final EndpointRegistry registry = registry();
    private final WorkerDirectory directory = new WorkerDirectory(registry);

    @AfterEach
    void closeRegistry() {
        registry.close();
    }

    @Test
    void should_callGrpcServiceAndVerifyInteraction_when_runnerExecutes() {
        WorkerStatus status = discover(RoleType.PREFILL);
        respond(response(RoleType.PREFILL, 100L, true).toBuilder()
                .setAvailableConcurrency(10)
                .setRunningQueryLen(5)
                .setWaitingQueryLen(3)
                .setStepLatencyMs(100)
                .setIterateCount(20)
                .setDpSize(2)
                .setTpSize(4)
                .build());

        poll(status);

        verify(grpc).getWorkerStatusAsync(
                "127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void should_not_update_task_list_when_status_version_is_unchanged() {
        WorkerStatus status = discover(RoleType.PREFILL);
        respond(response(RoleType.PREFILL, 100L, true));
        poll(status);
        WorkerStatus.CommittedWorkerStatus committed = status.committedWorkerStatus();
        EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setInputLength(100)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .setIsWaiting(true)
                .build();
        respond(response(RoleType.PREFILL, 100L, true).toBuilder()
                .addRunningTaskInfo(task).build());

        poll(status);

        assertTrue(status.committedEngineObservation().runningTaskList().isEmpty(),
                "same-version heartbeat must not replace the committed task list");
        assertSame(committed, status.committedWorkerStatus());
    }

    @Test
    void should_ignore_status_callback_from_expired_generation() {
        WorkerStatus expired = status(RoleType.VIT);
        WorkerStatus current = discover(RoleType.VIT);
        WorkerEndpoint currentEndpoint = RunnerTestSupport.publishEndpoint(
                registry, RoleType.VIT, ADDRESS, current);
        respond(response(RoleType.VIT, 100L, true));

        poll(expired);

        assertSame(currentEndpoint, registry.get(RoleType.VIT, ADDRESS));
        assertSame(current, currentEndpoint.getStatus());
        assertEquals(-1L, expired.appliedStatusCursor().statusVersion());
    }

    @Test
    void should_remove_endpoint_after_consecutive_grpc_failures() {
        WorkerStatus status = alive(RoleType.VIT);
        fail(new RuntimeException("unavailable"));

        pollThreeTimes(status);

        assertRetired(status);
    }

    @Test
    void should_markPrefillDeadWhenStatusCheckTimesOut() {
        WorkerStatus status = alive(RoleType.PREFILL);
        fail(Status.DEADLINE_EXCEEDED.asRuntimeException());

        pollThreeTimes(status);

        assertRetired(status);
        verify(reporter, times(3)).reportStatusCheckerFail(
                "test-model", BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT,
                RoleType.PREFILL);
    }

    @Test
    void should_markVitDeadWhenStatusCheckFailsWithoutTimeout() {
        WorkerStatus status = alive(RoleType.VIT);
        fail(new RuntimeException("connection refused"));

        pollThreeTimes(status);

        assertRetired(status);
    }

    @Test
    void should_applyExplicitDeadStatusFromVit() {
        WorkerStatus status = alive(RoleType.VIT);
        respond(response(RoleType.VIT, 1L, false));

        poll(status);

        assertRetired(status);
        assertFalse(status.pollHealth().reportedAlive());
    }

    @Test
    void should_notTrustDeadlineTokenInNonGrpcErrorMessage() {
        WorkerStatus status = alive(RoleType.VIT);
        fail(new RuntimeException("worker said DEADLINE_EXCEEDED but connection failed"));

        pollThreeTimes(status);

        assertRetired(status);
        verify(reporter, times(3)).reportStatusCheckerFail(
                "test-model", BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE,
                RoleType.VIT);
    }

    @Test
    void should_useLongerTimeoutForVitStatusCheck() {
        WorkerStatus status = discover(RoleType.VIT);
        respond(response(RoleType.VIT, 1L, true));

        pollWithPolicy(status, 20L, 1000L, true);

        verify(grpc).getWorkerStatusAsync(
                "127.0.0.1", 8081, -1L, 1000L, RoleType.VIT);
    }

    @Test
    void should_notShortenGlobalTimeoutForVitStatusCheck() {
        WorkerStatus status = discover(RoleType.VIT);
        respond(response(RoleType.VIT, 1L, true));

        pollWithPolicy(status, 5000L, 1000L, true);

        verify(grpc).getWorkerStatusAsync(
                "127.0.0.1", 8081, -1L, 5000L, RoleType.VIT);
    }

    @Test
    void should_keepLastVitAliveStateWhenStatusCheckTimesOut() {
        WorkerStatus status = alive(RoleType.VIT);
        WorkerEndpoint endpoint = registry.get(RoleType.VIT, ADDRESS);
        long lastSuccess = status.pollHealth().lastSuccessfulPollUs();
        fail(Status.DEADLINE_EXCEEDED.asRuntimeException());

        pollThreeTimes(status);

        assertTrue(status.isActiveGeneration());
        assertTrue(status.pollHealth().reportedAlive());
        assertSame(endpoint, registry.get(RoleType.VIT, ADDRESS));
        assertTrue(directory.isCurrentStatus(RoleType.VIT, ADDRESS, status));
        assertEquals(lastSuccess, status.pollHealth().lastSuccessfulPollUs());
        verify(reporter, times(3)).reportStatusCheckerFail(
                "test-model", BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, RoleType.VIT);
        verify(reporter, times(3)).reportStatusCheckerFail(
                anyString(), any(BalanceStatusEnum.class), any(RoleType.class));
    }

    @Test
    void should_markVitDeadWhenTimeoutRetentionIsDisabled() {
        WorkerStatus status = alive(RoleType.VIT);
        fail(Status.DEADLINE_EXCEEDED.asRuntimeException());

        for (int attempt = 0; attempt < 3; attempt++) {
            pollWithPolicy(status, 20L, 1000L, false);
        }

        assertRetired(status);
        assertEquals(3L, status.pollHealth().consecutiveTransportFailures());
    }

    @Test
    void retainedVitDeadlineStillExpiresFromLastSuccessfulPoll() {
        WorkerStatus status = alive(RoleType.VIT);
        long lastSuccess = status.pollHealth().lastSuccessfulPollUs();
        fail(Status.DEADLINE_EXCEEDED.asRuntimeException());
        pollThreeTimes(status);
        assertTrue(status.isActiveGeneration());
        assertEquals(lastSuccess, status.pollHealth().lastSuccessfulPollUs());

        // As in ExpirationCleanerTest, a zero fixture interval makes the real
        // cleaner cross the stale boundary without sleeping or changing clocks.
        FlexlbConfig config = new FlexlbConfig();
        config.getWorkerRegistry().getHealth().setStatusStaleAfterMs(0L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        new ExpirationCleaner(service, cache, directory).cleanExpiredWorkers();

        assertRetired(status);
        assertEquals(lastSuccess, status.pollHealth().lastSuccessfulPollUs());
    }

    private void pollWithPolicy(WorkerStatus status, long globalTimeoutMs,
                                long vitTimeoutMs, boolean retainVitAlive) {
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        new GrpcWorkerStatusRunner(
                "test-model", ADDRESS, "test-site", status.getRole(), "test-group",
                status, lease, directory, reporter, grpc, globalTimeoutMs,
                cache, Runnable::run, vitTimeoutMs, retainVitAlive).run();
    }

    private void pollThreeTimes(WorkerStatus status) {
        poll(status);
        poll(status);
        poll(status);
        assertEquals(3L, status.pollHealth().consecutiveTransportFailures());
    }

    private void poll(WorkerStatus status) {
        // Every poll owns a fresh one-shot lease, including consecutive failures.
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll();
        assertNotNull(lease);
        new GrpcWorkerStatusRunner(
                "test-model", ADDRESS, "test-site", status.getRole(), "test-group",
                status, lease, directory, reporter, grpc, 20L, cache, Runnable::run)
                .run();
    }

    private void assertRetired(WorkerStatus status) {
        // PollHealth retains the last engine report; routability belongs to the
        // exact generation and registry, not a mutable status.alive flag.
        assertFalse(status.isActiveGeneration());
        assertNull(registry.get(status.getRole(), ADDRESS));
        assertFalse(directory.isCurrentStatus(status.getRole(), ADDRESS, status));
    }

    private WorkerStatus alive(RoleType role) {
        WorkerStatus status = discover(role);
        RunnerTestSupport.publishEndpoint(registry, role, ADDRESS, status);
        return status;
    }

    private WorkerStatus discover(RoleType role) {
        return directory.currentOrDiscover(role, ADDRESS, () -> status(role));
    }

    private static WorkerStatus status(RoleType role) {
        return RunnerTestSupport.discovered(
                role, "test-group", "127.0.0.1", 8080, 8081, "test-site");
    }

    private void fail(Throwable failure) {
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.failedFuture(failure));
    }

    private void respond(EngineRpcService.WorkerStatusPB response) {
        when(grpc.getWorkerStatusAsync(
                anyString(), anyInt(), anyLong(), anyLong(), any()))
                .thenReturn(CompletableFuture.completedFuture(response));
    }

    private static EngineRpcService.WorkerStatusPB response(
            RoleType role, long version, boolean alive) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.valueOf("ROLE_TYPE_" + role.name()))
                .setStatusVersion(version)
                .setAlive(alive)
                .build();
    }

    private static EndpointRegistry registry() {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return RunnerTestSupport.endpointRegistry(config);
    }
}
