package org.flexlb.worker.status;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.schedule.EngineCancellation;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.mockito.Mockito;

import java.util.Map;

/**
 * Package-local fixtures for the frozen status/endpoint composition boundary.
 */
public final class RunnerTestSupport {
    private RunnerTestSupport() {}

    public static void awaitCondition(java.util.function.BooleanSupplier condition) {
        long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean()) {
            org.junit.jupiter.api.Assertions.assertTrue(
                    System.nanoTime() < deadline, "condition did not become true within 5 seconds");
            java.util.concurrent.locks.LockSupport.parkNanos(
                    java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(1));
            org.junit.jupiter.api.Assertions.assertFalse(
                    Thread.currentThread().isInterrupted(), "wait interrupted");
        }
    }

    public static EndpointRegistry endpointRegistry(ConfigService configService) {
        EngineCancellation delivery = Mockito.mock(EngineCancellation.class);
        var config = configService.loadBalanceConfig();
        return new EndpointRegistry(
                configService,
                new RequestRepository(),
                Mockito.mock(DeliveryMetricsReporter.class),
                new CapacityNotifier(),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class));
    }

    public static WorkerStatus alive(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        WorkerStatus status = discovered(role, group, ip, port, grpcPort, site);
        publish(status, response(status, true, 1L));
        return status;
    }

    public static WorkerStatus discovered(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        return WorkerStatus.createDiscovered(role, group, ip, port, grpcPort, site);
    }

    public static void publish(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    public static WorkerStatusResponse response(WorkerStatus status, boolean alive, long version) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(status.getRole());
        response.setAlive(alive);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        return response;
    }

    public static WorkerStatusResponse response(
            WorkerStatus status,
            boolean alive,
            long version,
            long availableKv,
            long totalKv,
            double stepLatencyMs,
            Map<String, TaskInfo> runningTasks) {
        WorkerStatusResponse response = response(status, alive, version);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStepLatencyMs(stepLatencyMs);
        response.setRunningTaskInfo(runningTasks);
        return response;
    }

    public static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry, RoleType role, String address, WorkerStatus status) {
        WorkerStatusResponse response = response(status, true, 1L);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            WorkerEndpoint endpoint = registry.updateEndpoint(address, status, prepared).endpoint();
            status.recordSuccessfulPoll(true);
            return endpoint;
        } finally {
            status.lock.unlock();
        }
    }
}
