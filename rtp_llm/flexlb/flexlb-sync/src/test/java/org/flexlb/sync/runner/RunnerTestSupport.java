package org.flexlb.sync.runner;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.mockito.Mockito;

import java.util.Map;

/** Package-local fixtures for the frozen status/endpoint composition boundary. */
public final class RunnerTestSupport {

    private static final EndpointEventProjector NOOP_EVENT_SINK =
            Mockito.mock(EndpointEventProjector.class);

    private RunnerTestSupport() {
    }

    public static EndpointEventProjector eventSink() {
        return NOOP_EVENT_SINK;
    }

    public static EndpointRegistry endpointRegistry(ConfigService configService) {
        DeliveryStrategy delivery = Mockito.mock(DeliveryStrategy.class);
        return new EndpointRegistry(
                configService,
                NOOP_EVENT_SINK,
                Mockito.mock(BatchSchedulerReporter.class),
                delivery,
                new PlacementAvailability());
    }

    public static WorkerStatus discovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        return WorkerStatus.createDiscovered(
                role, group, ip, port, grpcPort, site);
    }

    public static WorkerStatus alive(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        WorkerStatus status = discovered(
                role, group, ip, port, grpcPort, site);
        publish(status, response(status, true, 1L));
        return status;
    }

    public static WorkerStatusResponse response(
            WorkerStatus status, boolean alive, long version) {
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

    public static void publish(
            WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    public static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry,
            RoleType role,
            String address,
            WorkerStatus status) {
        WorkerStatusResponse response = response(status, true, 1L);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            WorkerEndpoint endpoint = registry
                    .publishPreparedEndpoint(address, status, prepared)
                    .endpoint();
            status.recordSuccessfulPoll(true);
            return endpoint;
        } finally {
            status.lock.unlock();
        }
    }

}
