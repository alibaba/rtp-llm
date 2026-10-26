package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestData;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EncoderStrategyTest {
    private final EndpointRegistry directory = mock(EndpointRegistry.class);
    private final EncoderStrategy strategy = new EncoderStrategy(directory);

    @Test
    void selectsLeastConcurrentWorkerWithoutUsingAvailableKvCache() {
        endpoint("busy", 8001, true, 40, 2, 2, 0);
        endpoint("zero-kv", 8004, true, 0, 1, 1, 0);
        endpoint("small-kv", 8002, true, 10, 1, 1, 0);
        endpoint("winner", 8003, true, 30, 1, 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("busy", "zero-kv", "small-kv", "winner"));

        try (EncoderSelection selected = strategy.select(context(), "group")) {
            assertEquals("zero-kv", selected.serverStatus().getServerIp());
            assertEquals(RoleType.ENCODER, selected.serverStatus().getRole());
        }
    }

    @Test
    void ignoresDeadWorkersAndIncludesPendingLocalRequests() {
        endpoint("dead", 8001, false, 100, 0, 0, 0);
        endpoint("negative", 8002, true, -1, 2, 0, 0);
        endpoint("pending", 8003, true, 100, 0, 0, 3);
        endpoint("winner", 8004, true, 1, 1, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("dead", "negative", "pending", "winner"));

        try (EncoderSelection selected = strategy.select(context(), "group")) {
            assertEquals("winner", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void returnsNoSelectionWhenNoLiveWorkerExists() {
        endpoint("dead", 8001, false, -1, 0, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER)).thenReturn(List.of("dead"));

        assertNull(strategy.select(context(), "group"));
    }

    @Test
    void selectsZeroKvWorkerWhenItHasLessConcurrentWork() {
        endpoint("positive", 8001, true, 100, 1, 0, 0);
        endpoint("zero", 8002, true, 0, 0, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("positive", "zero"));

        try (EncoderSelection selected = strategy.select(context(), "group")) {
            assertEquals("zero", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void knownCacheEstimateSelectsLessUncachedWorkEvenWhenItHasMoreRequests() {
        EncoderEndpoint longJob = endpoint("long-job", 8001, true, 0, 1, 0, 0);
        EncoderEndpoint shortJobs = endpoint("short-jobs", 8002, true, -1, 4, 0, 0);
        when(longJob.inflightUncachedTokenEstimate()).thenReturn(1000L);
        when(shortJobs.inflightUncachedTokenEstimate()).thenReturn(40L);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("long-job", "short-jobs"));

        try (EncoderSelection selected = strategy.select(context(1000L, 200L), "group")) {
            assertEquals("short-jobs", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void queueModeSkipsWorkersAtEncoderInflightLimit() {
        endpoint("at-cap", 8001, true, 0, 1, 1, 0);
        endpoint("winner", 8002, true, 0, 1, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("at-cap", "winner"));

        RequestData context = context();
        context.getConfig().getDispatcher().setMaxInflightPerEncoderWorker(2);

        try (EncoderSelection selected = strategy.select(context, "group")) {
            assertEquals("winner", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void queueModeReturnsNullWhenEveryWorkerIsAtEncoderInflightLimit() {
        endpoint("running", 8001, true, 0, 1, 0, 0);
        endpoint("pending", 8002, true, 0, 0, 0, 1);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("running", "pending"));

        RequestData context = context();
        context.getConfig().getDispatcher().setMaxInflightPerEncoderWorker(1);

        assertNull(strategy.select(context, "group"));
    }

    @Test
    void directModeDoesNotApplyEncoderInflightLimit() {
        endpoint("only-worker", 8001, true, 0, 1, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("only-worker"));

        RequestData context = context();
        context.getConfig().setScheduler(org.flexlb.config.ConfigTestSupport.direct());
        context.getConfig().getDispatcher().setMaxInflightPerEncoderWorker(1);

        try (EncoderSelection selected = strategy.select(context, "group")) {
            assertEquals("only-worker", selected.serverStatus().getServerIp());
        }
    }

    private EncoderEndpoint endpoint(
            String address,
            int port,
            boolean alive,
            long kv,
            long running,
            long waiting,
            int pending) {
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        EncoderEndpoint endpoint = mock(EncoderEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(directory.capture(RoleType.ENCODER, address)).thenReturn(pin);
        when(pin.endpoint()).thenReturn(endpoint);
        when(pin.generationId()).thenReturn(1L);
        when(pin.isOpen()).thenReturn(true);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn(address);
        when(endpoint.getHttpPort()).thenReturn(port);
        when(endpoint.pendingEncoderRequestCount()).thenReturn(pending);
        when(status.getGenerationId()).thenReturn(1L);
        when(status.isAlive()).thenReturn(alive);
        when(status.topologySnapshot())
                .thenReturn(
                        new WorkerStatus.TopologySnapshot(
                                "group", address, port, port + 1, "site"));
        when(status.committedEngineObservation()).thenReturn(engine);
        when(engine.availableKvCacheTokens()).thenReturn(kv);
        when(engine.runningQueryLen()).thenReturn(running);
        when(engine.waitingQueryLen()).thenReturn(waiting);
        return endpoint;
    }

    private static RequestData context() {
        RequestData context = new RequestData(new org.flexlb.config.FlexlbConfig());
        Request request = Request.builder().build();
        request = request.toBuilder().requestId("encoder-request").build();
        context.setRequest(request);
        return context;
    }

    private static RequestData context(long seqLen, long encoderCacheHitLen) {
        RequestData context = context();
        context.setRequest(context.getRequest().toBuilder().seqLen(seqLen).build());
        context.setRequest(
                context.getRequest().toBuilder().encoderCacheHitLen(encoderCacheHitLen).build());
        return context;
    }
}
