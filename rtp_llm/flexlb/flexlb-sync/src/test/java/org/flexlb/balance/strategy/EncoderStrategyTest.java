package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EncoderStrategyTest {
    private final WorkerDirectory directory = mock(WorkerDirectory.class);
    private final EncoderStrategy strategy = new EncoderStrategy(directory);

    @Test
    void selectsLeastConcurrentWorkerThenMostAvailableKvCache() {
        endpoint("busy", 8001, true, 40, 2, 2, 0);
        endpoint("zero-kv", 8004, true, 0, 1, 1, 0);
        endpoint("small-kv", 8002, true, 10, 1, 1, 0);
        endpoint("winner", 8003, true, 30, 1, 1, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("busy", "zero-kv", "small-kv", "winner"));

        try (SelectedRole selected = strategy.select(context(), "group")) {
            assertEquals("winner", selected.serverStatus().getServerIp());
            assertEquals(RoleType.ENCODER, selected.serverStatus().getRole());
        }
    }

    @Test
    void ignoresDeadAndNegativeKvWorkersAndIncludesPendingLocalRequests() {
        endpoint("dead", 8001, false, 100, 0, 0, 0);
        endpoint("negative", 8002, true, -1, 0, 0, 0);
        endpoint("pending", 8003, true, 100, 0, 0, 3);
        endpoint("winner", 8004, true, 1, 1, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("dead", "negative", "pending", "winner"));

        try (SelectedRole selected = strategy.select(context(), "group")) {
            assertEquals("winner", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void returnsNoSelectionWhenNoEligibleWorkerExists() {
        endpoint("negative", 8001, true, -1, 0, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("negative"));

        assertNull(strategy.select(context(), "group"));
    }

    @Test
    void selectsZeroKvWorkerWhenItHasLessConcurrentWork() {
        endpoint("positive", 8001, true, 100, 1, 0, 0);
        endpoint("zero", 8002, true, 0, 0, 0, 0);
        when(directory.endpointAddressSnapshot(RoleType.ENCODER))
                .thenReturn(List.of("positive", "zero"));

        try (SelectedRole selected = strategy.select(context(), "group")) {
            assertEquals("zero", selected.serverStatus().getServerIp());
        }
    }

    private void endpoint(String address, int port, boolean alive, long kv,
                          long running, long waiting, int pending) {
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        EncoderEndpoint endpoint = mock(EncoderEndpoint.class);
        WorkerStatus status = mock(WorkerStatus.class);
        WorkerStatus.EngineObservation engine = mock(WorkerStatus.EngineObservation.class);
        when(directory.captureEndpoint(RoleType.ENCODER, address)).thenReturn(pin);
        when(pin.endpoint()).thenReturn(endpoint);
        when(pin.generationId()).thenReturn(1L);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.getIp()).thenReturn(address);
        when(endpoint.getHttpPort()).thenReturn(port);
        when(endpoint.pendingEncoderRequestCount()).thenReturn(pending);
        when(status.getGenerationId()).thenReturn(1L);
        when(status.isAlive()).thenReturn(alive);
        when(status.topologySnapshot()).thenReturn(new WorkerStatus.TopologySnapshot(
                "group", address, port, port + 1, "site"));
        when(status.committedEngineObservation()).thenReturn(engine);
        when(engine.availableKvCacheTokens()).thenReturn(kv);
        when(engine.runningQueryLen()).thenReturn(running);
        when(engine.waitingQueryLen()).thenReturn(waiting);
    }

    private static BalanceContext context() {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId("encoder-request");
        context.setRequest(request);
        return context;
    }
}
