package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EmptyDecodeFinderTest {

    private final WorkerDirectory workerDirectory = mock(WorkerDirectory.class);
    private final EmptyDecodeFinder finder = new EmptyDecodeFinder(workerDirectory);

    @Test
    void decodeIsEmptyOnlyWhenEngineAndLocalRecordsAreAllZero() {
        assertTrue(EmptyDecodeFinder.isEmpty(view("10.0.0.1", 0, 0L, 0L)));
        assertFalse(EmptyDecodeFinder.isEmpty(view("10.0.0.1", 1, 0L, 0L)));
        assertFalse(EmptyDecodeFinder.isEmpty(view("10.0.0.1", 0, 1L, 0L)));
        assertFalse(EmptyDecodeFinder.isEmpty(view("10.0.0.1", 0, 0L, 1L)));
    }

    @Test
    void findReturnsFirstEmptyDecodeWithItsAddress() {
        DecodeEndpoint endpoint =
                new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, "g", "10.0.0.2", 8080, 8081, "test"),
                        mock(EndpointEventHandler.class));
        WorkerEndpoint.EndpointHold hold = endpoint.tryHold();
        DecodeEndpoint.DecodeRoutingView busy = view("10.0.0.1", 1, 0L, 0L, hold.generationId());
        DecodeEndpoint.DecodeRoutingView empty = view("10.0.0.2", 0, 0L, 0L, hold.generationId());
        when(workerDirectory.decodeRoutingSnapshot("g")).thenReturn(List.of(busy, empty));
        when(workerDirectory.holdDecodeEndpoint(empty)).thenReturn(hold);

        try (SelectedRole selected = finder.find("g", "r1")) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
            assertEquals(RoleType.DECODE, selected.serverStatus().getRole());
            assertEquals("r1", selected.serverStatus().getRequestId());
        } finally {
            endpoint.close();
        }
    }

    @Test
    void findReturnsNullWhenNoDecodeIsEmpty() {
        when(workerDirectory.decodeRoutingSnapshot("g")).thenReturn(List.of(view("10.0.0.1", 0, 1L, 0L)));

        assertNull(finder.find("g", "r1"));
    }

    private static DecodeEndpoint.DecodeRoutingView view(String ip, int totalLoad, long running, long waiting) {
        return view(ip, totalLoad, running, waiting, 1L);
    }

    private static DecodeEndpoint.DecodeRoutingView view(String ip,
                                                         int totalLoad,
                                                         long running,
                                                         long waiting,
                                                         long generationId) {
        WorkerStatus.EngineObservation engine = new WorkerStatus.EngineObservation(RoleType.DECODE, null, 1_000L,
                1_000L, Map.of(), 0.0, 0L, 1L, 1L, 0L, 0L, 0L, running, waiting);
        WorkerStatus.CommittedWorkerStatus committed =
                new WorkerStatus.CommittedWorkerStatus(engine, new WorkerStatus.AppliedStatusCursor(1L, 0L));
        DecodeEndpoint.CapacityUsage usage = new DecodeEndpoint.CapacityUsage(totalLoad, 1_000L, 1_000L, 0L, 0L);
        return new DecodeEndpoint.DecodeRoutingView(ip + ":8080", generationId,
                new WorkerStatus.TopologySnapshot("g", ip, 8080, 8081, "", "", 0, 1), committed, 0L, totalLoad, 0,
                usage, usage, 0L, 0L);
    }
}
