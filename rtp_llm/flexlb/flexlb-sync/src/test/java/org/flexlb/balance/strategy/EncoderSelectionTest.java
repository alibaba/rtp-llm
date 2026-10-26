package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EncoderSelectionTest {
    @Test
    void metadataRemainsFrozenAndReadableAfterPinRelease() {
        var endpoint = endpoint();
        var pin = endpoint.tryPinGeneration();
        var metadata = metadata(endpoint);
        try (var selection = EncoderSelection.create(pin, metadata)) {
            metadata.setRequestId("changed");
            metadata.setDpRank(9L);
            selection.serverStatus().setDpRank(7L);
            assertEquals("42", selection.serverStatus().getRequestId());
            assertEquals(3L, selection.serverStatus().getDpRank());
            selection.close();
            assertFalse(pin.isOpen());
            assertThrows(IllegalStateException.class, selection::generationPin);
            assertEquals(3L, selection.serverStatus().getDpRank());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void transferredPinStaysOpenUntilTheRouteCloses() {
        var endpoint = endpoint();
        var pin = endpoint.tryPinGeneration();
        var context = new RequestContext(new FlexlbConfig());
        context.data().setRequest(Request.builder().requestId("42").build());
        try (var route = new RequestRoute(context);
                var selection = EncoderSelection.create(pin, metadata(endpoint))) {
            assertEquals(RoleType.ENCODER, selection.transferTo(route).getRole());
            selection.close();
            assertTrue(pin.isOpen());
            assertThrows(IllegalStateException.class, () -> selection.transferTo(route));
            route.close();
            assertFalse(pin.isOpen());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void invalidMetadataConsumesTheIncomingPin() {
        var endpoint = endpoint();
        var pin = endpoint.tryPinGeneration();
        var metadata = metadata(endpoint);
        metadata.setRole(RoleType.VIT);
        try {
            assertThrows(
                    IllegalArgumentException.class, () -> EncoderSelection.create(pin, metadata));
            assertFalse(pin.isOpen());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    private static EncoderEndpoint endpoint() {
        return new EncoderEndpoint(
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "group", "encoder", 8000, 8001, null),
                new RequestRepository(),
                new PlacementAvailability());
    }

    private static ServerStatus metadata(EncoderEndpoint endpoint) {
        var status =
                ServerStatus.forWorker(
                        RoleType.ENCODER,
                        "42",
                        endpoint.getStatus().topologySnapshot(),
                        endpoint.getStatus().committedEngineObservation());
        status.setDpRank(3L);
        return status;
    }
}
