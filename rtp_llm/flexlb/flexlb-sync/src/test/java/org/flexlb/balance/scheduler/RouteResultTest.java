package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks that role results remain immutable when inputs, exported responses, or endpoint state later change.
 */
class RouteResultTest {
    @Test
    void frozenSelectionSurvivesInputResponseAndEndpointChanges() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "original", "chosen", 8080, 8090, null);
        var endpoint =
                EndpointTestSupport.realPrefill(
                        status,
                        SchedulingTestConfig.newConfig(),
                        org.mockito.Mockito.mock(DeliveryCoordinator.class),
                        new RequestRepository(),
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.DeliveryMetricsReporter.class));
        var roles = new java.util.ArrayList<RoleSelection>();
        try {
            var metadata =
                    ServerStatus.forWorker(
                            RoleType.PREFILL,
                            "42",
                            status.topologySnapshot(),
                            status.committedEngineObservation());
            var debug = new DebugInfo();
            debug.setHitCacheLen(7);
            metadata.setDebugInfo(debug);
            roles.add(new RoleSelection.Prefill(endpoint, metadata, 13, null));
            var result = new RouteResult(context, roles);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            new RouteResult(
                                    context,
                                    java.util.List.of(roles.getFirst(), roles.getFirst())));
            metadata.setGroup("changed");
            debug.setHitCacheLen(99);
            result.successResponse(false)
                    .getServerStatus()
                    .getFirst()
                    .getDebugInfo()
                    .setHitCacheLen(100);
            endpoint.close();

            var prefill =
                    assertInstanceOf(RoleSelection.Prefill.class, result.role(RoleType.PREFILL));
            assertSame(endpoint, prefill.endpoint());
            assertEquals(13, prefill.workMs());
            assertEquals(7, prefill.hitCacheTokens());
            assertEquals(
                    "original",
                    result.successResponse(true).getServerStatus().getFirst().getGroup());
            assertThrows(UnsupportedOperationException.class, () -> result.roles().clear());

            var another = new RouteResult(context, result.roles());
            assertNotSame(result, another);
            assertSame(context, result.ctx());
            assertSame(context, another.ctx());
            assertSame(result.role(RoleType.PREFILL), another.role(RoleType.PREFILL));
            assertNull(result.decodeReservation());
            assertNull(another.prefillReservation());
        } finally {
            endpoint.close();
            EndpointTestSupport.awaitRetirement(endpoint);
        }
    }
}
