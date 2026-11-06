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
                        org.mockito.Mockito.mock(DeliveryDispatcher.class),
                        new RequestRepository(),
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.DeliveryMetricsReporter.class));
        var builder = new RouteResult.Builder(context);
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
            builder.add(endpoint, metadata, 13, 17);
            var result = builder.prepare();
            assertThrows(IllegalStateException.class, builder::prepare);
            assertThrows(
                    IllegalStateException.class, () -> builder.add(endpoint, metadata, 13, 17));

            var rejected = new RouteResult.Builder(context);
            rejected.add(endpoint, metadata, 13, 17);
            assertThrows(
                    IllegalStateException.class, () -> rejected.add(endpoint, metadata, 13, 17));
            assertThrows(IllegalStateException.class, rejected::prepare);
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
            assertEquals(17, prefill.resourceVersion());
            assertEquals(7, prefill.hitCacheTokens());
            assertEquals(
                    "original",
                    result.successResponse(true).getServerStatus().getFirst().getGroup());
            assertThrows(UnsupportedOperationException.class, () -> result.roles().clear());

            var first = new RouteExecution(context, result);
            var second = new RouteExecution(context, result);
            assertNotSame(first, second);
            assertSame(first.result(), second.result());
            assertNull(first.decodeReservation());
            assertNull(second.prefillReservation());
        } finally {
            endpoint.close();
            EndpointTestSupport.awaitRetirement(endpoint);
        }
    }
}
