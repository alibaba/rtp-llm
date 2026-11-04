package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestData;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Checks Encoder admission and release of local request capacity.
 */
class EncoderEndpointTest {
    private final RequestLifecycle lifecycle = mock(RequestLifecycle.class);
    private final PlacementAvailability availability = mock(PlacementAvailability.class);

    @Test
    void retirementWaitsForAnAlreadyPinnedCommitAndPublishesOutsideWorkerLock() throws Exception {
        var endpoint = endpoint();
        var route = route("request", 10);
        var pin = endpoint.tryPinGeneration();
        assertNotNull(pin);
        var entered = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            assertFalse(endpoint.getStatus().lock.isHeldByCurrentThread());
                            ContextResult.EngineEvidence evidence = call.getArgument(1);
                            assertEquals(ContextResult.EngineKind.RETIRED, evidence.kind());
                            return null;
                        })
                .when(lifecycle)
                .onEngineEvidence(any(), any());
        try (var tasks = Executors.newSingleThreadExecutor()) {
            endpoint.getStatus().lock.lock();
            java.util.concurrent.Future<Boolean> commit;
            try {
                commit =
                        tasks.submit(
                                () -> {
                                    entered.countDown();
                                    return endpoint.commit(pin, route, 2);
                                });
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                endpoint.close();
                assertNull(endpoint.tryPinGeneration());
                verifyNoInteractions(lifecycle);
            } finally {
                endpoint.getStatus().lock.unlock();
            }
            assertTrue(commit.get(2, TimeUnit.SECONDS));
            pin.close();
            endpoint.awaitRetirement(2000L);
            assertEquals(0, endpoint.requestCount());
            verify(lifecycle).onEngineEvidence(eq(route.ctx()), any());
        } finally {
            pin.close();
            endpoint.close();
        }
    }

    @Test
    void closingSeveralPendingRequestsNotifiesCapacityOnce() {
        var endpoint = endpoint();
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.commit(pin, route("request-1", 10), 2));
            assertTrue(endpoint.commit(pin, route("request-2", 20), 2));
            assertFalse(endpoint.commit(pin, route("request-1", 10), 2));
        }
        assertEquals(2, endpoint.requestCount());
        endpoint.close();
        endpoint.awaitRetirement(2000L);
        assertEquals(0, endpoint.requestCount());
        verify(availability).changed(endpoint);
        verify(lifecycle, times(2)).onEngineEvidence(any(), any());
    }

    @Test
    void releaseRequiresTheExactRouteAndCannotRemoveAReplacement() {
        var endpoint = endpoint();
        var original = route("same-id", 10);
        var replacement = route("same-id", 20);
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.commit(pin, original, 1));
            endpoint.release(original);
            assertTrue(endpoint.commit(pin, replacement, 1));
            endpoint.release(original);
            assertEquals(1, endpoint.requestCount());
            assertEquals(20, endpoint.uncachedTokens());
            endpoint.release(replacement);
        } finally {
            endpoint.close();
            endpoint.awaitRetirement(2000L);
        }
    }

    private RequestRoute route(String id, long tokens) {
        var route = mock(RequestRoute.class);
        var context = mock(RequestContext.class);
        var data = mock(RequestData.class);
        when(route.requestId()).thenReturn(String.valueOf(id));
        when(route.ctx()).thenReturn(context);
        when(context.data()).thenReturn(data);
        when(data.getLifecycle()).thenReturn(lifecycle);
        when(data.getRequest())
                .thenReturn(Request.builder().requestId(String.valueOf(id)).seqLen(tokens).build());
        return route;
    }

    private EncoderEndpoint endpoint() {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "default", "127.0.0.1", 8001, 8002, "");
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        return new EncoderEndpoint(status, availability);
    }
}
