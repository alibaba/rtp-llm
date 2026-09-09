package org.flexlb.balance.endpoint;

import org.flexlb.debug.DebugQuery;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointDebugSnapshotTest {
    private static org.flexlb.balance.scheduler.ScheduledRequest request(long id) {
        var item = org.mockito.Mockito.mock(org.flexlb.balance.scheduler.ScheduledRequest.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(id);
        return item;
    }

    @Test
    void decodeSampleKeepsReservationIdentityAndCapacityAccounting() {
        var status = EndpointTestSupport.workerStatus(
                org.flexlb.dao.route.RoleType.DECODE, "10.0.0.1", 8080, 8081);
        var endpoint = new DecodeEndpoint(status, EndpointTestSupport.noopEventSink());
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            var first = endpoint.reservePinned(pin, 1L, 50L, 60L, 10);
            endpoint.reservePinned(pin, 2L, 50L, 60L, 20);
            var sample = endpoint.debugSnapshot(new DebugQuery(1, 1, null));
            assertTrue(sample.truncated());
            assertEquals(2, sample.metadata().get("reserved_total"));
            var exact = endpoint.debugSnapshot(new DebugQuery(1, 1, 1L));
            assertEquals(Long.toString(first.reservationToken()), exact.rows().getFirst().get("reservation_token"));
            assertEquals("reserved", exact.rows().getFirst().get("ownership"));
            assertEquals(2, endpoint.layeredAdmissionView().reserved().size());
        } finally {
            endpoint.close();
        }
    }

    @Test
    void prefillCopyIsBoundedImmutableAndDoesNotReleaseOwnership() {
        var lock = new ReentrantLock();
        var state = new PrefillState(lock, PrefillActiveIndex.disabled(), System::currentTimeMillis, () -> { });
        var first = state.reserveUnqueuedRoute(request(1), 10, 0).reservation();
        var second = state.reserveUnqueuedRoute(request(2), 10, 0).reservation();
        try {
            var page = state.debugSnapshot(new DebugQuery(1, 1, null));
            assertTrue(page.truncated());
            assertEquals(1, page.rows().size());
            assertEquals(2, page.metadata().get("retained_request_count"));
            var exact = state.debugSnapshot(new DebugQuery(1, 1, 2L));
            assertFalse(exact.truncated());
            assertEquals("2", exact.rows().getFirst().get("request_id"));
            second.close();
            assertEquals(1, exact.rows().size());
            assertTrue(state.debugSnapshot(new DebugQuery(1, 1, 2L)).rows().isEmpty());
        } finally {
            first.close();
            second.close();
        }
    }

    @Test
    void contendedOwnerReturnsBusyWithoutWaitingOrReportingEmptySuccess() throws Exception {
        var lock = new ReentrantLock();
        var state = new PrefillState(lock, PrefillActiveIndex.disabled(), System::currentTimeMillis, () -> { });
        var held = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newSingleThreadExecutor()) {
            var task = executor.submit(() -> {
                lock.lock();
                try {
                    held.countDown();
                    assertTrue(release.await(5, TimeUnit.SECONDS));
                } finally {
                    lock.unlock();
                }
                return null;
            });
            try {
                assertTrue(held.await(5, TimeUnit.SECONDS));
                assertEquals("busy", state.debugSnapshot(new DebugQuery(1, 1, null)).status());
            } finally {
                release.countDown();
            }
            task.get(5, TimeUnit.SECONDS);
        }
    }
}
