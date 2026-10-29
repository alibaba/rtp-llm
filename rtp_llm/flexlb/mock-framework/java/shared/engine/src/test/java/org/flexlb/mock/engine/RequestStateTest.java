package org.flexlb.mock.engine;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class RequestStateTest {
    @Test void cancellingOneBatchMemberRetainsItsResourceUntilForwardReleasesIt() {
        var closed = new int[2]; var published = new int[2]; var ended = new ArrayList<RequestFailure>();
        var first = new RequestState(Long.MAX_VALUE, ended::add);
        var second = new RequestState(Long.MAX_VALUE, ended::add);
        first.attach(() -> closed[0]++, () -> published[0]++);
        second.attach(() -> closed[1]++, () -> published[1]++);
        var firstUse = first.forward(); var secondUse = second.forward();
        first.finish(new RequestFailure(RequestFailure.Reason.CANCELLED, "cancel member"));
        assertArrayEquals(new int[]{0, 0}, closed); assertTrue(ended.isEmpty());
        assertThrows(IllegalStateException.class, () -> first.commit(1, 10));
        second.commit(1, 10); second.finish(null);
        firstUse.close(); firstUse.close(); secondUse.close();
        first.finish(null); second.finish(null);
        assertArrayEquals(new int[]{1, 1}, closed); assertArrayEquals(new int[]{0, 1}, published);
        assertEquals(2, ended.size()); assertEquals(RequestFailure.Reason.CANCELLED, ended.getFirst().reason());
        assertNull(ended.getLast()); assertEquals(1, second.snapshot().generated());
    }

    @Test void publicationFailureStillClosesOnceAndReportsFailure() {
        var closed = new AtomicInteger(); var ended = new ArrayList<RequestFailure>();
        var state = new RequestState(Long.MAX_VALUE, ended::add);
        state.attach(closed::incrementAndGet, () -> { throw new IllegalStateException("publish failed"); });
        state.finish(null); state.finish(null);
        assertEquals(1, closed.get()); assertEquals(1, ended.size());
        assertEquals(RequestFailure.Reason.INTERNAL, ended.getFirst().reason());
        assertEquals("publish failed", ended.getFirst().getCause().getMessage());
    }

    @Test void deadlineAtCommitDoesNotPublishSuccessfulCache() {
        var closed = new AtomicInteger(); var published = new AtomicInteger(); var ended = new ArrayList<RequestFailure>();
        var state = new RequestState(System.nanoTime() - 1, ended::add);
        state.attach(closed::incrementAndGet, published::incrementAndGet); state.finish(null);
        assertEquals(1, closed.get()); assertEquals(0, published.get());
        assertEquals(RequestFailure.Reason.TIMEOUT, ended.getFirst().reason());
    }

    @Test void transferPinDelaysCleanupUntilReaderExits() {
        var closed = new AtomicInteger(); var ended = new ArrayList<RequestFailure>();
        var state = new RequestState(Long.MAX_VALUE, ended::add);
        state.attach(closed::incrementAndGet, () -> {});
        var reader = state.pin();
        state.finish(null);
        assertEquals(0, closed.get()); assertTrue(ended.isEmpty());
        reader.close(); reader.close();
        assertEquals(1, closed.get()); assertEquals(1, ended.size());
    }
}
