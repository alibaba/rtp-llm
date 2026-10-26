package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.schedulev2.V2TestSupport;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class PrefillQueueRegistryTest {

    private final PrefillQueueRegistry queues = new PrefillQueueRegistry();
    private final PrefillEndpoint endpoint = V2TestSupport.prefillEndpoint("p1:8000");

    @Test
    void cachedStatusInitializesCapacityBeforeLoopStarts() {
        V2TestSupport.report(endpoint, 1L, 100L, 1000L, Map.of("external", TaskPhase.RUNNING));
        var loop = mock(PrefillQueueLoop.class);
        var createdLoops = new AtomicInteger();
        queues.start(queue -> {
            createdLoops.incrementAndGet();
            return loop;
        });

        endpoint.prefillQueue().initializeFromCachedStatus();

        PrefillQueue queue = queues.find(endpoint);
        assertSame(endpoint.prefillQueue(), queue);
        assertEquals(0, queue.waitingCount());
        assertEquals(1L, queue.inflightCount());
        assertEquals(1L, queue.unknownRequestCount());
        assertFalse(queue.hasDispatchSlot(1));
        assertEquals(0, createdLoops.get());
        verifyNoInteractions(loop);

        queues.start(endpoint);
        queues.start(endpoint);
        assertEquals(1, createdLoops.get());
        verify(loop).start();
        queues.close();
    }

    @Test
    void schedulerStartupStartsOnlyPublishedEndpoints() {
        PrefillEndpoint candidate = V2TestSupport.prefillEndpoint("p2:8000");
        queues.start(endpoint);
        var loop = mock(PrefillQueueLoop.class);
        var createdLoops = new AtomicInteger();

        queues.start(queue -> {
            assertSame(endpoint, queue.endpoint());
            createdLoops.incrementAndGet();
            return loop;
        });

        assertEquals(1, createdLoops.get());
        verify(loop).start();
        when(candidate.isRetired()).thenReturn(true);
        queues.retire(candidate);
        assertNull(queues.find(candidate));
        queues.close();
    }

    @Test
    void emptyRetiredGenerationCannotRemoveReplacementAtSameAddress() {
        queues.start(endpoint);
        PrefillQueue oldQueue = queues.find(endpoint);
        PrefillEndpoint replacement = V2TestSupport.prefillEndpoint(endpoint.ipPort());
        queues.start(replacement);
        PrefillQueue newQueue = queues.find(replacement);

        when(endpoint.isRetired()).thenReturn(true);
        queues.retire(endpoint);
        queues.retire(endpoint);

        assertNull(queues.find(endpoint));
        assertSame(newQueue, queues.find(replacement));
        assertNotSame(oldQueue, newQueue);
    }

    @Test
    void queueCanBeRegisteredBeforeSchedulerStarts() {
        queues.start(endpoint);
        assertSame(endpoint.prefillQueue(), queues.find(endpoint));
        assertEquals(1, queues.list().size());
    }
}
