package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QueueCapacityTest {

    @Test
    void tryIncreaseTotalQueueCountStopsAtLimitAndCountsEachFutureOnce() {
        QueueCapacity capacity = new QueueCapacity();
        CompletableFuture<Object> first = new CompletableFuture<>();
        CompletableFuture<Object> second = new CompletableFuture<>();

        assertTrue(capacity.tryIncreaseTotalQueueCount(first, 2L));
        assertTrue(capacity.tryIncreaseTotalQueueCount(first, 2L));
        assertEquals(1L, capacity.globalQueueAndEngineSubQueueCount());
        assertTrue(capacity.tryIncreaseTotalQueueCount(second, 2L));
        assertFalse(capacity.tryIncreaseTotalQueueCount(new CompletableFuture<>(), 2L));

        capacity.decreaseTotalQueueCount(first);
        capacity.decreaseTotalQueueCount(first);
        assertEquals(1L, capacity.globalQueueAndEngineSubQueueCount());
    }
}
