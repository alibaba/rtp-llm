package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.WorkerBatcher;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** API and ownership tests for the package-private request ledger. */
class PrefillRequestLedgerApiTest {

    @Test
    void exposesOnlyNarrowLifecycleAndSnapshotApi() {
        Set<String> exposedMethods = Arrays.stream(PrefillRequestLedger.class.getDeclaredMethods())
                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertEquals(Set.of(
                "tryAcquire", "release", "protect", "unprotect", "observe", "settle",
                "available", "count", "mutationVersion", "estimate", "evict", "maxAge"),
                exposedMethods);
    }

    @Test
    void ownsNoEndpointOrBatcherReference() {
        assertFalse(Arrays.stream(PrefillRequestLedger.class.getDeclaredFields())
                        .map(field -> field.getType())
                        .anyMatch(type -> type == PrefillEndpoint.class || type == WorkerBatcher.class),
                "ledger may notify through Runnable but must not retain orchestration objects");
    }

    @Test
    void capacityNotificationFollowsOnlyActualRemoval() {
        AtomicLong clock = new AtomicLong(1_000);
        AtomicInteger notifications = new AtomicInteger();
        PrefillRequestLedger ledger = new PrefillRequestLedger(
                notifications::incrementAndGet, clock::get, ignored -> {});

        assertTrue(ledger.tryAcquire(1L, 100, 1));
        assertTrue(ledger.release(1L));
        assertFalse(ledger.release(1L));

        assertTrue(ledger.tryAcquire(2L, 100, 1));
        assertTrue(ledger.settle(2L));
        assertFalse(ledger.settle(2L));

        assertTrue(ledger.tryAcquire(3L, 100, 1));
        clock.incrementAndGet();
        assertEquals(1, ledger.evict(0));
        assertEquals(0, ledger.evict(0));

        assertEquals(3, notifications.get());
        assertEquals(0, ledger.count());
        assertEquals(1, ledger.available(1));
    }
}
