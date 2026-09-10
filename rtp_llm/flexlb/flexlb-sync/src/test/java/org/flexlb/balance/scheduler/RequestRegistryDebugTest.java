package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.debug.DebugQuery;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RequestRegistryDebugTest {
    private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
    private final RequestRegistry registry;

    RequestRegistryDebugTest() {
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }

    @AfterEach
    void close() {
        if (registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @Test
    void retainedTombstoneIsVisibleWithoutOwningRequestResources() {
        long id = 9007199254740993L;
        var future = registry.register(RequestLifecycleTestSupport.context(config, id));
        var before = registry.debugSnapshot(new DebugQuery(5, 5, id));
        assertEquals(Long.toString(id), before.rows().getFirst().get("request_id"));
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        future.join();
        var after = registry.debugSnapshot(new DebugQuery(5, 5, id));
        var row = after.rows().getFirst();
        assertEquals("TOMBSTONE", row.get("storage_phase"));
        assertEquals(before.rows().getFirst().get("request_generation"), row.get("request_generation"));
        for (String field : row.keySet()) {
            if (field.startsWith("has_")) {
                assertEquals(false, row.get(field), field);
            }
        }
        assertEquals(0, registry.liveRequestCount());
        var slot = registry.requestSlot(id);
        assertTrue(registry.removeExactTombstone(slot, Long.MAX_VALUE));
        assertTrue(registry.debugSnapshot(new DebugQuery(5, 5, id)).rows().isEmpty());
        registry.register(RequestLifecycleTestSupport.context(config, id));
        assertNotEquals(row.get("request_generation"), registry.debugSnapshot(new DebugQuery(5, 5, id))
                .rows().getFirst().get("request_generation"));
    }

    @Test
    void limitBoundsTraversalAndDoesNotChangeLiveOwnership() {
        for (int i = 1; i <= 10; i++) {
            registry.register(RequestLifecycleTestSupport.context(config, i));
        }
        var page = registry.debugSnapshot(new DebugQuery(2, 3, null));
        assertEquals(2, page.rows().size());
        assertEquals(2, page.scannedCount());
        assertTrue(page.truncated());
        assertEquals("partial", page.status());
        assertEquals(10, registry.liveRequestCount());
        assertThrows(UnsupportedOperationException.class,
                () -> page.rows().getFirst().put("storage_phase", "TOMBSTONE"));
    }
}
