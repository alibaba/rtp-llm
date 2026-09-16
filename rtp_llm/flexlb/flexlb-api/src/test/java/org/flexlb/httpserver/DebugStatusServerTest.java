package org.flexlb.httpserver;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.debug.DebugPage;
import org.flexlb.debug.DebugQuery;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verifyNoInteractions;

class DebugStatusServerTest {
    private final RequestRegistry registry = mock(RequestRegistry.class);
    private final RequestScheduler scheduler = mock(RequestScheduler.class);
    private final EndpointRegistry endpoints = mock(EndpointRegistry.class);
    private final DebugSnapshotService service = new DebugSnapshotService(registry, scheduler, endpoints);
    private final DebugStatusServer server = new DebugStatusServer(service, new ObjectMapper());
    private final WebTestClient client = WebTestClient.bindToRouterFunction(server.debugRoutes()).build();

    @AfterEach
    void close() {
        server.close();
    }

    @Test
    void exactLookupPreservesLongIdentityAndSerializesOffRequestThread() {
        AtomicReference<String> thread = new AtomicReference<>();
        when(registry.debugSnapshot(any())).thenAnswer(call -> {
            DebugQuery query = call.getArgument(0);
            thread.set(Thread.currentThread().getName());
            return page(List.of(Map.of("request_id", query.requestId().toString(),
                    "storage_phase", "TOMBSTONE")), "ok", false, 1);
        });
        client.get().uri("/rtp_llm/debug/requests/9007199254740993?include=scheduler")
                .exchange().expectStatus().isOk().expectHeader().valueEquals("Cache-Control", "no-store")
                .expectBody().jsonPath("$.components.scheduler.rows[0].request_id")
                .isEqualTo("9007199254740993")
                .jsonPath("$.components.scheduler.rows[0].storage_phase").isEqualTo("TOMBSTONE");
        assertEquals("flexlb-debug-snapshot", thread.get());
    }

    @Test
    void invalidQueriesNeverVisitOwners() {
        for (String query : List.of("limit=0", "limit=5001", "scan_limit=20001", "include=missing",
                "endpoint_limit=257", "include=", "limit=10&scan_limit=1")) {
            client.get().uri("/rtp_llm/debug/snapshot?" + query).exchange().expectStatus().isBadRequest();
        }
        verifyNoInteractions(registry, scheduler, endpoints);
    }

    @Test
    void ownerFailureIsExplicitPartialNotEmptySuccess() {
        when(registry.debugSnapshot(any())).thenThrow(new IllegalStateException("owner failed"));
        client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange().expectStatus().isOk()
                .expectBody().jsonPath("$.status").isEqualTo("partial")
                .jsonPath("$.components.scheduler.status").isEqualTo("unavailable");
    }

    @Test
    void budgetIsSharedAcrossOwnersAndExhaustionIsExplicit() {
        when(registry.debugSnapshot(any())).thenReturn(page(List.of(Map.of("request_id", "1")), "ok", false, 1));
        var snapshot = service.capture(new DebugQuery(1, 1, null), Set.of("scheduler", "queues"), 1);
        assertEquals("partial", snapshot.status());
        assertEquals("budget_exhausted", snapshot.components().get("queues").status());
        verifyNoInteractions(scheduler);
    }

    @Test
    void rejectedExecutionIsHttpFailureNotAnEmptySnapshot() {
        server.close();
        client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange()
                .expectStatus().isEqualTo(503).expectBody().jsonPath("$.error").isEqualTo("capture_unavailable");
    }

    private static DebugPage page(List<Map<String, Object>> rows, String status, boolean truncated, int scanned) {
        return new DebugPage("per_entry", status, 1, 2, scanned, truncated, rows, Map.of());
    }
}
