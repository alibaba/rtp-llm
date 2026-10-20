package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.util.List;

import static org.mockito.Mockito.*;

class MasterDiagnosticsServerTest {
    private final EndpointRegistry registry = mock(EndpointRegistry.class);
    private final RequestScheduler scheduler = mock(RequestScheduler.class);
    private final MasterDiagnosticsServer server = new MasterDiagnosticsServer(registry, scheduler);
    private final WebTestClient client = WebTestClient.bindToRouterFunction(server.diagnosticRoutes()).build();

    private void directory(int count, boolean stable) {
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        var rows = java.util.stream.IntStream.range(0, count)
                .mapToObj(i -> new EndpointRegistry.PrefillRoutingEntry("127.0.0.1:" + (8080 + i), endpoint)).toList();
        var directory = new EndpointRegistry.DiagnosticDirectory(rows, List.of());
        when(registry.diagnosticDirectory()).thenReturn(directory);
        when(registry.diagnosticDirectoryUnchanged(directory)).thenReturn(stable);
    }

    @Test
    void unknownHealthIsNullAndStringGenerationIsPreserved() {
        directory(1, true);
        client.get().uri("/rtp_llm/debug/snapshot").exchange().expectStatus().isOk()
                .expectHeader().valueEquals("Cache-Control", "no-store")
                .expectBody().jsonPath("$.schema_version").isEqualTo(2)
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(true)
                .jsonPath("$.components.workers.rows[0].generation").value(value -> org.junit.jupiter.api.Assertions.assertInstanceOf(String.class, value))
                .jsonPath("$.components.workers.rows[0].applied_observation_age_ms").isEmpty()
                .jsonPath("$.components.workers.rows[0].last_reported_alive").isEmpty()
                .jsonPath("$.components.workers.rows[0].indexed_cache_version").isEmpty()
                .jsonPath("$.components.workers.rows[0].index_initialized").isEqualTo(false);
    }

    @Test
    void rowBudgetCannotMasqueradeAsAnEmptyFleet() {
        directory(2, true);
        client.get().uri("/rtp_llm/debug/snapshot?limit=1").exchange().expectStatus().isOk()
                .expectBody().jsonPath("$.status").isEqualTo("partial")
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(false)
                .jsonPath("$.components.workers.coverage.returned_count").isEqualTo(1);
    }

    @Test
    void topologyChangeInvalidatesAbsenceInference() {
        directory(0, false);
        client.get().uri("/rtp_llm/debug/snapshot").exchange().expectStatus().isOk()
                .expectBody().jsonPath("$.components.workers.reason").isEqualTo("directory_changed")
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(false);
    }

    @Test
    void schedulerOnlyDoesNotTraverseWorkers() {
        client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange().expectStatus().isOk();
        verifyNoInteractions(registry);
    }

    @Test
    void unsupportedOrUnboundedQueriesAreRejected() {
        for (String query : List.of("include=requests", "include=workers,workers", "limit=0", "limit=2049",
                "budget_ms=251", "limit=no", "limit=1&limit=2", "scan_keys=true")) {
            client.get().uri("/rtp_llm/debug/snapshot?" + query).exchange().expectStatus().isBadRequest();
        }
        verifyNoInteractions(registry, scheduler);
    }

    @Test
    void appliedStateHasItsOwnAge() {
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setStatusVersion(1L);
        response.setAlive(true);
        status.lock.lock();
        try {
            var observation = status.freezeStatusResponse(response);
            status.recordSuccessfulPoll(true);
            status.publishPreparedStatus(status.prepareNewStatus(observation));
            status.recordAppliedObservation(observation);
        } finally {
            status.lock.unlock();
        }
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        var directory = new EndpointRegistry.DiagnosticDirectory(
                List.of(new EndpointRegistry.PrefillRoutingEntry(status.getIpPort(), endpoint)), List.of());
        when(registry.diagnosticDirectory()).thenReturn(directory);
        when(registry.diagnosticDirectoryUnchanged(directory)).thenReturn(true);
        client.get().uri("/rtp_llm/debug/snapshot").exchange().expectStatus().isOk()
                .expectBody().jsonPath("$.components.workers.rows[0].applied_status_version").isEqualTo("1")
                .jsonPath("$.components.workers.rows[0].applied_observation_age_ms").isNumber();
    }
}
