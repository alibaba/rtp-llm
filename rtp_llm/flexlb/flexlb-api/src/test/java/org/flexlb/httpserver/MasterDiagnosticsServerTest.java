package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.PrefillTrafficTrend;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.grace.ApplicationLifecycle;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class MasterDiagnosticsServerTest {
    private final EndpointRegistry registry = mock(EndpointRegistry.class);
    private final RequestScheduler scheduler = mock(RequestScheduler.class);
    private final MasterDiagnosticsServer server =
            new MasterDiagnosticsServer(registry, scheduler);
    private final WebTestClient client =
            WebTestClient.bindToRouterFunction(server.diagnosticRoutes()).build();

    @Test
    void trafficEndpointReadsPublishedStateWithoutTraversingWorkers() {
        PrefillTrafficTrend.Snapshot trend = new PrefillTrafficTrend.Snapshot(
                PrefillTrafficTrend.State.RECOVERING,
                100, 300, 10000000, 30000000, 35000, true, 5000);
        when(scheduler.prefillTrafficSnapshot()).thenReturn(Map.of("g", trend));

        client.get().uri("/rtp_llm/debug/prefill_traffic").exchange()
                .expectStatus().isOk()
                .expectHeader().valueEquals("Cache-Control", "no-store")
                .expectBody()
                .jsonPath("$.schema_version").isEqualTo(1)
                .jsonPath("$.captured_at_ms").isNumber()
                .jsonPath("$.pools.g.state").isEqualTo("RECOVERING")
                .jsonPath("$.pools.g.arrivalQps").isEqualTo(100.0)
                .jsonPath("$.pools.g.grace").isEqualTo(true)
                .jsonPath("$.pools.g.graceRemainingMs").isEqualTo(5000);

        verifyNoInteractions(registry);
    }

    @Test
    void trafficEndpointIsEmptyWithoutQueueCoordinator() {
        when(scheduler.prefillTrafficSnapshot()).thenReturn(Map.of());

        client.get().uri("/rtp_llm/debug/prefill_traffic").exchange()
                .expectStatus().isOk()
                .expectBody().jsonPath("$.pools").isEmpty();

        verifyNoInteractions(registry);
    }

    private void prefillDirectory(
            Map<String, PrefillEndpoint> captured,
            Map<String, PrefillEndpoint> verified) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        captured.put("127.0.0.1:8080", endpoint);
        verified.put("127.0.0.1:8080", endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(captured, verified);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());
    }

    @Test
    void unknownHealthIsNullAndStringGenerationIsPreserved() {
        Map<String, PrefillEndpoint> captured = new LinkedHashMap<>();
        Map<String, PrefillEndpoint> verified = new LinkedHashMap<>();
        prefillDirectory(captured, verified);

        client.get().uri("/rtp_llm/debug/snapshot").exchange()
                .expectStatus().isOk()
                .expectHeader().valueEquals("Cache-Control", "no-store")
                .expectBody()
                .jsonPath("$.schema_version").isEqualTo(2)
                .jsonPath("$.instance_id").isNotEmpty()
                .jsonPath("$.snapshot_id").isNotEmpty()
                .jsonPath("$.status").isEqualTo("ok")
                .jsonPath("$.components.workers.status").isEqualTo("ok")
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(true)
                .jsonPath("$.components.workers.rows[0].role").isEqualTo("prefill")
                .jsonPath("$.components.workers.rows[0].group").isEqualTo("g")
                .jsonPath("$.components.workers.rows[0].generation")
                .value(value -> assertInstanceOf(String.class, value))
                .jsonPath("$.components.workers.rows[0].lifecycle_state").isEqualTo("active")
                .jsonPath("$.components.workers.rows[0].applied_status_version").isEmpty()
                .jsonPath("$.components.workers.rows[0].applied_latest_finished_task_version").isEmpty()
                .jsonPath("$.components.workers.rows[0].poll_response_age_ms").isEmpty()
                .jsonPath("$.components.workers.rows[0].last_reported_alive").isEmpty()
                .jsonPath("$.components.workers.rows[0].healthy").isEqualTo(false)
                .jsonPath("$.components.workers.rows[0].cached_key_count").isEmpty()
                .jsonPath("$.components.workers.rows[0].index_initialized").isEqualTo(false);
    }

    @Test
    void decodeRowsExposeEngineObservationWithoutCacheIndex() {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.DECODE, "g", "127.0.0.1", 7070, 9070, "s");
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        Map<String, DecodeEndpoint> captured = new LinkedHashMap<>();
        captured.put("127.0.0.1:7070", endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(Map.of());
        when(registry.snapshotDecodeEndpoints()).thenReturn(captured);

        client.get().uri("/rtp_llm/debug/snapshot").exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.components.workers.rows[0].worker_id").isEqualTo("decode:127.0.0.1:7070")
                .jsonPath("$.components.workers.rows[0].role").isEqualTo("decode")
                .jsonPath("$.components.workers.rows[0].waiting_query_len").isEqualTo(0)
                .jsonPath("$.components.workers.rows[0].running_query_len").isEqualTo(0)
                .jsonPath("$.components.workers.rows[0].cached_key_count").doesNotExist()
                .jsonPath("$.components.workers.rows[0].index_initialized").doesNotExist();
    }

    @Test
    void rowBudgetCannotMasqueradeAsAnEmptyFleet() {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        Map<String, PrefillEndpoint> captured = new LinkedHashMap<>();
        captured.put("127.0.0.1:8080", endpoint);
        captured.put("127.0.0.1:8081", endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(captured);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        client.get().uri("/rtp_llm/debug/snapshot?limit=1").exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.status").isEqualTo("partial")
                .jsonPath("$.components.workers.status").isEqualTo("partial")
                .jsonPath("$.components.workers.reason").isEqualTo("capture_budget")
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(false)
                .jsonPath("$.components.workers.coverage.truncated").isEqualTo(true)
                .jsonPath("$.components.workers.coverage.returned_count").isEqualTo(1);
    }

    @Test
    void topologyChangeInvalidatesAbsenceInference() {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        Map<String, PrefillEndpoint> appeared = new LinkedHashMap<>();
        appeared.put("127.0.0.1:8080", endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(Map.of(), appeared);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        client.get().uri("/rtp_llm/debug/snapshot").exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.components.workers.reason").isEqualTo("directory_changed")
                .jsonPath("$.components.workers.coverage.complete").isEqualTo(false)
                .jsonPath("$.components.workers.coverage.truncated").isEqualTo(false);
    }

    @Test
    void schedulerOnlyDoesNotTraverseWorkers() {
        when(scheduler.getQueuedRequestCount()).thenReturn(7);
        when(scheduler.getInflightSize()).thenReturn(3);

        client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.status").isEqualTo("ok")
                .jsonPath("$.components.workers").doesNotExist()
                .jsonPath("$.components.scheduler.status").isEqualTo("ok")
                .jsonPath("$.components.scheduler.data.queued_requests").isEqualTo(7)
                .jsonPath("$.components.scheduler.data.inflight_requests").isEqualTo(3);

        verifyNoInteractions(registry);
    }

    @Test
    void unsupportedOrUnboundedQueriesAreRejected() {
        for (String query : List.of("include=requests", "include=workers,workers", "limit=0",
                "limit=2049", "budget_ms=251", "limit=no", "limit=1&limit=2", "scan_keys=true")) {
            client.get().uri("/rtp_llm/debug/snapshot?" + query).exchange()
                    .expectStatus().isBadRequest();
        }
        verifyNoInteractions(registry, scheduler);
    }

    @Test
    void appliedStateIsPublishedThroughTheStatusContract() {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        publishAppliedStatus(status);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        Map<String, PrefillEndpoint> captured = new LinkedHashMap<>();
        captured.put(status.getIpPort(), endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(captured);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        client.get().uri("/rtp_llm/debug/snapshot").exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.components.workers.rows[0].applied_status_version").isEqualTo("1")
                .jsonPath("$.components.workers.rows[0].applied_latest_finished_task_version").isEqualTo("1")
                .jsonPath("$.components.workers.rows[0].applied_observation_age_ms").doesNotExist()
                .jsonPath("$.components.workers.rows[0].poll_response_age_ms").isNumber()
                .jsonPath("$.components.workers.rows[0].last_reported_alive").isEqualTo(true)
                .jsonPath("$.components.workers.rows[0].healthy").isEqualTo(true)
                .jsonPath("$.components.workers.rows[0].consecutive_worker_failures").isEqualTo(0)
                .jsonPath("$.components.workers.rows[0].waiting_query_len").isEqualTo(6)
                .jsonPath("$.components.workers.rows[0].running_query_len").isEqualTo(4)
                .jsonPath("$.components.workers.rows[0].available_kv_cache_tokens").isEqualTo(512)
                .jsonPath("$.components.workers.rows[0].total_kv_cache_tokens").isEqualTo(1024)
                .jsonPath("$.components.workers.rows[0].received_cache_version").isEqualTo("9")
                .jsonPath("$.components.workers.rows[0].indexed_cache_version").isEqualTo("9")
                .jsonPath("$.components.workers.rows[0].index_initialized").isEqualTo(true)
                .jsonPath("$.components.workers.rows[0].cached_key_count").isEqualTo(4);
    }

    private static void publishAppliedStatus(WorkerStatus status) {
        WorkerStatus.PollLease lease = status.tryBeginStatusPoll(1L);
        assertTrue(status.tryBeginStatusApplication(lease));
        status.recordStatusRpcResult(lease, true, true, null);
        lease.close();
        status.lock.lock();
        try {
            WorkerStatus.StatusObservation observation = status.bindStatusObservation(
                    new WorkerStatus.EngineObservation(
                            RoleType.PREFILL, null, 512, 1024, Map.of(), 5.0, 3, 1, 1, 0,
                            8192, 4096, 4, 6),
                    true, 1L, 1L, Map.of());
            status.publishPreparedStatus(status.prepareNewStatus(observation, lease));
        } finally {
            status.lock.unlock();
            status.finishStatusApplication(lease);
        }
        status.cacheMutationLock.lock();
        try {
            status.publishCacheStatus(
                    CacheStatus.builder().version(9L).cacheKeySize(4L).build());
            status.publishCacheIndexedVersion(9L);
        } finally {
            status.cacheMutationLock.unlock();
        }
    }

    @Test
    void oneCaptureAtATimeRejectsConcurrentSnapshots() throws Exception {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g", "127.0.0.1", 8080, 9090, "s");
        CountDownLatch blocked = new CountDownLatch(1);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenAnswer(invocation -> {
            blocked.await();
            return status;
        });
        Map<String, PrefillEndpoint> captured = new LinkedHashMap<>();
        captured.put("127.0.0.1:8080", endpoint);
        when(registry.snapshotPrefillEndpoints()).thenReturn(captured);
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Void> first = executor.submit(() -> {
                client.get().uri("/rtp_llm/debug/snapshot?include=workers").exchange()
                        .expectStatus().isOk();
                return null;
            });
            verify(registry, timeout(2000)).snapshotPrefillEndpoints();
            client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange()
                    .expectStatus().isEqualTo(429)
                    .expectBody().jsonPath("$.status").isEqualTo("busy");
            blocked.countDown();
            first.get(5, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void diagnosticRoutesLeaveLegacyOperationalRoutesUntouched() {
        client.get().uri("/rtp_llm/queue_snapshot").exchange()
                .expectStatus().isNotFound();
        client.get().uri("/rtp_llm/inflight_status").exchange()
                .expectStatus().isNotFound();
        client.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange()
                .expectStatus().isOk();

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        when(registry.snapshotPrefillEndpoints()).thenReturn(Map.of());
        when(registry.snapshotDecodeEndpoints()).thenReturn(Map.of());
        HttpLoadBalanceServer loadBalance = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class), configService, scheduler, registry,
                mock(WorkerDirectory.class), mock(ApplicationLifecycle.class),
                new ServerScheduleLatencyRecorder());
        RouterFunction<?> merged =
                server.diagnosticRoutes().andOther(loadBalance.loadBalancePrefill());

        WebTestClient mergedClient = WebTestClient.bindToRouterFunction(merged).build();
        mergedClient.get().uri("/rtp_llm/inflight_status").exchange()
                .expectStatus().isOk();
        mergedClient.get().uri("/rtp_llm/debug/snapshot?include=scheduler").exchange()
                .expectStatus().isOk();
    }
}
