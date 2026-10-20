package org.flexlb.httpserver;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Semaphore;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

/** Bounded, read-only P/D diagnostics. Legacy operational APIs keep their contracts. */
@Component
public final class MasterDiagnosticsServer {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int MAX_BYTES = 1_048_576;
    private final String instanceId = UUID.randomUUID().toString();
    private final Semaphore captures = new Semaphore(1);
    private final EndpointRegistry registry;
    private final RequestScheduler scheduler;

    public MasterDiagnosticsServer(EndpointRegistry registry, RequestScheduler scheduler) {
        this.registry = registry;
        this.scheduler = scheduler;
    }

    @Bean
    public RouterFunction<ServerResponse> diagnosticRoutes() {
        return route().GET("/rtp_llm/debug/snapshot", this::snapshot).build();
    }

    private Mono<ServerResponse> snapshot(ServerRequest request) {
        final Set<String> include;
        final int limit;
        final int budgetMs;
        try {
            if (!Set.of("include", "limit", "budget_ms").containsAll(request.queryParams().keySet())
                    || request.queryParams().values().stream().anyMatch(v -> v.size() != 1)) {
                throw new IllegalArgumentException("unknown or repeated parameter");
            }
            List<String> components = Arrays.asList(request.queryParam("include")
                    .orElse("workers,scheduler").split(",", -1));
            include = Set.copyOf(components);
            if (include.size() != components.size()
                    || !Set.of("workers", "scheduler").containsAll(include)) {
                throw new IllegalArgumentException("unsupported component");
            }
            limit = parameter(request, "limit", 1024, 2048);
            budgetMs = parameter(request, "budget_ms", 100, 250);
        } catch (IllegalArgumentException invalid) {
            return ServerResponse.badRequest().bodyValue(Map.of("error", invalid.getMessage()));
        }
        if (!captures.tryAcquire()) {
            return ServerResponse.status(429).bodyValue(Map.of("status", "busy"));
        }
        try {
            long start = System.currentTimeMillis();
            long expires = System.nanoTime() + budgetMs * 1_000_000L;
            Map<String, Object> components = new LinkedHashMap<>();
            if (include.contains("workers")) components.put("workers", workers(limit, expires));
            if (include.contains("scheduler")) {
                long began = System.currentTimeMillis();
                components.put("scheduler", Map.of(
                        "status", "ok", "capture_started_at_ms", began,
                        "capture_finished_at_ms", System.currentTimeMillis(),
                        "data", Map.of("queued_requests", scheduler.getQueuedRequestCount(),
                                "inflight_requests", scheduler.getInflightSize())));
            }
            boolean complete = components.values().stream()
                    .allMatch(c -> "ok".equals(((Map<?, ?>) c).get("status")));
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("schema_version", 2);
            result.put("instance_id", instanceId);
            result.put("snapshot_id", UUID.randomUUID().toString());
            result.put("capture_started_at_ms", start);
            result.put("capture_finished_at_ms", System.currentTimeMillis());
            result.put("status", complete ? "ok" : "partial");
            result.put("capabilities", Map.of(
                    "workers", Map.of("version", 1, "features", List.of(
                            "poll_health", "applied_observation", "cache_progress")),
                    "scheduler", Map.of("version", 1, "features", List.of("counts"))));
            result.put("components", components);
            String body = JSON.writeValueAsString(result);
            if (body.getBytes(StandardCharsets.UTF_8).length > MAX_BYTES) {
                return ServerResponse.status(503).bodyValue(Map.of("status", "unavailable", "reason", "byte_budget"));
            }
            return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                    .header("Cache-Control", "no-store").bodyValue(body);
        } catch (JsonProcessingException failure) {
            return ServerResponse.status(503).bodyValue(Map.of("status", "unavailable", "reason", "serialization"));
        } finally {
            captures.release();
        }
    }

    private static int parameter(ServerRequest request, String name, int defaultValue, int maximum) {
        int value = Integer.parseInt(request.queryParam(name).orElse(Integer.toString(defaultValue)));
        if (value <= 0 || value > maximum) throw new IllegalArgumentException("invalid " + name);
        return value;
    }

    private Map<String, Object> workers(int limit, long expires) {
        long start = System.currentTimeMillis();
        EndpointRegistry.DiagnosticDirectory directory = registry.diagnosticDirectory();
        List<Map<String, Object>> rows = new ArrayList<>();
        String reason = null;
        for (var entry : directory.prefill()) {
            if (rows.size() >= limit || System.nanoTime() >= expires) {
                reason = "capture_budget";
                break;
            }
            rows.add(worker(entry.address(), "prefill", entry.endpoint()));
        }
        if (reason == null) {
            for (var entry : directory.decode()) {
                if (rows.size() >= limit || System.nanoTime() >= expires) {
                    reason = "capture_budget";
                    break;
                }
                rows.add(worker(entry.getKey(), "decode", entry.getValue()));
            }
        }
        if (!registry.diagnosticDirectoryUnchanged(directory)) reason = "directory_changed";
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", reason == null ? "ok" : "partial");
        result.put("reason", reason);
        result.put("capture_started_at_ms", start);
        result.put("capture_finished_at_ms", System.currentTimeMillis());
        result.put("consistency", "immutable_role_directories_independent_worker_sources");
        result.put("coverage", Map.of("scope", "prefill_decode_endpoint_registry",
                "complete", reason == null, "truncated", "capture_budget".equals(reason),
                "returned_count", rows.size()));
        result.put("rows", rows);
        return result;
    }

    private static Map<String, Object> worker(String address, String role, WorkerEndpoint endpoint) {
        WorkerStatus status = endpoint.getStatus();
        WorkerStatus.PollHealth poll = status.pollHealth();
        WorkerStatus.AppliedObservation applied = status.appliedObservation();
        long nowUs = System.nanoTime() / 1000;
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("worker_id", role + ":" + address);
        row.put("address", address);
        row.put("role", role);
        row.put("generation", Long.toString(status.getGenerationId()));
        row.put("lifecycle_state", status.isActiveGeneration() ? "active" : "retiring");
        row.put("last_reported_alive", poll.successfulPollIntervalUs() == 0 ? null : poll.reportedAlive());
        row.put("poll_response_age_ms", poll.successfulPollIntervalUs() == 0 ? null : age(nowUs, poll.lastSuccessfulPollUs()));
        row.put("applied_observation_age_ms", age(nowUs, applied.observedAtUs()));
        row.put("applied_status_version", applied.statusVersion() < 0 ? null : Long.toString(applied.statusVersion()));
        if ("prefill".equals(role)) {
            var cache = status.getCacheStatus();
            long indexedVersion = status.diagnosticCacheIndexedVersion();
            row.put("received_cache_version", cache == null ? null
                    : Long.toString(cache.getVersion()));
            row.put("indexed_cache_version", indexedVersion < 0 ? null : Long.toString(indexedVersion));
            row.put("index_initialized", indexedVersion >= 0);
            row.put("cached_key_count", cache == null ? null : cache.getCacheKeySize());
        }
        return row;
    }

    private static Double age(long nowUs, long observedUs) {
        return observedUs == 0 ? null : Math.max(0L, nowUs - observedUs) / 1000.0;
    }
}
