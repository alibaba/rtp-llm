package org.flexlb.httpserver;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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

/**
 * Bounded, read-only master diagnostics over the live P/D fleet.
 * A single capture runs at a time and is limited by row count, wall-clock
 * budget and a response byte cap; legacy operational routes keep their
 * contracts.
 */
@Component
public class MasterDiagnosticsServer {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int MAX_BYTES = 1_048_576;
    private final String instanceId = UUID.randomUUID().toString();
    private final Semaphore captures = new Semaphore(1);
    private final EndpointRegistry endpointRegistry;
    private final RequestScheduler requestScheduler;

    public MasterDiagnosticsServer(EndpointRegistry endpointRegistry,
                                   RequestScheduler requestScheduler) {
        this.endpointRegistry = endpointRegistry;
        this.requestScheduler = requestScheduler;
    }

    @Bean
    public RouterFunction<ServerResponse> diagnosticRoutes() {
        return route()
                .GET("/rtp_llm/debug/snapshot", this::snapshot)
                .GET("/rtp_llm/debug/prefill_traffic", this::prefillTraffic)
                .build();
    }

    private Mono<ServerResponse> prefillTraffic(ServerRequest request) {
        return ServerResponse.ok()
                .header("Cache-Control", "no-store")
                .bodyValue(Map.of(
                        "schema_version", 1,
                        "captured_at_ms", System.currentTimeMillis(),
                        "pools", requestScheduler.prefillTrafficSnapshot()));
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
            if (include.contains("workers")) {
                components.put("workers", workers(limit, expires));
            }
            if (include.contains("scheduler")) {
                long began = System.currentTimeMillis();
                components.put("scheduler", Map.of(
                        "status", "ok", "capture_started_at_ms", began,
                        "capture_finished_at_ms", System.currentTimeMillis(),
                        "data", Map.of("queued_requests", requestScheduler.getQueuedRequestCount(),
                                "inflight_requests", requestScheduler.getInflightSize())));
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
                            "poll_health", "engine_observation", "applied_status_cursor",
                            "cache_index")),
                    "scheduler", Map.of("version", 1, "features", List.of("counts"))));
            result.put("components", components);
            String body = JSON.writeValueAsString(result);
            if (body.getBytes(StandardCharsets.UTF_8).length > MAX_BYTES) {
                return ServerResponse.status(503)
                        .bodyValue(Map.of("status", "unavailable", "reason", "byte_budget"));
            }
            return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                    .header("Cache-Control", "no-store").bodyValue(body);
        } catch (JsonProcessingException failure) {
            return ServerResponse.status(503)
                    .bodyValue(Map.of("status", "unavailable", "reason", "serialization"));
        } finally {
            captures.release();
        }
    }

    private static int parameter(ServerRequest request, String name, int defaultValue, int maximum) {
        int value = Integer.parseInt(request.queryParam(name).orElse(Integer.toString(defaultValue)));
        if (value <= 0 || value > maximum) {
            throw new IllegalArgumentException("invalid " + name);
        }
        return value;
    }

    private Map<String, Object> workers(int limit, long expires) {
        long start = System.currentTimeMillis();
        Map<String, PrefillEndpoint> prefill = endpointRegistry.snapshotPrefillEndpoints();
        Map<String, DecodeEndpoint> decode = endpointRegistry.snapshotDecodeEndpoints();
        List<Map<String, Object>> rows = new ArrayList<>();
        String reason = null;
        for (Map.Entry<String, PrefillEndpoint> entry : prefill.entrySet()) {
            if (rows.size() >= limit || System.nanoTime() >= expires) {
                reason = "capture_budget";
                break;
            }
            rows.add(worker(entry.getKey(), "prefill", entry.getValue()));
        }
        if (reason == null) {
            for (Map.Entry<String, DecodeEndpoint> entry : decode.entrySet()) {
                if (rows.size() >= limit || System.nanoTime() >= expires) {
                    reason = "capture_budget";
                    break;
                }
                rows.add(worker(entry.getKey(), "decode", entry.getValue()));
            }
        }
        if (reason == null && !directoryUnchanged(prefill, decode)) {
            reason = "directory_changed";
        }
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

    private boolean directoryUnchanged(Map<String, PrefillEndpoint> prefill,
                                       Map<String, DecodeEndpoint> decode) {
        return endpointRegistry.snapshotPrefillEndpoints().equals(prefill)
                && endpointRegistry.snapshotDecodeEndpoints().equals(decode);
    }

    private static Map<String, Object> worker(String address, String role, WorkerEndpoint endpoint) {
        WorkerStatus status = endpoint.getStatus();
        WorkerStatus.PollHealth poll = status.pollHealth();
        WorkerStatus.EngineObservation engine = status.committedEngineObservation();
        WorkerStatus.AppliedStatusCursor cursor = status.appliedStatusCursor();
        long nowUs = System.nanoTime() / 1000;
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("worker_id", role + ":" + address);
        row.put("address", address);
        row.put("role", role);
        row.put("group", status.getGroup());
        row.put("generation", Long.toString(status.getGenerationId()));
        row.put("lifecycle_state", status.isActiveGeneration() ? "active" : "retiring");
        row.put("waiting_query_len", engine.waitingQueryLen());
        row.put("running_query_len", engine.runningQueryLen());
        row.put("available_kv_cache_tokens", engine.availableKvCacheTokens());
        row.put("total_kv_cache_tokens", engine.totalKvCacheTokens());
        row.put("last_reported_alive", poll.lastSuccessfulPollUs() == 0L ? null : poll.reportedAlive());
        row.put("poll_response_age_ms", age(nowUs, poll.lastSuccessfulPollUs()));
        row.put("consecutive_worker_failures", poll.consecutiveWorkerFailures());
        row.put("healthy", poll.healthy());
        row.put("applied_status_version", cursor.statusVersion() < 0L
                ? null : Long.toString(cursor.statusVersion()));
        row.put("applied_latest_finished_task_version", cursor.latestFinishedTaskVersion() < 0L
                ? null : Long.toString(cursor.latestFinishedTaskVersion()));
        if ("prefill".equals(role)) {
            WorkerStatus.CacheIndexSnapshot cache = status.cacheIndexSnapshot();
            row.put("received_cache_version", cache.cacheStatus() == null ? null
                    : Long.toString(cache.cacheStatus().getVersion()));
            row.put("indexed_cache_version", cache.indexInitialized()
                    ? Long.toString(cache.indexedVersion()) : null);
            row.put("index_initialized", cache.indexInitialized());
            row.put("cached_key_count", cache.cacheStatus() == null
                    ? null : cache.cacheStatus().getCacheKeySize());
        }
        return row;
    }

    private static Double age(long nowUs, long observedUs) {
        return observedUs == 0 ? null : Math.max(0L, nowUs - observedUs) / 1000.0;
    }
}
