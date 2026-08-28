package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Lightweight HTTP control server for the Java mock engine cluster.
 *
 * <p>Provides 11 endpoints mirroring the legacy Python mock control API:
 * snapshot, inject, clear_inject, health, requests, set_perf, set_kv_pressure,
 * set_queue_depth, stop_engine, start_engine, and metrics.
 *
 * <p>Python compatibility layer (Phase 2): all POST endpoints accept dual
 * addressing — either {@code {"engine": "<name>"}} resolved by engine name
 * (e.g. "prefill-0") or {@code {"port": N}} resolved by gRPC port, matching
 * the legacy Python mock control plane which addresses engines by name.
 * Response schemas follow Python: /snapshot wraps engines in
 * {@code {"engines": [...], "cluster_counters": {...}}}, /requests is keyed
 * by engine name, /health returns {@code {"status": "ok"}}, and /metrics
 * emits the Python metric names with matching labels (aggregated by role by
 * default, per-engine with {@code ?per_engine=true}). The pre-existing Java
 * request formats and legacy metric series are retained for backward
 * compatibility.
 *
 * <p>Uses JDK built-in {@link HttpServer} — no additional Maven dependencies.
 */
final class MockControlServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HttpServer httpServer;
    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private final Map<Integer, Server> serversByPort;
    private final EventLoopGroup bossGroup;
    private final EventLoopGroup workerGroup;

    MockControlServer(Map<Integer, JavaMockEngineCluster.FastRpcService> services,
                      Map<Integer, Server> serversByPort,
                      EventLoopGroup bossGroup,
                      EventLoopGroup workerGroup,
                      String host,
                      int httpPort) throws IOException {
        this.services = services;
        this.serversByPort = serversByPort;
        this.bossGroup = bossGroup;
        this.workerGroup = workerGroup;
        this.httpServer = HttpServer.create(new InetSocketAddress(host, httpPort), 0);
        httpServer.createContext("/snapshot", this::handleSnapshot);
        httpServer.createContext("/inject", this::handleInject);
        httpServer.createContext("/clear_inject", this::handleClearInject);
        httpServer.createContext("/health", this::handleHealth);
        httpServer.createContext("/requests", this::handleRequests);
        httpServer.createContext("/set_perf", this::handleSetPerf);
        httpServer.createContext("/set_kv_pressure", this::handleSetKvPressure);
        httpServer.createContext("/set_queue_depth", this::handleSetQueueDepth);
        httpServer.createContext("/stop_engine", this::handleStopEngine);
        httpServer.createContext("/start_engine", this::handleStartEngine);
        httpServer.createContext("/cancel_request", this::handleCancelRequest);
        httpServer.createContext("/metrics", this::handleMetrics);
        httpServer.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "mock-control-http");
            t.setDaemon(true);
            return t;
        }));
    }

    void start() {
        httpServer.start();
    }

    void stop() {
        httpServer.stop(0);
    }

    int getPort() {
        return httpServer.getAddress().getPort();
    }

    // ────────────────── Dual addressing (Python compat) ──────────────────

    /** Carries an HTTP error status + message back to the handler wrapper. */
    private static final class ApiException extends Exception {
        final int status;
        final Map<String, Object> response;

        ApiException(int status, String message) {
            super(message);
            this.status = status;
            this.response = Map.of("error", message);
        }

        ApiException(
                int status,
                String message,
                Map<String, Object> response) {
            super(message);
            this.status = status;
            this.response = response;
        }
    }

    @FunctionalInterface
    private interface ServicePostAction {
        Map<String, Object> apply(
                JsonNode body,
                JavaMockEngineCluster.FastRpcService service) throws Exception;
    }

    /**
     * Python-compatible dual addressing: resolve the target engine by name
     * ({@code "engine"} field, e.g. "prefill-0" — the same naming scheme as
     * the cluster) or by gRPC port ({@code "port"} field, the
     * original Java scheme). Errors if neither is present or the engine is
     * unknown.
     */
    private JavaMockEngineCluster.FastRpcService resolveService(JsonNode body) throws ApiException {
        JsonNode engineNode = body.path("engine");
        if (engineNode.isTextual() && !engineNode.asText().isEmpty()) {
            String engineName = engineNode.asText();
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                if (engineName.equals(service.getEngineName())) {
                    return service;
                }
            }
            throw new ApiException(404, "engine '" + engineName + "' not found");
        }
        if (body.has("port")) {
            int port = body.path("port").asInt();
            JavaMockEngineCluster.FastRpcService service = services.get(port);
            if (service == null) {
                throw new ApiException(404, "engine not found for port " + port);
            }
            return service;
        }
        throw new ApiException(400, "request must contain 'engine' or 'port'");
    }

    private void handleServicePost(
            HttpExchange exchange,
            ServicePostAction action) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Map.of("error", "Method Not Allowed"));
            return;
        }
        try {
            JsonNode body = MAPPER.readTree(exchange.getRequestBody());
            JavaMockEngineCluster.FastRpcService service = resolveService(body);
            sendJson(exchange, 200, action.apply(body, service));
        } catch (ApiException apiFailure) {
            sendJson(exchange, apiFailure.status, apiFailure.response);
        } catch (Exception failure) {
            // Always answer malformed JSON and unexpected handler failures;
            // if a response was already committed there is nothing left to do.
            try {
                sendJson(exchange, 500,
                        Map.of("error", String.valueOf(failure)));
            } catch (IOException ignored) {
                // Response already committed.
            }
        }
    }

    private static Map<String, Object> successResponse(
            JavaMockEngineCluster.FastRpcService service) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "ok");
        response.put("engine", service.getEngineName());
        response.put("port", service.getGrpcPort());
        return response;
    }

    // ────────────────── Endpoint handlers ──────────────────

    private void handleSnapshot(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Map.of("error", "Method Not Allowed"));
            return;
        }
        // Python cluster.snapshot() shape: {"engines": [...], "cluster_counters": {...}}
        List<Map<String, Object>> engines = new ArrayList<>();
        for (JavaMockEngineCluster.FastRpcService service : orderedServices()) {
            engines.add(service.getSnapshot());
        }
        Map<String, Object> response = new LinkedHashMap<>();
        // Sampling timestamp for cross-source alignment with client epoch_ms
        // fields and the java_mock_stats ts_epoch_ms field (additive, top-level).
        response.put("ts_epoch_ms", System.currentTimeMillis());
        response.put("engines", engines);
        // The Java cluster runs engines in-process (no remote decode forwarding),
        // so the gRPC forwarding counters are always zero — kept for schema parity.
        Map<String, Object> clusterCounters = new LinkedHashMap<>();
        clusterCounters.put("grpc_error_count", 0);
        clusterCounters.put("grpc_retry_count", 0);
        clusterCounters.put("grpc_cancel_forward_count", 0);
        response.put("cluster_counters", clusterCounters);
        sendJson(exchange, 200, response);
    }

    private void handleInject(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            if (body.has("type")) {
                // Original Java fault-injection format ({"port"/"engine", "type", "enabled", ...}).
                String type = body.path("type").asText();
                boolean enabled = body.path("enabled").asBoolean(true);
                FaultInjectionConfig.Builder builder = service.getFaultConfig().toBuilder();
                switch (type) {
                    case "enqueue_error" -> builder.failOnEnqueue(enabled);
                    case "generate_error" -> builder.generateError(enabled);
                    case "fetch_error" -> builder.fetchError(enabled);
                    case "no_respond" -> builder.noRespond(enabled);
                    case "kv_pressure" -> builder.kvPressureTokens(enabled ? body.path("tokens").asLong(500_000) : 0);
                    case "queue_depth" -> builder.queueDepthLimit(enabled ? body.path("depth").asInt(10) : 0);
                    case "crash_after" -> builder.crashAfterNRequests(enabled ? body.path("n").asInt(5) : 0);
                    case "enqueue_delay" -> builder.enqueueDelayMs(enabled ? body.path("delay_ms").asLong(0) : 0);
                    case "generate_delay" -> builder.generateDelayMs(enabled ? body.path("delay_ms").asLong(0) : 0);
                    default -> throw new ApiException(
                            400, "unknown injection type: " + type);
                }
                service.setFaultConfig(builder.build());
                Map<String, Object> response = successResponse(service);
                response.put("type", type);
                return response;
            }

            // Python format (_http_inject / set_injection):
            // {"engine": "prefill-0", "config": {"enqueue_error": bool, "fetch_error": bool,
            //                                     "generate_error": bool, "no_respond": bool}}
            // Python REPLACES the whole inject config (not merge), so build a fresh one.
            JsonNode cfg = body.path("config");
            FaultInjectionConfig injected = FaultInjectionConfig.builder()
                    .failOnEnqueue(cfg.path("enqueue_error").asBoolean(false))
                    .fetchError(cfg.path("fetch_error").asBoolean(false))
                    .generateError(cfg.path("generate_error").asBoolean(false))
                    .noRespond(cfg.path("no_respond").asBoolean(false))
                    .build();
            service.setFaultConfig(injected);
            return successResponse(service);
        });
    }

    private void handleClearInject(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            service.clearFaultConfig();
            return successResponse(service);
        });
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Map.of("error", "Method Not Allowed"));
            return;
        }
        int total = services.size();
        long healthy = services.values().stream().filter(s -> !s.isStopped()).count();
        Map<String, Object> response = new LinkedHashMap<>();
        // Python returns {"status": "ok"}; extra fields kept for the existing
        // EngineCrashRecoveryTest and richer diagnostics (superset is compatible).
        response.put("status", "ok");
        response.put("healthy", healthy == total);
        response.put("engines", total);
        sendJson(exchange, 200, response);
    }

    private void handleRequests(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Map.of("error", "Method Not Allowed"));
            return;
        }
        // Python /requests returns a dict keyed by engine name, each value being
        // the request lifecycle map of that engine.
        Map<String, Object> response = new LinkedHashMap<>();
        for (JavaMockEngineCluster.FastRpcService service : orderedServices()) {
            response.put(service.getEngineName(), service.getRequestLifecycleSnapshot());
        }
        sendJson(exchange, 200, response);
    }

    private void handleSetPerf(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            MockPerformanceModel perf = service.getPerformance();
            // Python fields (_http_set_perf):
            if (body.has("prefill_fixed_ms")) {
                perf.setOverrideFixedPrefillMs(body.get("prefill_fixed_ms").asDouble());
            }
            if (body.has("decode_scale")) {
                perf.setOverrideDecodeScale(body.get("decode_scale").asDouble());
            }
            if (body.has("max_prefill_concurrency")) {
                service.setMaxPrefillConcurrency(body.get("max_prefill_concurrency").asInt());
            }
            // Original Java fields retained:
            if (body.has("prefill_ms")) {
                perf.setOverrideFixedPrefillMs(body.get("prefill_ms").asDouble());
            }
            if (body.has("decode_step_ms")) {
                perf.setOverrideDecodeStepMs(body.get("decode_step_ms").asDouble());
            }
            if (body.has("jitter_pct")) {
                perf.setJitterPct(body.get("jitter_pct").asDouble());
            }
            return successResponse(service);
        });
    }

    private void handleSetKvPressure(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            if (body.has("active_kv_tokens")) {
                // Python semantics (_http_set_kv_pressure): ABSOLUTE
                // value — state._active_kv_tokens = value.
                service.setAbsoluteActiveKvTokens(body.get("active_kv_tokens").asLong(0));
            } else {
                // Original Java semantics: additive pressure tokens.
                long tokens = body.path("tokens").asLong(0);
                FaultInjectionConfig.Builder builder = service.getFaultConfig().toBuilder();
                builder.kvPressureTokens(tokens);
                service.setFaultConfig(builder.build());
            }
            return successResponse(service);
        });
    }

    private void handleSetQueueDepth(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            // NOTE (intentional divergence): the legacy Python queue_depth was a fake
            // display value (only bumped the snapshot "waiting" counter); Java
            // implements it as real enqueue rejection. Field name kept for compatibility.
            int depth = body.has("queue_depth")
                    ? body.get("queue_depth").asInt(0)
                    : body.path("depth").asInt(0);
            FaultInjectionConfig.Builder builder = service.getFaultConfig().toBuilder();
            builder.queueDepthLimit(depth);
            service.setFaultConfig(builder.build());
            return successResponse(service);
        });
    }

    private void handleStopEngine(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            int port = service.getGrpcPort();
            service.setStopped(true);
            Server server = serversByPort.get(port);
            if (server != null) {
                server.shutdownNow();
            }
            Map<String, Object> response = successResponse(service);
            response.put("action", "stopped");
            return response;
        });
    }

    private void handleStartEngine(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            int port = service.getGrpcPort();
            service.clearFaultConfig();
            service.resetEnqueueCount();
            // Release the port fully before rebinding: shut the old server down
            // and await termination, then build+bind the new one. Only after the
            // new server is actually serving do we flip the stopped flag and
            // swap serversByPort — a bind failure leaves a consistent state
            // (stopped=true, no phantom server) instead of stopped=false with
            // nothing listening.
            Server existing = serversByPort.get(port);
            if (existing != null && !existing.isShutdown()) {
                existing.shutdown();
                try {
                    if (!existing.awaitTermination(5, TimeUnit.SECONDS)) {
                        existing.shutdownNow();
                        existing.awaitTermination(5, TimeUnit.SECONDS);
                    }
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    existing.shutdownNow();
                }
            }
            Server server;
            try {
                server = NettyServerBuilder.forPort(port)
                        .bossEventLoopGroup(bossGroup)
                        .workerEventLoopGroup(workerGroup)
                        .channelType(NioServerSocketChannel.class)
                        .directExecutor()
                        .maxInboundMessageSize(16 * 1024 * 1024)
                        .addService(service)
                        .build()
                        .start();
            } catch (Exception e) {
                service.setStopped(true);
                throw new ApiException(500,
                        "failed to start engine on port " + port + ": " + e);
            }
            service.setStopped(false);
            serversByPort.put(port, server);
            Map<String, Object> response = successResponse(service);
            response.put("action", "started");
            return response;
        });
    }

    /**
     * Test-only cancel injection (C1-②, accepted-eviction 8429 wiring): POST
     * {@code {"engine"|"port": ..., "request_id": ...}} drives the same
     * three-branch {@link JavaMockEngineCluster.FastRpcService#cancelRequest}
     * contract as the in-process MockEngineCancelChannel — on the found branch
     * the request is removed and the CANCELLED completion surfaces in the next
     * WorkerStatus finished list (iron rule 4: WorkerStatus stays the sole
     * release-confirmation source). Response: {@code {status, found, phase,
     * already_finished}} with {@code phase} as the TaskPhase enum name (null
     * unless found). A Decode target returns HTTP 501 / {@code UNIMPLEMENTED},
     * matching the production role contract.
     */
    private void handleCancelRequest(HttpExchange exchange) throws IOException {
        handleServicePost(exchange, (body, service) -> {
            if (!body.has("request_id")) {
                throw new ApiException(400, "request must contain 'request_id'");
            }
            long requestId = parseRequestId(body.get("request_id"));
            JavaMockEngineCluster.CancelResult result;
            try {
                result = service.cancelRequest(requestId);
            } catch (UnsupportedOperationException unsupported) {
                throw new ApiException(
                        501,
                        unsupported.getMessage(),
                        Map.of(
                                "status", "UNIMPLEMENTED",
                                "error", unsupported.getMessage()));
            }
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", result.found() ? "ACCEPTED" : "NOT_FOUND");
            response.put("found", result.found());
            response.put("phase", result.phase() == null ? null : result.phase().name());
            response.put("already_finished", result.alreadyFinished());
            response.put("engine", service.getEngineName());
            response.put("port", service.getGrpcPort());
            response.put("request_id", requestId);
            return response;
        });
    }

    /**
     * Strict request_id parsing (P2-3): Jackson's {@code asLong()} coerces
     * non-numeric values to 0, silently turning a caller schema bug into a
     * cancel of request 0 (not_found). Accept integral numbers and integral
     * decimal strings only; everything else is a 400.
     */
    private static long parseRequestId(JsonNode node) throws ApiException {
        if (node.isIntegralNumber() && node.canConvertToLong()) {
            return node.asLong();
        }
        if (node.isTextual()) {
            try {
                return Long.parseLong(node.asText().trim());
            } catch (NumberFormatException ignored) {
                // fall through to the 400 below
            }
        }
        throw new ApiException(400, "'request_id' must be an integer, got: " + node);
    }

    private void handleMetrics(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            sendJson(exchange, 405, Map.of("error", "Method Not Allowed"));
            return;
        }
        String query = exchange.getRequestURI().getQuery();
        boolean perEngine = query != null && query.contains("per_engine=true");

        // Take one snapshot per engine; reused for both Python-style and legacy series.
        List<Map<String, Object>> snaps = new ArrayList<>();
        List<JavaMockEngineCluster.FastRpcService> engineServices = orderedServices();
        for (JavaMockEngineCluster.FastRpcService service : engineServices) {
            snaps.add(service.getSnapshot());
        }

        StringBuilder sb = new StringBuilder();
        appendMetricsMeta(sb);

        if (perEngine) {
            appendPerEngineMetrics(sb, engineServices, snaps);
        } else {
            appendAggregatedMetrics(sb, snaps);
        }
        appendLegacyMetrics(sb, engineServices);

        // Cluster-level counters (Java in-process model: always 0, schema-compatible).
        sb.append("flexlb_mock_grpc_error_count 0\n");
        sb.append("flexlb_mock_grpc_retry_count 0\n");
        sb.append("flexlb_mock_grpc_cancel_forward_count 0\n");

        sendText(exchange, 200, sb.toString());
    }

    /** Services sorted by gRPC port for deterministic output (Python lists engines in creation order). */
    private List<JavaMockEngineCluster.FastRpcService> orderedServices() {
        List<JavaMockEngineCluster.FastRpcService> ordered = new ArrayList<>(services.values());
        ordered.sort(java.util.Comparator.comparingInt(JavaMockEngineCluster.FastRpcService::getGrpcPort));
        return ordered;
    }

    // ────────────────── Metrics builders ──────────────────

    /**
     * HELP/TYPE lines for the union of the Python metric set (legacy
     * ~L825-877 / ~L1344-1396) and the retained legacy Java series.
     */
    private static void appendMetricsMeta(StringBuilder sb) {
        String[][] meta = {
                {"mock_engine_up", "1 if engine is running, 0 if stopped", "gauge"},
                {"mock_engine_running", "current running requests", "gauge"},
                {"mock_engine_waiting", "current waiting requests", "gauge"},
                {"mock_engine_accepted_total", "total accepted requests", "counter"},
                {"mock_engine_completed_total", "total completed requests", "counter"},
                {"mock_engine_cancelled_total", "total cancelled requests", "counter"},
                {"mock_engine_cache_keys", "number of cache keys", "gauge"},
                {"mock_engine_cache_evictions_total", "total cache evictions", "counter"},
                {"mock_engine_active_kv_tokens", "active KV cache tokens", "gauge"},
                {"mock_engine_available_kv_tokens", "available KV cache tokens", "gauge"},
                {"mock_engine_rpc_total", "total RPC calls by method", "counter"},
                {"mock_engine_prefill_ms_avg", "average prefill execution time in ms", "gauge"},
                {"mock_engine_prefill_ms_p99", "p99 prefill execution time in ms", "gauge"},
                {"mock_engine_prefill_ms_count", "number of prefill samples", "gauge"},
                {"mock_engine_decode_ms_avg", "average decode execution time in ms", "gauge"},
                {"mock_engine_decode_ms_p99", "p99 decode execution time in ms", "gauge"},
                {"mock_engine_decode_ms_count", "number of decode samples", "gauge"},
                {"flexlb_mock_grpc_error_count", "Total gRPC errors in remote decode", "counter"},
                {"flexlb_mock_grpc_retry_count", "Total gRPC retries in remote decode", "counter"},
                {"flexlb_mock_grpc_cancel_forward_count", "Total cancel forwarded to remote engines", "counter"},
                // Legacy Java-only series (retained, not part of the Python set).
                {"mock_engine_running_tasks", "Current running tasks", "gauge"},
                {"mock_engine_inflight_count", "Current inflight count", "gauge"},
                {"mock_engine_kv_tokens_used", "KV cache tokens in use", "gauge"},
                {"mock_engine_heap_used_bytes", "JVM heap used in bytes", "gauge"},
        };
        for (String[] m : meta) {
            sb.append("# HELP ").append(m[0]).append(' ').append(m[1]).append('\n');
            sb.append("# TYPE ").append(m[0]).append(' ').append(m[2]).append('\n');
        }
    }

    /**
     * Per-engine series mirroring Python {@code /metrics?per_engine=true}
     * (Python-compat format): labels engine_name/role/grpc_port/engine_ip.
     * This is the mode the Grafana dashboard + Prometheus scrape config use.
     */
    private static void appendPerEngineMetrics(
            StringBuilder sb,
            List<JavaMockEngineCluster.FastRpcService> engineServices,
            List<Map<String, Object>> snaps) {
        for (int i = 0; i < engineServices.size(); i++) {
            JavaMockEngineCluster.FastRpcService service = engineServices.get(i);
            Map<String, Object> snap = snaps.get(i);
            String labels = String.format("engine_name=\"%s\",role=\"%s\",grpc_port=\"%d\",engine_ip=\"%s\"",
                    escapeLabel(service.getEngineName()),
                    escapeLabel(service.getRoleName().toLowerCase()),
                    service.getGrpcPort(),
                    escapeLabel(service.getHost()));
            sb.append(String.format("mock_engine_up{%s} %d%n", labels, service.isStopped() ? 0 : 1));
            sb.append(String.format("mock_engine_running{%s} %s%n", labels, snap.get("running")));
            sb.append(String.format("mock_engine_waiting{%s} %s%n", labels, snap.get("waiting")));
            sb.append(String.format("mock_engine_accepted_total{%s} %s%n", labels, snap.get("accepted")));
            sb.append(String.format("mock_engine_completed_total{%s} %s%n", labels, snap.get("completed")));
            sb.append(String.format("mock_engine_cancelled_total{%s} %s%n", labels, snap.get("cancelled_count")));
            sb.append(String.format("mock_engine_cache_keys{%s} %s%n", labels, snap.get("cache_keys")));
            sb.append(String.format("mock_engine_cache_evictions_total{%s} %s%n", labels, snap.get("cache_evictions")));
            sb.append(String.format("mock_engine_active_kv_tokens{%s} %s%n", labels, snap.get("active_kv_tokens")));
            sb.append(String.format("mock_engine_available_kv_tokens{%s} %s%n", labels, snap.get("available_kv_tokens")));
            @SuppressWarnings("unchecked")
            Map<String, Object> rpcCounts = (Map<String, Object>) snap.get("rpc_counts");
            for (Map.Entry<String, Object> entry : rpcCounts.entrySet()) {
                sb.append(String.format("mock_engine_rpc_total{%s,rpc_method=\"%s\"} %s%n",
                        labels, escapeLabel(entry.getKey()), entry.getValue()));
            }
            sb.append(String.format("mock_engine_prefill_ms_avg{%s} %.1f%n", labels, asDouble(snap.get("prefill_ms_avg"))));
            sb.append(String.format("mock_engine_prefill_ms_p99{%s} %.1f%n", labels, asDouble(snap.get("prefill_ms_p99"))));
            sb.append(String.format("mock_engine_prefill_ms_count{%s} %s%n", labels, snap.get("prefill_ms_count")));
            sb.append(String.format("mock_engine_decode_ms_avg{%s} %.1f%n", labels, asDouble(snap.get("decode_ms_avg"))));
            sb.append(String.format("mock_engine_decode_ms_p99{%s} %.1f%n", labels, asDouble(snap.get("decode_ms_p99"))));
            sb.append(String.format("mock_engine_decode_ms_count{%s} %s%n", labels, snap.get("decode_ms_count")));
        }
    }

    /**
     * Default mode: series aggregated per role with a single {@code role} label,
     * mirroring the legacy Python generate_aggregated_prometheus_metrics
     * ~L882-958) — sums for counters/gauges, weighted avg and max-of-p99 for
     * latency, sorted rpc_method labels.
     */
    private static void appendAggregatedMetrics(StringBuilder sb, List<Map<String, Object>> snaps) {
        Map<String, List<Map<String, Object>>> buckets = new LinkedHashMap<>();
        buckets.put("prefill", new ArrayList<>());
        buckets.put("decode", new ArrayList<>());
        for (Map<String, Object> snap : snaps) {
            Object role = snap.get("role");
            if (role != null && buckets.containsKey(role.toString())) {
                buckets.get(role.toString()).add(snap);
            }
        }
        for (Map.Entry<String, List<Map<String, Object>>> bucket : buckets.entrySet()) {
            List<Map<String, Object>> group = bucket.getValue();
            if (group.isEmpty()) {
                continue;
            }
            String label = "role=\"" + bucket.getKey() + "\"";

            long up = group.stream().filter(e -> !Boolean.TRUE.equals(e.get("stopped"))).count();
            sb.append(String.format("mock_engine_up{%s} %d%n", label, up));
            sb.append(String.format("mock_engine_running{%s} %d%n", label, sumLong(group, "running")));
            sb.append(String.format("mock_engine_waiting{%s} %d%n", label, sumLong(group, "waiting")));
            sb.append(String.format("mock_engine_accepted_total{%s} %d%n", label, sumLong(group, "accepted")));
            sb.append(String.format("mock_engine_completed_total{%s} %d%n", label, sumLong(group, "completed")));
            sb.append(String.format("mock_engine_cancelled_total{%s} %d%n", label, sumLong(group, "cancelled_count")));
            sb.append(String.format("mock_engine_cache_keys{%s} %d%n", label, sumLong(group, "cache_keys")));
            sb.append(String.format("mock_engine_cache_evictions_total{%s} %d%n", label, sumLong(group, "cache_evictions")));
            sb.append(String.format("mock_engine_active_kv_tokens{%s} %d%n", label, sumLong(group, "active_kv_tokens")));
            sb.append(String.format("mock_engine_available_kv_tokens{%s} %d%n", label, sumLong(group, "available_kv_tokens")));

            Map<String, Long> rpcTotals = new TreeMap<>();
            for (Map<String, Object> e : group) {
                @SuppressWarnings("unchecked")
                Map<String, Object> rpcCounts = (Map<String, Object>) e.get("rpc_counts");
                if (rpcCounts != null) {
                    for (Map.Entry<String, Object> entry : rpcCounts.entrySet()) {
                        rpcTotals.merge(entry.getKey(), ((Number) entry.getValue()).longValue(), Long::sum);
                    }
                }
            }
            for (Map.Entry<String, Long> entry : rpcTotals.entrySet()) {
                sb.append(String.format("mock_engine_rpc_total{role=\"%s\",rpc_method=\"%s\"} %d%n",
                        bucket.getKey(), escapeLabel(entry.getKey()), entry.getValue()));
            }

            appendLatencyAggregates(sb, label, group, "prefill");
            appendLatencyAggregates(sb, label, group, "decode");
        }
    }

    private static void appendLatencyAggregates(StringBuilder sb, String label,
                                                List<Map<String, Object>> group, String kind) {
        long totalCount = sumLong(group, kind + "_ms_count");
        double avg = 0.0;
        if (totalCount > 0) {
            double weighted = 0.0;
            for (Map<String, Object> e : group) {
                weighted += asDouble(e.get(kind + "_ms_avg")) * asLong(e.get(kind + "_ms_count"));
            }
            avg = weighted / totalCount;
        }
        double p99 = 0.0;
        for (Map<String, Object> e : group) {
            p99 = Math.max(p99, asDouble(e.get(kind + "_ms_p99")));
        }
        sb.append(String.format("mock_engine_%s_ms_avg{%s} %.1f%n", kind, label, avg));
        sb.append(String.format("mock_engine_%s_ms_p99{%s} %.1f%n", kind, label, p99));
        sb.append(String.format("mock_engine_%s_ms_count{%s} %d%n", kind, label, totalCount));
    }

    /** Retained legacy Java series with port/role labels (pre-Phase-2 format). */
    private static void appendLegacyMetrics(StringBuilder sb,
                                            List<JavaMockEngineCluster.FastRpcService> engineServices) {
        Runtime runtime = Runtime.getRuntime();
        long heapUsed = runtime.totalMemory() - runtime.freeMemory();
        for (JavaMockEngineCluster.FastRpcService service : engineServices) {
            // Lowercase role keeps the legacy series label-consistent with the
            // Python-compat aggregated series (role="prefill"/"decode").
            String labels = String.format("port=\"%d\",role=\"%s\"",
                    service.getGrpcPort(), service.getRoleName().toLowerCase());
            sb.append(String.format("mock_engine_running_tasks{%s} %d%n", labels, service.getRunningCount()));
            sb.append(String.format("mock_engine_accepted_total{%s} %d%n", labels, service.getAcceptedCount()));
            sb.append(String.format("mock_engine_completed_total{%s} %d%n", labels, service.getCompletedCount()));
            sb.append(String.format("mock_engine_inflight_count{%s} %d%n", labels, service.getInflightCount()));
            sb.append(String.format("mock_engine_kv_tokens_used{%s} %d%n", labels, service.getActiveKvTokens()));
            sb.append(String.format("mock_engine_heap_used_bytes{%s} %d%n", labels, heapUsed));
        }
    }

    private static long sumLong(List<Map<String, Object>> group, String key) {
        long sum = 0;
        for (Map<String, Object> e : group) {
            sum += asLong(e.get(key));
        }
        return sum;
    }

    private static long asLong(Object value) {
        return value instanceof Number ? ((Number) value).longValue() : 0L;
    }

    private static double asDouble(Object value) {
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }

    private static String escapeLabel(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    // ────────────────── Utility methods ──────────────────

    private static void sendJson(HttpExchange exchange, int status, Object body) throws IOException {
        String json = MAPPER.writeValueAsString(body);
        byte[] bytes = json.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void sendText(HttpExchange exchange, int status, String text) throws IOException {
        byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }
}
