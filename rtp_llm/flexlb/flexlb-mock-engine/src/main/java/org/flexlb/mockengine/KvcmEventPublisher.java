package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

/**
 * Test-engine implementation of cpp/cache/events/KVCMPublisher.cc's Meta HTTP
 * protocol. Only logical device-cache keys are published; location URIs describe
 * simulated HBM and cannot be used to transfer real KV tensors.
 */
final class KvcmEventPublisher implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    // Bundle engines share transport threads and connection pools. Disabled
    // publishing never enters this factory or allocates an HTTP client.
    private static final Map<Integer, HttpClient> HTTP_CLIENTS = new ConcurrentHashMap<>();
    record Config(URI endpoint, String group, String instance, String model,
                  int queueCapacity, int batchSize, int flushMs, int heartbeatMs,
                  int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs, String storageType) {
        Config(URI endpoint, String group, String instance, String model,
               int queueCapacity, int batchSize, int flushMs, int heartbeatMs,
               int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs) {
            this(endpoint, group, instance, model, queueCapacity, batchSize, flushMs, heartbeatMs,
                    snapshotMs, timeoutMs, retryMs, maxRetryMs, "ST_EVENT_REPORT");
        }
        Config {
            if (!("http".equals(endpoint.getScheme()) || "https".equals(endpoint.getScheme())) || endpoint.getHost() == null
                    || endpoint.getUserInfo() != null || endpoint.getQuery() != null || endpoint.getFragment() != null
                    || group.isBlank() || instance.isBlank() || model.isBlank()
                    || !("ST_EVENT_REPORT".equals(storageType) || "ST_EVENT_REPORT_L1P5".equals(storageType))
                    || queueCapacity <= 0 || batchSize <= 0 || flushMs <= 0 || heartbeatMs <= 0
                    || snapshotMs <= 0 || timeoutMs <= 0 || retryMs <= 0 || maxRetryMs < retryMs)
                throw new IllegalArgumentException("invalid KVCM event configuration");
        }

        static Config fromEnv(Map<String, String> env) {
            String type = env.getOrDefault("KV_CACHE_EVENT_PUBLISHER_TYPE", "none");
            if (type.equals("none")) return null;
            if (!type.equals("kvcm")) throw new IllegalArgumentException("unknown cache event publisher type");
            String endpoint = env.getOrDefault("KV_CACHE_EVENT_MANAGER_ENDPOINT", "");
            if (!endpoint.contains("://") && !endpoint.isBlank()) endpoint = "http://" + endpoint;
            String deployment = env.getOrDefault("BIZ_NAME", "") + "/" + env.getOrDefault("DEPLOYMENT_NAME", "");
            String instance = env.getOrDefault("KV_CACHE_EVENT_INSTANCE_ID", deployment.equals("/") ? "" : deployment);
            // A bundle has many independent engines. One global owner override
            // would let one engine's snapshot erase another engine's keys.
            if (!env.getOrDefault("KV_CACHE_EVENT_HOST_IP_PORT", "").isBlank())
                throw new IllegalArgumentException("Mock derives a unique host:HTTP-port for every engine; remove KV_CACHE_EVENT_HOST_IP_PORT");
            return new Config(URI.create(endpoint.replaceAll("/+$", "")),
                    env.getOrDefault("KV_CACHE_EVENT_INSTANCE_GROUP", env.getOrDefault("RECO_INSTANCE_GROUP", "")),
                    instance, env.getOrDefault("MODEL_TYPE", "mock"),
                    integer(env, "QUEUE_CAPACITY", 65536), integer(env, "BATCH_SIZE", 256),
                    integer(env, "FLUSH_MS", 100), integer(env, "HEARTBEAT_MS", 5000),
                    integer(env, "SNAPSHOT_MS", 60000), integer(env, "TIMEOUT_MS", 1500),
                    integer(env, "RETRY_MS", 100), integer(env, "MAX_RETRY_MS", 5000),
                    env.getOrDefault("MOCK_KVCM_STORAGE_TYPE", "ST_EVENT_REPORT"));
        }

        private static int integer(Map<String, String> env, String name, int fallback) {
            return Integer.parseInt(env.getOrDefault("MOCK_KVCM_" + name, Integer.toString(fallback)));
        }
    }

    @FunctionalInterface interface Transport { boolean post(String route, String body) throws Exception; }
    private final Config config;
    private final MockLruBlockCache cache;
    private final BooleanSupplier available;
    private final String host;
    private final String instanceId;
    private final int blockSize;
    private final Transport transport;
    private final MutationQueue queue;
    private final AtomicLong dirty = new AtomicLong(1);
    private final AtomicLong overflows = new AtomicLong();
    private final AtomicLong failures = new AtomicLong();
    private final AtomicLong reports = new AtomicLong();
    private final AtomicLong snapshots = new AtomicLong();
    private final AtomicLong traces = new AtomicLong();
    private volatile String state = "STARTING";
    private volatile boolean closed;
    private Thread worker;
    private boolean registered;
    private long cutoff, reconciled;
    private record Pending(String body, long version, long generation) {}
    private Pending pending;

    /** Cache-side offer never waits for a consumer, including snapshot pruning. */
    private static final class MutationQueue {
        private final int capacity;
        private final ArrayDeque<MockLruBlockCache.Mutation> entries = new ArrayDeque<>();
        private final ReentrantLock lock = new ReentrantLock();
        private volatile int depth;
        MutationQueue(int capacity) { this.capacity = capacity; }
        boolean offer(MockLruBlockCache.Mutation mutation) {
            if (!lock.tryLock()) return false;
            try {
                if (entries.size() == capacity) return false;
                entries.addLast(mutation);
                depth = entries.size();
                return true;
            } finally { lock.unlock(); }
        }
        void drainTo(List<MockLruBlockCache.Mutation> batch, int limit) {
            lock.lock();
            try {
                while (!entries.isEmpty() && batch.size() < limit) batch.add(entries.removeFirst());
                depth = entries.size();
            } finally { lock.unlock(); }
        }
        void discardThrough(long version) {
            lock.lock();
            try {
                entries.removeIf(mutation -> mutation.version() <= version);
                depth = entries.size();
            } finally { lock.unlock(); }
        }
        int size() { return depth; }
    }

    KvcmEventPublisher(Config config, MockLruBlockCache cache, String host, int blockSize,
                       BooleanSupplier available, Transport transport) {
        this.config = config;
        this.cache = cache;
        this.host = host;
        // Match KvcmWorkerMetadataResolver's namespace + block-size lookup.
        // Per-engine ownership belongs in host_ip_port, not separate instances.
        this.instanceId = config.instance() + "_" + blockSize;
        this.blockSize = blockSize;
        this.available = available;
        this.transport = transport == null ? httpTransport(config) : transport;
        queue = new MutationQueue(config.queueCapacity());
    }

    private static Transport httpTransport(Config config) {
        HttpClient client = HTTP_CLIENTS.computeIfAbsent(config.timeoutMs(), timeout -> HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeout)).version(HttpClient.Version.HTTP_1_1).build());
        return (route, body) -> {
            HttpRequest request = HttpRequest.newBuilder(URI.create(config.endpoint().toString() + route))
                    .timeout(Duration.ofMillis(config.timeoutMs())).header("Content-Type", "application/json")
                    .header("Accept", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build();
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() >= 200 && response.statusCode() < 300 && responseOk(response.body());
        };
    }

    static boolean responseOk(String response) {
        try {
            JsonNode root = JSON.readTree(response);
            if (!codeOk(root.path("header").path("status").path("code"))) return false;
            JsonNode items = root.has("item_results") ? root.get("item_results") : root.get("itemResults");
            if (items != null) {
                if (!items.isArray()) return false;
                for (JsonNode item : items) if (!codeOk(item)) return false;
            }
            return true;
        } catch (Exception error) { return false; }
    }

    private static boolean codeOk(JsonNode node) {
        if (node.isObject()) return codeOk(node.has("code") ? node.get("code") : node.path("status"));
        return node.isIntegralNumber() ? node.asLong() == 1
                : node.isTextual() && (node.asText().equals("OK") || node.asText().equals("1"));
    }

    void start() {
        cache.setMutationListener(this::enqueue);
        worker = Thread.ofVirtual().name("mock-kvcm-" + host).start(this::run);
    }

    private void enqueue(MockLruBlockCache.Mutation mutation) {
        if (closed) return;
        if (mutation.key() == null) dirty.incrementAndGet();
        else if (!queue.offer(mutation)) { overflows.incrementAndGet(); dirty.incrementAndGet(); }
    }

    Map<String, Object> status() {
        return Map.of("state", state, "host_ip_port", host, "queue_depth", queue.size(),
                "overflows", overflows.get(), "failures", failures.get(), "reports", reports.get(),
                "snapshots", snapshots.get());
    }

    private ObjectNode registration() {
        ObjectNode root = JSON.createObjectNode();
        root.put("trace_id", trace()).put("instance_group", config.group()).put("instance_id", instanceId)
                .put("block_size", blockSize);
        root.putObject("model_deployment").put("model_name", config.model()).put("dtype", "fp16")
                .put("use_mla", false).put("tp_size", 1).put("dp_size", 1).put("pp_size", 1);
        // One simulated HBM pool; size is a mock descriptor, not tensor bytes.
        root.putArray("location_spec_infos").addObject().put("name", "mock-hbm").put("size", 1);
        root.putArray("location_spec_groups").addObject().put("name", "default")
                .putArray("spec_names").add("mock-hbm");
        return root;
    }

    private String trace() { return "mock-" + host + "-" + traces.incrementAndGet(); }
    private ObjectNode report() {
        ObjectNode root = JSON.createObjectNode();
        root.put("trace_id", trace()).put("instance_id", instanceId).put("host_ip_port", host)
                .put("storage_type", config.storageType());
        root.putArray("events");
        return root;
    }

    private ObjectNode event(ObjectNode report, String type, String payload) {
        return ((ArrayNode) report.get("events")).addObject().put("event_type", type).putObject(payload);
    }
    private void specs(ObjectNode block) {
        block.putArray("specs").addObject().put("name", "mock-hbm").put("uri", "mock://" + host);
    }
    private ObjectNode control(String type, String payload) {
        ObjectNode root = report();
        ObjectNode value = event(root, type, payload);
        if (type.equals("EVENT_NODE_REGISTER")) value.putArray("mediums").add("hbm");
        if (type.equals("EVENT_HEARTBEAT")) value.putObject("system_status").put("engine", "rtp-llm-mock").put("dp_rank", "0");
        return root;
    }

    private boolean post(String route, String body) throws Exception {
        boolean ok = transport.post(route, body);
        if (ok) reports.incrementAndGet();
        return ok;
    }
    private boolean control(String type) throws Exception {
        String payload = switch (type) {
            case "EVENT_NODE_REGISTER" -> "node_register";
            case "EVENT_HOST_DOWN" -> "host_down";
            default -> "heartbeat";
        };
        return post("/api/reportEvent", control(type, payload).toString());
    }

    private void run() {
        int backoff = config.retryMs();
        long heartbeat = 0, snapshotAt = 0;
        try {
            while (!closed) {
                try {
                    if (!available.getAsBoolean()) {
                        if (registered) { registered = false; control("EVENT_HOST_DOWN"); }
                        pending = null;
                        state = "STOPPED";
                        Thread.sleep(config.flushMs());
                        continue;
                    }
                    if (!registered) {
                        if (!post("/api/registerInstance", registration().toString()) || !control("EVENT_NODE_REGISTER"))
                            throw new IllegalStateException("registration rejected");
                        registered = true;
                        dirty.incrementAndGet();
                    }
                    long now = System.nanoTime();
                    if (now >= snapshotAt) {
                        dirty.incrementAndGet();
                        snapshotAt = now + Duration.ofMillis(config.snapshotMs()).toNanos();
                    }
                    if (pending != null || reconciled != dirty.get()) {
                        state = "RESYNCING";
                        if (pending == null) {
                            long generation = dirty.get();
                            var snapshot = cache.keySnapshot();
                            ObjectNode root = report();
                            ArrayNode blocks = event(root, "EVENT_BLOCK_SNAPSHOT", "block_snapshot").putArray("blocks");
                            for (Long key : snapshot.keys()) {
                                ObjectNode block = blocks.addObject().put("block_key", key.toString()).put("medium", "hbm");
                                specs(block);
                            }
                            pending = new Pending(root.toString(), snapshot.version(), generation);
                        }
                        // Retry the identical snapshot until acknowledged; do not
                        // replace an uncertain request with a newer snapshot.
                        if (!post("/api/reportEvent", pending.body())) throw new IllegalStateException("snapshot rejected");
                        cutoff = pending.version();
                        queue.discardThrough(cutoff);
                        reconciled = pending.generation();
                        pending = null;
                        snapshots.incrementAndGet();
                        if (now >= heartbeat) {
                            if (!control("EVENT_HEARTBEAT")) throw new IllegalStateException("heartbeat rejected");
                            heartbeat = System.nanoTime() + Duration.ofMillis(config.heartbeatMs()).toNanos();
                        }
                        if (reconciled != dirty.get()) Thread.sleep(config.retryMs());
                    } else {
                        var batch = new ArrayList<MockLruBlockCache.Mutation>();
                        queue.drainTo(batch, config.batchSize());
                        // KVCM processes ADD before DELETE in a batch: retain
                        // only the last transition for each key (real parity).
                        var last = new LinkedHashMap<Long, MockLruBlockCache.Mutation>();
                        for (var mutation : batch) if (mutation.version() > cutoff) last.put(mutation.key(), mutation);
                        if (!last.isEmpty()) {
                            ObjectNode root = report();
                            for (var mutation : last.values()) {
                                ObjectNode value = event(root, mutation.added() ? "EVENT_BLOCK_ADD" : "EVENT_BLOCK_DELETE",
                                        mutation.added() ? "block_add" : "block_delete");
                                value.put("block_key", mutation.key().toString()).put("medium", "hbm");
                                if (mutation.added()) specs(value); else value.putArray("spec_names").add("mock-hbm");
                            }
                            if (!post("/api/reportEvent", root.toString())) throw new IllegalStateException("mutation rejected");
                        }
                        if (now >= heartbeat) {
                            if (!control("EVENT_HEARTBEAT")) throw new IllegalStateException("heartbeat rejected");
                            heartbeat = now + Duration.ofMillis(config.heartbeatMs()).toNanos();
                        }
                        state = "READY";
                        Thread.sleep(config.flushMs());
                    }
                    backoff = config.retryMs();
                } catch (InterruptedException error) { break;
                } catch (Exception error) {
                    failures.incrementAndGet();
                    state = "DEGRADED";
                    registered = false;
                    dirty.incrementAndGet();
                    Thread.sleep(backoff);
                    backoff = (int) Math.min((long) config.maxRetryMs(), (long) backoff * 2);
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } finally {
            // Clear interrupt so a bounded best-effort final HTTP request can run.
            Thread.interrupted();
            if (registered) try { control("EVENT_HOST_DOWN"); } catch (Exception ignored) { }
            state = "CLOSED";
        }
    }

    @Override public void close() {
        closed = true;
        cache.setMutationListener(null);
        if (worker != null) {
            worker.interrupt();
            try { worker.join(config.timeoutMs() + 1000L); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        }
    }
}
