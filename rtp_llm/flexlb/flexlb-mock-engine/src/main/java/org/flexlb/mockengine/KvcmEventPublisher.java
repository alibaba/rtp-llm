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
import java.util.concurrent.TimeUnit;
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
    record Group(String tag, String type, boolean reuse, long sizeBytes) {
        Group {
            if (tag == null || !tag.matches("[A-Za-z0-9_-]+") || !List.of("FULL", "SWA", "LINEAR").contains(type)
                    || sizeBytes <= 0) throw new IllegalArgumentException("invalid KVCM group descriptor");
        }
        boolean published() { return reuse && !type.equals("LINEAR"); }
    }
    record Discovery(boolean vipserver, String address) {}
    record ModelMetadata(String dtype, boolean useMla, int tp, int dp, int pp, int dpRank) {
        ModelMetadata {
            if (dtype == null || dtype.isBlank() || tp < 1 || dp < 1 || pp != 1 || dpRank < 0 || dpRank >= dp)
                throw new IllegalArgumentException("invalid model metadata; publisher requires pp=1");
        }
    }
    record Config(URI endpoint, String group, String instance, String model,
                  int queueCapacity, int batchSize, int flushMs, int heartbeatMs,
                  int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs, String storageType,
                  List<Group> groups, Discovery discovery, int snapshotTimeoutMs, int maxRetryAfterMs,
                  boolean appendBlockSize, ModelMetadata metadata) {
        Config(URI endpoint, String group, String instance, String model, int queueCapacity, int batchSize,
               int flushMs, int heartbeatMs, int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs, String storageType,
               List<Group> groups, Discovery discovery, int snapshotTimeoutMs, int maxRetryAfterMs, boolean appendBlockSize) {
            this(endpoint, group, instance, model, queueCapacity, batchSize, flushMs, heartbeatMs, snapshotMs, timeoutMs,
                    retryMs, maxRetryMs, storageType, groups, discovery, snapshotTimeoutMs, maxRetryAfterMs,
                    appendBlockSize, new ModelMetadata("fp16", false, 1, 1, 1, 0));
        }
        Config(URI endpoint, String group, String instance, String model, int queueCapacity, int batchSize,
               int flushMs, int heartbeatMs, int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs) {
            this(endpoint, group, instance, model, queueCapacity, batchSize, flushMs, heartbeatMs,
                    snapshotMs, timeoutMs, retryMs, maxRetryMs, "ST_EVENT_REPORT_L1P5");
        }
        Config(URI endpoint, String group, String instance, String model, int queueCapacity, int batchSize,
               int flushMs, int heartbeatMs, int snapshotMs, int timeoutMs, int retryMs, int maxRetryMs, String storageType) {
            this(endpoint, group, instance, model, queueCapacity, batchSize, flushMs, heartbeatMs,
                    snapshotMs, timeoutMs, retryMs, maxRetryMs, storageType,
                    List.of(new Group("full", "FULL", true, 4096)), null, 30000, 300000, false);
        }
        Config {
            if ((endpoint == null && (discovery == null || discovery.address() == null || discovery.address().isBlank()))
                    || (endpoint != null && (!("http".equals(endpoint.getScheme()) || "https".equals(endpoint.getScheme()))
                    || endpoint.getHost() == null || endpoint.getUserInfo() != null || endpoint.getQuery() != null || endpoint.getFragment() != null))
                    || group.isBlank() || instance.isBlank() || model.isBlank()
                    || !("ST_EVENT_REPORT".equals(storageType) || "ST_EVENT_REPORT_L1P5".equals(storageType))
                    || queueCapacity <= 0 || batchSize <= 0 || flushMs <= 0 || heartbeatMs <= 0 || snapshotMs <= 0
                    || timeoutMs <= 0 || retryMs <= 0 || maxRetryMs < retryMs || snapshotTimeoutMs <= 0 || maxRetryAfterMs <= 0)
                throw new IllegalArgumentException("invalid KVCM event configuration");
            groups = groups.stream().filter(Group::published).toList();
            if (groups.isEmpty() || groups.stream().map(Group::tag).distinct().count() != groups.size())
                throw new IllegalArgumentException("KVCM requires distinct reusable FULL/SWA groups");
        }
        static Config fromEnv(Map<String, String> env) {
            String type = env.getOrDefault("KV_CACHE_EVENT_PUBLISHER_TYPE", "none");
            if (type.equals("none")) return null;
            if (!type.equals("kvcm")) throw new IllegalArgumentException("unknown cache event publisher type");
            String endpoint = env.getOrDefault("KV_CACHE_EVENT_MANAGER_ENDPOINT", "");
            if (!endpoint.contains("://") && !endpoint.isBlank()) endpoint = "http://" + endpoint;
            String deployment = env.getOrDefault("BIZ_NAME", "") + "/" + env.getOrDefault("DEPLOYMENT_NAME", "");
            String instance = env.getOrDefault("KV_CACHE_EVENT_INSTANCE_ID", deployment.equals("/") ? "" : deployment);
            if (!env.getOrDefault("KV_CACHE_EVENT_HOST_IP_PORT", "").isBlank())
                throw new IllegalArgumentException("Mock derives a unique host:HTTP-port for every engine; remove KV_CACHE_EVENT_HOST_IP_PORT");
            boolean vip = env.getOrDefault("RECO_ENABLE_VIPSERVER", "0").equals("1");
            Discovery discovery = new Discovery(vip, env.getOrDefault(vip ? "RECO_VIPSERVER_DOMAIN" : "RECO_SERVER_ADDRESS", ""));
            List<Group> groups = new ArrayList<>();
            try {
                JsonNode input = JSON.readTree(env.getOrDefault("MOCK_KVCM_GROUPS_JSON",
                        "[{\"tag\":\"full\",\"type\":\"FULL\",\"size_bytes\":4096}]"));
                if (!input.isArray()) throw new IllegalArgumentException("groups must be an array");
                for (JsonNode item : input) groups.add(new Group(item.path("tag").asText(), item.path("type").asText(),
                        item.path("reuse").asBoolean(true), item.path("size_bytes").asLong()));
            } catch (java.io.IOException error) { throw new IllegalArgumentException("invalid MOCK_KVCM_GROUPS_JSON", error); }
            return new Config(endpoint.isBlank() ? null : URI.create(endpoint.replaceAll("/+$", "")),
                    env.getOrDefault("KV_CACHE_EVENT_INSTANCE_GROUP", env.getOrDefault("RECO_INSTANCE_GROUP", "")),
                    instance, env.getOrDefault("MODEL_TYPE", "mock"),
                    integer(env, "QUEUE_CAPACITY", 100000), integer(env, "BATCH_SIZE", 1000),
                    integer(env, "FLUSH_MS", 20), integer(env, "HEARTBEAT_MS", 1000),
                    integer(env, "SNAPSHOT_MS", 300000), integer(env, "TIMEOUT_MS", 1500),
                    integer(env, "RETRY_MS", 500), integer(env, "MAX_RETRY_MS", 16000),
                    env.getOrDefault("MOCK_KVCM_STORAGE_TYPE", "ST_EVENT_REPORT_L1P5"), groups, discovery,
                    integer(env, "SNAPSHOT_TIMEOUT_MS", 30000), integer(env, "MAX_RETRY_AFTER_MS", 300000),
                    Boolean.parseBoolean(env.getOrDefault("MOCK_KVCM_APPEND_BLOCK_SIZE", "false")),
                    new ModelMetadata(env.getOrDefault("MOCK_KVCM_DTYPE", "fp16"),
                            Boolean.parseBoolean(env.getOrDefault("MOCK_KVCM_USE_MLA", "false")),
                            integer(env, "TP_SIZE", 1), integer(env, "DP_SIZE", 1), integer(env, "PP_SIZE", 1), integer(env, "DP_RANK", 0)));
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
    private Thread worker, heartbeatWorker;
    private final java.util.concurrent.locks.ReentrantReadWriteLock lifecycle = new java.util.concurrent.locks.ReentrantReadWriteLock();
    private final AtomicLong snapshotEpoch = new AtomicLong(), requestedSnapshotEpoch = new AtomicLong();
    private final AtomicLong retryDeadline = new AtomicLong();
    private final ThreadLocal<Feedback> feedback = new ThreadLocal<>();
    private KvcmEndpointResolver endpointResolver;
    private volatile boolean registered;
    private boolean hostDownPending;
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
        this.instanceId = config.instance() + (config.appendBlockSize() ? "_" + blockSize : "");
        cache.configurePublicationGroups(config.groups().size());
        this.blockSize = blockSize;
        this.available = available;
        this.transport = transport == null ? httpTransport(config) : transport;
        queue = new MutationQueue(config.queueCapacity());
    }

    private Transport httpTransport(Config config) {
        endpointResolver = new KvcmEndpointResolver(config, instanceId);
        HttpClient client = HTTP_CLIENTS.computeIfAbsent(config.timeoutMs(), timeout -> HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeout)).version(HttpClient.Version.HTTP_1_1).build());
        return (route, body) -> {
            URI endpoint = endpointResolver.endpoint();
            boolean snapshot = body.contains("EVENT_BLOCK_SNAPSHOT");
            HttpRequest request = HttpRequest.newBuilder(URI.create(endpoint.toString() + route))
                    .timeout(Duration.ofMillis(snapshot ? config.snapshotTimeoutMs() : config.timeoutMs()))
                    .header("Content-Type", "application/json").header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body)).build();
            try {
                var response = client.send(request, HttpResponse.BodyHandlers.ofString());
                Feedback result = parseFeedback(response.body());
                feedback.set(result);
                boolean ok = response.statusCode() >= 200 && response.statusCode() < 300 && result.valid() && result.ok();
                if (!ok) endpointResolver.invalidate(endpoint);
                return ok;
            } catch (Exception error) { endpointResolver.invalidate(endpoint); throw error; }
        };
    }

    record Feedback(boolean valid, boolean ok, boolean registrationRequired, boolean snapshotRequired, long retryAfterMs) {}
    static Feedback parseFeedback(String response) {
        try {
            JsonNode root = JSON.readTree(response);
            JsonNode status = root.path("header").path("status").path("code");
            if (status.isMissingNode() || !(status.isNumber() || status.isTextual() || status.isObject()))
                return new Feedback(false, false, false, false, 0);
            boolean ok = codeOk(status), registration = registrationCode(status), snapshot = matches(status, 14, "SNAPSHOT_REQUIRED");
            JsonNode hint = root.has("snapshot_required") ? root.get("snapshot_required") : root.get("snapshotRequired");
            if (hint != null) { if (!hint.isBoolean()) return new Feedback(false, false, false, false, 0); snapshot |= hint.asBoolean(); }
            long delay = 0;
            JsonNode retry = root.has("retry_after_ms") ? root.get("retry_after_ms") : root.get("retryAfterMs");
            if (retry != null) {
                if (!(retry.isIntegralNumber() || retry.isTextual())) return new Feedback(false, false, false, false, 0);
                var parsed = new java.math.BigInteger(retry.asText());
                if (parsed.signum() < 0 || parsed.bitLength() > 64) return new Feedback(false, false, false, false, 0);
                if (!retry.asText().matches("[0-9]+")) return new Feedback(false, false, false, false, 0);
                delay = parsed.min(java.math.BigInteger.valueOf(Long.MAX_VALUE)).longValue();
            }
            JsonNode items = root.has("item_results") ? root.get("item_results") : root.get("itemResults");
            if (items != null) {
                if (!items.isArray()) return new Feedback(false, false, false, false, 0);
                for (JsonNode item : items) { ok &= codeOk(item); registration |= registrationCode(item); snapshot |= matches(item, 14, "SNAPSHOT_REQUIRED"); }
            }
            return new Feedback(true, ok, registration, snapshot, delay);
        } catch (Exception error) { return new Feedback(false, false, false, false, 0); }
    }
    private static boolean matches(JsonNode node, int code, String name) {
        if (node.isObject()) return matches(node.has("code") ? node.get("code") : node.path("status"), code, name);
        return node.isIntegralNumber() ? node.asLong() == code : node.isTextual() && (node.asText().equals(name) || node.asText().equals(Integer.toString(code)));
    }
    private static boolean registrationCode(JsonNode node) {
        return matches(node, 8, "INSTANCE_NOT_EXIST") || matches(node, 9, "SERVER_NOT_LEADER") || matches(node, 10, "NODE_NOT_REGISTERED");
    }

    static boolean responseOk(String response) { var f = parseFeedback(response); return f.valid() && f.ok(); }

    private static boolean codeOk(JsonNode node) {
        if (node.isObject()) return codeOk(node.has("code") ? node.get("code") : node.path("status"));
        return node.isIntegralNumber() ? node.asLong() == 1
                : node.isTextual() && (node.asText().equals("OK") || node.asText().equals("1"));
    }

    void start() {
        cache.setMutationListener(this::enqueue);
        worker = Thread.ofVirtual().name("mock-kvcm-" + host).start(this::run);
        heartbeatWorker = Thread.ofVirtual().name("mock-kvcm-heartbeat-" + host).start(this::heartbeatLoop);
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
                .put("block_size", blockSize).put("default_query_type", "QT_PREFIX_MATCH");
        var metadata = config.metadata();
        root.putObject("model_deployment").put("model_name", config.model()).put("dtype", metadata.dtype())
                .put("use_mla", metadata.useMla()).put("tp_size", metadata.tp()).put("dp_size", metadata.dp()).put("pp_size", metadata.pp());
        ArrayNode specs = root.putArray("location_spec_infos");
        ArrayNode names = root.putArray("location_spec_groups").addObject().put("name", "default").putArray("spec_names");
        for (Group group : config.groups()) {
            specs.addObject().put("name", specName(group)).put("size", group.sizeBytes());
            names.add(specName(group));
        }
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
    private String specName(Group group) { return "rtp_llm_hbm_" + blockSize + "_" + group.tag(); }
    private void spec(ArrayNode specs, int index) {
        Group group = config.groups().get(index);
        specs.addObject().put("name", specName(group)).put("uri", "event_report://" + host + "/hbm?size=" + group.sizeBytes());
    }
    private ObjectNode control(String type, String payload) {
        ObjectNode root = report();
        ObjectNode value = event(root, type, payload);
        if (type.equals("EVENT_NODE_REGISTER")) value.putArray("mediums").add("hbm");
        if (type.equals("EVENT_HEARTBEAT")) value.putObject("system_status").put("engine", "rtp-llm-mock").put("dp_rank", Integer.toString(config.metadata().dpRank()));
        return root;
    }

    private boolean post(String route, String body) throws Exception {
        long epoch = snapshotEpoch.get();
        feedback.remove();
        boolean ok;
        try { ok = transport.post(route, body); }
        catch (Exception error) { registered = false; retryDeadline.set(0); throw error; }
        Feedback result = feedback.get();
        if (result != null) {
            if (!result.valid() || result.registrationRequired() || (!ok && result.ok())) {
                registered = false;
                retryDeadline.set(0);
            }
            if (result.snapshotRequired()) {
                if (ok) requestedSnapshotEpoch.accumulateAndGet(epoch + 1, Math::max);
                else dirty.incrementAndGet();
            }
            if (result.retryAfterMs() > 0 && !body.contains("EVENT_HEARTBEAT"))
                retryDeadline.accumulateAndGet(System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(
                        Math.min(result.retryAfterMs(), config.maxRetryAfterMs())), Math::max);
        } else if (!ok) { registered = false; retryDeadline.set(0); }
        if (ok) reports.incrementAndGet();
        return ok;
    }
    private void heartbeatLoop() {
        try {
            while (!closed) {
                Thread.sleep(config.heartbeatMs());
                lifecycle.readLock().lock();
                try {
                    if (!closed && registered && available.getAsBoolean() && !control("EVENT_HEARTBEAT")) {
                        failures.incrementAndGet(); dirty.incrementAndGet(); state = "DEGRADED";
                    }
                } catch (Exception error) {
                    if (!closed) { failures.incrementAndGet(); dirty.incrementAndGet(); state = "DEGRADED"; }
                }
                finally { lifecycle.readLock().unlock(); }
            }
        } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
    }
    boolean setGroupResident(long key, String tag, boolean resident) {
        for (int i = 0; i < config.groups().size(); i++)
            if (config.groups().get(i).tag().equals(tag)) return cache.setGroupResident(key, i, resident);
        throw new IllegalArgumentException("unknown or excluded KVCM group");
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
        long snapshotAt = 0;
        try {
            while (!closed) {
                try {
                    if (!available.getAsBoolean()) {
                        lifecycle.writeLock().lock();
                        try {
                            registered = false;
                            if (hostDownPending) {
                                if (!control("EVENT_HOST_DOWN")) throw new IllegalStateException("host down rejected");
                                hostDownPending = false;
                            }
                        } finally { lifecycle.writeLock().unlock(); }
                        pending = null;
                        state = "STOPPED";
                        Thread.sleep(config.flushMs());
                        continue;
                    }
                    long pause = retryDeadline.get() - System.nanoTime();
                    if (pause > 0) { Thread.sleep(Math.min(100, Math.max(1, TimeUnit.NANOSECONDS.toMillis(pause)))); continue; }
                    if (!registered) {
                        // New registration needs current authoritative state, never replay an old owner's pending snapshot.
                        pending = null;
                        if (!post("/api/registerInstance", registration().toString()))
                            throw new IllegalStateException("registration rejected");
                        // An uncertain node-register acknowledgement may still
                        // have created this owner. Keep cleanup pending until
                        // HOST_DOWN is acknowledged, including during close.
                        hostDownPending = true;
                        if (!control("EVENT_NODE_REGISTER")) throw new IllegalStateException("node registration rejected");
                        registered = true;
                        dirty.incrementAndGet();
                    }
                    long now = System.nanoTime();
                    if (now >= snapshotAt) {
                        dirty.incrementAndGet();
                        snapshotAt = now + Duration.ofMillis(config.snapshotMs()).toNanos();
                    }
                    if (pending != null || reconciled != dirty.get() || requestedSnapshotEpoch.get() > snapshotEpoch.get()) {
                        state = "RESYNCING";
                        if (pending == null) {
                            long generation = dirty.get();
                            var snapshot = cache.componentSnapshot();
                            ObjectNode root = report();
                            ArrayNode blocks = event(root, "EVENT_BLOCK_SNAPSHOT", "block_snapshot").putArray("blocks");
                            for (var entry : snapshot.groups().entrySet()) {
                                ObjectNode block = blocks.addObject().put("block_key", entry.getKey().toString()).put("medium", "hbm");
                                ArrayNode specs = block.putArray("specs");
                                for (int group : entry.getValue()) spec(specs, group);
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
                        snapshotEpoch.incrementAndGet();
                        if (reconciled != dirty.get()) Thread.sleep(config.retryMs());
                    } else {
                        var batch = new ArrayList<MockLruBlockCache.Mutation>();
                        queue.drainTo(batch, config.batchSize());
                        // KVCM processes ADD before DELETE in a batch: retain
                        // only the last transition for each (key, group) (real parity).
                        var last = new LinkedHashMap<String, MockLruBlockCache.Mutation>();
                        for (var mutation : batch) if (mutation.version() > cutoff) last.put(mutation.key() + ":" + mutation.groupIndex(), mutation);
                        if (!last.isEmpty()) {
                            ObjectNode root = report();
                            for (var mutation : last.values()) {
                                ObjectNode value = event(root, mutation.added() ? "EVENT_BLOCK_ADD" : "EVENT_BLOCK_DELETE",
                                        mutation.added() ? "block_add" : "block_delete");
                                value.put("block_key", mutation.key().toString()).put("medium", "hbm");
                                if (mutation.added()) spec(value.putArray("specs"), mutation.groupIndex());
                                else value.putArray("spec_names").add(specName(config.groups().get(mutation.groupIndex())));
                            }
                            if (!post("/api/reportEvent", root.toString())) throw new IllegalStateException("mutation rejected");
                        }
                        state = "READY";
                        Thread.sleep(config.flushMs());
                    }
                    backoff = config.retryMs();
                } catch (InterruptedException error) { break;
                } catch (Exception error) {
                    failures.incrementAndGet();
                    state = "DEGRADED";
                    dirty.incrementAndGet();
                    Thread.sleep(backoff);
                    backoff = (int) Math.min((long) config.maxRetryMs(), (long) backoff * 2);
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } finally {
            closed = true;
            if (heartbeatWorker != null) {
                heartbeatWorker.interrupt();
                try { heartbeatWorker.join(config.timeoutMs() + 1000L); }
                catch (InterruptedException ignored) { Thread.interrupted(); }
            }
            Thread.interrupted();
            lifecycle.writeLock().lock();
            try { if (hostDownPending) control("EVENT_HOST_DOWN"); } catch (Exception ignored) { }
            finally { lifecycle.writeLock().unlock(); }
            if (endpointResolver != null) endpointResolver.close();
            state = "CLOSED";
        }
    }

    @Override public void close() {
        closed = true;
        cache.setMutationListener(null);
        if (worker != null) {
            worker.interrupt();
            try { worker.join(Math.max(config.timeoutMs(), config.snapshotTimeoutMs()) + 2000L); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        }
    }
}
