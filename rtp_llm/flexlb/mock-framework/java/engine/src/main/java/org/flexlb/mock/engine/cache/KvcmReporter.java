package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.LongSupplier;

/**
 * KVCM Meta reporting only. Cache locks enqueue immutable changes; snapshots and heartbeats never block mutation reports.
 */
public final class KvcmReporter implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private static final int QUEUE_CAPACITY = 100000, BATCH_SIZE = 1000;

    private final Configuration config;
    private final Identity identity;
    private final KvCacheManager cache;
    private final LongSupplier ackClock;

    private long normalAckDeadline;
    private boolean heartbeatSnapshotHint;

    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(1500)).build();
    private final ScheduledExecutorService worker = Executors.newSingleThreadScheduledExecutor(
        Thread.ofVirtual().name("kvcm-metadata").factory());
    // Registration and mutation reports have one owner; snapshots are best effort.
    private final ScheduledExecutorService heartbeats = Executors.newSingleThreadScheduledExecutor(
        Thread.ofVirtual().name("kvcm-heartbeat").factory());
    private final ScheduledExecutorService snapshots = Executors.newSingleThreadScheduledExecutor(
        Thread.ofVirtual().name("kvcm-snapshot").factory());
    private final Object registrationLock = new Object();
    private final Object snapshotSendLock = new Object();

    private volatile long generation;
    private long retryAt;
    private int failures;

    private final AtomicBoolean registrationRequired = new AtomicBoolean();
    private final ArrayBlockingQueue<CacheMetadata.Change> changes = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final AtomicBoolean snapshotRequired = new AtomicBoolean(true);
    private final AutoCloseable observation;

    private long lastSnapshot;
    private volatile boolean registered, closed, remoteEnabled;

    private volatile String failure = "", heartbeatFailure = "", snapshotFailure = "";

    public KvcmReporter(Configuration config, Identity identity, KvCacheManager cache) {
        this(config, identity, cache, System::nanoTime);
    }

    KvcmReporter(Configuration config, Identity identity, KvCacheManager cache, LongSupplier ackClock) {
        this.ackClock = Objects.requireNonNull(ackClock);
        this.config = Objects.requireNonNull(config); this.identity = Objects.requireNonNull(identity);
        this.cache = Objects.requireNonNull(cache);
        observation = config.events() ? cache.observeMetadata(change -> {
            if (!changes.offer(change)) snapshotRequired.set(true);
        }) : () -> {};
        worker.scheduleWithFixedDelay(this::tick, 0, 20, TimeUnit.MILLISECONDS);
        snapshots.scheduleWithFixedDelay(this::sendSnapshot, 0, 20, TimeUnit.MILLISECONDS);
        heartbeats.scheduleWithFixedDelay(this::heartbeat, 0, 1, TimeUnit.SECONDS);
    }

    public Map<String, Object> status() {
        return Map.of("payload_io", false, "registered", registered && !registrationRequired.get(), "remote_enabled", remoteEnabled,
            "last_error", !failure.isEmpty() ? failure : !heartbeatFailure.isEmpty() ? heartbeatFailure : snapshotFailure, "pending_events", changes.size());
    }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        try { observation.close(); } catch (Exception error) { throw new IllegalStateException(error); }
        // Complete an in-flight registration before sending HOST_DOWN; interrupt only on timeout.
        heartbeats.shutdownNow();
        snapshots.shutdownNow();
        worker.shutdown();
        try {
            if (!worker.awaitTermination(3, TimeUnit.SECONDS)) {
                worker.shutdownNow();
                if (!worker.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KVCM reporter did not stop");
            }
            if (!heartbeats.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KVCM heartbeat worker did not stop");
            if (!snapshots.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KVCM snapshot worker did not stop");
            URI endpoint = config.endpoint().current();
            if (registered && endpoint != null) try { accepted(post(endpoint, "/api/reportEvent", control("HOST_DOWN")), false); }
            catch (Exception error) { failure = error.toString(); }
        } catch (InterruptedException error) {
            worker.shutdownNow(); Thread.currentThread().interrupt();
        } finally { try { config.endpoint().close(); } finally { http.shutdownNow(); } }
    }

    private void heartbeat() {
        long current = generation;
        if (closed || !registered || registrationRequired.get()) return;
        try {
            feedback(post("/api/reportEvent", control("HEARTBEAT")), true, true, current);
            heartbeatFailure = "";
        } catch (Exception error) {
            heartbeatFailure = error.toString();
            recoverAfterMissingAck(current);
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        }
    }

    private void feedback(JsonNode response, boolean heartbeat, boolean normalAck, long expectedGeneration) {
        synchronized (registrationLock) {
            if (generation != expectedGeneration || !registered || registrationRequired.get()) return;
            boolean hint = accepted(response, heartbeat);
            if (normalAck) {
                normalAckDeadline = ackClock.getAsLong() + TimeUnit.SECONDS.toNanos(60);
                if (config.events() && hint && (!heartbeat || !heartbeatSnapshotHint)) snapshotRequired.set(true);
            }
            if (heartbeat) heartbeatSnapshotHint = hint;
        }
    }

    private boolean accepted(JsonNode response, boolean heartbeat) {
        var status = response.path("header").path("status");
        if (!status.isObject() || !status.has("code")) throw new IllegalStateException("Invalid KVCM status");
        var items = response.has("item_results") ? response.path("item_results") : response.path("itemResults");
        if (!items.isMissingNode() && !items.isArray()) throw new IllegalStateException("Invalid KVCM item results");
        var hint = response.has("snapshot_required") ? response.path("snapshot_required") : response.path("snapshotRequired");
        if (!hint.isMissingNode() && !hint.isBoolean()) throw new IllegalStateException("Invalid KVCM snapshot hint");
        var codes = new ArrayList<String>();
        codes.add(code(status));
        for (var item : items) codes.add(code(item));
        boolean ok = true, registration = false, snapshot = hint.asBoolean();
        for (String value : codes) {
            ok &= Set.of("1", "OK").contains(value);
            registration |= Set.of("8", "9", "10", "INSTANCE_NOT_EXIST", "SERVER_NOT_LEADER", "NODE_NOT_REGISTERED").contains(value);
            snapshot |= Set.of("14", "SNAPSHOT_REQUIRED").contains(value);
        }
        // Consume all recovery signals before reporting any individual failure.
        if (registration) { registrationRequired.set(true); normalAckDeadline = 0; }
        if (!ok) throw new IllegalStateException("KVCM rejected metadata: " + codes);
        if (heartbeat) {
            var extra = response.has("extra_info") ? response.path("extra_info") : response.path("extraInfo");
            if (extra.isTextual()) try {
                var body = JSON.readTree(extra.textValue());
                var flag = body == null ? null : body.get("rtp_remote_cache_enabled");
                if (flag != null && flag.isBoolean()) remoteEnabled = flag.booleanValue();
            } catch (java.io.IOException ignored) { /* Advisory control cannot invalidate an event ACK. */ }
        }
        return snapshot;
    }

    private static String code(JsonNode value) {
        if (value.isObject()) return code(value.has("code") ? value.path("code") : value.path("status"));
        return value.asText();
    }

    private JsonNode post(String path, ObjectNode request) throws Exception {
        return post(config.endpoint().get(), path, request);
    }

    private JsonNode post(URI target, String path, ObjectNode request) throws Exception {
        request.put("trace_id", UUID.randomUUID().toString());
        return post(target, path, JSON.writeValueAsString(request), 1500);
    }

    private JsonNode post(URI target, String path, String request, int timeoutMillis) throws Exception {
        try {
            var response = http.send(HttpRequest.newBuilder(URI.create(target.toString().replaceAll("/$", "") + path))
                .timeout(Duration.ofMillis(timeoutMillis)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(request)).build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200)
                throw new IllegalStateException("Invalid KVCM HTTP response: " + response.statusCode());
            var body = JSON.readTree(response.body());
            if (notLeader(body.path("header").path("status"))) config.endpoint().invalidate(target);
            var items = body.has("item_results") ? body.path("item_results") : body.path("itemResults");
            for (var item : items) if (notLeader(item)) config.endpoint().invalidate(target);
            return body;
        } catch (Exception error) {
            config.endpoint().invalidate(target);
            throw error;
        }
    }

    private static boolean notLeader(JsonNode value) {
        return Set.of("9", "SERVER_NOT_LEADER").contains(code(value));
    }

    private ObjectNode control(String kind) {
        var root = report(); var body = ((ArrayNode) root.path("events")).addObject().put("event_type", "EVENT_" + kind)
            .putObject(kind.toLowerCase(Locale.ROOT));
        if (kind.equals("NODE_REGISTER")) { var media = body.putArray("mediums"); if (config.events()) media.add("hbm"); }
        if (kind.equals("HEARTBEAT")) body.putObject("system_status").put("engine", "rtp-llm").put("dp_rank", "0");
        return root;
    }

    private ObjectNode report() {
        var root = JSON.createObjectNode().put("instance_id", identity.instance()).put("host_ip_port", identity.address())
            .put("storage_type", "ST_EVENT_REPORT_L1P5");
        root.putArray("events"); return root;
    }

    private void recoverAfterMissingAck(long expectedGeneration) {
        synchronized (registrationLock) {
            if (generation == expectedGeneration && normalAckDeadline != 0
                    && ackClock.getAsLong() - normalAckDeadline >= 0) {
                normalAckDeadline = 0;
                registrationRequired.set(true);
            }
        }
    }

    private void sendSnapshot() {
        long current = generation;
        if (closed || !config.events() || !registered || registrationRequired.get()) return;
        long now = System.nanoTime();
        if (!snapshotRequired.getAndSet(false)
                && now - lastSnapshot < TimeUnit.MILLISECONDS.toNanos(config.snapshotMillis())) return;
        lastSnapshot = now;
        try {
            String body = snapshot();
            synchronized (snapshotSendLock) {
                if (closed || current != generation || !registered || registrationRequired.get()) return;
                feedback(post(config.endpoint().get(), "/api/reportEvent", body, 30000), false, false, current);
            }
            snapshotFailure = "";
        } catch (Exception error) {
            // Native Publisher discards this round, including rate limits. Never replay it or hold up deltas.
            snapshotFailure = error.toString();
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        }
    }

    private String snapshot() throws java.io.IOException {
        var snapshot = cache.metadataKeys();
        var text = new java.io.StringWriter();
        try (var out = JSON.getFactory().createGenerator(text)) {
            out.writeStartObject();
            out.writeStringField("trace_id", UUID.randomUUID().toString());
            out.writeStringField("instance_id", identity.instance());
            out.writeStringField("host_ip_port", identity.address());
            out.writeStringField("storage_type", "ST_EVENT_REPORT_L1P5");
            out.writeArrayFieldStart("events"); out.writeStartObject();
            out.writeStringField("event_type", "EVENT_BLOCK_SNAPSHOT");
            out.writeObjectFieldStart("block_snapshot");
            out.writeNumberField("key_count", snapshot.keys().length);
            out.writeArrayFieldStart("blocks");
            for (long key : snapshot.keys()) {
                out.writeStartObject(); out.writeStringField("block_key", Long.toString(key));
                out.writeStringField("medium", "hbm");
                out.writeArrayFieldStart("specs"); out.writeStartObject();
                out.writeStringField("name", identity.spec());
                out.writeStringField("uri", "event_report://" + identity.address() + "/hbm");
                out.writeEndObject(); out.writeEndArray(); out.writeEndObject();
            }
            out.writeEndArray(); out.writeEndObject(); out.writeEndObject();
            out.writeEndArray(); out.writeEndObject();
        }
        return text.toString();
    }

    private void tick() {
        if (closed) return;
        synchronized (registrationLock) {
            if (registrationRequired.getAndSet(false)) {
                registered = false;
                generation++;
                normalAckDeadline = 0;
                heartbeatSnapshotHint = false;
            }
        }
        if (System.nanoTime() < retryAt) return;
        try {
            if (!registered) {
                // Match native Publisher: drain an old upload before re-registering. Deltas never take this lock.
                synchronized (snapshotSendLock) {
                    if (closed) return;
                    accepted(post("/api/registerInstance", registration()), false);
                    accepted(post("/api/reportEvent", control("NODE_REGISTER")), true);
                    synchronized (registrationLock) {
                        changes.clear();
                        snapshotRequired.set(true);
                        registered = true;
                    }
                }
            }
            if (config.events() && !registrationRequired.get()) {
                var report = mutations();
                if (!report.path("events").isEmpty())
                    feedback(post("/api/reportEvent", report), false, true, generation);
            }
            failures = 0; failure = "";
        } catch (Exception error) {
            snapshotRequired.set(true);
            failure = error.toString();
            retryAt = System.nanoTime() + retryDelay(failures++);
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        }
    }

    private ObjectNode registration() {
        var root = JSON.createObjectNode().put("instance_group", identity.group()).put("instance_id", identity.instance())
            .put("block_size", identity.blockSize()).put("default_query_type", "QT_PREFIX_MATCH");
        root.putObject("model_deployment").put("model_name", identity.model()).put("dtype", identity.dtype())
            .put("use_mla", identity.mla()).put("tp_size", identity.cpSize()).put("dp_size", 1).put("pp_size", 1);
        root.putArray("location_spec_infos").addObject().put("name", identity.spec());
        root.putArray("location_spec_groups").addObject().put("name", "default").putArray("spec_names").add(identity.spec());
        return root;
    }

    private ObjectNode mutations() {
        // Native Meta groups ADD before DELETE. Send only the final transition per key in a batch.
        var batch = new LinkedHashMap<Long, CacheMetadata.Change>();
        for (int i = 0; i < BATCH_SIZE; i++) {
            var change = changes.poll(); if (change == null) break;
            batch.put(change.key(), change);
        }
        var root = report(); var events = (ArrayNode) root.path("events");
        for (var change : batch.values()) {
            boolean add = change.added();
            var event = events.addObject().put("event_type", add ? "EVENT_BLOCK_ADD" : "EVENT_BLOCK_DELETE");
            var body = event.putObject(add ? "block_add" : "block_delete")
                .put("block_key", Long.toString(change.key())).put("medium", "hbm");
            if (add) spec(body); else body.putArray("spec_names").add(identity.spec());
        }
        return root;
    }

    private void spec(ObjectNode block) {
        block.putArray("specs").addObject().put("name", identity.spec())
            .put("uri", "event_report://" + identity.address() + "/hbm");
    }

    private static long retryDelay(int failures) {
        return TimeUnit.MILLISECONDS.toNanos(500L << Math.min(failures, 5));
    }

    public record Configuration(KvcmDiscovery endpoint, boolean events, long snapshotMillis) {
        public Configuration {
            Objects.requireNonNull(endpoint);
            if (events && snapshotMillis < 1) throw new IllegalArgumentException("Invalid KVCM snapshot interval");
        }
        public Configuration(URI endpoint, boolean events, long snapshotMillis) {
            this(KvcmDiscovery.direct(endpoint), events, snapshotMillis);
        }
        public static URI validateEndpoint(URI endpoint) {
            if (endpoint == null || !("http".equals(endpoint.getScheme()) || "https".equals(endpoint.getScheme()))
                    || endpoint.getHost() == null || endpoint.getRawQuery() != null
                    || endpoint.getRawFragment() != null || endpoint.getUserInfo() != null)
                throw new IllegalArgumentException("Invalid KVCM Meta HTTP endpoint");
            return endpoint;
        }
    }

    /**
     * Caller describes the capability it actually implements; no model/layout inference in the reporter.
     */
    public record Identity(String instance, String group, String address, String model, String dtype,
                           boolean mla, int blockSize, int cpSize, String spec) {
        public Identity {
            for (String value : List.of(instance, group, address, model, dtype, spec))
                if (value.isBlank()) throw new IllegalArgumentException("Empty KVCM identity");
            if (blockSize < 1 || cpSize < 1) throw new IllegalArgumentException("Invalid KVCM layout");
        }
    }
}
