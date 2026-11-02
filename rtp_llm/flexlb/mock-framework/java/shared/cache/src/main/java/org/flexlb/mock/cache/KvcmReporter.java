package org.flexlb.mock.cache;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import static org.flexlb.mock.cache.CacheMetadata.Medium.*;

/** KVCM Meta reporting only. Cache locks enqueue immutable changes; all HTTP runs here. */
public final class KvcmReporter implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int QUEUE_CAPACITY = 100000, BATCH_SIZE = 1000;
    public record Configuration(Supplier<URI> endpoint, boolean events, long snapshotMillis) {
        public Configuration {
            Objects.requireNonNull(endpoint);
            if (snapshotMillis < 1) throw new IllegalArgumentException("Invalid KVCM snapshot interval");
        }
        public Configuration(URI endpoint, boolean events, long snapshotMillis) {
            this(fixed(endpoint), events, snapshotMillis);
        }
        private static Supplier<URI> fixed(URI endpoint) {
            validateEndpoint(endpoint);
            return () -> endpoint;
        }
        public static URI validateEndpoint(URI endpoint) {
            if (endpoint == null || !("http".equals(endpoint.getScheme()) || "https".equals(endpoint.getScheme()))
                    || endpoint.getHost() == null || endpoint.getRawQuery() != null
                    || endpoint.getRawFragment() != null || endpoint.getUserInfo() != null)
                throw new IllegalArgumentException("Invalid KVCM Meta HTTP endpoint");
            return endpoint;
        }
    }
    /** Caller describes the capability it actually implements; no model/layout inference in the reporter. */
    public record Identity(String instance, String group, String address, String model, String dtype,
                           boolean mla, int blockSize, int tp, int dp, String spec) {
        public Identity {
            for (String value : List.of(instance, group, address, model, dtype, spec))
                if (value.isBlank()) throw new IllegalArgumentException("Empty KVCM identity");
            if (blockSize < 1 || tp < 1 || dp < 1) throw new IllegalArgumentException("Invalid KVCM layout");
        }
    }
    private final Configuration config;
    private final Identity identity;
    private final KvCache cache;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(500)).build();
    private final ScheduledExecutorService worker = Executors.newSingleThreadScheduledExecutor(
        Thread.ofVirtual().name("kvcm-metadata").factory());
    private final ArrayBlockingQueue<CacheMetadata.Change> changes = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private final AtomicBoolean snapshotRequired = new AtomicBoolean(true);
    private final AutoCloseable observation;
    private long snapshotVersion = -1, lastSnapshot, lastHeartbeat, retryAt;
    private volatile boolean registered, closed, remoteEnabled;
    private volatile String failure = "";

    public KvcmReporter(Configuration config, Identity identity, KvCache cache) {
        this.config = Objects.requireNonNull(config); this.identity = Objects.requireNonNull(identity);
        this.cache = Objects.requireNonNull(cache);
        observation = config.events() ? cache.observeMetadata(change -> {
            if (change.before() == HBM || change.after() == HBM)
                if (!changes.offer(change)) snapshotRequired.set(true);
        }) : () -> {};
        worker.scheduleWithFixedDelay(this::tick, 0, 20, TimeUnit.MILLISECONDS);
    }
    private ObjectNode registration() {
        var root = JSON.createObjectNode().put("instance_group", identity.group()).put("instance_id", identity.instance())
            .put("block_size", identity.blockSize()).put("default_query_type", "QT_PREFIX_MATCH");
        root.putObject("model_deployment").put("model_name", identity.model()).put("dtype", identity.dtype())
            .put("use_mla", identity.mla()).put("tp_size", identity.tp()).put("dp_size", identity.dp()).put("pp_size", 1);
        root.putArray("location_spec_infos").addObject().put("name", identity.spec());
        root.putArray("location_spec_groups").addObject().put("name", "default").putArray("spec_names").add(identity.spec());
        return root;
    }
    private ObjectNode report() {
        var root = JSON.createObjectNode().put("instance_id", identity.instance()).put("host_ip_port", identity.address())
            .put("storage_type", "ST_EVENT_REPORT_L1P5");
        root.putArray("events"); return root;
    }
    private ObjectNode control(String kind) {
        var root = report(); var body = ((ArrayNode) root.path("events")).addObject().put("event_type", "EVENT_" + kind)
            .putObject(kind.toLowerCase(Locale.ROOT));
        if (kind.equals("NODE_REGISTER")) { var media = body.putArray("mediums"); if (config.events()) media.add("hbm"); }
        if (kind.equals("HEARTBEAT")) body.putObject("system_status").put("engine", "rtp-llm").put("dp_rank", "0");
        return root;
    }
    private void spec(ObjectNode block) {
        block.putArray("specs").addObject().put("name", identity.spec())
            .put("uri", "event_report://" + identity.address() + "/hbm");
    }
    private ObjectNode snapshot(CacheMetadata.Snapshot snapshot) {
        var root = report(); var blocks = ((ArrayNode) root.path("events")).addObject().put("event_type", "EVENT_BLOCK_SNAPSHOT")
            .putObject("block_snapshot").put("key_count", snapshot.keys(HBM).size()).putArray("blocks");
        for (long key : new TreeSet<>(snapshot.keys(HBM)))
            spec(blocks.addObject().put("block_key", Long.toString(key)).put("medium", "hbm"));
        return root;
    }
    private ObjectNode mutations() {
        // Native Meta groups ADD before DELETE. Send only the final transition per key in a batch.
        var batch = new LinkedHashMap<Long, CacheMetadata.Change>();
        for (int i = 0; i < BATCH_SIZE; i++) {
            var change = changes.poll(); if (change == null) break;
            if (change.version() > snapshotVersion) batch.put(change.key(), change);
        }
        var root = report(); var events = (ArrayNode) root.path("events");
        for (var change : batch.values()) {
            boolean add = change.after() == HBM;
            var event = events.addObject().put("event_type", add ? "EVENT_BLOCK_ADD" : "EVENT_BLOCK_DELETE");
            var body = event.putObject(add ? "block_add" : "block_delete")
                .put("block_key", Long.toString(change.key())).put("medium", "hbm");
            if (add) spec(body); else body.putArray("spec_names").add(identity.spec());
        }
        return root;
    }
    private JsonNode post(String path, ObjectNode request) throws Exception {
        request.put("trace_id", UUID.randomUUID().toString());
        var target = Configuration.validateEndpoint(config.endpoint().get());
        var response = http.send(HttpRequest.newBuilder(URI.create(target.toString().replaceAll("/$", "") + path))
            .timeout(Duration.ofMillis(1500)).header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(request))).build(), HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200 || response.body().length() > 1048576)
            throw new IllegalStateException("Invalid KVCM HTTP response: " + response.statusCode());
        return JSON.readTree(response.body());
    }
    private static String code(JsonNode value) {
        if (value.isObject()) return code(value.has("code") ? value.path("code") : value.path("status"));
        return value.asText();
    }
    private void checkCode(JsonNode value) {
        String code = code(value);
        if (Set.of("8", "9", "10", "INSTANCE_NOT_EXIST", "SERVER_NOT_LEADER", "NODE_NOT_REGISTERED").contains(code))
            registered = false;
        if (!Set.of("1", "OK").contains(code)) throw new IllegalStateException("KVCM rejected metadata: " + code);
    }
    private void accepted(JsonNode response, boolean heartbeat) {
        checkCode(response.path("header").path("status"));
        var items = response.has("item_results") ? response.path("item_results") : response.path("itemResults");
        if (!items.isMissingNode() && !items.isArray()) throw new IllegalStateException("Invalid KVCM item results");
        for (var item : items) checkCode(item);
        if (response.path("snapshot_required").asBoolean() || response.path("snapshotRequired").asBoolean())
            snapshotRequired.set(true);
        if (heartbeat) {
            var extra = response.has("extra_info") ? response.path("extra_info") : response.path("extraInfo");
            if (extra.isTextual()) try {
                var body = JSON.readTree(extra.textValue());
                var flag = body == null ? null : body.get("rtp_remote_cache_enabled");
                if (flag != null && flag.isBoolean()) remoteEnabled = flag.booleanValue();
            } catch (java.io.IOException ignored) { /* Advisory control cannot invalidate an event ACK. */ }
        }
    }
    private void tick() {
        if (closed || System.nanoTime() < retryAt) return;
        try {
            if (!registered) {
                accepted(post("/api/registerInstance", registration()), false);
                accepted(post("/api/reportEvent", control("NODE_REGISTER")), true);
                registered = true; snapshotRequired.set(true); lastHeartbeat = 0;
            }
            if (System.nanoTime() - lastHeartbeat >= TimeUnit.SECONDS.toNanos(1)) {
                accepted(post("/api/reportEvent", control("HEARTBEAT")), true); lastHeartbeat = System.nanoTime();
            }
            if (config.events()) {
                if (System.nanoTime() - lastSnapshot >= TimeUnit.MILLISECONDS.toNanos(config.snapshotMillis()))
                    snapshotRequired.set(true);
                if (snapshotRequired.getAndSet(false)) {
                    var snapshot = cache.metadata();
                    accepted(post("/api/reportEvent", snapshot(snapshot)), false);
                    snapshotVersion = snapshot.version(); lastSnapshot = System.nanoTime();
                } else {
                    var report = mutations();
                    if (!report.path("events").isEmpty()) accepted(post("/api/reportEvent", report), false);
                }
            }
            failure = "";
        } catch (Exception error) {
            snapshotRequired.set(true); failure = error.toString();
            retryAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(500);
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        }
    }
    public Map<String, Object> status() {
        return Map.of("payload_io", false, "registered", registered, "remote_enabled", remoteEnabled,
            "last_error", failure, "pending_events", changes.size());
    }
    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        try { observation.close(); } catch (Exception error) { throw new IllegalStateException(error); }
        worker.shutdown();
        try {
            if (!worker.awaitTermination(3, TimeUnit.SECONDS)) {
                worker.shutdownNow();
                if (!worker.awaitTermination(3, TimeUnit.SECONDS)) throw new IllegalStateException("KVCM reporter did not stop");
            }
            if (registered) try { accepted(post("/api/reportEvent", control("HOST_DOWN")), false); }
            catch (Exception error) { failure = error.toString(); }
        } catch (InterruptedException error) {
            worker.shutdownNow(); Thread.currentThread().interrupt();
        } finally { http.close(); }
    }
}
