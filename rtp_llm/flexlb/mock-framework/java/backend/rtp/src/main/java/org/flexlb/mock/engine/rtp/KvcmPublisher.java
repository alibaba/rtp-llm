package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.cache.KvCache;

/** Native KVCM Meta HTTP control/events. Never advertises a payload storage backend. */
final class KvcmPublisher implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final RtpConfig config;
    private final KvCache cache;
    private final URI endpoint;
    private final String instance, group, host, spec;
    private final boolean events;
    private final long snapshotMillis;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(500)).build();
    private final ScheduledExecutorService worker = Executors.newSingleThreadScheduledExecutor(Thread.ofVirtual().name("kvcm-control").factory());
    private Set<Long> published = Set.of();
    private boolean registered, snapshotRequired;
    private long lastSnapshot;
    private volatile boolean closed, remoteEnabled;
    private volatile String failure = "";

    static KvcmPublisher start(RtpConfig config, KvCache cache, String address) {
        var input = config.business();
        String type = input.path("kv_cache_event_publisher_type").asText("none");
        if (!Set.of("none", "kvcm").contains(type)) throw new IllegalArgumentException("Invalid kv_cache_event_publisher_type");
        boolean control = RtpConfig.nativeBool(input,"enable_remote_cache",false);
        if (type.equals("none") && (!control || input.path("kv_cache_event_manager_endpoint").asText().isBlank())) return null;
        var publisher = new KvcmPublisher(config, cache, address, type.equals("kvcm"));
        publisher.worker.scheduleWithFixedDelay(publisher::tick, 0, 1, TimeUnit.SECONDS);
        return publisher;
    }

    private KvcmPublisher(RtpConfig config, KvCache cache, String address, boolean events) {
        this.config = config; this.cache = cache; this.events = events;
        var input = config.business();
        endpoint = URI.create(input.path("kv_cache_event_manager_endpoint").asText());
        if (!Set.of("http", "https").contains(endpoint.getScheme()) || endpoint.getHost() == null
                || endpoint.getRawQuery() != null || endpoint.getRawFragment() != null || endpoint.getUserInfo() != null)
            throw new IllegalArgumentException("KVCM events require a direct Meta HTTP endpoint");
        instance = input.path("kv_cache_event_instance_id").asText(config.id() + "-" + UUID.randomUUID());
        group = input.path("kv_cache_event_instance_group").asText("default");
        host = input.path("kv_cache_event_host_ip_port").asText(address);
        spec = "rtp_llm_hbm_" + config.blockSize() + "_full";
        snapshotMillis = RtpConfig.nativeInt(input,"kv_cache_event_snapshot_interval_ms",600000,1,Integer.MAX_VALUE);
        if (instance.isBlank() || group.isBlank() || host.isBlank()) throw new IllegalArgumentException("Missing KVCM event identity");
    }

    private ObjectNode registration() {
        var root = JSON.createObjectNode().put("instance_group",group).put("instance_id",instance)
            .put("block_size",config.blockSize()).put("default_query_type","QT_PREFIX_MATCH");
        root.putObject("model_deployment").put("model_name",config.business().path("model_name").asText("mock-full-kv"))
            .put("dtype","FP16").put("use_mla",false).put("tp_size",config.tpSize()).put("dp_size",config.dpSize()).put("pp_size",1);
        root.putArray("location_spec_infos").addObject().put("name",spec);
        root.putArray("location_spec_groups").addObject().put("name","default").putArray("spec_names").add(spec);
        return root;
    }
    private ObjectNode report() {
        var root = JSON.createObjectNode().put("instance_id",instance).put("host_ip_port",host).put("storage_type","ST_EVENT_REPORT_L1P5");
        root.putArray("events"); return root;
    }
    private ObjectNode control(String kind) {
        var root = report(); var event = ((ArrayNode)root.path("events")).addObject().put("event_type","EVENT_" + kind);
        var body = event.putObject(kind.toLowerCase(Locale.ROOT));
        if (kind.equals("NODE_REGISTER")) { var mediums = body.putArray("mediums"); if (events) mediums.add("hbm"); }
        if (kind.equals("HEARTBEAT")) body.putObject("system_status").put("engine","rtp-llm").put("dp_rank","0");
        return root;
    }
    private ObjectNode snapshot(Set<Long> keys) {
        var root = report(); var blocks = ((ArrayNode)root.path("events")).addObject().put("event_type","EVENT_BLOCK_SNAPSHOT")
            .putObject("block_snapshot").putArray("blocks");
        for (long key : new TreeSet<>(keys)) {
            var block = blocks.addObject().put("block_key",Long.toString(key)).put("medium","hbm");
            block.putArray("specs").addObject().put("name",spec).put("uri","event_report://" + host + "/hbm");
        }
        return root;
    }
    private ObjectNode mutations(Set<Long> keys) {
        var root = report(); var changes = (ArrayNode) root.path("events");
        for (long key : new TreeSet<>(published)) if (!keys.contains(key)) {
            changes.addObject().put("event_type","EVENT_BLOCK_DELETE").putObject("block_delete")
                .put("block_key",Long.toString(key)).put("medium","hbm").putArray("spec_names").add(spec);
        }
        for (long key : new TreeSet<>(keys)) if (!published.contains(key)) {
            changes.addObject().put("event_type","EVENT_BLOCK_ADD").putObject("block_add")
                .put("block_key",Long.toString(key)).put("medium","hbm").putArray("specs")
                .addObject().put("name",spec).put("uri","event_report://" + host + "/hbm");
        }
        return root;
    }
    private JsonNode post(String path, ObjectNode request) throws Exception {
        request.put("trace_id",UUID.randomUUID().toString());
        var response = http.send(HttpRequest.newBuilder(URI.create(endpoint.toString().replaceAll("/$","") + path))
            .timeout(Duration.ofSeconds(1)).header("Content-Type","application/json")
            .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(request))).build(),HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200 || response.body().length() > 1048576) throw new IllegalStateException("KVCM HTTP response invalid");
        return JSON.readTree(response.body());
    }
    private static String code(JsonNode value) {
        if (value.isObject()) return code(value.has("code") ? value.path("code") : value.path("status"));
        return value.asText();
    }
    private void accepted(JsonNode response, boolean heartbeat) {
        String status = code(response.path("header").path("status"));
        if (Set.of("8","9","10","INSTANCE_NOT_EXIST","SERVER_NOT_LEADER","NODE_NOT_REGISTERED").contains(status)) {
            registered = false; throw new IllegalStateException("KVCM registration expired: " + status);
        }
        if (Set.of("14","SNAPSHOT_REQUIRED").contains(status)) {
            snapshotRequired = true; throw new IllegalStateException("KVCM requires a new snapshot");
        }
        else if (!Set.of("1","OK").contains(status)) throw new IllegalStateException("KVCM rejected report: " + status);
        var items = response.has("item_results") ? response.path("item_results") : response.path("itemResults");
        if (!items.isMissingNode() && !items.isArray()) throw new IllegalStateException("Invalid KVCM item results");
        for (var item : items) if (!Set.of("1","OK").contains(code(item))) {
            if (Set.of("8","9","10","INSTANCE_NOT_EXIST","SERVER_NOT_LEADER","NODE_NOT_REGISTERED").contains(code(item))) registered = false;
            snapshotRequired = true; throw new IllegalStateException("KVCM rejected event item");
        }
        snapshotRequired |= response.path("snapshot_required").asBoolean() || response.path("snapshotRequired").asBoolean();
        if (heartbeat) {
            var extra = response.has("extra_info") ? response.path("extra_info") : response.path("extraInfo");
            if (extra.isTextual()) try {
                var body = JSON.readTree(extra.textValue());
                var flag = body == null ? null : body.get("rtp_remote_cache_enabled");
                if (flag != null && flag.isBoolean()) remoteEnabled = flag.booleanValue();
            } catch (java.io.IOException ignored) { /* Native extra_info is advisory. */ }
        }
    }
    private synchronized void tick() {
        if (closed) return;
        try {
            if (!registered) {
                accepted(post("/api/registerInstance",registration()),false);
                accepted(post("/api/reportEvent",control("NODE_REGISTER")),true);
                registered = true; snapshotRequired = events;
            }
            if (events) {
                var keys = cache.snapshot(-1,true).keys();
                if (snapshotRequired || System.nanoTime() - lastSnapshot >= TimeUnit.MILLISECONDS.toNanos(snapshotMillis)) {
                    snapshotRequired = false;
                    accepted(post("/api/reportEvent",snapshot(keys)),false);
                    lastSnapshot = System.nanoTime(); published = Set.copyOf(keys);
                } else if (!keys.equals(published)) {
                    accepted(post("/api/reportEvent",mutations(keys)),false); published = Set.copyOf(keys);
                }
            }
            accepted(post("/api/reportEvent",control("HEARTBEAT")),true); failure = "";
        } catch (Exception error) {
            snapshotRequired = events; failure = error.toString();
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
        }
    }
    Map<String,Object> status() { return Map.of("payload_io",false,"remote_enabled",remoteEnabled,"last_error",failure); }
    @Override public void close() {
        worker.shutdownNow();
        synchronized (this) {
            closed = true;
            if (registered) try { post("/api/reportEvent",control("HOST_DOWN")); } catch (Exception ignored) { }
        }
        http.close();
    }
}
