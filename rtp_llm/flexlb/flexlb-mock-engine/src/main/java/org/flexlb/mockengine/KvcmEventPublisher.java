package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Opt-in metadata-only publisher. Mirrors C++ KVCMPublisher lifecycle; no KV payload transfer. */
final class KvcmEventPublisher implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private record Mutation(long key, boolean add) {}
    private final MockLruBlockCache cache;
    private final URI endpoint;
    private final String group, instance, host, storageType;
    private final int blockSize;
    private final ArrayBlockingQueue<Mutation> queue;
    private final AtomicBoolean dirty = new AtomicBoolean(true);
    private final AtomicLong adds = new AtomicLong(), deletes = new AtomicLong(), heartbeats = new AtomicLong();
    private final AtomicLong sent = new AtomicLong(), failures = new AtomicLong(), snapshots = new AtomicLong();
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private final Thread worker;
    private volatile boolean running = true;
    private volatile String lastError = "";
    private volatile boolean registered;

    static KvcmEventPublisher fromEnvironment(MockLruBlockCache cache, String host, int blockSize) {
        String endpoint = System.getenv("MOCK_KVCM_ENDPOINT");
        if (endpoint == null || endpoint.isBlank()) return null;
        try {
            return new KvcmEventPublisher(cache, endpoint,
                    required("MOCK_KVCM_INSTANCE_GROUP"), required("MOCK_KVCM_INSTANCE_PREFIX") + "-" + host,
                    host, blockSize, required("MOCK_KVCM_STORAGE_TYPE"), 8192);
        } catch (RuntimeException error) {
            System.err.println("KVCM publishing disabled: " + error);
            return null;
        }
    }

    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException(name + " is required");
        return value;
    }

    KvcmEventPublisher(MockLruBlockCache cache, String endpoint, String group, String instance,
                       String host, int blockSize, String storageType, int capacity) {
        this.cache = cache;
        this.endpoint = URI.create(endpoint.replaceAll("/+$", ""));
        if (!List.of("http", "https").contains(this.endpoint.getScheme()))
            throw new IllegalArgumentException("KVCM endpoint requires http(s)");
        if (!List.of("ST_EVENT_REPORT_L1P5", "ST_EVENT_REPORT_L2").contains(storageType))
            throw new IllegalArgumentException("explicit supported event storage type required");
        this.group = group; this.instance = instance; this.host = host;
        this.blockSize = blockSize; this.storageType = storageType;
        this.queue = new ArrayBlockingQueue<>(capacity);
        cache.setMutationListener((key, add) -> {
            if (!queue.offer(new Mutation(key, add))) dirty.set(true);
        }, () -> dirty.set(true));
        worker = new Thread(this::run, "kvcm-events-" + host);
        worker.setDaemon(true);
        worker.start();
    }

    Map<String, Object> status() {
        var value = new LinkedHashMap<String, Object>();
        value.put("adds", adds.get()); value.put("deletes", deletes.get()); value.put("heartbeats", heartbeats.get());
        value.putAll(Map.of("instance_id", instance, "host_ip_port", host, "registered", registered,
                "sent_batches", sent.get(), "snapshots", snapshots.get(), "failures", failures.get(),
                "queue_size", queue.size(), "last_error", lastError));
        return value;
    }

    private Map<String, Object> spec() {
        return Map.of("name", "mock_hbm", "uri", "rtp-llm://" + host + "/hbm");
    }

    private Map<String, Object> block(long key) {
        return Map.of("block_key", Long.toString(key), "medium", "hbm", "specs", List.of(spec()));
    }

    private Map<String, Object> event(String type, String field, Object body) {
        return Map.of("event_type", type, field, body);
    }

    private JsonNode post(String route, Object body) throws Exception {
        var request = HttpRequest.newBuilder(endpoint.resolve("/api/" + route))
                .timeout(Duration.ofSeconds(3)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(body))).build();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        JsonNode json = JSON.readTree(response.body());
        if (response.statusCode() != 200 || !ok(json.path("header").path("status")))
            throw new IllegalStateException("KVCM " + route + ": " + response.statusCode() + " " + json);
        JsonNode items = json.has("item_results") ? json.path("item_results") : json.path("itemResults");
        for (JsonNode item : items) if (!ok(item)) throw new IllegalStateException("KVCM item failed: " + item);
        return json;
    }

    static boolean ok(JsonNode node) {
        if (node.isObject()) return node.has("code") ? ok(node.path("code")) : ok(node.path("status"));
        return "OK".equals(node.asText()) || "1".equals(node.asText());
    }

    private JsonNode report(List<Map<String, Object>> events) throws Exception {
        JsonNode response = post("reportEvent", Map.of("trace_id", "mock-" + java.util.UUID.randomUUID(),
                "instance_id", instance, "host_ip_port", host, "events", events, "storage_type", storageType));
        sent.incrementAndGet();
        for (var event : events) {
            switch (event.get("event_type").toString()) {
                case "EVENT_BLOCK_ADD" -> adds.incrementAndGet();
                case "EVENT_BLOCK_DELETE" -> deletes.incrementAndGet();
                case "EVENT_HEARTBEAT" -> heartbeats.incrementAndGet();
                default -> { }
            }
        }
        if (response.path("snapshot_required").asBoolean()) dirty.set(true);
        return response;
    }

    private void register() throws Exception {
        post("registerInstance", Map.of("trace_id", "mock-register", "instance_group", group,
                "instance_id", instance, "block_size", blockSize,
                "model_deployment", Map.of("model_name", "mock-model", "dtype", "BF16", "use_mla", false,
                        "tp_size", 1, "dp_size", 1, "pp_size", 1),
                "location_spec_infos", List.of(Map.of("name", "mock_hbm", "size", 4096)),
                "location_spec_groups", List.of(Map.of("name", "default", "spec_names", List.of("mock_hbm")))));
        report(List.of(event("EVENT_NODE_REGISTER", "node_register", Map.of("mediums", List.of("hbm")))));
        registered = true;
        dirty.set(true);
    }

    private void snapshot() throws Exception {
        List<Map<String, Object>> blocks;
        // Capture a cut of the state and discard only events already represented by that cut.
        // Mutations after this lock is released stay queued while the HTTP request is in flight.
        synchronized (cache) {
            blocks = cache.snapshotKeys().stream().map(this::block).toList();
            queue.clear();
            dirty.set(false);
        }
        JsonNode response = report(List.of(event("EVENT_BLOCK_SNAPSHOT", "block_snapshot", Map.of("blocks", blocks))));
        if (response.path("snapshot_required").asBoolean()) throw new IllegalStateException("snapshot not committed");
        snapshots.incrementAndGet();
    }

    private void run() {
        long heartbeat = 0, reconcile = 0, backoff = 100;
        while (running) {
            try {
                if (!registered) register();
                long now = System.nanoTime();
                if (dirty.get() || now >= reconcile) {
                    snapshot();
                    reconcile = now + Duration.ofSeconds(30).toNanos();
                } else {
                    var batch = new ArrayList<Mutation>();
                    queue.drainTo(batch, 512);
                    // KVCM groups ADD before DELETE: preserve the last transition per key in each unsent batch.
                    var last = new LinkedHashMap<Long, Boolean>();
                    for (Mutation m : batch) last.put(m.key(), m.add());
                    var events = new ArrayList<Map<String, Object>>();
                    last.forEach((key, add) -> events.add(add
                            ? event("EVENT_BLOCK_ADD", "block_add", block(key))
                            : event("EVENT_BLOCK_DELETE", "block_delete", Map.of("block_key", Long.toString(key),
                                    "medium", "hbm", "spec_names", List.of("mock_hbm")))));
                    if (!events.isEmpty()) report(events);
                }
                if (now >= heartbeat) {
                    report(List.of(event("EVENT_HEARTBEAT", "heartbeat", Map.of("system_status", Map.of("engine", "rtp-llm", "dp_rank", "0")))));
                    heartbeat = now + Duration.ofSeconds(2).toNanos();
                }
                lastError = "";
                backoff = 100;
                Thread.sleep(25);
            } catch (InterruptedException e) {
                if (running) { dirty.set(true); registered = false; }
            } catch (Exception e) {
                failures.incrementAndGet(); lastError = e.toString();
                dirty.set(true); registered = false;
                try { Thread.sleep(backoff); } catch (InterruptedException ignored) { }
                backoff = Math.min(5000, backoff * 2);
            }
        }
    }

    @Override public void close() {
        running = false;
        worker.interrupt();
        try {
            worker.join(4000);
            if (worker.isAlive()) throw new IllegalStateException("KVCM publisher did not stop");
            if (registered) report(List.of(event("EVENT_HOST_DOWN", "host_down", Map.of())));
        } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
          catch (Exception e) { lastError = e.toString(); }
        synchronized (cache) { cache.setMutationListener((key, add) -> {}, () -> {}); }
        client.close();
    }
}
