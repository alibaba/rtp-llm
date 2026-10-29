package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.runtime.ApiServer;
import org.flexlb.mock.client.ClientProtocol.*;

/** Each JVM executes locally; only client-0 coordinates the one current plan. */
public final class ClientNode implements ClientProtocol, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final String id;
    private final Path topology, directory;
    private final ClientMetrics metrics = new ClientMetrics();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private final ApiServer api;
    private LoadClient worker;
    private long workerSequence, sequence, nextRequestId = 1;
    private Group group;
    private boolean collected = true;
    private volatile boolean closed;

    public ClientNode(String id, Path topology, Path directory) throws IOException {
        this.id = id; this.topology = topology; this.directory = directory;
        api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).bind(ClientProtocol.class, this).metrics(metrics::scrape).start();
    }
    public String url() { return api.url(); }
    private static void save(Path path, Object value) throws IOException {
        Files.createDirectories(path.getParent());
        Path temp = Files.createTempFile(path.getParent(), ".client-", ".json");
        try {
            JSON.writerWithDefaultPrettyPrinter().writeValue(temp.toFile(), value);
            Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temp); }
    }
    private JsonNode call(String url, String operation, Object args) throws Exception {
        var body = Map.of("interface", operation, "args", args, "context", Map.of(
            "run", "foreground", "query_time", Instant.now().toString(), "timeout_ms", 2000, "observation", Map.of()));
        var request = HttpRequest.newBuilder(URI.create(url + "/invoke")).timeout(Duration.ofSeconds(3))
            .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofByteArray(JSON.writeValueAsBytes(body))).build();
        var response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) throw new IOException(operation + " at " + url + ": HTTP " + response.statusCode());
        var result = JSON.readTree(response.body());
        if (!result.isObject() || result.size() != 1 || !result.has("result")) throw new IOException("Invalid Client response");
        return result.get("result");
    }
    @Override public synchronized Ack prepare(Shard args) throws Exception {
        if (closed || args.sequence() <= workerSequence || worker != null && !worker.state().done())
            throw new IllegalStateException("Previous Client playback has not ended or stale prepare");
        save(directory.resolve("playback/" + args.sequence() + "/plan.json"), args);
        if (worker != null) worker.close();
        worker = new LoadClient(args, metrics, topology); workerSequence = args.sequence();
        return new Ack(true);
    }
    private synchronized LoadClient worker(long expected) {
        if (worker == null || expected != workerSequence) throw new IllegalStateException("Stale/unknown worker command");
        return worker;
    }
    @Override public Ack begin(Begin args) { worker(args.sequence()).begin(args.start_epoch_ms()); return new Ack(true); }
    @Override public Ack halt(Sequence args) { worker(args.sequence()).halt(); return new Ack(true); }
    @Override public State state(Sequence args) {
        var result = worker(args.sequence()).state();
        if (result.done()) try { save(directory.resolve("playback/" + args.sequence() + "/result.json"), result); }
        catch (IOException error) { throw new UncheckedIOException(error); }
        return result;
    }

    @Override public Ack start(Start args) throws Exception {
        Group created;
        synchronized (this) {
            if (!id.equals("client-0") || closed) throw new IllegalStateException("Start must target live client-0");
            if (group != null && (!group.finished.isDone() || !collected))
                throw new IllegalStateException("Call stop to collect the previous playback before starting again");
            var nodes = JSON.readTree(topology.toFile());
            var peers = new ArrayList<String>();
            for (var node : nodes) if (node.path("role").asText().equals("client")) peers.add(node.path("control").asText());
            if (peers.isEmpty() || new HashSet<>(peers).size() != peers.size() || !peers.contains(url())
                    || args.concurrency() < peers.size())
                throw new IllegalArgumentException("Need unique Client endpoints and at least one request/concurrency slot per Client");
            long firstId = nextRequestId;
            created = new Group(++sequence, List.copyOf(peers), args, firstId);
            group = created; collected = false;
        }
        try {
            save(directory.resolve("coordinator/" + created.number + "/plan.json"), Map.of("clients", created.peers, "config", args));
            var parts = RequestFiles.split(args, directory.resolve("coordinator/" + created.number + "/shards"), created.peers.size(), () -> created.stop || closed);
            long total = parts.stream().mapToLong(RequestFiles.Part::count).reduce(0, Math::addExact);
            synchronized (this) { nextRequestId = Math.addExact(created.firstId, total); }
            save(directory.resolve("coordinator/" + created.number + "/input.json"), Map.of("count", total, "shards", parts));
            for (int i = 0; i < created.peers.size(); i++) {
                RequestFiles.checkInterrupted();
                if (created.stop || closed) throw new IOException("Client start stopped during preparation");
                var part = parts.get(i);
                int slots = args.concurrency() / created.peers.size() + (i < args.concurrency() % created.peers.size() ? 1 : 0);
                call(created.peers.get(i), "worker/prepare", new Shard(created.number, args.masters(), part.path(),
                    part.count(), i, created.peers.size(), created.firstId, part.sha256(), args.qps(), slots,
                    args.queue_capacity(), args.request_timeout_ms(), args.drain_timeout_ms()));
            }
            long startAt = System.currentTimeMillis() + Math.max(1000, created.peers.size() * 100L);
            for (String peer : created.peers) {
                RequestFiles.checkInterrupted();
                if (created.stop || closed) throw new IOException("Client start stopped before begin");
                call(peer, "worker/begin", new Begin(created.number, startAt));
            }
            created.thread = Thread.ofVirtual().name("client-coordinator").start(created::supervise);
            return new Ack(true);
        } catch (Exception error) {
            created.error("Start failed: " + error);
            created.stop = true;
            created.thread = Thread.ofVirtual().name("client-coordinator").start(created::supervise);
            throw error;
        }
    }
    @Override public Summary stop(Empty args) throws Exception {
        Group current;
        synchronized (this) {
            if (!id.equals("client-0")) throw new IllegalStateException("Stop must target client-0");
            current = group;
        }
        if (current == null) return new Summary(Map.of(), List.of());
        current.stop = true;
        var result = current.finished.get(current.config.drain_timeout_ms() + 15000 + current.peers.size() * 6000L, TimeUnit.MILLISECONDS);
        synchronized (this) { collected = true; }
        return result;
    }
    private final class Group {
        final long number, firstId;
        final List<String> peers;
        final Start config;
        final CompletableFuture<Summary> finished = new CompletableFuture<>();
        final List<String> errors = new ArrayList<>();
        volatile boolean stop;
        Thread thread;
        Group(long number, List<String> peers, Start config, long firstId) {
            this.number = number; this.peers = peers; this.config = config; this.firstId = firstId;
        }
        void error(String error) { if (errors.isEmpty()) errors.add(error); }
        void supervise() {
            var states = new LinkedHashMap<String, State>();
            var unavailable = new HashSet<String>();
            boolean halted = false;
            long stoppingDeadline = Long.MAX_VALUE;
            try {
                while (true) {
                    if ((stop || !errors.isEmpty()) && !halted) {
                        halted = true;
                        stoppingDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(config.drain_timeout_ms() + 10000);
                        // Stop all peers concurrently; one unreachable peer must not serialize the rest.
                        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
                            var calls = new ArrayList<Future<?>>();
                            for (String peer : peers) calls.add(pool.submit(() -> {
                                try { call(peer, "worker/halt", new Sequence(number)); }
                                catch (Exception failure) { throw new CompletionException(failure); }
                            }));
                            for (var pending : calls) try { pending.get(); }
                            catch (ExecutionException failure) { error("Stop propagation failed: " + failure.getCause()); }
                        }
                    }
                    for (String peer : peers) {
                        if (unavailable.contains(peer) || states.containsKey(peer) && states.get(peer).done()) continue;
                        try {
                            var state = JSON.treeToValue(call(peer, "worker/state", new Sequence(number)), State.class);
                            states.put(peer, state);
                            if (!state.errors().isEmpty()) error(peer + ": " + state.errors().getFirst());
                        } catch (Exception failure) { error(peer + ": " + failure); unavailable.add(peer); }
                    }
                    if (states.size() == peers.size() && states.values().stream().allMatch(State::done) && unavailable.isEmpty()) break;
                    if (halted && (System.nanoTime() >= stoppingDeadline
                            || peers.stream().allMatch(p -> unavailable.contains(p) || states.containsKey(p) && states.get(p).done()))) break;
                    Thread.sleep(100);
                }
                if (!unavailable.isEmpty() || states.size() != peers.size() || states.values().stream().anyMatch(s -> !s.done()))
                    error("Client results incomplete");
            } catch (Exception failure) { error("Coordinator failed: " + failure); }
            var result = new Summary(Map.copyOf(states), List.copyOf(errors));
            try { save(directory.resolve("coordinator/" + number + "/result.json"), result); }
            catch (IOException failure) {
                error("Cannot save Client result: " + failure);
                result = new Summary(Map.copyOf(states), List.copyOf(errors));
            }
            finished.complete(result);
        }
    }
    @Override public void close() {
        LoadClient owned;
        synchronized (this) { if (closed) return; closed = true; owned = worker; if (group != null) group.stop = true; }
        if (owned != null) owned.close();
        api.close(); http.shutdownNow();
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use runner environment");
        var config = JSON.readTree(Path.of(required("MOCK_CONFIG_FILE")).toFile());
        if (!config.path("component").path("config").isEmpty()) throw new IllegalArgumentException("Client configuration belongs to client/start");
        var directory = Path.of(required("MOCK_OUTPUT_DIR"));
        try (var node = new ClientNode(required("MOCK_NODE_ID"), Path.of(required("MOCK_TOPOLOGY_FILE")), directory)) {
            var hook = new Thread(node::close, "client-shutdown"); Runtime.getRuntime().addShutdownHook(hook);
            save(directory.resolve("effective-config.json"), Map.of("id", node.id));
            save(directory.resolve("endpoints.json"), Map.of("id", node.id, "role", "client", "control", node.url(),
                "rpc", URI.create(node.url()).getAuthority(), "metrics", node.url() + "/metrics"));
            node.api.publishReady(Path.of(required("MOCK_READY_FILE")));
            try { new CountDownLatch(1).await(); }
            finally { try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException ignored) { } }
        }
    }
    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + name);
        return value;
    }
}
