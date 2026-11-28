package org.flexlb.mock.frontend;

import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.protocol.ControlHttpServer;
import org.flexlb.mock.protocol.JsonFiles;
import org.flexlb.mock.frontend.FrontendProtocol.*;

/**
 * Exposes playback endpoints and assembles the local worker. Frontend-0 also owns the current cluster replay.
 */
public final class FrontendNode implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final String id;

    private final int imageToken;
    private final int physicalBlockSize;
    private final int prefillCpSize;
    private final long cacheKeySeed;

    private final Path topology, directory;
    private final FrontendMetrics metrics = new FrontendMetrics();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private final ControlHttpServer api;

    private volatile ReplayWorker worker;

    private long sequence, nextRequestId = 1;

    private ReplayCoordinator coordinator;

    private volatile boolean closed;

    public FrontendNode(String id, Path topology, Path directory) throws IOException {
        this(id, topology, directory, org.flexlb.mock.protocol.MultimodalProtocol.IMAGE_TOKEN, 0, 0L, 1);
    }

    FrontendNode(String id, Path topology, Path directory, int imageToken, int physicalBlockSize, long seed, int prefillCpSize) throws IOException {
        this.imageToken = imageToken; this.physicalBlockSize = physicalBlockSize; this.cacheKeySeed = seed;
        this.prefillCpSize = prefillCpSize;
        this.id = id; this.topology = topology; this.directory = directory;
        api = new ControlHttpServer(new InetSocketAddress("127.0.0.1", 0)).route("frontend/start", Start.class, Ack.class, this::startReplay)
            .route("frontend/stop", Empty.class, Summary.class, this::stopReplay)
            .route("worker/prepare", FrontendReplayPlan.class, Ack.class, this::prepareWorker)
            .route("worker/begin", Begin.class, Ack.class, this::begin)
            .route("worker/halt", Sequence.class, Ack.class, this::halt)
            .route("worker/state", Sequence.class, State.class, this::state)
            .metrics(metrics::scrape).start();
    }

    public synchronized Ack prepareWorker(FrontendReplayPlan args) throws Exception {
        if (closed || worker != null && (args.sequence() <= worker.sequence() || !worker.state().done()))
            throw new IllegalStateException("Previous Frontend playback has not ended or stale prepare");
        var playback = directory.resolve("playback/" + args.sequence());
        Files.createDirectories(playback.getParent()); Files.createDirectory(playback);
        save(playback.resolve("plan.json"), args);
        if (worker != null) worker.close();
        worker = new ReplayWorker(args, metrics, topology, playback.resolve("requests.jsonl"), id, imageToken, physicalBlockSize, cacheKeySeed, prefillCpSize);
        return new Ack(true);
    }

    public Ack begin(Begin args) { requireWorker(args.sequence()).begin(args.start_epoch_ms()); return new Ack(true); }

    public Ack halt(Sequence args) { requireWorker(args.sequence()).halt(); return new Ack(true); }

    public State state(Sequence args) {
        return requireWorker(args.sequence()).state();
    }

    public Ack startReplay(Start args) throws Exception {
        ReplayCoordinator created;
        synchronized (this) {
            if (!id.equals("frontend-0") || closed) throw new IllegalStateException("Start must target live frontend-0");
            if (coordinator != null && !coordinator.canStartNext())
                throw new IllegalStateException("Call stop to collect the previous playback before starting again");
            var nodes = JSON.readTree(topology.toFile());
            var peers = new ArrayList<String>();
            for (var node : nodes) if (node.path("role").asText().equals("frontend")) peers.add(node.path("control").asText());
            if (peers.isEmpty() || new HashSet<>(peers).size() != peers.size() || !peers.contains(url())
                    || args.total_concurrency() < peers.size())
                throw new IllegalArgumentException("Need unique Frontend endpoints and at least one request/concurrency slot per Frontend");
            long firstId = nextRequestId;
            created = new ReplayCoordinator(++sequence, List.copyOf(peers), args, firstId);
            coordinator = created;
        }
        created.start();
        return new Ack(true);
    }

    public String url() { return api.url(); }

    public Summary stopReplay(Empty args) throws Exception {
        ReplayCoordinator current;
        synchronized (this) {
            if (!id.equals("frontend-0")) throw new IllegalStateException("Stop must target frontend-0");
            current = coordinator;
        }
        return current == null ? new Summary(Map.of(), List.of()) : current.stopAndCollect();
    }

    @Override public void close() {
        ReplayWorker owned;
        synchronized (this) { if (closed) return; closed = true; owned = worker; if (coordinator != null) coordinator.requestStop(); }
        if (owned != null) owned.close();
        api.close(); http.shutdownNow();
    }

    void publishReady(Path path) throws IOException { api.publishReady(path); }

    private synchronized void reserveRequestIds(long count) {
        nextRequestId = Math.addExact(nextRequestId, count);
    }

    private static void save(Path path, Object value) throws IOException {
        Files.createDirectories(path.getParent());
        JsonFiles.writeAtomically(path, value, JSON.writerWithDefaultPrettyPrinter(), true);
    }

    private ReplayWorker requireWorker(long expected) {
        var current = worker;
        if (current == null || expected != current.sequence()) throw new IllegalStateException("Stale/unknown worker command");
        return current;
    }

    /**
     * Coordinates one round across Frontend processes and collects their final states.
     */
    private final class ReplayCoordinator {
        private final long number, firstId;

        private final List<String> peers;
        private final Start config;
        private final CompletableFuture<Summary> finished = new CompletableFuture<>();
        private final List<String> errors = new ArrayList<>();

        private volatile boolean stop;
        private volatile boolean collected;

        ReplayCoordinator(long number, List<String> peers, Start config, long firstId) {
            this.number = number; this.peers = peers; this.config = config; this.firstId = firstId;
        }

        void start() throws Exception {
            var args = config;
            try {
                var planPath = directory.resolve("coordinator/" + number + "/plan.json");
                var plan = new LinkedHashMap<String, Object>();
                plan.put("frontends", peers);
                plan.put("config", args);
                save(planPath, plan);
                long total;
                String identity;
                try (var input = new IndexedTrafficReader(Path.of(args.requests_file()))) {
                    if (input.missingOutput > 0 && args.fallback_max_output_tokens() == null)
                        throw new IOException("Dataset requires fallback_max_output_tokens for " + input.missingOutput + " requests");
                    total = input.count;
                    identity = input.identity;
                    if (total < peers.size()) throw new IOException("Need at least one request per Frontend");
                    ReplayWorker.checkPlaybackEnd(total - 1, input.lastArrival, args.qps(), args.replay_speed(), args.warmup());
                }
                reserveRequestIds(total);
                plan.put("count", total);
                plan.put("sha256", identity);
                save(planPath, plan);
                int frontends = peers.size();
                for (int i = 0; i < frontends; i++) {
                    if (stop || closed) throw new IOException("Frontend start stopped during preparation");
                    long count = FrontendReplayPlan.requestCount(total, i, frontends);
                    int slots = args.total_concurrency() / frontends + (i < args.total_concurrency() % frontends ? 1 : 0);
                    call(peers.get(i), "worker/prepare", new FrontendReplayPlan(number, args.masters(), args.requests_file(),
                        count, i, frontends, firstId, identity, args.qps(), args.replay_speed(), slots,
                        args.queue_capacity(), args.request_timeout_ms(), args.drain_timeout_ms(), args.reuse_cache(),
                        args.fallback_max_output_tokens(), args.warmup(), args.duplicate_every()), 10_000);
                }
                long startAt = System.currentTimeMillis() + Math.max(1000, peers.size() * 100L);
                for (String peer : peers) {
                    if (stop || closed) throw new IOException("Frontend start stopped before begin");
                    call(peer, "worker/begin", new Begin(number, startAt));
                }
                Thread.ofVirtual().name("frontend-coordinator").start(this::supervise);
            } catch (Exception failure) {
                error("Start failed: " + failure);
                requestStop();
                Thread.ofVirtual().name("frontend-coordinator").start(this::supervise);
                throw failure;
            }
        }

        void requestStop() { stop = true; }

        boolean canStartNext() { return finished.isDone() && collected; }

        Summary stopAndCollect() throws Exception {
            requestStop();
            var result = finished.get(config.drain_timeout_ms() + 15000 + peers.size() * 6000L, TimeUnit.MILLISECONDS);
            collected = true;
            return result;
        }

        private void supervise() {
            var states = new LinkedHashMap<String, State>();
            var unavailable = new HashSet<String>();
            var stateTimeouts = new HashMap<String, Integer>();
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
                            var state = JSON.treeToValue(call(peer, "worker/state", new Sequence(number), 5_000), State.class);
                            states.put(peer, state);
                            stateTimeouts.remove(peer);
                            if (!state.errors().isEmpty()) error(peer + ": " + state.errors().getFirst());
                        } catch (Exception failure) {
                            if (failure instanceof HttpTimeoutException && stateTimeouts.merge(peer, 1, Integer::sum) < 3) continue;
                            error(peer + ": " + failure);
                            unavailable.add(peer);
                        }
                    }
                    if (states.size() == peers.size() && states.values().stream().allMatch(State::done) && unavailable.isEmpty()) break;
                    if (halted && (System.nanoTime() >= stoppingDeadline
                            || peers.stream().allMatch(p -> unavailable.contains(p) || states.containsKey(p) && states.get(p).done()))) break;
                    Thread.sleep(100);
                }
                if (!unavailable.isEmpty() || states.size() != peers.size() || states.values().stream().anyMatch(s -> !s.done()))
                    error("Frontend results incomplete");
            } catch (Exception failure) { error("Coordinator failed: " + failure); }
            var result = new Summary(Map.copyOf(states), List.copyOf(errors));
            try { save(directory.resolve("coordinator/" + number + "/result.json"), result); }
            catch (IOException failure) {
                error("Cannot save Frontend result: " + failure);
                result = new Summary(Map.copyOf(states), List.copyOf(errors));
            }
            finished.complete(result);
        }

        private JsonNode call(String url, String operation, Object args) throws Exception {
            return call(url, operation, args, 2_000);
        }

        private JsonNode call(String url, String operation, Object args, int timeoutMillis) throws Exception {
            var body = Map.of("interface", operation, "args", args, "timeout_ms", timeoutMillis);
            var request = HttpRequest.newBuilder(URI.create(url + "/invoke")).timeout(Duration.ofMillis(timeoutMillis + 1_000L))
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofByteArray(JSON.writeValueAsBytes(body))).build();
            var response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 504 && operation.equals("worker/state"))
                throw new HttpTimeoutException(operation + " at " + url + ": HTTP 504");
            if (response.statusCode() != 200) throw new IOException(operation + " at " + url + ": HTTP " + response.statusCode());
            var result = JSON.readTree(response.body());
            if (!result.isObject() || result.size() != 1 || !result.has("result")) throw new IOException("Invalid Frontend response");
            return result.get("result");
        }

        private void error(String error) { if (errors.isEmpty()) errors.add(error); }
    }
}
