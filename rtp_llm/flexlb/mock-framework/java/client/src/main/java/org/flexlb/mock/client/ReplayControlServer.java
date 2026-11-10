package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import org.flexlb.mock.protocol.ControlServer;
import org.flexlb.mock.client.ClientProtocol.*;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;

/**
 * Each JVM executes locally; only client-0 coordinates the one current plan.
 */
public final class ReplayControlServer implements ClientProtocol, AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();

    private final String id;

    private final int imageToken;
    private final int blockSize;
    private final int prefillCpSize;
    private final long cacheKeySeed;

    private final Path topology, directory;
    private final ClientMetrics metrics = new ClientMetrics();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
    private final ControlServer api;

    private Session session;

    private long sequence, nextRequestId = 1;

    private Group group;

    private boolean collected = true;
    private volatile boolean closed;

    public ReplayControlServer(String id, Path topology, Path directory) throws IOException {
        this(id, topology, directory, org.flexlb.mock.protocol.MediaInput.IMAGE_TOKEN, 0, 0L, 1);
    }

    private ReplayControlServer(String id, Path topology, Path directory, int imageToken, int blockSize, long seed, int prefillCpSize) throws IOException {
        this.imageToken = imageToken; this.blockSize = blockSize; this.cacheKeySeed = seed;
        this.prefillCpSize = prefillCpSize;
        this.id = id; this.topology = topology; this.directory = directory;
        api = new ControlServer(new InetSocketAddress("127.0.0.1", 0)).route("client/start", Start.class, Ack.class, this::start)
            .route("client/stop", Empty.class, Summary.class, this::stop)
            .route("worker/prepare", ClientReplayPlan.class, Ack.class, this::prepare)
            .route("worker/begin", Begin.class, Ack.class, this::begin)
            .route("worker/halt", Sequence.class, Ack.class, this::halt)
            .route("worker/state", Sequence.class, State.class, this::state)
            .metrics(metrics::scrape).start();
    }

    @Override public synchronized Ack prepare(ClientReplayPlan args) throws Exception {
        if (closed || session != null && (args.sequence() <= session.plan.sequence() || !session.state().done()))
            throw new IllegalStateException("Previous Client playback has not ended or stale prepare");
        var playback = directory.resolve("playback/" + args.sequence());
        Files.createDirectories(playback.getParent()); Files.createDirectory(playback);
        save(playback.resolve("plan.json"), args);
        if (session != null) session.close();
        session = new Session(args, metrics, topology, playback.resolve("requests.jsonl"), id, imageToken, blockSize, cacheKeySeed, prefillCpSize);
        return new Ack(true);
    }

    @Override public Ack begin(Begin args) { session(args.sequence()).begin(args.start_epoch_ms()); return new Ack(true); }

    @Override public Ack halt(Sequence args) { session(args.sequence()).halt(); return new Ack(true); }

    @Override public State state(Sequence args) {
        var result = session(args.sequence()).state();
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
                    || args.total_concurrency() < peers.size())
                throw new IllegalArgumentException("Need unique Client endpoints and at least one request/concurrency slot per Client");
            long firstId = nextRequestId;
            created = new Group(++sequence, List.copyOf(peers), args, firstId);
            group = created; collected = false;
        }
        try {
            save(directory.resolve("coordinator/" + created.number + "/plan.json"), Map.of("clients", created.peers, "config", args));
            long total;
            String identity;
            try (var input = new IndexedTrafficReader(Path.of(args.requests_file()))) {
                if (input.missingOutput > 0 && args.fallback_max_output_tokens() == null)
                    throw new IOException("Dataset requires fallback_max_output_tokens for " + input.missingOutput + " requests");
                total = input.count;
                identity = input.identity;
                if (total < created.peers.size()) throw new IOException("Need at least one request per Client");
                Session.delayNanos(total - 1, input.lastArrival, args.qps(), args.replay_speed());
            }
            synchronized (this) { nextRequestId = Math.addExact(created.firstId, total); }
            save(directory.resolve("coordinator/" + created.number + "/input.json"),
                Map.of("count", total, "path", args.requests_file(), "sha256", identity));
            int clients = created.peers.size();
            for (int i = 0; i < clients; i++) {
                if (created.stop || closed) throw new IOException("Client start stopped during preparation");
                long count = ClientProtocol.requestCount(total, i, clients);
                int slots = args.total_concurrency() / clients + (i < args.total_concurrency() % clients ? 1 : 0);
                call(created.peers.get(i), "worker/prepare", new ClientReplayPlan(created.number, args.masters(), args.requests_file(),
                    count, i, clients, created.firstId, identity, args.qps(), args.replay_speed(), slots,
                    args.queue_capacity(), args.request_timeout_ms(), args.drain_timeout_ms(), args.reuse_cache(),
                    args.fallback_max_output_tokens()));
            }
            long startAt = System.currentTimeMillis() + Math.max(1000, created.peers.size() * 100L);
            for (String peer : created.peers) {
                if (created.stop || closed) throw new IOException("Client start stopped before begin");
                call(peer, "worker/begin", new Begin(created.number, startAt));
            }
            Thread.ofVirtual().name("client-coordinator").start(created::supervise);
            return new Ack(true);
        } catch (Exception error) {
            created.error("Start failed: " + error);
            created.stop = true;
            Thread.ofVirtual().name("client-coordinator").start(created::supervise);
            throw error;
        }
    }

    public String url() { return api.url(); }

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

    @Override public void close() {
        Session owned;
        synchronized (this) { if (closed) return; closed = true; owned = session; if (group != null) group.stop = true; }
        if (owned != null) owned.close();
        api.close(); http.shutdownNow();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use runner environment");
        var config = JSON.readTree(Path.of(required("MOCK_CONFIG_FILE")).toFile());
        var settings = config.path("component").path("config");
        settings.fieldNames().forEachRemaining(name -> {
            if (!java.util.Set.of("image_token_id", "cache_key_hash_seed", "seq_size_per_block", "prefill_cp_size").contains(name)) throw new IllegalArgumentException("Unknown Client setting: " + name);
        });
        var configuredImageToken = settings.get("image_token_id");
        if (configuredImageToken != null && (!configuredImageToken.isIntegralNumber()
                || !configuredImageToken.canConvertToInt() || configuredImageToken.intValue() < 0))
            throw new IllegalArgumentException("Client image_token_id must be nonnegative int32");
        int imageToken = settings.path("image_token_id").asInt(org.flexlb.mock.protocol.MediaInput.IMAGE_TOKEN);
        int blockSize = settings.path("seq_size_per_block").asInt(0);
        if (settings.has("seq_size_per_block") && (!settings.path("seq_size_per_block").isIntegralNumber() || !settings.path("seq_size_per_block").canConvertToInt() || blockSize < 1))
            throw new IllegalArgumentException("Client seq_size_per_block must be positive int32");
        var configuredCp = settings.get("prefill_cp_size");
        if (configuredCp != null && (!configuredCp.isIntegralNumber() || !configuredCp.canConvertToInt() || configuredCp.intValue() < 0))
            throw new IllegalArgumentException("Client prefill_cp_size must be nonnegative int32");
        int prefillCpSize = Math.max(1, settings.path("prefill_cp_size").asInt(1));
        if (prefillCpSize > 1 && blockSize == 0)
            throw new IllegalArgumentException("Client prefill_cp_size requires seq_size_per_block");
        int routeBlockSize = Math.multiplyExact(blockSize, prefillCpSize);
        long seed = org.flexlb.mock.protocol.ConfigInput.signedLong(settings, "cache_key_hash_seed", 0);
        var directory = Path.of(required("MOCK_OUTPUT_DIR"));
        try (var node = new ReplayControlServer(required("MOCK_NODE_ID"), Path.of(required("MOCK_TOPOLOGY_FILE")), directory, imageToken, blockSize, seed, prefillCpSize)) {
            var hook = new Thread(node::close, "client-shutdown"); Runtime.getRuntime().addShutdownHook(hook);
            var effective = JSON.createObjectNode().put("id", node.id).put("seq_size_per_block", blockSize).put("cache_key_hash_seed", seed)
                .put("prefill_cp_size", prefillCpSize).put("cache_key_block_size", routeBlockSize);
            if (configuredImageToken != null) effective.put("image_token_id", imageToken);
            save(directory.resolve("effective-config.json"), effective);
            save(directory.resolve("endpoints.json"), Map.of("id", node.id, "role", "client", "control", node.url(),
                "rpc", URI.create(node.url()).getAuthority(), "metrics", node.url() + "/metrics"));
            node.api.publishReady(Path.of(required("MOCK_READY_FILE")));
            try { new CountDownLatch(1).await(); }
            finally { try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException ignored) { } }
        }
    }

    private static void save(Path path, Object value) throws IOException {
        Files.createDirectories(path.getParent());
        Path temp = Files.createTempFile(path.getParent(), ".client-", ".json");
        try {
            JSON.writerWithDefaultPrettyPrinter().writeValue(temp.toFile(), value);
            Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temp); }
    }

    private JsonNode call(String url, String operation, Object args) throws Exception {
        var body = Map.of("interface", operation, "args", args, "timeout_ms", 2000);
        var request = HttpRequest.newBuilder(URI.create(url + "/invoke")).timeout(Duration.ofSeconds(3))
            .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofByteArray(JSON.writeValueAsBytes(body))).build();
        var response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) throw new IOException(operation + " at " + url + ": HTTP " + response.statusCode());
        var result = JSON.readTree(response.body());
        if (!result.isObject() || result.size() != 1 || !result.has("result")) throw new IOException("Invalid Client response");
        return result.get("result");
    }

    private synchronized Session session(long expected) {
        if (session == null || expected != session.plan.sequence()) throw new IllegalStateException("Stale/unknown worker command");
        return session;
    }

    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + name);
        return value;
    }

    /**
     * Coordinates one round across Client processes and collects their final states.
     */
    private final class Group {
        final long number, firstId;

        final List<String> peers;
        final Start config;
        final CompletableFuture<Summary> finished = new CompletableFuture<>();
        final List<String> errors = new ArrayList<>();

        volatile boolean stop;

        Group(long number, List<String> peers, Start config, long firstId) {
            this.number = number; this.peers = peers; this.config = config; this.firstId = firstId;
        }

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

        void error(String error) { if (errors.isEmpty()) errors.add(error); }
    }

    /**
     * Owns one round of local playback, from pacing through drain.
     */
    static final class Session implements AutoCloseable {
        private final ClientReplayPlan plan;
        private final BackendRpcClient client;
        private final RequestJournal journal;
        private final MasterDiscovery discovery;
        private final ClientMetrics metrics;
        private final ArrayBlockingQueue<Item> queue;
        private final Map<Long, BackendRpcClient.Call> active = new HashMap<>();
        private final Map<String, Long> outcomes = new TreeMap<>();
        private final List<String> errors = new ArrayList<>();
        private final CompletableFuture<State> completion = new CompletableFuture<>();

        private final int imageToken;
        private final int blockSize;

        private Thread player, dispatcher;

        private boolean begun, producerDone;
        private volatile boolean stopped;

        private final AtomicLong produced = new AtomicLong();

        private long issued, finished, drainDeadline = Long.MAX_VALUE;

        Session(ClientReplayPlan plan, ClientMetrics metrics, Path topology, Path evidence, String id,
                     int imageToken, int blockSize, long seed, int prefillCpSize) throws Exception {
            this.imageToken = imageToken; this.blockSize = blockSize;
            this.plan = plan; this.metrics = metrics;
            queue = new ArrayBlockingQueue<>(plan.queue_capacity());
            for (var outcome : BackendRpcClient.Outcome.values()) outcomes.put(outcome.name(), 0L);
            discovery = new MasterDiscovery(topology, plan.masters());
            try { journal = new RequestJournal(evidence, id, plan); }
            catch (Exception error) { discovery.close(); throw error; }
            client = new BackendRpcClient(discovery::routes, journal::record, imageToken, seed, prefillCpSize);
            // A lost begin command must not leave a prepared worker alive forever.
            dispatcher = Thread.ofVirtual().name("client-dispatch").start(this::dispatch);
        }

        @Override public void close() {
            halt();
            List<BackendRpcClient.Call> calls;
            synchronized (this) { calls = List.copyOf(active.values()); }
            calls.forEach(BackendRpcClient.Call::cancel);
            try { completion.get(8, TimeUnit.SECONDS); }
            catch (Exception error) { dispatcher.interrupt(); }
        }

        synchronized void begin(long epochMillis) {
            if (begun || stopped || completion.isDone()) throw new IllegalStateException("Client already begun or stopped");
            long delay = epochMillis - System.currentTimeMillis();
            if (delay < 0 || delay > 30000) throw new IllegalArgumentException("Start time must be within the next 30 seconds");
            begun = true;
            long start = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(delay);
            player = Thread.ofVirtual().name("client-player").start(() -> {
                try { play(plan, start, imageToken, this::offer); }
                catch (InterruptedException interrupted) { if (!isStopped()) fail("Player interrupted unexpectedly"); }
                catch (java.io.InterruptedIOException interrupted) { if (!isStopped()) fail(interrupted.toString()); }
                catch (java.io.IOException | RuntimeException error) { fail(error.toString()); }
                finally { synchronized (Session.this) { finishInput(); } }
            });
            LockSupport.unpark(dispatcher);
        }

        static void play(ClientReplayPlan plan, long startNanos, int imageToken, Consumer<Item> submit) throws IOException, InterruptedException {
            try (var indexed = new IndexedTrafficReader(Path.of(plan.requests_file()))) {
                long expected = ClientProtocol.requestCount(indexed.count, plan.first_index(), plan.stride());
                if (!indexed.identity.equals(plan.sha256()) || expected != plan.count())
                    throw new IOException("Indexed traffic changed after planning");
                delayNanos(indexed.count - 1, indexed.lastArrival, plan.qps(), plan.replay_speed());
                for (long position = 0; position < plan.count(); position++) {
                    if (Thread.interrupted()) throw new InterruptedException();
                    long index = plan.first_index() + position * plan.stride();
                    var entry = indexed.entry(index, imageToken);
                    int outputLimit = playbackLimit(entry, plan.fallback_max_output_tokens());
                    long due = startNanos + delayNanos(index, entry.arrivalMs(), plan.qps(), plan.replay_speed());
                    long left;
                    while ((left = due - System.nanoTime()) > 0) TimeUnit.NANOSECONDS.sleep(left);
                    if (Thread.interrupted()) throw new InterruptedException();
                    submit.accept(new Item(plan.first_request_id() + index, entry, outputLimit));
                }
            }
        }

        static long delayNanos(long index, long arrivalMs, Double qps, Double speed) throws IOException {
            double seconds = speed == null ? index / qps : arrivalMs / (1000.0 * speed);
            if (!Double.isFinite(seconds) || seconds < 0 || seconds > 86400)
                throw new IOException("Playback exceeds 24 hours");
            return (long) (seconds * 1_000_000_000.0);
        }

        static int playbackLimit(IndexedTrafficReader.Entry entry, Integer fallback) throws IOException {
            if (entry.observedOutputTokens() == 0)
                throw new IOException("Indexed traffic: zero observed output cannot be replayed as generation");
            if (entry.observedOutputTokens() > 0) return entry.observedOutputTokens();
            if (entry.maxOutputTokens() > 0) return entry.maxOutputTokens();
            if (fallback == null) throw new IOException("Indexed traffic: fallback_max_output_tokens is required");
            return fallback;
        }

        synchronized void fail(String error) {
            if (errors.isEmpty()) errors.add(error);
            halt();
        }

        synchronized void halt() {
            if (completion.isDone() || stopped) return;
            stopped = true; queue.clear();
            if (player != null) player.interrupt();
            finishInput();
        }

        synchronized State state() { return completion.isDone() ? completion.join() : snapshot(false); }

        State await(long timeoutMillis) throws Exception { return completion.get(timeoutMillis, TimeUnit.MILLISECONDS); }

        private synchronized boolean isStopped() { return stopped; }

        private void offer(Item item) {
            if (stopped) return;
            if (!queue.offer(item)) {
                synchronized (this) { fail("Playback queue full: capacity=" + plan.queue_capacity() + ", inflight=" + active.size()); }
                throw new IllegalStateException("Playback queue full");
            }
            produced.incrementAndGet();
            // Stop may race an offer; it can never reopen this queue or issue that request.
            if (stopped) queue.clear();
            LockSupport.unpark(dispatcher);
        }

        private void finishInput() {
            if (!producerDone) {
                producerDone = true;
                drainDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(plan.drain_timeout_ms());
            }
            LockSupport.unpark(dispatcher);
        }

        private void dispatch() {
            long prepareDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            try {
                while (true) {
                    discovery.check();
                    if (journal.failure() != null) fail(journal.failure());
                    List<BackendRpcClient.Call> cancel = List.of();
                    synchronized (this) {
                        if (!begun && !stopped && System.nanoTime() >= prepareDeadline) fail("Timed out waiting for begin");
                        if (producerDone && queue.isEmpty() && active.isEmpty()) break;
                        if (producerDone && System.nanoTime() >= drainDeadline) {
                            fail("Client drain timed out");
                            cancel = List.copyOf(active.values());
                        } else if (begun && !stopped && !queue.isEmpty() && active.size() < plan.concurrency()) {
                            var item = queue.remove();
                            var entry = item.entry();
                            var input = entry.input().toBuilder().setRequestId(item.requestId());
                            input.getGenerateConfigBuilder().setReuseCache(Boolean.TRUE.equals(plan.reuse_cache()))
                                .setMaxNewTokens(item.outputLimit()).setIgnoreEos(entry.observedOutputTokens() > 0)
                                .setCanUsePdSeparation(true).setReturnIncremental(true).setIsStreaming(true);
                            var generateInput = input.build();
                            var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(item.requestId())
                                .setCacheKeyBlockSize(blockSize).setSeqLen(input.getTokenIdsCount())
                                .setMaxNewTokens(input.getGenerateConfig().getMaxNewTokens()).setPriority(entry.priority())
                                .setGenerateInput(generateInput.toByteString()).build();
                            var call = client.send(request, generateInput, Duration.ofMillis(plan.request_timeout_ms()), entry.expectedImageTokens());
                            active.put(item.requestId(), call); issued++; metrics.issued();
                            call.completion().whenComplete((result, error) -> {
                                synchronized (Session.this) {
                                    active.remove(item.requestId()); finished++;
                                    if (error != null) fail("Request execution failed: " + error);
                                    else { metrics.finished(result); outcomes.merge(result.outcome().name(), 1L, Long::sum); }
                                    LockSupport.unpark(dispatcher);
                                }
                            });
                            continue;
                        }
                    }
                    // Native completion callbacks acquire this object's monitor; never cancel while holding it.
                    cancel.forEach(BackendRpcClient.Call::cancel);
                    LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(100));
                    if (Thread.interrupted()) throw new InterruptedException();
                }
            } catch (InterruptedException error) {
                fail("Client dispatcher interrupted"); Thread.currentThread().interrupt();
            } catch (RuntimeException error) { fail("Client dispatcher failed: " + error); }
            finally {
                if (player != null) try { player.join(1000); }
                catch (InterruptedException error) { Thread.currentThread().interrupt(); }
                client.close(); discovery.close();
                try { journal.finish(issued, finished); }
                catch (java.io.IOException error) { fail(error.getMessage()); }
                synchronized (this) { completion.complete(snapshot(true)); LockSupport.unpark(dispatcher); }
            }
        }

        private State snapshot(boolean done) {
            return new State(done, stopped, produced.get(), issued, finished, plan.count() - issued,
                queue.size(), active.size(), Map.copyOf(outcomes), List.copyOf(errors));
        }

        record Item(long requestId, IndexedTrafficReader.Entry entry, int outputLimit) {}
    }
}
