package org.flexlb.httpserver;

import com.github.luben.zstd.ZstdOutputStream;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.util.Logger;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Properties;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/** One bounded capture window per Master process. Only the writer sees the anonymization key. */
final class ScheduleCapture {
    private static final int QUEUE_SIZE = 1024;
    private static final int MAX_KEYS = 8192;
    private static final int MAX_NODES = 2_000_000;
    private static final int MAX_ATTEMPTS = 250_000;
    private static final long DRAIN_MS = 90_000;

    private sealed interface Event permits Arrival, Result {}
    private record Arrival(long attempt, long offsetMs, long requestId, long blockSize, long inputTokens,
                           int outputLimit, int priority, boolean vitOnly, List<Long> keys) implements Event {}
    private record Result(long attempt, String status, int code) implements Event {}
    private record Key(long blockSize, long rawKey) {}
    private record Node(int id, int parent) {}

    private final ArrayBlockingQueue<Event> queue = new ArrayBlockingQueue<>(QUEUE_SIZE);
    private long startMs;
    private long endMs;
    private long sealMs;
    private final LongSupplier clock;
    private boolean closed;
    private long arrivals;
    private long droppedArrivals;
    private long droppedResults;

    static ScheduleCapture fromEnvironment() {
        String directory = System.getenv("FLEXLB_PREFIX_CAPTURE_DIR");
        if (directory == null || directory.isBlank()) return null;
        return new ScheduleCapture(Path.of(directory));
    }

    private ScheduleCapture(Path directory) {
        this.clock = System::currentTimeMillis;
        Thread.ofPlatform().daemon().name("master-prefix-capture").start(() -> awaitArm(directory));
    }

    ScheduleCapture(Path directory, long startMs, int seconds, byte[] key, LongSupplier clock) throws Exception {
        this.clock = clock;
        Mac mac = arm(startMs, seconds, key);
        Thread.ofPlatform().daemon().name("master-prefix-capture").start(() -> write(directory, mac));
    }

    private void awaitArm(Path directory) {
        Path control = directory.resolve("capture.properties");
        try {
            while (!Files.isRegularFile(control)) Thread.sleep(500);
            var config = new Properties();
            try (var reader = Files.newBufferedReader(control)) { config.load(reader); }
            long start = Long.parseLong(config.getProperty("start_ms"));
            int seconds = Integer.parseInt(config.getProperty("seconds"));
            byte[] key = HexFormat.of().parseHex(config.getProperty("key"));
            Mac mac = arm(start, seconds, key);
            write(directory, mac);
        } catch (Exception | LinkageError e) {
            Logger.error("Master prefix capture could not arm: {}", e.getClass().getSimpleName());
        }
    }

    private synchronized Mac arm(long start, int seconds, byte[] key) throws Exception {
        if (start <= 0 || seconds < 1 || seconds > 1800 || key.length != 32 || start < clock.getAsLong()) {
            throw new IllegalArgumentException("Invalid prefix capture window or key length");
        }
        long end = Math.addExact(start, seconds * 1000L);
        long seal = Math.addExact(end, DRAIN_MS);
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key, "HmacSHA256"));
        startMs = start;
        endMs = end;
        sealMs = seal;
        return mac;
    }

    synchronized long arrive(FlexlbScheduleRequestPB request) {
        long now = clock.getAsLong();
        if (closed || now < startMs || now >= endMs || request.getForwardHop() != 0) return 0;
        long attempt = ++arrivals;
        if (attempt > MAX_ATTEMPTS || request.getBlockCacheKeysCount() > MAX_KEYS
                || !queue.offer(new Arrival(attempt, now - startMs, request.getRequestId(),
                request.getCacheKeyBlockSize(), request.getSeqLen(), request.getMaxNewTokens(),
                request.getPriority(), request.getVitRouteOnly(), request.getBlockCacheKeysList()))) {
            droppedArrivals++;
            return 0;
        }
        return attempt;
    }

    synchronized void complete(long attempt, String status, int code) {
        if (!closed && !queue.offer(new Result(attempt, status, code))) droppedResults++;
    }

    private void write(Path directory, Mac mac) {
        Path output = directory.resolve("prefix-" + startMs + "-" + ProcessHandle.current().pid() + ".jsonl.zst");
        Path partial = output.resolveSibling(output.getFileName() + ".part");
        var nodes = new HashMap<Key, Node>();
        try {
            try (var out = new BufferedWriter(new OutputStreamWriter(
                    new ZstdOutputStream(Files.newOutputStream(partial, java.nio.file.StandardOpenOption.CREATE_NEW), 3),
                    StandardCharsets.UTF_8))) {
                out.write("[\"header\",1," + startMs + "," + (endMs - startMs) + ",\""
                        + anonymous(mac, -1, 0) + "\",\"" + UUID.randomUUID() + "\"]\n");
                while (true) {
                    var event = queue.poll(200, TimeUnit.MILLISECONDS);
                    if (event != null) {
                        switch (event) {
                            case Arrival arrival -> writeArrival(out, mac, nodes, arrival);
                            case Result result -> out.write("[\"result\"," + result.attempt + ",\""
                                    + result.status + "\"," + result.code + "]\n");
                        }
                    }
                    String trailer = sealIfDrained();
                    if (trailer != null) {
                        out.write(trailer);
                        break;
                    }
                }
            }
            Files.move(partial, output, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (Exception | LinkageError e) {
            synchronized (this) {
                closed = true;
                queue.clear();
            }
            Logger.error("Master prefix capture incomplete: {}", e.getClass().getSimpleName());
        }
    }

    private synchronized String sealIfDrained() {
        if (!queue.isEmpty() || clock.getAsLong() < sealMs) return null;
        closed = true;
        return "[\"end\"," + arrivals + "," + droppedArrivals + "," + droppedResults + "]\n";
    }

    private static void writeArrival(BufferedWriter out, Mac mac, HashMap<Key, Node> nodes,
                                     Arrival arrival) throws Exception {
        int parent = 0;
        if (!arrival.vitOnly) {
            if (arrival.blockSize <= 0 && !arrival.keys.isEmpty()) {
                throw new IllegalStateException("Missing logical block size");
            }
            for (long key : arrival.keys) {
                var identity = new Key(arrival.blockSize, key);
                var node = nodes.get(identity);
                if (node == null) {
                    if (nodes.size() >= MAX_NODES) throw new IllegalStateException("Capture tree reached its node limit");
                    node = new Node(nodes.size() + 1, parent);
                    nodes.put(identity, node);
                    out.write("[\"node\"," + node.id + "," + parent + ",\""
                            + anonymous(mac, arrival.blockSize, key) + "\"]\n");
                } else if (node.parent != parent) {
                    throw new IllegalStateException("Inconsistent prefix path");
                }
                parent = node.id;
            }
        }
        out.write("[\"request\"," + arrival.attempt + ",\"" + anonymous(mac, 0, arrival.requestId) + "\","
                + arrival.offsetMs + "," + arrival.blockSize + "," + arrival.inputTokens + ","
                + arrival.outputLimit + "," + arrival.priority + "," + arrival.vitOnly + "," + parent + "]\n");
    }

    private static String anonymous(Mac mac, long namespace, long id) {
        byte[] digest = mac.doFinal(ByteBuffer.allocate(16).putLong(namespace).putLong(id).array());
        return HexFormat.of().formatHex(digest, 0, 16);
    }
}
