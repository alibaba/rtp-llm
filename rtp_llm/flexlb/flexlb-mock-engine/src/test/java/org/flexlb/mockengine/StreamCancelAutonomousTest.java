package org.flexlb.mockengine;

import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.Status;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Autonomous client-gone cancellation (production C++ engine alignment):
 * when the client's GenerateStream / FetchResponse stream breaks mid-flight,
 * the engine itself must clean the request up within a short window (the
 * production engine checks IsCancelled in its per-token loop).
 *
 * <p>Coverage matrix (the four required behaviours):
 * <ol>
 *   <li><b>prefill direct break</b> — the prefill is executing the request:
 *       runningTasks entry dropped, cancelled_rids recorded, typed CANCELLED
 *       terminal published, census counts it — all within 1s of the break;</li>
 *   <li><b>P→D propagation</b> — the break lands on the prefill AFTER the
 *       decode hand-off (both a broken GenerateStream and a broken
 *       FetchResponse): the decode cleans its slot/KV bookkeeping and records
 *       its own CANCELLED terminal, the prefill publishes the typed terminal
 *       carrying the request identity (tryCancelDownstream contract);</li>
 *   <li><b>decode direct break</b> — NON_BATCH style direct-to-decode stream:
 *       cleanup plus the terminal reported IMMEDIATELY (no stale-inflight TTL
 *       wait — the 5s decode step is still far from finishing);</li>
 *   <li><b>race idempotence + no false positives</b> — client-gone vs the
 *       explicit Cancel RPC in both orderings publish exactly one terminal
 *       (the cancel() claim), and a stream that completes NORMALLY never
 *       triggers the detector (the listener re-checks isCancelled()).</li>
 * </ol>
 *
 * <p>Real Netty gRPC servers are required (in-process calls run on
 * {@code Context.ROOT}, which the detector deliberately ignores), with the
 * break driven by a raw {@link ClientCall#cancel} so exactly one stream dies
 * while any sibling stream on the same engine stays alive.
 */
class StreamCancelAutonomousTest {

    /** Fresh range above the 62000-63900 blocks used by the existing suites. */
    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(64000);

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private Map<Integer, Server> serversByPort;
    private JavaMockEngineCluster.ClusterStats stats;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (serversByPort != null) {
            for (Server server : serversByPort.values()) {
                server.shutdownNow();
            }
            serversByPort = null;
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            services = null;
        }
        if (bossGroup != null) {
            bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
            bossGroup = null;
        }
        if (workerGroup != null) {
            workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
            workerGroup = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
            scheduler = null;
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Tests
    // ════════════════════════════════════════════════════════════════

    @Test
    void prefillDirectStreamBreakCleansUpWithinOneSecond() throws Exception {
        // 4s prefill keeps the request executing on the prefill engine while
        // the stream breaks; the 1s detection window is far inside it.
        startCluster(model("4000", 10), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort());
        long rid = 7001L;

        try (ClientStream stream = ClientStream.generateStream(
                prefillPort(), MockEngineTestSupport.input(rid, 16))) {
            awaitCondition(() -> prefill.getRunningCount() >= 1, 2_000,
                    "prefill request should be running before the break");
            assertEquals(0, prefill.getCancelledCount(),
                    "sanity: no cancel before the break");

            stream.breakStream();

            // ≤1s autonomous cleanup — the core production-alignment budget.
            awaitCondition(() -> prefill.getCancelledCount() == 1, 1_000,
                    "prefill should self-cancel after the client stream broke"
                            + " (running=" + prefill.getRunningCount() + ")");
            assertEquals(0, prefill.getRunningCount(),
                    "runningTasks entry must be dropped after the break");
            assertEquals(0, prefill.getInflightCount(),
                    "pendingRequests must net to zero after the break");
            assertEquals(1, clientGoneCensus(),
                    "census must count exactly one effective client-gone cancel");
            assertTrue(cancelledRids(prefill).contains(rid),
                    "cancelled_rids must record " + rid + ": " + cancelledRids(prefill));

            // Typed CANCELLED terminal published exactly once for the master.
            assertEquals(1, countCancelTerminals(prefill, rid),
                    "exactly one typed CANCELLED terminal expected");
        }
    }

    @Test
    void generateStreamBreakAfterHandoffPropagatesCancelToDecode() throws Exception {
        // 10ms prefill (hand-off happens almost immediately) + 5s decode step:
        // the break lands while the DECODE executes.
        startCluster(model("10", 5000), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort());
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort());
        long rid = 7002L;

        try (ClientStream stream = ClientStream.generateStream(
                prefillPort(), MockEngineTestSupport.inputWithDecode(rid, 16, decodePort()))) {
            awaitCondition(() -> decode.getRunningCount() >= 1, 2_000,
                    "decode should run the request after the P->D hand-off");

            stream.breakStream();

            awaitCondition(() -> decode.getCancelledCount() == 1, 1_000,
                    "decode should be cancelled by the propagated client-gone"
                            + " (running=" + decode.getRunningCount() + ")");
            // Decode-side cleanup: slot, KV, inflight all released.
            assertEquals(0, decode.getRunningCount());
            assertEquals(0, decode.getInflightCount());
            assertEquals(0, decode.getActiveKvTokens(),
                    "decode KV must be released by the propagated cancel");
            assertTrue(cancelledRids(decode).contains(rid),
                    "decode cancelled_rids must record " + rid);
            // The prefill never ran cancel() itself — it only published the
            // typed terminal (production: the P context cancel propagates,
            // the prefill's own accounting stays clean).
            assertEquals(0, prefill.getCancelledCount(),
                    "prefill must not double-count the propagated cancel");

            // Both engines publish exactly one typed CANCELLED terminal for
            // the request: decode locally, prefill via recordClientGoneCanceled.
            assertEquals(1, countCancelTerminals(decode, rid),
                    "decode must publish one typed CANCELLED terminal");
            assertEquals(1, countCancelTerminals(prefill, rid),
                    "prefill must publish the propagated typed CANCELLED terminal");
            assertEquals(1, clientGoneCensus(),
                    "census must count the propagated cancel exactly once");
        }
    }

    @Test
    void fetchResponseBreakPropagatesCancelToDecode() throws Exception {
        // The BATCH-dispatcher shape: the client's FetchResponse stream is
        // glued to the ORIGINAL prefill; breaking it must drive the same
        // P->D propagation while the sibling GenerateStream stays alive.
        startCluster(model("10", 5000), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort());
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort());
        long rid = 7003L;

        try (ClientStream generateStream = ClientStream.generateStream(
                prefillPort(), MockEngineTestSupport.inputWithDecode(rid, 16, decodePort()))) {
            awaitCondition(() -> decode.getRunningCount() >= 1, 2_000,
                    "decode should run the request after the P->D hand-off");

            try (ClientStream fetch = ClientStream.fetchResponse(prefillPort(),
                    EngineRpcService.FetchRequestPB.newBuilder()
                            .setRequestId(rid)
                            .build())) {
                // Wait until the fetch RPC actually reached the engine's
                // handler (listener armed), then break ONLY the fetch stream.
                awaitCondition(() -> rpcCount(prefill, "fetch_response") >= 1, 2_000,
                        "fetchResponse should reach the engine handler");
                Thread.sleep(100); // handler registration completes after the counter
                fetch.breakStream();
            }

            awaitCondition(() -> decode.getCancelledCount() == 1, 1_000,
                    "decode should be cancelled by the broken fetch stream");
            assertEquals(0, decode.getRunningCount());
            assertEquals(0, decode.getActiveKvTokens());
            assertEquals(0, prefill.getCancelledCount());
            assertEquals(1, countCancelTerminals(prefill, rid),
                    "prefill must publish the propagated typed terminal");
            assertEquals(1, countCancelTerminals(decode, rid));
            assertEquals(1, clientGoneCensus(),
                    "only the broken fetch stream may count as client-gone");
        }
        // The still-alive GenerateStream receives the CANCELLED error frame
        // and closes normally (terminal frame semantics) — no explosion, and
        // its own context close at channel shutdown must stay a no-op.
        awaitCondition(() -> decode.getInflightCount() == 0 && prefill.getInflightCount() == 0,
                1_000, "both engines must quiesce");
        assertEquals(1, clientGoneCensus(),
                "closing the sibling stream after cleanup must not re-count");
    }

    @Test
    void decodeDirectStreamBreakReportsTerminalImmediately() throws Exception {
        // NON_BATCH style: GenerateStream straight to the decode engine. The
        // 5s decode step means a NORMAL terminal is ~5s away — a terminal
        // inside the 1s window proves the engine reported early instead of
        // waiting out the stale-inflight TTL.
        startCluster(model("10", 5000), 0, 1);
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort());
        long rid = 7004L;

        try (ClientStream stream = ClientStream.generateStream(
                decodePort(), MockEngineTestSupport.input(rid, 16))) {
            awaitCondition(() -> decode.getRunningCount() >= 1, 2_000,
                    "decode should run the direct request");

            stream.breakStream();

            long breakNanos = System.nanoTime();
            awaitCondition(() -> decode.getCancelledCount() == 1, 1_000,
                    "decode must self-cancel within the 1s window");
            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - breakNanos);
            assertTrue(elapsedMs < 5_000,
                    "terminal at " + elapsedMs + "ms must beat the 5s decode step");

            assertEquals(0, decode.getRunningCount());
            assertEquals(0, decode.getInflightCount());
            assertEquals(0, decode.getActiveKvTokens(),
                    "decode KV must be released");
            assertTrue(cancelledRids(decode).contains(rid));
            assertEquals(1, countCancelTerminals(decode, rid),
                    "early typed CANCELLED terminal published exactly once");
            assertEquals(1, clientGoneCensus());
        }
    }

    @Test
    void clientGoneRacesExplicitCancelInBothOrdersIsIdempotent() throws Exception {
        startCluster(model("10", 5000), 0, 1);
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort());

        // ── Order A: stream breaks FIRST, explicit Cancel arrives second ──
        long ridA = 7005L;
        try (ClientStream stream = ClientStream.generateStream(
                decodePort(), MockEngineTestSupport.input(ridA, 16))) {
            awaitCondition(() -> decode.getRunningCount() >= 1, 2_000,
                    "decode should run the request (order A leg)");
            stream.breakStream();
            awaitCondition(() -> decode.getCancelledCount() == 1, 1_000,
                    "client-gone cancel should claim the terminal first");

            // The late explicit Cancel finds the claim armed → no-op.
            assertNull(decode.cancel(ridA),
                    "explicit cancel after client-gone must no-op (claim already armed)");
            assertEquals(1, decode.getCancelledCount(),
                    "cancelled_count must stay at 1 (idempotent claim)");
            assertEquals(1, countCancelTerminals(decode, ridA),
                    "exactly one typed CANCELLED terminal despite the double cancel");
            assertEquals(1, clientGoneCensus(),
                    "only the client-gone path may count");
        }

        awaitCondition(() -> decode.getInflightCount() == 0, 1_000,
                "engine must quiesce between the two race legs");

        // ── Order B: explicit Cancel FIRST, stream breaks second ──
        long ridB = 7006L;
        try (ClientStream stream = ClientStream.generateStream(
                decodePort(), MockEngineTestSupport.input(ridB, 16))) {
            awaitCondition(() -> decode.getRunningCount() >= 1, 2_000,
                    "decode should run the request (order B leg)");
            assertNotNull(decode.cancel(ridB),
                    "explicit cancel should claim the terminal (request running)");
            assertEquals(2, decode.getCancelledCount(),
                    "two distinct requests cancelled so far");

            stream.breakStream();
            // Give the break time to propagate — if the detector mis-fired
            // it would show up inside this window.
            Thread.sleep(500);

            assertEquals(2, decode.getCancelledCount(),
                    "the late break must not cancel twice");
            assertEquals(1, countCancelTerminals(decode, ridB),
                    "exactly one typed terminal for ridB");
            assertEquals(1, clientGoneCensus(),
                    "a break on an already-terminal request must stay a no-op");
            assertEquals(0, decode.getInflightCount());
        }
    }

    @Test
    void normalCompletionNeverTriggersClientGoneDetector() throws Exception {
        // Full healthy P->D request (~60ms end to end). The stream closes
        // NORMALLY — grpc notifies the CancellationListener on close too, so
        // only the isCancelled() re-check keeps this from being a false
        // positive. That is exactly what this test pins.
        startCluster(model("10", 50), 1, 1);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort());
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort());
        long rid = 7007L;

        try (ClientStream stream = ClientStream.generateStream(
                prefillPort(), MockEngineTestSupport.inputWithDecode(rid, 16, decodePort()))) {
            assertTrue(stream.closed.await(10, TimeUnit.SECONDS),
                    "healthy stream must complete");
            assertEquals(Status.Code.OK, stream.closeStatus.getCode(),
                    "healthy stream must close OK, got " + stream.closeStatus);
            assertTrue(stream.frames.get() >= 1,
                    "healthy stream must deliver at least one frame");

            // Let any (buggy) post-close listener notification land before
            // asserting the census stayed clean.
            Thread.sleep(300);
            assertEquals(0, prefill.getCancelledCount(),
                    "normal completion must not be cancelled (prefill)");
            assertEquals(0, decode.getCancelledCount(),
                    "normal completion must not be cancelled (decode)");
            assertEquals(0, clientGoneCensus(),
                    "normal stream close must never count as client-gone");
            assertEquals(0, prefill.getInflightCount());
            assertEquals(0, decode.getInflightCount());
            assertEquals(0, countCancelTerminals(prefill, rid)
                    + countCancelTerminals(decode, rid),
                    "no CANCELLED terminal for a healthy request");
        }
    }

    @Test
    void statsLineCarriesClientGoneCensusKey() {
        // Emission surface: the census joins the parsed java_mock_stats keys.
        String line = JavaMockEngineCluster.buildStatsLine(
                List.of(), new JavaMockEngineCluster.ClusterStats());
        assertTrue(line.contains(" cancel_census_client_gone=")
                        || line.contains("java_mock_stats cancel_census_client_gone="),
                "stats line must carry cancel_census_client_gone: " + line);
    }

    // ════════════════════════════════════════════════════════════════
    //  Cluster bootstrap (DynamicEngineScaleTest pattern, no control server)
    // ════════════════════════════════════════════════════════════════

    private int basePort;
    /** Actual decode port — with 0 prefill engines the decode takes basePort+0. */
    private int decodePortValue;

    private int prefillPort() {
        return basePort;
    }

    private int decodePort() {
        return decodePortValue;
    }

    private MockPerformanceModel model(String prefillFormula, double decodeStepMs)
            throws IOException {
        return MockEngineTestSupport.performanceModel(
                tempDir, prefillFormula, 1.0, decodeStepMs);
    }

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        basePort = allocatePortBlock();
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "stream-cancel-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(4);
        services = new ConcurrentHashMap<>();
        serversByPort = new ConcurrentHashMap<>();
        stats = new JavaMockEngineCluster.ClusterStats();
        JavaMockEngineCluster.Config config = new JavaMockEngineCluster.Config();
        config.host = "127.0.0.1";
        config.uniqueEngineIps = false;

        for (int i = 0; i < nPrefill; i++) {
            JavaMockEngineCluster.startEngine(config, model, serversByPort, bossGroup,
                    workerGroup, services, scheduler, stats, "prefill", "prefill-" + i,
                    basePort + i, i);
        }
        for (int i = 0; i < nDecode; i++) {
            JavaMockEngineCluster.startEngine(config, model, serversByPort, bossGroup,
                    workerGroup, services, scheduler, stats, "decode", "decode-" + i,
                    basePort + nPrefill + i, nPrefill + i);
        }
        decodePortValue = basePort + nPrefill;
    }

    /**
     * Claim the next 32-port block with every port bindable right now
     * (mirrors {@code DynamicEngineScaleTest.allocatePortBlock}).
     */
    private static int allocatePortBlock() {
        for (int attempt = 0; attempt < 20; attempt++) {
            int basePort = PORT_ALLOCATOR.getAndAdd(32);
            boolean allFree = true;
            for (int port = basePort; port < basePort + 32; port++) {
                try (ServerSocket probe = new ServerSocket()) {
                    probe.bind(new InetSocketAddress(port), 1);
                } catch (IOException e) {
                    allFree = false;
                    break;
                }
            }
            if (allFree) {
                return basePort;
            }
        }
        throw new IllegalStateException("no bindable 32-port block after 20 attempts");
    }

    private long clientGoneCensus() {
        return stats.cancelCensusClientGone.sum();
    }

    @SuppressWarnings("unchecked")
    private static List<Long> cancelledRids(JavaMockEngineCluster.FastRpcService service) {
        return (List<Long>) service.getSnapshot().get("cancelled_rids");
    }

    /** Per-RPC counter from the public snapshot surface (rpc_counts map). */
    private static int rpcCount(JavaMockEngineCluster.FastRpcService service, String rpc) {
        @SuppressWarnings("unchecked")
        Map<String, Object> counts =
                (Map<String, Object>) service.getSnapshot().get("rpc_counts");
        return ((Number) counts.get(rpc)).intValue();
    }

    /** Typed CANCELLED terminals for {@code rid} visible in the next WorkerStatus poll. */
    private static long countCancelTerminals(JavaMockEngineCluster.FastRpcService service,
                                             long rid) {
        return MockEngineTestSupport.workerStatus(service, 0)
                .getFinishedTaskListList().stream()
                .filter(task -> task.getRequestId() == rid)
                .filter(task -> task.hasErrorInfo()
                        && task.getErrorInfo().getErrorCode()
                                == EngineRpcService.ErrorCodePB.CANCELLED.getNumber())
                .count();
    }

    private static void awaitCondition(BooleanSupplier condition, long timeoutMs, String message)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(10);
        }
        throw new AssertionError(message);
    }

    private interface BooleanSupplier {
        boolean getAsBoolean();
    }

    // ════════════════════════════════════════════════════════════════
    //  Client-side stream handle with surgical break control
    // ════════════════════════════════════════════════════════════════

    /**
     * One client RPC stream over a dedicated channel, driven through the raw
     * {@link ClientCall} API so {@link #breakStream()} cancels EXACTLY this
     * call (channel-level shutdown would kill sibling streams on the same
     * engine — the fetch-propagation test relies on that precision).
     */
    private static final class ClientStream implements AutoCloseable {
        private final ManagedChannel channel;
        private final ClientCall<?, EngineRpcService.GenerateOutputsPB> call;
        private final AtomicInteger frames = new AtomicInteger();
        private final CountDownLatch closed = new CountDownLatch(1);
        private volatile Status closeStatus;

        private ClientStream(ManagedChannel channel,
                             ClientCall<?, EngineRpcService.GenerateOutputsPB> call) {
            this.channel = channel;
            this.call = call;
        }

        static ClientStream generateStream(int port, EngineRpcService.GenerateInputPB input) {
            return open(port, RpcServiceGrpc.getGenerateStreamCallMethod(), input);
        }

        static ClientStream fetchResponse(int port, EngineRpcService.FetchRequestPB request) {
            return open(port, RpcServiceGrpc.getFetchResponseMethod(), request);
        }

        private static <ReqT> ClientStream open(
                int port,
                MethodDescriptor<ReqT, EngineRpcService.GenerateOutputsPB> method,
                ReqT request) {
            ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", port)
                    .usePlaintext()
                    .build();
            ClientCall<ReqT, EngineRpcService.GenerateOutputsPB> call =
                    channel.newCall(method, CallOptions.DEFAULT);
            ClientStream stream = new ClientStream(channel, call);
            call.start(new ClientCall.Listener<>() {
                @Override
                public void onMessage(EngineRpcService.GenerateOutputsPB value) {
                    stream.frames.incrementAndGet();
                }

                @Override
                public void onClose(Status status, Metadata trailers) {
                    stream.closeStatus = status;
                    stream.closed.countDown();
                }
            }, new Metadata());
            // Flow-control permits for inbound messages: a raw ClientCall
            // delivers nothing until request() arms it — forgetting this
            // made the server-side onNext succeed while the client held every
            // frame (frames=0, no onClose) and the healthy-stream test hang.
            call.request(Integer.MAX_VALUE);
            call.sendMessage(request);
            call.halfClose();
            return stream;
        }

        /** Simulate the client dying: cancel exactly this call mid-flight. */
        void breakStream() {
            call.cancel("client stream gone", null);
        }

        @Override
        public void close() {
            channel.shutdownNow();
        }
    }
}
