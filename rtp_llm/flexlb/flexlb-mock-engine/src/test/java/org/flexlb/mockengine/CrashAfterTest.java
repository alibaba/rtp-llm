package org.flexlb.mockengine;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPost;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * crash_after TRUE-CRASH semantics (memory wipe + gRPC port kill) — the
 * behavioural lock for the semantics distinction:
 *
 * <ul>
 *   <li>{@code stop_engine} = network-level outage: port closed, pools
 *       KEPT for in-place continuation;</li>
 *   <li>{@code crash_after} = process-level death: port closed AND every
 *       per-engine container (running tasks, queues, response streams,
 *       leases, KV LRU, bounded histories, un-acked completions, admission
 *       gauges, counters) wiped — recovery equals a reboot from zero.</li>
 * </ul>
 *
 * <p>Both close the gRPC port, so the master walks the same 3-strike retire
 * path either way; the difference must show up in the recovered engine's
 * memory, asserted here. The crash-triggering EnqueueBatch still answers
 * with an EMPTY ack (no successes, no errors) so the master-side
 * BATCH_ACK_UNCERTAIN fence contract is preserved; the port kill itself is
 * delayed by {@link JavaMockEngineCluster.FastRpcService#CRASH_PORT_KILL_DELAY_MS}
 * so that ack flushes to the wire while the socket is still open.
 *
 * <p>Also locks the epoch fence: prefill batch start / completion callbacks
 * and decode step ticks already queued on the shared scheduler when the
 * crash lands cannot be unscheduled — they must drop out without touching
 * the wiped state (no resurrection of running tasks, no LRU handover, no
 * completion publication).
 */
class CrashAfterTest {

    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(63300);

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private Map<Integer, Server> serversByPort;

    @AfterEach
    void tearDown() throws InterruptedException {
        cleanupCluster();
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 1: n=3 → two successes, the 3rd crashes; wipe + port kill +
    //  /start_engine recovery on clean state
    // ════════════════════════════════════════════════════════════════

    @Test
    void crashWipesMemoryAndKillsGrpcPortThenRecovers() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");
        int prefillPort = startPrefillClusterWithGrpc(model);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort);

        try {
            // Two requests carrying hash-channel block keys: once their 10ms
            // executions finish, the keys land in the LRU — exactly the KV
            // memory the crash must wipe (stop_engine would keep it).
            for (int i = 1; i <= 2; i++) {
                EngineRpcService.EnqueueBatchResponsePB response = enqueue(prefill,
                        batch(3000 + i, slot(0,
                                inputWithBlockKeys(i, 10, List.of(100L + i, 200L + i)))));
                assertEquals(1, response.getSuccessesCount(),
                        "request " + i + " should succeed pre-crash");
            }
            awaitInflightZero(prefill, 2_000);
            Map<String, Object> before = prefill.getSnapshot();
            assertFalse(((List<?>) before.get("cache_key_set")).isEmpty(),
                    "pre-crash sanity: completed requests' keys must sit in the LRU");

            // Arm the crash on the 3rd EnqueueBatch and deliver it.
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort
                            + ",\"type\":\"crash_after\",\"enabled\":true,\"n\":3}");
            assertEquals(3, prefill.getFaultConfig().getCrashAfterNRequests());

            EngineRpcService.EnqueueBatchResponsePB crashAck = enqueue(prefill,
                    batch(3003, slot(0, inputWithBlockKeys(3, 10, List.of(103L)))));
            assertEquals(0, crashAck.getSuccessesCount(),
                    "crash-triggering ack must carry NO successes (empty ack)");
            assertEquals(0, crashAck.getErrorsCount(),
                    "crash-triggering ack must carry NO errors either (uncertain fence)");
            assertTrue(prefill.isStopped(), "engine must be stopped (crashed) after the 3rd enqueue");

            // Memory wipe: no running tasks, no inflight, no held blocks,
            // empty LRU, fresh-process counters. Snapshot BEFORE the probe
            // request below — the stopped-branch rejection itself bumps the
            // rpc counter on the dead instance.
            Map<String, Object> wiped = prefill.getSnapshot();
            assertEquals(0, intField(wiped, "running"), "crash wipes running tasks");
            assertEquals(0, intField(wiped, "inflight"), "crash wipes inflight");
            assertEquals(0, intField(wiped, "held_blocks"), "crash wipes held blocks");
            assertTrue(((List<?>) wiped.get("cache_key_set")).isEmpty(),
                    "crash wipes the KV LRU (completed keys gone — the stop_engine contrast)");
            assertEquals(0, intField(wiped, "accepted"),
                    "crash resets the accept counter (fresh process)");
            assertEquals(0, intField(wiped, "completed"),
                    "crash resets the completed counter (fresh process)");
            assertEquals(0, rpcEnqueue(wiped),
                    "crash resets the rpc counters");
            assertEquals(0, intField(wiped, "waiting"), "crash wipes the wait queues");

            // The 4th in-process request sees the stopped rejection: empty ack.
            EngineRpcService.EnqueueBatchResponsePB fourth = enqueue(prefill,
                    batch(3004, slot(0, input(4, 10))));
            assertEquals(0, fourth.getSuccessesCount());
            assertEquals(0, fourth.getErrorsCount());

            // Port kill: after the CRASH_PORT_KILL_DELAY_MS grace a REAL gRPC
            // channel gets connection-refused (UNAVAILABLE) — the master's
            // health poller sees the same dead port.
            Thread.sleep(JavaMockEngineCluster.FastRpcService.CRASH_PORT_KILL_DELAY_MS + 400);
            assertTrue(portRefusesGrpc(prefillPort),
                    "gRPC port must refuse connections after the crash");

            // Recovery: /start_engine rebuilds the gRPC server on CLEAN state
            // (also disarms the fault config and resets the enqueue counter).
            httpPost(controlServer.getPort(), "/start_engine",
                    "{\"port\":" + prefillPort + "}");
            assertFalse(prefill.isStopped(), "engine must run again after /start_engine");
            assertEquals(0, prefill.getFaultConfig().getCrashAfterNRequests(),
                    "start_engine disarms the crash_after fault config");

            Map<String, Object> recovered = prefill.getSnapshot();
            assertEquals(0, intField(recovered, "running"));
            assertEquals(0, intField(recovered, "inflight"));
            assertTrue(((List<?>) recovered.get("cache_key_set")).isEmpty(),
                    "recovered engine starts from an EMPTY cache (reboot, not resume)");

            // Fresh traffic flows — through the rebuilt port too.
            EngineRpcService.EnqueueBatchResponsePB after = enqueue(prefill,
                    batch(3999, slot(0, input(99, 10))));
            assertEquals(1, after.getSuccessesCount(),
                    "enqueue must succeed after recovery");
            assertTrue(portServesGrpc(prefillPort),
                    "rebuilt gRPC port must serve traffic");
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 2: mid-execution crash — queued completion callbacks must not
    //  resurrect the wiped state (epoch fence)
    // ════════════════════════════════════════════════════════════════

    @Test
    void inFlightCallbacksCannotResurrectWipedState() throws Exception {
        // 300ms prefills: the first two requests are mid-execution when the
        // crash lands, so their completion callbacks are still queued on the
        // shared scheduler (they cannot be unscheduled — only fenced).
        MockPerformanceModel model = performanceModel(tempDir, "300");
        int prefillPort = startPrefillClusterWithGrpc(model);
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort);

        try {
            for (int i = 1; i <= 2; i++) {
                EngineRpcService.EnqueueBatchResponsePB response = enqueue(prefill,
                        batch(4000 + i, slot(0,
                                inputWithBlockKeys(i, 10, List.of(300L + i)))));
                assertEquals(1, response.getSuccessesCount(),
                        "request " + i + " should be admitted pre-crash");
            }

            // Arm + trigger while both 300ms executions are still running.
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort
                            + ",\"type\":\"crash_after\",\"enabled\":true,\"n\":3}");
            EngineRpcService.EnqueueBatchResponsePB crashAck = enqueue(prefill,
                    batch(4003, slot(0, input(3, 10))));
            assertEquals(0, crashAck.getSuccessesCount());
            assertTrue(prefill.isStopped());

            // The wipe is IMMEDIATE, mid-execution: both running requests
            // vanish (a real process death takes the scheduler state with it).
            Map<String, Object> wiped = prefill.getSnapshot();
            assertEquals(0, intField(wiped, "running"),
                    "mid-execution crash wipes the running tasks");
            assertEquals(0, intField(wiped, "inflight"),
                    "mid-execution crash wipes inflight bookkeeping");
            assertEquals(0, intField(wiped, "held_blocks"),
                    "mid-execution crash releases held blocks to nothing");

            // Wait PAST the 300ms execution window: the late completion
            // callbacks must NOT resurrect anything (epoch fence).
            Thread.sleep(900);
            Map<String, Object> after = prefill.getSnapshot();
            assertEquals(0, intField(after, "running"),
                    "late completion callback must not re-add running tasks");
            assertEquals(0, intField(after, "inflight"),
                    "late completion callback must not touch inflight");
            assertTrue(((List<?>) after.get("cache_key_set")).isEmpty(),
                    "late completion callback must not hand blocks to the LRU");
            assertEquals(0, intField(after, "completed"),
                    "late completion callback must not publish completions");
            assertEquals(0, intField(after, "accepted"),
                    "counters stay at the fresh-process zero");
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Cluster bootstrap (startClusterWithGrpc pattern + setGrpcServer,
    //  which the true-crash port kill requires)
    // ════════════════════════════════════════════════════════════════

    /**
     * One prefill engine behind a REAL gRPC server (plus the control-plane
     * HTTP server). The service is pointed at its own server via
     * {@code setGrpcServer} exactly like production startEngine does —
     * without it the crash would wipe memory but never kill the port.
     */
    private int startPrefillClusterWithGrpc(MockPerformanceModel model) throws IOException {
        int port = allocatePortBlock(1);
        scheduler = Executors.newScheduledThreadPool(4, r -> {
            Thread t = new Thread(r, "crash-after-test-scheduler");
            t.setDaemon(true);
            return t;
        });
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(4);
        services = new ConcurrentHashMap<>();
        serversByPort = new ConcurrentHashMap<>();

        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                port, services, scheduler, model, 6_144,
                new JavaMockEngineCluster.ClusterStats());
        services.put(port, service);
        Server server = NettyServerBuilder.forPort(port)
                .bossEventLoopGroup(bossGroup)
                .workerEventLoopGroup(workerGroup)
                .channelType(NioServerSocketChannel.class)
                .directExecutor()
                .maxInboundMessageSize(16 * 1024 * 1024)
                .addService(service)
                .build()
                .start();
        serversByPort.put(port, server);
        service.setGrpcServer(server);

        controlServer = new MockControlServer(
                services, serversByPort, bossGroup, workerGroup, "127.0.0.1", 0);
        controlServer.start();
        return port;
    }

    private void cleanupCluster() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
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
            bossGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS);
            bossGroup = null;
        }
        if (workerGroup != null) {
            workerGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS);
            workerGroup = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(2, TimeUnit.SECONDS);
            scheduler = null;
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Helpers
    // ════════════════════════════════════════════════════════════════

    private static int intField(Map<String, Object> snapshot, String key) {
        Object value = snapshot.get(key);
        return value instanceof Number number ? number.intValue() : -1;
    }

    @SuppressWarnings("unchecked")
    private static int rpcEnqueue(Map<String, Object> snapshot) {
        Object counts = snapshot.get("rpc_counts");
        if (counts instanceof Map<?, ?> map) {
            Object value = ((Map<String, Object>) map).get("enqueue_batch");
            return value instanceof Number number ? number.intValue() : -1;
        }
        return -1;
    }

    private static void awaitInflightZero(JavaMockEngineCluster.FastRpcService service,
                                          long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        assertEquals(0, service.getInflightCount(), "inflight must drain to zero");
    }

    /**
     * Claim the next free port (same hermetic pattern as the comprehensive
     * suite's allocatePortBlock, single-engine sized).
     */
    private static int allocatePortBlock(int needed) {
        for (int attempt = 0; attempt < 20; attempt++) {
            int basePort = PORT_ALLOCATOR.getAndAdd(10);
            boolean allFree = true;
            for (int i = 0; i < needed; i++) {
                try (ServerSocket probe = new ServerSocket(basePort + i)) {
                    probe.setReuseAddress(true);
                } catch (IOException e) {
                    allFree = false;
                    break;
                }
            }
            if (allFree) {
                return basePort;
            }
        }
        throw new IllegalStateException("no bindable port after 20 attempts");
    }

    /** True when the port refuses a REAL gRPC call (UNAVAILABLE / refused). */
    private static boolean portRefusesGrpc(int port) {
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", port)
                .usePlaintext()
                .build();
        try {
            RpcServiceGrpc.RpcServiceBlockingStub stub = RpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(2, TimeUnit.SECONDS);
            stub.enqueueBatch(EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                    .setBatchId(1L)
                    .build());
            return false; // the call went through — the port is still alive
        } catch (StatusRuntimeException e) {
            return e.getStatus().getCode() == Status.Code.UNAVAILABLE;
        } finally {
            channel.shutdownNow();
        }
    }

    /** True when the port serves a REAL gRPC call. */
    private static boolean portServesGrpc(int port) {
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", port)
                .usePlaintext()
                .build();
        try {
            RpcServiceGrpc.RpcServiceBlockingStub stub = RpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(2, TimeUnit.SECONDS);
            EngineRpcService.EnqueueBatchResponsePB response = stub.enqueueBatch(
                    EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                            .setBatchId(2L)
                            .build());
            return response != null; // stopped engines answer empty acks — still "serving"
        } catch (StatusRuntimeException e) {
            return false;
        } finally {
            channel.shutdownNow();
        }
    }
}
