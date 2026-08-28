package org.flexlb.httpserver;

import ch.qos.logback.classic.Level;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.slf4j.LoggerFactory;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Root-cause regression for follower request threads blocked on a slow Master. */
class FollowerAsyncForwardingNettyTest {

    private static final int EXECUTOR_CORE_SIZE =
            Integer.getInteger("flexlb.forwarding.capacity.core", 4);
    private static final int EXECUTOR_MAX_SIZE =
            Integer.getInteger("flexlb.forwarding.capacity.max", 4);
    private static final int EXECUTOR_QUEUE_SIZE =
            Integer.getInteger("flexlb.forwarding.capacity.queue", 4);
    private static final int REQUEST_COUNT =
            Integer.getInteger("flexlb.forwarding.capacity.requests", 32);
    private static final int TARGET_QPS =
            Integer.getInteger("flexlb.forwarding.capacity.qps", 500);
    private static final long LEADER_DELAY_MS =
            Long.getLong("flexlb.forwarding.capacity.leader-delay-ms", 1000L);
    private static final long WARMUP_REQUEST_ID = 80_000L;
    private static final long FIRST_REQUEST_ID = 81_000L;

    private ch.qos.logback.classic.Logger nettyLogger;
    private ch.qos.logback.classic.Logger grpcLogger;
    private Level previousNettyLogLevel;
    private Level previousGrpcLogLevel;

    @BeforeEach
    void suppressTransportLogs() {
        nettyLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.netty");
        grpcLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("io.grpc");
        previousNettyLogLevel = nettyLogger.getLevel();
        previousGrpcLogLevel = grpcLogger.getLevel();
        nettyLogger.setLevel(Level.WARN);
        grpcLogger.setLevel(Level.WARN);
    }

    @AfterEach
    void restoreTransportLogs() {
        nettyLogger.setLevel(previousNettyLogLevel);
        grpcLogger.setLevel(previousGrpcLogLevel);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void slowMasterDoesNotOccupyOrQueueFollowerRequestThreads() throws Exception {
        try (SlowMaster master = SlowMaster.start(REQUEST_COUNT);
             Follower follower = Follower.start(master.httpAddress(), REQUEST_COUNT);
             Client client = Client.connect(follower.grpcPort())) {
            client.warmUp(WARMUP_REQUEST_ID);
            awaitCondition(
                    () -> follower.activeRequests.getCount() == 0,
                    Duration.ofSeconds(2),
                    "warm-up request token was not released");

            long trafficStartNanos = System.nanoTime();
            for (int index = 0; index < REQUEST_COUNT; index++) {
                paceUntil(trafficStartNanos, index);
                client.schedule(FIRST_REQUEST_ID + index);
            }

            try {
                boolean allRequestsForwarded =
                        master.awaitAllRequests(Duration.ofSeconds(8));
                boolean requestExecutorIdle = follower.waitForRequestExecutorIdle(
                        Duration.ofSeconds(2));
                printCapacitySummary(
                        follower, master, allRequestsForwarded, requestExecutorIdle);

                assertTrue(allRequestsForwarded,
                        () -> "slow Master received " + master.receivedRequestCount()
                                + " of " + REQUEST_COUNT
                                + " requests; follower rejections="
                                + follower.rejections.get());
                assertTrue(requestExecutorIdle,
                        () -> "follower executor did not become idle: active="
                                + follower.requestExecutor.getActiveCount()
                                + ", queued="
                                + follower.requestExecutor.getQueue().size());

                assertEquals(0, follower.rejections.get(),
                        "async forwarding must not reject follower request tasks");
                assertEquals(REQUEST_COUNT, follower.activeRequests.getCount(),
                        "pending RPCs must retain, but not prematurely release, request tokens");
                master.assertExactlyOnceAndOneHop();
                verify(follower.routeService, never()).route(any());
            } finally {
                master.releaseResponses();
            }

            assertTrue(client.awaitTerminals(Duration.ofSeconds(8)),
                    () -> "only " + client.terminalCount() + " client RPCs terminated");
            assertTrue(client.errors.isEmpty(),
                    () -> "unexpected client errors: " + client.errors);
            client.assertExactlyOnce();
            awaitCondition(
                    () -> follower.activeRequests.getCount() == 0,
                    Duration.ofSeconds(2),
                    "active request tokens were not released after Master responses");
            follower.awaitRequestExecutorIdle(Duration.ofSeconds(2));
            assertTrue(follower.waitForCallbackExecutorIdle(Duration.ofSeconds(2)),
                    "forward callback executor did not drain");
            assertEquals(0, follower.channelRejections.get());
        }
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    @EnabledIfSystemProperty(
            named = "flexlb.forwarding.capacity.benchmark",
            matches = "true")
    void benchmarkFollowerCapacityAgainstDelayedMaster() throws Exception {
        try (DelayedMaster master = DelayedMaster.start(LEADER_DELAY_MS);
             Follower follower = Follower.start(
                     master.httpAddress(), EXECUTOR_QUEUE_SIZE);
             BenchmarkClient client = BenchmarkClient.connect(
                     follower.grpcPort(), REQUEST_COUNT)) {
            client.warmUp(WARMUP_REQUEST_ID);
            awaitCondition(
                    () -> follower.activeRequests.getCount() == 0,
                    Duration.ofSeconds(2),
                    "benchmark warm-up request token was not released");

            GcSnapshot gcBefore = GcSnapshot.capture();
            long trafficStartNanos = System.nanoTime();
            long rssBeforeBytes = currentRssBytes();
            int threadsBefore = ManagementFactory.getThreadMXBean().getThreadCount();
            boolean allTerminated;
            long issuanceEndNanos;
            long terminalEndNanos;
            CapacitySampler sampler = new CapacitySampler(follower, rssBeforeBytes);
            sampler.start();
            try {
                for (int index = 0; index < REQUEST_COUNT; index++) {
                    paceUntil(trafficStartNanos, index);
                    client.schedule(FIRST_REQUEST_ID + index, index);
                }
                issuanceEndNanos = System.nanoTime();
                allTerminated = client.awaitTerminals(Duration.ofSeconds(45));
                terminalEndNanos = System.nanoTime();
            } finally {
                sampler.close();
            }

            boolean tokensDrained = waitUntil(
                    () -> follower.activeRequests.getCount() == 0,
                    Duration.ofSeconds(5));
            boolean executorDrained = follower.waitForRequestExecutorIdle(
                    Duration.ofSeconds(5));
            boolean callbackExecutorDrained = follower.waitForCallbackExecutorIdle(
                    Duration.ofSeconds(5));
            GcSnapshot gcAfter = GcSnapshot.capture();
            CapacityResult result = CapacityResult.from(
                    client,
                    master,
                    follower,
                    sampler,
                    gcBefore,
                    gcAfter,
                    trafficStartNanos,
                    issuanceEndNanos,
                    terminalEndNanos,
                    rssBeforeBytes,
                    threadsBefore,
                    allTerminated,
                    tokensDrained,
                    executorDrained,
                    callbackExecutorDrained);
            result.printJson();

            assertTrue(allTerminated,
                    () -> "benchmark left " + client.remainingTerminals()
                            + " client RPCs pending; first_error=" + client.firstError());
            assertEquals(0, master.duplicateRequests(),
                    "Master must never receive a request_id twice");
            assertEquals(0, master.invalidForwardHops(),
                    "every forwarded request must carry forward_hop=1");
            assertEquals(0, client.duplicateResponses(),
                    "client must receive at most one response per unary RPC");
            assertEquals(0, client.duplicateTerminals(),
                    "client must receive exactly one terminal callback per unary RPC");
            assertEquals(0, follower.channelRejections.get(),
                    "forward callback executor must not reject completions");
            assertTrue(tokensDrained, "benchmark leaked active request tokens");
            assertTrue(executorDrained, "follower request executor did not drain");
            assertTrue(callbackExecutorDrained,
                    "forward callback executor did not drain");
            verify(follower.routeService, never()).route(any());
        }
    }

    private static void printCapacitySummary(Follower follower,
                                             SlowMaster master,
                                             boolean allRequestsForwarded,
                                             boolean requestExecutorIdle) {
        System.out.printf(
                "Follower forwarding capacity: core=%d max=%d queue_capacity=%d qps=%d "
                        + "requests=%d master_received=%d pool_size=%d active=%d queued=%d "
                        + "rejections=%d active_tokens=%d all_forwarded=%s executor_idle=%s%n",
                EXECUTOR_CORE_SIZE,
                EXECUTOR_MAX_SIZE,
                follower.requestQueueCapacity,
                TARGET_QPS,
                REQUEST_COUNT,
                master.receivedRequestCount(),
                follower.requestExecutor.getPoolSize(),
                follower.requestExecutor.getActiveCount(),
                follower.requestExecutor.getQueue().size(),
                follower.rejections.get(),
                follower.activeRequests.getCount(),
                allRequestsForwarded,
                requestExecutorIdle);
    }

    private static void paceUntil(long trafficStartNanos, int requestIndex) {
        if (TARGET_QPS <= 0) {
            return;
        }
        long targetNanos = trafficStartNanos
                + (long) requestIndex * TimeUnit.SECONDS.toNanos(1) / TARGET_QPS;
        long remainingNanos;
        while ((remainingNanos = targetNanos - System.nanoTime()) > 0) {
            LockSupport.parkNanos(remainingNanos);
        }
    }

    private static void awaitCondition(BooleanSupplier condition,
                                       Duration timeout,
                                       String failureMessage) {
        assertTrue(waitUntil(condition, timeout), failureMessage);
    }

    private static boolean waitUntil(BooleanSupplier condition, Duration timeout) {
        long deadlineNanos = System.nanoTime() + timeout.toNanos();
        while (!condition.getAsBoolean() && System.nanoTime() < deadlineNanos) {
            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(10));
        }
        return condition.getAsBoolean();
    }

    private static long currentRssBytes() {
        Path status = Path.of("/proc/self/status");
        if (!Files.isReadable(status)) {
            return -1L;
        }
        try {
            List<String> lines = Files.readAllLines(status);
            for (String line : lines) {
                if (line.startsWith("VmRSS:")) {
                    String[] fields = line.trim().split("\\s+");
                    return Long.parseLong(fields[1]) * 1024L;
                }
            }
        } catch (Exception ignored) {
            // RSS is an optional Linux-only benchmark metric.
        }
        return -1L;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long requestId) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .setSeqLen(1024)
                .setGenerateTimeout(TimeUnit.SECONDS.toMillis(10))
                .build();
    }

    private static final class SlowMaster implements AutoCloseable {
        private final CountDownLatch slowRequests;
        private final Map<Long, AtomicInteger> requestCounts = new ConcurrentHashMap<>();
        private final Map<Long, Integer> forwardHops = new ConcurrentHashMap<>();
        private final Queue<PendingResponse> pendingResponses = new ConcurrentLinkedQueue<>();
        private final AtomicBoolean responsesReleased = new AtomicBoolean(false);
        private final Server server;

        private SlowMaster(int expectedSlowRequests) throws Exception {
            slowRequests = new CountDownLatch(expectedSlowRequests);
            server = NettyServerBuilder
                    .forAddress(new InetSocketAddress("127.0.0.1", 0))
                    .directExecutor()
                    .addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                        @Override
                        public void schedule(
                                FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB>
                                        responseObserver) {
                            if (request.getRequestId() == WARMUP_REQUEST_ID) {
                                respond(responseObserver);
                                return;
                            }
                            requestCounts.computeIfAbsent(
                                    request.getRequestId(), ignored -> new AtomicInteger())
                                    .incrementAndGet();
                            forwardHops.put(request.getRequestId(), request.getForwardHop());
                            pendingResponses.add(new PendingResponse(responseObserver));
                            slowRequests.countDown();
                            if (responsesReleased.get()) {
                                drainResponses();
                            }
                        }
                    })
                    .build()
                    .start();
        }

        static SlowMaster start(int expectedSlowRequests) throws Exception {
            return new SlowMaster(expectedSlowRequests);
        }

        String httpAddress() {
            return "127.0.0.1:"
                    + (server.getPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
        }

        boolean awaitAllRequests(Duration timeout) throws InterruptedException {
            return slowRequests.await(timeout.toMillis(), TimeUnit.MILLISECONDS);
        }

        int receivedRequestCount() {
            return requestCounts.values().stream()
                    .mapToInt(AtomicInteger::get)
                    .sum();
        }

        void assertExactlyOnceAndOneHop() {
            assertEquals(REQUEST_COUNT, requestCounts.size());
            for (long requestId = FIRST_REQUEST_ID;
                 requestId < FIRST_REQUEST_ID + REQUEST_COUNT;
                 requestId++) {
                assertEquals(1, requestCounts.getOrDefault(
                                requestId, new AtomicInteger()).get(),
                        "Master received a missing or duplicate request_id=" + requestId);
                assertEquals(1, forwardHops.getOrDefault(requestId, -1),
                        "forward_hop must be incremented exactly once for request_id="
                                + requestId);
            }
        }

        void releaseResponses() {
            responsesReleased.set(true);
            drainResponses();
        }

        private void drainResponses() {
            PendingResponse pending;
            while ((pending = pendingResponses.poll()) != null) {
                respond(pending.observer());
            }
        }

        private static void respond(
                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
            observer.onNext(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                    .setSuccess(true)
                    .setCode(200)
                    .setEnqueuedByMaster(true)
                    .build());
            observer.onCompleted();
        }

        @Override
        public void close() throws InterruptedException {
            releaseResponses();
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
        }

        private record PendingResponse(
                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        }
    }

    private static final class DelayedMaster implements AutoCloseable {
        private final long responseDelayMs;
        private final Map<Long, AtomicInteger> requestCounts = new ConcurrentHashMap<>();
        private final AtomicInteger receivedRequests = new AtomicInteger();
        private final AtomicInteger duplicateRequests = new AtomicInteger();
        private final AtomicInteger invalidForwardHops = new AtomicInteger();
        private final ScheduledExecutorService responseTimer =
                Executors.newScheduledThreadPool(4);
        private final Server server;

        private DelayedMaster(long responseDelayMs) throws Exception {
            this.responseDelayMs = responseDelayMs;
            server = NettyServerBuilder
                    .forAddress(new InetSocketAddress("127.0.0.1", 0))
                    .directExecutor()
                    .addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                        @Override
                        public void schedule(
                                FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                                StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB>
                                        responseObserver) {
                            if (request.getRequestId() == WARMUP_REQUEST_ID) {
                                SlowMaster.respond(responseObserver);
                                return;
                            }
                            int occurrences = requestCounts.computeIfAbsent(
                                            request.getRequestId(), ignored -> new AtomicInteger())
                                    .incrementAndGet();
                            receivedRequests.incrementAndGet();
                            if (occurrences > 1) {
                                duplicateRequests.incrementAndGet();
                            }
                            if (request.getForwardHop() != 1) {
                                invalidForwardHops.incrementAndGet();
                            }
                            responseTimer.schedule(
                                    () -> SlowMaster.respond(responseObserver),
                                    responseDelayMs,
                                    TimeUnit.MILLISECONDS);
                        }
                    })
                    .build()
                    .start();
        }

        static DelayedMaster start(long responseDelayMs) throws Exception {
            return new DelayedMaster(responseDelayMs);
        }

        String httpAddress() {
            return "127.0.0.1:"
                    + (server.getPort() - FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET);
        }

        int receivedRequests() {
            return receivedRequests.get();
        }

        int duplicateRequests() {
            return duplicateRequests.get();
        }

        int invalidForwardHops() {
            return invalidForwardHops.get();
        }

        @Override
        public void close() throws InterruptedException {
            responseTimer.shutdownNow();
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private static final class Follower implements AutoCloseable {
        private final AtomicInteger rejections = new AtomicInteger();
        private final AtomicInteger channelRejections = new AtomicInteger();
        private final RouteService routeService = mock(RouteService.class);
        private final ActiveRequestCounter activeRequests = new ActiveRequestCounter();
        private final int requestQueueCapacity;
        private final ThreadPoolExecutor requestExecutor;
        private final ThreadPoolExecutor channelExecutor;
        private final EventLoopGroup channelEventLoop;
        private final FlexlbGrpcForwarder forwarder;
        private final Server server;

        private Follower(String masterHttpAddress, int requestQueueCapacity) throws Exception {
            this.requestQueueCapacity = requestQueueCapacity;
            LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
            when(consistency.isNeedConsistency()).thenReturn(true);
            when(consistency.isMaster()).thenReturn(false);
            when(consistency.getMasterHostIpPort()).thenReturn(masterHttpAddress);
            when(consistency.getLocalHostIp()).thenReturn("127.0.0.2");

            ConfigService configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
            EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);
            channelEventLoop = new NioEventLoopGroup(1);
            // Match the production forwarder callback executor width so this
            // benchmark isolates the follower's inbound request executor.
            channelExecutor = new ThreadPoolExecutor(
                    16,
                    16,
                    5L,
                    TimeUnit.MINUTES,
                    new LinkedBlockingQueue<>(2000),
                    runnable -> new Thread(
                            runnable, "follower-forward-callback-test"),
                    (runnable, executor) -> {
                        channelRejections.incrementAndGet();
                        new ThreadPoolExecutor.AbortPolicy()
                                .rejectedExecution(runnable, executor);
                    });
            forwarder = new FlexlbGrpcForwarder(
                    consistency,
                    configService,
                    healthReporter,
                    channelEventLoop,
                    channelExecutor);

            FlexlbServiceImpl service = new FlexlbServiceImpl(
                    routeService,
                    consistency,
                    healthReporter,
                    activeRequests,
                    forwarder,
                    configService,
                    mock(BatchSchedulerReporter.class),
                    mock(ServerScheduleLatencyRecorder.class),
                    mock(RequestSchedulerReporter.class));

            requestExecutor = new ThreadPoolExecutor(
                    EXECUTOR_CORE_SIZE,
                    EXECUTOR_MAX_SIZE,
                    0L,
                    TimeUnit.MILLISECONDS,
                    new ArrayBlockingQueue<>(requestQueueCapacity),
                    runnable -> new Thread(runnable, "follower-root-cause-test"),
                    (runnable, executor) -> {
                        rejections.incrementAndGet();
                        new ThreadPoolExecutor.AbortPolicy()
                                .rejectedExecution(runnable, executor);
                    });
            server = NettyServerBuilder
                    .forAddress(new InetSocketAddress("127.0.0.1", 0))
                    .executor(requestExecutor)
                    .addService(service)
                    .build()
                    .start();
        }

        static Follower start(String masterHttpAddress, int requestQueueCapacity)
                throws Exception {
            return new Follower(masterHttpAddress, requestQueueCapacity);
        }

        int grpcPort() {
            return server.getPort();
        }

        void awaitRequestExecutorIdle(Duration timeout) {
            awaitCondition(
                    () -> requestExecutor.getActiveCount() == 0
                            && requestExecutor.getQueue().isEmpty(),
                    timeout,
                    "follower executor did not become idle: active="
                            + requestExecutor.getActiveCount()
                            + ", queued=" + requestExecutor.getQueue().size());
        }

        boolean waitForRequestExecutorIdle(Duration timeout) {
            return waitUntil(
                    () -> requestExecutor.getActiveCount() == 0
                            && requestExecutor.getQueue().isEmpty(),
                    timeout);
        }

        boolean waitForCallbackExecutorIdle(Duration timeout) {
            return waitUntil(
                    () -> channelExecutor.getActiveCount() == 0
                            && channelExecutor.getQueue().isEmpty(),
                    timeout);
        }

        @Override
        public void close() throws InterruptedException {
            forwarder.shutdown();
            server.shutdownNow();
            server.awaitTermination(5, TimeUnit.SECONDS);
            requestExecutor.shutdownNow();
            channelExecutor.shutdownNow();
            channelEventLoop.shutdownGracefully().sync();
        }
    }

    private static final class Client implements AutoCloseable {
        private final ManagedChannel channel;
        private final FlexlbServiceGrpc.FlexlbServiceStub asyncStub;
        private final FlexlbServiceGrpc.FlexlbServiceBlockingStub blockingStub;
        private final CountDownLatch terminals = new CountDownLatch(REQUEST_COUNT);
        private final Map<Long, AtomicInteger> responseCounts = new ConcurrentHashMap<>();
        private final Map<Long, AtomicInteger> completionCounts = new ConcurrentHashMap<>();
        private final Queue<Throwable> errors = new ConcurrentLinkedQueue<>();

        private Client(int grpcPort) {
            channel = NettyChannelBuilder
                    .forAddress("127.0.0.1", grpcPort)
                    .usePlaintext()
                    .build();
            asyncStub = FlexlbServiceGrpc.newStub(channel);
            blockingStub = FlexlbServiceGrpc.newBlockingStub(channel);
        }

        static Client connect(int grpcPort) {
            return new Client(grpcPort);
        }

        void warmUp(long requestId) {
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = blockingStub
                    .withDeadlineAfter(5, TimeUnit.SECONDS)
                    .schedule(request(requestId));
            assertTrue(response.getSuccess());
        }

        void schedule(long requestId) {
            asyncStub.withDeadlineAfter(30, TimeUnit.SECONDS)
                    .schedule(request(requestId), new StreamObserver<>() {
                        @Override
                        public void onNext(
                                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
                            responseCounts.computeIfAbsent(
                                    requestId, ignored -> new AtomicInteger())
                                    .incrementAndGet();
                        }

                        @Override
                        public void onError(Throwable error) {
                            errors.add(error);
                            terminals.countDown();
                        }

                        @Override
                        public void onCompleted() {
                            completionCounts.computeIfAbsent(
                                    requestId, ignored -> new AtomicInteger())
                                    .incrementAndGet();
                            terminals.countDown();
                        }
                    });
        }

        boolean awaitTerminals(Duration timeout) throws InterruptedException {
            return terminals.await(timeout.toMillis(), TimeUnit.MILLISECONDS);
        }

        long terminalCount() {
            return REQUEST_COUNT - terminals.getCount();
        }

        void assertExactlyOnce() {
            assertEquals(REQUEST_COUNT, responseCounts.size());
            assertEquals(REQUEST_COUNT, completionCounts.size());
            for (long requestId = FIRST_REQUEST_ID;
                 requestId < FIRST_REQUEST_ID + REQUEST_COUNT;
                 requestId++) {
                assertEquals(1, responseCounts.getOrDefault(
                                requestId, new AtomicInteger()).get(),
                        "client received a missing or duplicate response for request_id="
                                + requestId);
                assertEquals(1, completionCounts.getOrDefault(
                                requestId, new AtomicInteger()).get(),
                        "client received a missing or duplicate completion for request_id="
                                + requestId);
            }
        }

        @Override
        public void close() throws InterruptedException {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private static final class BenchmarkClient implements AutoCloseable {
        private final ManagedChannel channel;
        private final FlexlbServiceGrpc.FlexlbServiceStub asyncStub;
        private final FlexlbServiceGrpc.FlexlbServiceBlockingStub blockingStub;
        private final CountDownLatch terminals;
        private final long[] requestStartedNanos;
        private final long[] successfulLatencyNanos;
        private final AtomicIntegerArray responseClaims;
        private final AtomicIntegerArray terminalClaims;
        private final AtomicInteger terminalCount = new AtomicInteger();
        private final AtomicInteger successCount = new AtomicInteger();
        private final AtomicInteger applicationFailureCount = new AtomicInteger();
        private final AtomicInteger rpcErrorCount = new AtomicInteger();
        private final AtomicInteger duplicateResponseCount = new AtomicInteger();
        private final AtomicInteger duplicateTerminalCount = new AtomicInteger();
        private final AtomicReference<String> firstError = new AtomicReference<>("");

        private BenchmarkClient(int grpcPort, int requestCount) {
            channel = NettyChannelBuilder
                    .forAddress("127.0.0.1", grpcPort)
                    .usePlaintext()
                    .build();
            asyncStub = FlexlbServiceGrpc.newStub(channel);
            blockingStub = FlexlbServiceGrpc.newBlockingStub(channel);
            terminals = new CountDownLatch(requestCount);
            requestStartedNanos = new long[requestCount];
            successfulLatencyNanos = new long[requestCount];
            responseClaims = new AtomicIntegerArray(requestCount);
            terminalClaims = new AtomicIntegerArray(requestCount);
        }

        static BenchmarkClient connect(int grpcPort, int requestCount) {
            return new BenchmarkClient(grpcPort, requestCount);
        }

        void warmUp(long requestId) {
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = blockingStub
                    .withDeadlineAfter(5, TimeUnit.SECONDS)
                    .schedule(request(requestId));
            assertTrue(response.getSuccess());
        }

        void schedule(long requestId, int requestIndex) {
            requestStartedNanos[requestIndex] = System.nanoTime();
            long deadlineMs = Math.max(30_000L, LEADER_DELAY_MS * 5L);
            asyncStub.withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS)
                    .schedule(request(requestId), new StreamObserver<>() {
                        @Override
                        public void onNext(
                                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
                            if (!responseClaims.compareAndSet(requestIndex, 0, 1)) {
                                duplicateResponseCount.incrementAndGet();
                                return;
                            }
                            if (response.getSuccess()) {
                                successfulLatencyNanos[requestIndex] =
                                        System.nanoTime() - requestStartedNanos[requestIndex];
                                successCount.incrementAndGet();
                            } else {
                                applicationFailureCount.incrementAndGet();
                                firstError.compareAndSet("", "code=" + response.getCode()
                                        + " message=" + response.getErrorMessage());
                            }
                        }

                        @Override
                        public void onError(Throwable error) {
                            rpcErrorCount.incrementAndGet();
                            firstError.compareAndSet("", error.toString());
                            finishTerminal();
                        }

                        @Override
                        public void onCompleted() {
                            finishTerminal();
                        }

                        private void finishTerminal() {
                            if (!terminalClaims.compareAndSet(requestIndex, 0, 1)) {
                                duplicateTerminalCount.incrementAndGet();
                                return;
                            }
                            terminalCount.incrementAndGet();
                            terminals.countDown();
                        }
                    });
        }

        boolean awaitTerminals(Duration timeout) throws InterruptedException {
            return terminals.await(timeout.toMillis(), TimeUnit.MILLISECONDS);
        }

        long remainingTerminals() {
            return terminals.getCount();
        }

        int terminalCount() {
            return terminalCount.get();
        }

        int successCount() {
            return successCount.get();
        }

        int applicationFailureCount() {
            return applicationFailureCount.get();
        }

        int rpcErrorCount() {
            return rpcErrorCount.get();
        }

        int duplicateResponses() {
            return duplicateResponseCount.get();
        }

        int duplicateTerminals() {
            return duplicateTerminalCount.get();
        }

        String firstError() {
            return firstError.get();
        }

        long[] successfulLatencies() {
            return Arrays.stream(successfulLatencyNanos)
                    .filter(value -> value > 0)
                    .sorted()
                    .toArray();
        }

        @Override
        public void close() throws InterruptedException {
            channel.shutdownNow();
            channel.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private static final class CapacitySampler implements AutoCloseable {
        private final Follower follower;
        private final ScheduledExecutorService timer =
                Executors.newSingleThreadScheduledExecutor();
        private final AtomicInteger maxPoolSize = new AtomicInteger();
        private final AtomicInteger maxActive = new AtomicInteger();
        private final AtomicInteger maxQueue = new AtomicInteger();
        private final AtomicInteger maxCallbackPoolSize = new AtomicInteger();
        private final AtomicInteger maxCallbackActive = new AtomicInteger();
        private final AtomicInteger maxCallbackQueue = new AtomicInteger();
        private final AtomicInteger maxJvmThreads = new AtomicInteger();
        private final AtomicLong maxRssBytes;

        private CapacitySampler(Follower follower, long initialRssBytes) {
            this.follower = follower;
            maxRssBytes = new AtomicLong(initialRssBytes);
        }

        void start() {
            sample();
            timer.scheduleAtFixedRate(this::sample, 0, 100, TimeUnit.MILLISECONDS);
        }

        private void sample() {
            try {
                maxPoolSize.accumulateAndGet(
                        follower.requestExecutor.getPoolSize(), Math::max);
                maxActive.accumulateAndGet(
                        follower.requestExecutor.getActiveCount(), Math::max);
                maxQueue.accumulateAndGet(
                        follower.requestExecutor.getQueue().size(), Math::max);
                maxCallbackPoolSize.accumulateAndGet(
                        follower.channelExecutor.getPoolSize(), Math::max);
                maxCallbackActive.accumulateAndGet(
                        follower.channelExecutor.getActiveCount(), Math::max);
                maxCallbackQueue.accumulateAndGet(
                        follower.channelExecutor.getQueue().size(), Math::max);
                maxJvmThreads.accumulateAndGet(
                        ManagementFactory.getThreadMXBean().getThreadCount(), Math::max);
                long rssBytes = currentRssBytes();
                if (rssBytes >= 0) {
                    maxRssBytes.accumulateAndGet(rssBytes, Math::max);
                }
            } catch (RuntimeException ignored) {
                // Sampling must not perturb or abort the benchmark workload.
            }
        }

        @Override
        public void close() {
            sample();
            timer.shutdownNow();
        }
    }

    private record GcSnapshot(long collections, long collectionTimeMs) {
        static GcSnapshot capture() {
            long collections = 0;
            long collectionTimeMs = 0;
            for (GarbageCollectorMXBean collector
                    : ManagementFactory.getGarbageCollectorMXBeans()) {
                if (collector.getCollectionCount() >= 0) {
                    collections += collector.getCollectionCount();
                }
                if (collector.getCollectionTime() >= 0) {
                    collectionTimeMs += collector.getCollectionTime();
                }
            }
            return new GcSnapshot(collections, collectionTimeMs);
        }
    }

    private record CapacityResult(
            double offeredQps,
            double terminalQps,
            double successQps,
            double p95Ms,
            double p99Ms,
            double p999Ms,
            int masterReceived,
            int terminals,
            int successes,
            int applicationFailures,
            int rpcErrors,
            int duplicateResponses,
            int duplicateTerminals,
            int rejections,
            int callbackRejections,
            int maxPoolSize,
            int maxActive,
            int maxQueue,
            int finalActive,
            int finalQueue,
            int maxCallbackPoolSize,
            int maxCallbackActive,
            int maxCallbackQueue,
            int finalCallbackActive,
            int finalCallbackQueue,
            int threadsBefore,
            int maxJvmThreads,
            long rssBeforeBytes,
            long maxRssBytes,
            long gcCollections,
            long gcTimeMs,
            long finalActiveTokens,
            boolean allTerminated,
            boolean tokensDrained,
            boolean executorDrained,
            boolean callbackExecutorDrained) {

        static CapacityResult from(
                BenchmarkClient client,
                DelayedMaster master,
                Follower follower,
                CapacitySampler sampler,
                GcSnapshot gcBefore,
                GcSnapshot gcAfter,
                long trafficStartNanos,
                long issuanceEndNanos,
                long terminalEndNanos,
                long rssBeforeBytes,
                int threadsBefore,
                boolean allTerminated,
                boolean tokensDrained,
                boolean executorDrained,
                boolean callbackExecutorDrained) {
            long[] successLatencies = client.successfulLatencies();
            return new CapacityResult(
                    perSecond(REQUEST_COUNT, issuanceEndNanos - trafficStartNanos),
                    perSecond(client.terminalCount(), terminalEndNanos - trafficStartNanos),
                    perSecond(client.successCount(), terminalEndNanos - trafficStartNanos),
                    percentileMillis(successLatencies, 0.95),
                    percentileMillis(successLatencies, 0.99),
                    percentileMillis(successLatencies, 0.999),
                    master.receivedRequests(),
                    client.terminalCount(),
                    client.successCount(),
                    client.applicationFailureCount(),
                    client.rpcErrorCount(),
                    client.duplicateResponses(),
                    client.duplicateTerminals(),
                    follower.rejections.get(),
                    follower.channelRejections.get(),
                    sampler.maxPoolSize.get(),
                    sampler.maxActive.get(),
                    sampler.maxQueue.get(),
                    follower.requestExecutor.getActiveCount(),
                    follower.requestExecutor.getQueue().size(),
                    sampler.maxCallbackPoolSize.get(),
                    sampler.maxCallbackActive.get(),
                    sampler.maxCallbackQueue.get(),
                    follower.channelExecutor.getActiveCount(),
                    follower.channelExecutor.getQueue().size(),
                    threadsBefore,
                    sampler.maxJvmThreads.get(),
                    rssBeforeBytes,
                    sampler.maxRssBytes.get(),
                    Math.max(0, gcAfter.collections() - gcBefore.collections()),
                    Math.max(0, gcAfter.collectionTimeMs() - gcBefore.collectionTimeMs()),
                    follower.activeRequests.getCount(),
                    allTerminated,
                    tokensDrained,
                    executorDrained,
                    callbackExecutorDrained);
        }

        private static double perSecond(long count, long elapsedNanos) {
            return elapsedNanos <= 0
                    ? 0.0
                    : count * 1_000_000_000.0 / elapsedNanos;
        }

        private static double percentileMillis(long[] sortedNanos, double percentile) {
            if (sortedNanos.length == 0) {
                return 0.0;
            }
            int index = Math.max(
                    0,
                    (int) Math.ceil(sortedNanos.length * percentile) - 1);
            return sortedNanos[index] / 1_000_000.0;
        }

        void printJson() {
            long rssDeltaBytes = rssBeforeBytes < 0 || maxRssBytes < 0
                    ? -1L
                    : maxRssBytes - rssBeforeBytes;
            System.out.printf(Locale.ROOT,
                    "FOLLOWER_FORWARDING_CAPACITY_SUMMARY "
                            + "{\"core\":%d,\"max\":%d,\"queue_capacity\":%d,"
                            + "\"target_qps\":%d,\"leader_delay_ms\":%d,\"requests\":%d,"
                            + "\"offered_qps\":%.2f,\"terminal_qps\":%.2f,"
                            + "\"success_qps\":%.2f,\"success_p95_ms\":%.3f,"
                            + "\"success_p99_ms\":%.3f,\"success_p999_ms\":%.3f,"
                            + "\"master_received\":%d,\"terminals\":%d,\"successes\":%d,"
                            + "\"application_failures\":%d,\"rpc_errors\":%d,"
                            + "\"duplicate_responses\":%d,\"duplicate_terminals\":%d,"
                            + "\"rejections\":%d,\"callback_rejections\":%d,"
                            + "\"max_pool_size\":%d,"
                            + "\"max_active\":%d,\"max_queue\":%d,"
                            + "\"final_active\":%d,\"final_queue\":%d,"
                            + "\"max_callback_pool_size\":%d,"
                            + "\"max_callback_active\":%d,"
                            + "\"max_callback_queue\":%d,"
                            + "\"final_callback_active\":%d,"
                            + "\"final_callback_queue\":%d,"
                            + "\"threads_before\":%d,\"max_jvm_threads\":%d,"
                            + "\"thread_delta\":%d,\"rss_before_bytes\":%d,"
                            + "\"max_rss_bytes\":%d,\"rss_delta_bytes\":%d,"
                            + "\"gc_collections\":%d,\"gc_time_ms\":%d,"
                            + "\"final_active_tokens\":%d,\"all_terminated\":%s,"
                            + "\"tokens_drained\":%s,\"executor_drained\":%s,"
                            + "\"callback_executor_drained\":%s}%n",
                    EXECUTOR_CORE_SIZE,
                    EXECUTOR_MAX_SIZE,
                    EXECUTOR_QUEUE_SIZE,
                    TARGET_QPS,
                    LEADER_DELAY_MS,
                    REQUEST_COUNT,
                    offeredQps,
                    terminalQps,
                    successQps,
                    p95Ms,
                    p99Ms,
                    p999Ms,
                    masterReceived,
                    terminals,
                    successes,
                    applicationFailures,
                    rpcErrors,
                    duplicateResponses,
                    duplicateTerminals,
                    rejections,
                    callbackRejections,
                    maxPoolSize,
                    maxActive,
                    maxQueue,
                    finalActive,
                    finalQueue,
                    maxCallbackPoolSize,
                    maxCallbackActive,
                    maxCallbackQueue,
                    finalCallbackActive,
                    finalCallbackQueue,
                    threadsBefore,
                    maxJvmThreads,
                    maxJvmThreads - threadsBefore,
                    rssBeforeBytes,
                    maxRssBytes,
                    rssDeltaBytes,
                    gcCollections,
                    gcTimeMs,
                    finalActiveTokens,
                    allTerminated,
                    tokensDrained,
                    executorDrained,
                    callbackExecutorDrained);
        }
    }
}
