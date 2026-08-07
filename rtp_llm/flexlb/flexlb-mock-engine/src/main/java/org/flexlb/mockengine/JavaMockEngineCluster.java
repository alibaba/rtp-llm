package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Formula-driven Engine gRPC cluster for FlexLB control-plane and capacity tests.
 *
 * <p>Requests are queued using their input/output token shape, configured prefill formula,
 * cache hits, and decode batch curve. It models service timing and queue pressure rather
 * than GPU kernels. All listening ports share Netty event loops.
 *
 * <p>An HTTP control server ({@link MockControlServer}) provides fault injection,
 * engine stop/start, and Prometheus metrics endpoints.
 */
public final class JavaMockEngineCluster {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    /** Default KV cache token capacity per engine (Python --prefill/--decode-total-kv-tokens default). */
    static final long DEFAULT_TOTAL_KV_TOKENS = 6_291_456L;
    /** Default decode available_concurrency reported to the master (previously hard-coded 132). */
    static final int DEFAULT_DECODE_MAX_CONCURRENCY = 132;

    private JavaMockEngineCluster() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        MockPerformanceModel performance = MockPerformanceModel.load(
                config.performanceFile, config.masterConfigFile);
        if (config.blockSize > 0) {
            // Python compat: perf_cfg.setdefault("block_size", args.block_size)
            performance.setBlockSize(config.blockSize);
        }
        ClusterStats stats = new ClusterStats();
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup(config.eventLoopThreads);
        Map<Integer, Server> serversByPort = new ConcurrentHashMap<>();
        Map<Integer, FastRpcService> services = new ConcurrentHashMap<>();
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(config.completionThreads, runnable -> {
            Thread thread = new Thread(runnable, "java-mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });

        MockControlServer controlServer;
        try {
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    0, config.nPrefill, "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL);
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    config.nPrefill, config.nDecode, "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE);
            writeDiscoveryFiles(config);
            controlServer = new MockControlServer(
                    services, serversByPort, bossGroup, workerGroup, config.host, config.baseGrpcPort - 1);
            controlServer.start();
        } catch (Throwable error) {
            scheduler.shutdownNow();
            for (FastRpcService s : services.values()) s.shutdown();
            shutdown(serversByPort, bossGroup, workerGroup);
            throw error;
        }

        scheduler.scheduleAtFixedRate(() -> {
            int prefillPending = services.values().stream()
                    .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                    .mapToInt(service -> service.pendingRequests.get()).sum();
            int maxPrefillPending = services.values().stream()
                    .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                    .mapToInt(service -> service.pendingRequests.get()).max().orElse(0);
            int decodeRunning = services.values().stream()
                    .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                    .mapToInt(service -> service.activeDecodeRequests.get()).sum();
            long prefillBatches = stats.prefillBatches.sum();
            double avgBatchSize = prefillBatches == 0
                    ? 0.0 : stats.prefillBatchRequests.sum() / (double) prefillBatches;
            double avgBatchMs = prefillBatches == 0
                    ? 0.0 : stats.prefillBatchExecutionMs.sum() / (double) prefillBatches;
            Runtime runtime = Runtime.getRuntime();
            long heapUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
            long heapMaxMb = runtime.maxMemory() / (1024 * 1024);
            System.out.printf(
                    "java_mock_stats enqueue_rpcs=%d enqueued_requests=%d status_rpcs=%d cache_rpcs=%d "
                            + "prefill_batches=%d avg_batch_size=%.2f max_batch_size=%d "
                            + "avg_batch_ms=%.2f max_batch_ms=%d prefill_pending=%d "
                            + "max_prefill_pending=%d decode_running=%d heap_used_mb=%d heap_max_mb=%d "
                            + "generate_stream_rpcs=%d fetch_response_rpcs=%d cancel_rpcs=%d%n",
                    stats.enqueueRpcs.sum(), stats.enqueuedRequests.sum(),
                    stats.statusRpcs.sum(), stats.cacheRpcs.sum(),
                    prefillBatches, avgBatchSize, stats.maxPrefillBatchSize.get(),
                    avgBatchMs, stats.maxPrefillBatchExecutionMs.get(),
                    prefillPending, maxPrefillPending, decodeRunning, heapUsedMb, heapMaxMb,
                    stats.generateStreamRpcs.sum(), stats.fetchResponseRpcs.sum(), stats.cancelRpcs.sum());
        },
                5, 5, TimeUnit.SECONDS);

        scheduler.scheduleAtFixedRate(() -> {
            for (FastRpcService service : services.values()) {
                service.checkLeakDrain(60_000_000_000L);
            }
        },
                30, 30, TimeUnit.SECONDS);

        scheduler.scheduleAtFixedRate(() -> {
            for (FastRpcService service : services.values()) {
                service.periodicCleanup();
            }
        },
                60, 60, TimeUnit.SECONDS);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            controlServer.stop();
            scheduler.shutdownNow();
            for (FastRpcService service : services.values()) {
                service.shutdown();
            }
            shutdown(serversByPort, bossGroup, workerGroup);
        }, "java-mock-engine-shutdown"));

        System.out.printf("Java mock engine ready: prefill=%d decode=%d ports=%d-%d eventLoops=%d performance=%s completionThreads=%d%n",
                config.nPrefill, config.nDecode, config.baseGrpcPort,
                config.baseGrpcPort + config.nPrefill + config.nDecode - 1,
                config.eventLoopThreads, config.performanceFile, config.completionThreads);
        System.out.printf("HTTP control server listening on port %d%n", config.baseGrpcPort - 1);
        new CountDownLatch(1).await();
    }

    private static void startRole(Config config,
                                  MockPerformanceModel performance,
                                  Map<Integer, Server> serversByPort,
                                  EventLoopGroup bossGroup,
                                  EventLoopGroup workerGroup,
                                  Map<Integer, FastRpcService> services,
                                  ScheduledExecutorService scheduler,
                                  ClusterStats stats,
                                  int portOffset,
                                  int count,
                                  String roleName,
                                  EngineRpcService.RoleTypePB roleType) throws IOException {
        for (int i = 0; i < count; i++) {
            int grpcPort = config.baseGrpcPort + portOffset + i;
            int cacheCapacity = roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                    ? config.prefillCacheBlocks : config.decodeCacheBlocks;
            FastRpcService service = new FastRpcService(
                    roleName + "-" + i, config.host, roleName, roleType, grpcPort,
                    services, scheduler, performance, cacheCapacity, stats,
                    config.totalKvTokens, config.decodeMaxConcurrency);
            services.put(grpcPort, service);
            Server server = NettyServerBuilder.forPort(grpcPort)
                    .bossEventLoopGroup(bossGroup)
                    .workerEventLoopGroup(workerGroup)
                    .channelType(NioServerSocketChannel.class)
                    // SO_REUSEADDR on the listening (server) socket: allows rebind
                    // after kill while the port is in TIME_WAIT. grpc 1.65.0
                    // NettyServerBuilder.withOption() maps to ServerBootstrap.option(),
                    // which sets the option on the parent (listening) channel.
                    .withOption(ChannelOption.SO_REUSEADDR, true)
                    // SO_REUSEADDR on accepted (child) sockets — defense in depth.
                    .withChildOption(ChannelOption.SO_REUSEADDR, true)
                    .directExecutor()
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .addService(service)
                    .build()
                    .start();
            serversByPort.put(grpcPort, server);
        }
    }

    private static void shutdown(Map<Integer, Server> serversByPort,
                                 EventLoopGroup bossGroup,
                                 EventLoopGroup workerGroup) {
        for (Server server : serversByPort.values()) {
            server.shutdownNow();
        }
        bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
    }

    static void writeDiscoveryFiles(Config config) throws IOException {
        String prefillAddresses = addressList(config.host, config.baseGrpcPort, config.nPrefill);
        String decodeAddresses = addressList(
                config.host, config.baseGrpcPort + config.nPrefill, config.nDecode);

        Map<String, Object> prefillEndpoint = new LinkedHashMap<>();
        prefillEndpoint.put("address", config.prefillDomain);
        prefillEndpoint.put("protocol", "http");
        prefillEndpoint.put("path", "/");
        Map<String, Object> decodeEndpoint = new LinkedHashMap<>();
        decodeEndpoint.put("address", config.decodeDomain);
        decodeEndpoint.put("protocol", "http");
        decodeEndpoint.put("path", "/");
        Map<String, Object> roleEndpoint = new LinkedHashMap<>();
        roleEndpoint.put("group", "mock");
        roleEndpoint.put("prefill_endpoint", prefillEndpoint);
        roleEndpoint.put("decode_endpoint", decodeEndpoint);
        Map<String, Object> serviceConfig = new LinkedHashMap<>();
        serviceConfig.put("service_id", "aigc.text-generation.generation.engine_service");
        serviceConfig.put("load_balance", true);
        serviceConfig.put("role_endpoints", List.of(roleEndpoint));

        Map<String, String> env = new LinkedHashMap<>();
        env.put("MODEL_SERVICE_CONFIG", OBJECT_MAPPER.writeValueAsString(serviceConfig));
        env.put("DOMAIN_ADDRESS:" + config.prefillDomain, prefillAddresses);
        env.put("DOMAIN_ADDRESS:" + config.decodeDomain, decodeAddresses);

        List<Map<String, Object>> engines = new ArrayList<>(config.nPrefill + config.nDecode);
        addEngineRecords(engines, config, 0, config.nPrefill, "prefill");
        addEngineRecords(engines, config, config.nPrefill, config.nDecode, "decode");

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("prefill_domain", config.prefillDomain);
        payload.put("decode_domain", config.decodeDomain);
        payload.put("env", env);
        payload.put("engines", engines);

        Path endpointPath = Path.of(config.endpointFile);
        Files.createDirectories(endpointPath.toAbsolutePath().getParent());
        OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValue(endpointPath.toFile(), payload);

        if (config.envFile != null) {
            Path envPath = Path.of(config.envFile);
            Files.createDirectories(envPath.toAbsolutePath().getParent());
            List<String> lines = new ArrayList<>();
            lines.add("# Generated by JavaMockEngineCluster");
            lines.add("env \\");
            env.forEach((key, value) -> lines.add("  '" + key + "=" + value + "' \\"));
            lines.add("  <your-flexlb-api-start-command>");
            Files.write(envPath, lines);
        }
    }

    private static String addressList(String host, int firstGrpcPort, int count) {
        StringBuilder addresses = new StringBuilder(count * 20);
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                addresses.append(',');
            }
            addresses.append(host).append(':').append(firstGrpcPort + i - 1);
        }
        return addresses.toString();
    }

    private static void addEngineRecords(List<Map<String, Object>> engines,
                                         Config config,
                                         int portOffset,
                                         int count,
                                         String role) {
        for (int i = 0; i < count; i++) {
            int grpcPort = config.baseGrpcPort + portOffset + i;
            Map<String, Object> engine = new LinkedHashMap<>();
            engine.put("name", role + "-" + i);
            engine.put("role", role);
            engine.put("ip", config.host);
            engine.put("grpc_port", grpcPort);
            engine.put("http_port", grpcPort - 1);
            engine.put("grpc_addr", config.host + ":" + grpcPort);
            engine.put("http_addr", config.host + ":" + (grpcPort - 1));
            engines.add(engine);
        }
    }

    static final class FastRpcService extends RpcServiceGrpc.RpcServiceImplBase {
        /** Bound for the per-engine request_lifecycle map (Python _prune_lifecycle cap). */
        private static final int LIFECYCLE_CAP = 10_000;
        /** Bound for the cancelled_rids history exposed in the Python snapshot schema. */
        private static final int CANCELLED_RID_CAP = 10_000;
        /** Window of recent execution times used for prefill_ms/decode_ms avg+p99 (Python keeps 100). */
        private static final int RECENT_TIME_CAP = 100;

        private final String engineName;
        private final String host;
        private final String roleName;
        private final EngineRpcService.RoleTypePB roleType;
        private final int grpcPort;
        private final Map<Integer, FastRpcService> services;
        private final ScheduledExecutorService scheduler;
        private final MockPerformanceModel performance;
        private final MockLruBlockCache cache;
        private final ClusterStats stats;
        private final long totalKvTokens;
        private final int decodeMaxConcurrency;
        // Per-method RPC counters (Python _rpc_counts, snapshot "rpc_counts").
        private final AtomicLong rpcEnqueueBatch = new AtomicLong();
        private final AtomicLong rpcGenerateStream = new AtomicLong();
        private final AtomicLong rpcFetchResponse = new AtomicLong();
        private final AtomicLong rpcCancel = new AtomicLong();
        // Bounded request lifecycle map keyed by request id (Python _request_lifecycle).
        private final LinkedHashMap<Long, Map<String, Object>> requestLifecycles = new LinkedHashMap<>();
        // Bounded cancelled rid history (Python _cancelled / snapshot "cancelled_rids").
        private final LinkedHashSet<Long> cancelledRidHistory = new LinkedHashSet<>();
        // Recent execution times for snapshot prefill_ms_*/decode_ms_* fields.
        private final ArrayDeque<Double> recentPrefillTimes = new ArrayDeque<>();
        private final ArrayDeque<Double> recentDecodeTimes = new ArrayDeque<>();
        // Python /set_perf max_prefill_concurrency. When null, the legacy per-dp-rank
        // serialization is used; once explicitly configured a global lane pool of this
        // size models the Python prefill semaphore.
        private volatile int maxPrefillConcurrency = 1;
        private volatile AtomicLong[] prefillLanes = null;
        private final AtomicLong statusVersion = new AtomicLong();
        private final AtomicLong completionVersion = new AtomicLong();
        private final AtomicLong cacheVersion = new AtomicLong(1);
        private final Map<Integer, AtomicLong> nextPrefillAvailableNanosByDp = new ConcurrentHashMap<>();
        private final AtomicLong activeKvTokens = new AtomicLong();
        private final AtomicInteger pendingRequests = new AtomicInteger();
        private final AtomicInteger waitingPrefillRequests = new AtomicInteger();
        private final AtomicInteger activePrefillBatches = new AtomicInteger();
        private final AtomicInteger activeDecodeRequests = new AtomicInteger();
        // ── Decode wait queue + hard concurrency gate (change 1) ──
        // Pending decode requests waiting for a concurrency slot. Drained by the
        // decode completion callback after a running request finishes.
        // decodeMaxConcurrency becomes a real hard gate: activeDecodeRequests is
        // reserved under decodeQueueLock at admission/drain time so it can never
        // exceed the cap, and a completion hands its freed slot to one queued
        // request atomically (no lost slot, no over-admission).
        private final ArrayDeque<DecodePendingTask> decodePendingQueue = new ArrayDeque<>();
        private final Object decodeQueueLock = new Object();
        // ── Prefill batch-level wait queue (change 2) ──
        // Pending prefill batches waiting for a maxPrefillConcurrency slot. Drained
        // by the prefill completion callback. waitingPrefillRequests now reports the
        // real queued depth (queued requests) instead of lane-delayed batches.
        private final ArrayDeque<PrefillPendingBatch> prefillPendingQueue = new ArrayDeque<>();
        private final Object prefillQueueLock = new Object();
        private final ConcurrentLinkedQueue<VersionedTask> completions = new ConcurrentLinkedQueue<>();
        private final Map<Long, EngineRpcService.TaskInfoPB> runningTasks = new ConcurrentHashMap<>();
        private final Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>> responseQueues = new ConcurrentHashMap<>();
        private final Map<Long, String> requestStates = new ConcurrentHashMap<>();
        /** Safety-net TTL for cancelled markers never consumed by a completion callback. */
        private static final long CANCELLED_MARKER_TTL_SECONDS = 600;

        /** rid -> insertion time (System.nanoTime); consumed by completion callbacks. */
        private final Map<Long, Long> cancelledRequests = new ConcurrentHashMap<>();

        private volatile FaultInjectionConfig faultConfig = FaultInjectionConfig.builder().build();
        private final AtomicInteger enqueueCount = new AtomicInteger();
        private volatile boolean stopped = false;
        private final AtomicBoolean leakDetected = new AtomicBoolean(false);
        private final AtomicLong lastEnqueueTime = new AtomicLong(System.nanoTime());
        private final AtomicLong acceptedCount = new AtomicLong();
        private final AtomicLong completedCount = new AtomicLong();
        private final AtomicLong cancelledCount = new AtomicLong();
        private final ExecutorService responseExecutor;

        /** Test/default constructor: derives engine name from role+port, default KV capacity. */
        FastRpcService(String roleName,
                       EngineRpcService.RoleTypePB roleType,
                       int grpcPort,
                       Map<Integer, FastRpcService> services,
                       ScheduledExecutorService scheduler,
                       MockPerformanceModel performance,
                       int cacheCapacity,
                       ClusterStats stats) {
            this(roleName.toLowerCase() + "-" + grpcPort, "127.0.0.1", roleName, roleType,
                    grpcPort, services, scheduler, performance, cacheCapacity, stats,
                    DEFAULT_TOTAL_KV_TOKENS, DEFAULT_DECODE_MAX_CONCURRENCY);
        }

        FastRpcService(String engineName,
                       String host,
                       String roleName,
                       EngineRpcService.RoleTypePB roleType,
                       int grpcPort,
                       Map<Integer, FastRpcService> services,
                       ScheduledExecutorService scheduler,
                       MockPerformanceModel performance,
                       int cacheCapacity,
                       ClusterStats stats,
                       long totalKvTokens,
                       int decodeMaxConcurrency) {
            this.engineName = engineName;
            this.host = host;
            this.totalKvTokens = totalKvTokens;
            this.decodeMaxConcurrency = decodeMaxConcurrency;
            this.roleName = roleName.toUpperCase();
            this.roleType = roleType;
            this.grpcPort = grpcPort;
            this.services = services;
            this.scheduler = scheduler;
            this.performance = performance;
            this.cache = new MockLruBlockCache(cacheCapacity);
            this.responseExecutor = Executors.newCachedThreadPool(r -> {
                Thread thread = new Thread(r, "mock-response-poller-" + grpcPort);
                thread.setDaemon(true);
                return thread;
            });
            this.stats = stats;
        }

        @Override
        public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                 StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
            stats.enqueueRpcs.increment();
            rpcEnqueueBatch.incrementAndGet();
            EngineRpcService.EnqueueBatchResponsePB.Builder response =
                    EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());

            if (stopped) {
                observer.onNext(response.build());
                observer.onCompleted();
                return;
            }

            if (faultConfig.isFailOnEnqueue()) {
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                    for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                        response.addErrorsBuilder()
                                .setRequestId(input.getInput().getRequestId())
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorMessage(faultConfig.getEnqueueErrorMessage())
                                        .build());
                    }
                }
                observer.onNext(response.build());
                observer.onCompleted();
                return;
            }

            if (faultConfig.getQueueDepthLimit() > 0
                    && pendingRequests.get() >= faultConfig.getQueueDepthLimit()) {
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                    for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                        response.addErrorsBuilder()
                                .setRequestId(input.getInput().getRequestId())
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorMessage("queue depth limit exceeded")
                                        .build());
                    }
                }
                observer.onNext(response.build());
                observer.onCompleted();
                return;
            }

            int enqueueTotal = enqueueCount.incrementAndGet();
            if (faultConfig.getCrashAfterNRequests() > 0
                    && enqueueTotal >= faultConfig.getCrashAfterNRequests()) {
                stopped = true;
                observer.onNext(response.build());
                observer.onCompleted();
                return;
            }

            Runnable process = () -> {
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                    List<MockPerformanceModel.RequestShape> shapes = new ArrayList<>(slot.getRequestsCount());
                    for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                        stats.enqueuedRequests.increment();
                        acceptedCount.incrementAndGet();
                        long requestId = input.getInput().getRequestId();
                        response.addSuccessesBuilder().setRequestId(requestId);
                        shapes.add(performance.shape(input.getInput(), cache));
                        responseQueues.computeIfAbsent(requestId, k -> new LinkedBlockingQueue<>());
                        requestStates.put(requestId, "running");
                        recordLifecycleStart(requestId, request.getBatchId(), "enqueue_batch");
                    }
                    schedulePrefillCompletion(shapes, request.getBatchId(), slot.getDpRank());
                }
                observer.onNext(response.build());
                observer.onCompleted();
            };

            lastEnqueueTime.set(System.nanoTime());

            if (faultConfig.getEnqueueDelayMs() > 0) {
                scheduler.schedule(process, faultConfig.getEnqueueDelayMs(), TimeUnit.MILLISECONDS);
            } else {
                process.run();
            }
        }

        @Override
        public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                    StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
            stats.statusRpcs.increment();
            long requestedVersion = request.getLatestFinishedVersion();
            VersionedTask head;
            while ((head = completions.peek()) != null && head.version <= requestedVersion) {
                completions.poll();
            }
            long latestVersion = completionVersion.get();
            long runningCount = runningTasks.values().stream()
                    .filter(task -> task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .count();
            long usedKv = Math.min(totalKvTokens, activeKvTokens.get() + faultConfig.getKvPressureTokens());
            EngineRpcService.WorkerStatusPB.Builder status = EngineRpcService.WorkerStatusPB.newBuilder()
                    .setAlive(!stopped)
                    .setRole(roleName)
                    .setRoleType(roleType)
                    .setAvailableConcurrency(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                            // Python worker_status: max(0, _max_prefill_concurrency - len(_running))
                            ? Math.max(0, maxPrefillConcurrency - activePrefillBatches.get())
                            : Math.max(0, decodeMaxConcurrency - (int) runningCount))
                    // waitingQueryLen now reports the real pending-queue depth for
                    // BOTH roles: prefill queued requests (waitingPrefillRequests)
                    // and decode queued requests (decodePendingQueue size). Previously
                    // decode always reported 0. Reuses the existing proto field.
                    .setWaitingQueryLen(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                            ? waitingPrefillRequests.get() : decodePendingQueueSize())
                    .setRunningQueryLen((int) runningCount)
                    .setAvailableKvCache(totalKvTokens - usedKv)
                    .setTotalKvCache(totalKvTokens)
                    .setStatusVersion(statusVersion.incrementAndGet())
                    .setLatestFinishedVersion(latestVersion)
                    .setDpSize(1)
                    .setTpSize(1)
                    .setDpRank(0);
            status.addAllRunningTaskInfo(runningTasks.values());
            for (VersionedTask completion : completions) {
                if (completion.version > requestedVersion && completion.version <= latestVersion) {
                    status.addFinishedTaskList(completion.task);
                }
            }
            observer.onNext(status.build());
            observer.onCompleted();
        }

        @Override
        public void generateStreamCall(EngineRpcService.GenerateInputPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            stats.generateStreamRpcs.increment();
            rpcGenerateStream.incrementAndGet();

            if (faultConfig.isGenerateError()) {
                observer.onError(new RuntimeException("injected generate_error"));
                return;
            }
            // Python compat: inject_config["enqueue_error"] also makes generate_stream
            // raise (the Python mock checked enqueue_error in generate_stream too).
            if (faultConfig.isFailOnEnqueue()) {
                observer.onError(new RuntimeException("injected enqueue_error"));
                return;
            }
            if (faultConfig.isNoRespond()) {
                return;
            }
            if (stopped) {
                observer.onError(new RuntimeException("engine stopped"));
                return;
            }

            long requestId = request.getRequestId();
            MockPerformanceModel.RequestShape shape = performance.shape(request, cache);
            acceptedCount.incrementAndGet();
            lastEnqueueTime.set(System.nanoTime());
            requestStates.put(requestId, "running");
            recordLifecycleStart(requestId, -1, "generate_stream");

            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    responseQueues.computeIfAbsent(requestId, k -> new LinkedBlockingQueue<>());

            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                if (!scheduleDecodeCompletion(shape, -1, queue)) {
                    // Backpressure: decode pending queue full — reject so the
                    // caller (master/client) perceives decode overload and can
                    // retry elsewhere. Clean up the per-request state set up above.
                    responseQueues.remove(requestId);
                    requestStates.put(requestId, "rejected");
                    observer.onError(new RuntimeException("decode queue full (backpressure)"));
                    return;
                }
            } else {
                schedulePrefillCompletion(List.of(shape), -1, 0);
            }

            // Use a separate executor for blocking poll to avoid starving the
            // completion scheduler which is responsible for producing responses.
            responseExecutor.execute(() -> {
                try {
                    EngineRpcService.GenerateOutputsPB output = queue.poll(60, TimeUnit.SECONDS);
                    if (output != null) {
                        observer.onNext(output);
                    }
                    observer.onCompleted();
                } catch (InterruptedException e) {
                    observer.onError(e);
                }
            });
        }

        @Override
        public void fetchResponse(EngineRpcService.FetchRequestPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            stats.fetchResponseRpcs.increment();
            rpcFetchResponse.incrementAndGet();

            long requestId = request.getRequestId();

            if (faultConfig.isFetchError()) {
                observer.onNext(EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(requestId)
                        .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder()
                                .addFinished(false)
                                .build())
                        .build());
                observer.onError(new RuntimeException("injected fetch_error"));
                return;
            }

            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    responseQueues.computeIfAbsent(requestId, k -> new LinkedBlockingQueue<>());

            // Use a separate executor for blocking poll to avoid starving the
            // completion scheduler which is responsible for producing responses.
            responseExecutor.execute(() -> {
                try {
                    EngineRpcService.GenerateOutputsPB output = queue.poll(60, TimeUnit.SECONDS);
                    if (output != null) {
                        observer.onNext(output);
                    }
                    observer.onCompleted();
                } catch (InterruptedException e) {
                    observer.onError(e);
                }
            });
        }

        void cancel(long requestId) {
            stats.cancelRpcs.increment();
            rpcCancel.incrementAndGet();
            cancelledRequests.put(requestId, System.nanoTime());
            addCancelledRid(requestId);
            recordLifecycleEnd(requestId, true);
            // If the request is still parked in the decode pending queue (not yet
            // running), it has NOT been counted in activeDecodeRequests nor
            // activeKvTokens, so those must not be decremented. removeIf under the
            // admission lock atomically determines queued-vs-running: if the task is
            // still in the queue it is removed here (wasQueued=true); if it was
            // already drained into a running slot, removeIf finds nothing and the
            // request is treated as running (decrement activeDecodeRequests below).
            boolean wasQueuedDecode = false;
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                synchronized (decodeQueueLock) {
                    wasQueuedDecode = decodePendingQueue.removeIf(
                            t -> t.shape().input().getRequestId() == requestId);
                }
            }
            // For prefill, partial batches are left in the queue (other requests in
            // the batch may still be alive); the drain's anyAlive check drops fully-
            // cancelled batches and the completion's !alreadyCancelled guard
            // prevents double-decrement of pendingRequests for cancelled members.
            DecodePendingTask drainNext = null;
            EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
            if (removed != null) {
                pendingRequests.decrementAndGet();
                if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE && !wasQueuedDecode) {
                    activeDecodeRequests.decrementAndGet();
                    activeKvTokens.addAndGet(-removed.getInputLength());
                    // Drain the queue to hand the freed slot to a queued request.
                    // Without this, a cancelled running request's slot would sit
                    // idle until the next completion fires, stalling queued
                    // requests. Reuses the same skip-cancelled + pollFirst +
                    // activeDecodeRequests++ pattern as the completion drain.
                    synchronized (decodeQueueLock) {
                        while (!decodePendingQueue.isEmpty()) {
                            DecodePendingTask candidate = decodePendingQueue.peekFirst();
                            if (!runningTasks.containsKey(candidate.shape().input().getRequestId())) {
                                decodePendingQueue.pollFirst();
                                continue;
                            }
                            activeDecodeRequests.incrementAndGet();
                            drainNext = decodePendingQueue.pollFirst();
                            break;
                        }
                    }
                }
            }
            requestStates.put(requestId, "cancelled");
            cancelledCount.incrementAndGet();
            long version = completionVersion.incrementAndGet();
            EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(requestId)
                    .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(EngineRpcService.ErrorCodePB.CANCELLED.getNumber())
                            .setErrorMessage("cancelled by client")
                            .build())
                    .setEndTimeMs(System.currentTimeMillis())
                    .setDpRank(0)
                    .build();
            completions.add(new VersionedTask(version, task));
            statusVersion.incrementAndGet();
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue = responseQueues.get(requestId);
            if (queue != null) {
                queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(requestId)
                        .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                .setErrorCode(EngineRpcService.ErrorCodePB.CANCELLED)
                                .setErrorMessage("cancelled by client")
                                .build())
                        .build());
                // The poller already holds a reference to the queue, so it is safe
                // to remove it from the map after offering the cancel response.
                responseQueues.remove(requestId);
            }
            // Run the next queued decode request outside the lock (slot already
            // reserved by the drain above) to avoid holding decodeQueueLock while
            // scheduling.
            if (drainNext != null) {
                lastEnqueueTime.set(System.nanoTime());
                runDecode(drainNext.shape(), drainNext.batchId(), drainNext.responseQueue());
            }
        }

        private void schedulePrefillCompletion(List<MockPerformanceModel.RequestShape> shapes,
                                               long batchId,
                                               int dpRank) {
            if (shapes.isEmpty()) {
                return;
            }
            // Hard gate on maxPrefillConcurrency (change 2): a batch is either admitted
            // immediately (occupies a concurrency slot) or parked in the pending
            // queue until a running batch finishes. activePrefillBatches is reserved
            // under prefillQueueLock at admission/drain time so admission and the
            // completion-callback drain cannot over-admit into the same slot.
            synchronized (prefillQueueLock) {
                if (activePrefillBatches.get() < maxPrefillConcurrency) {
                    activePrefillBatches.incrementAndGet();
                    pendingRequests.addAndGet(shapes.size());
                    for (MockPerformanceModel.RequestShape shape : shapes) {
                        runningTasks.put(shape.input().getRequestId(),
                                task(shape, batchId, dpRank,
                                        EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    }
                } else {
                    prefillPendingQueue.addLast(new PrefillPendingBatch(shapes, batchId, dpRank));
                    waitingPrefillRequests.addAndGet(shapes.size());
                    pendingRequests.addAndGet(shapes.size());
                    for (MockPerformanceModel.RequestShape shape : shapes) {
                        runningTasks.put(shape.input().getRequestId(),
                                task(shape, batchId, dpRank,
                                        EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    }
                    return;
                }
            }
            runPrefillBatch(shapes, batchId, dpRank);
        }

        /**
         * Execute a prefill batch through the lane time-axis serialization and
         * schedule its completion. The concurrency slot is already reserved by the
         * caller (schedulePrefillCompletion admission or pending-queue drain), so
         * this method must NOT touch activePrefillBatches, pendingRequests, nor
         * waitingPrefillRequests — only lane timing, runningTask phase, and the
         * completion callback (which frees the slot and drains the queue).
         */
        private void runPrefillBatch(List<MockPerformanceModel.RequestShape> shapes,
                                     long batchId,
                                     int dpRank) {
            long executionMs = performance.prefillMs(shapes);
            long generateDelayMs = faultConfig.getGenerateDelayMs();
            long now = System.nanoTime();
            long executionNanos = TimeUnit.MILLISECONDS.toNanos(executionMs + generateDelayMs);
            // When max_prefill_concurrency was explicitly configured via /set_perf, a
            // global lane pool models the Python prefill semaphore; otherwise keep the
            // legacy per-dp-rank serialization.
            AtomicLong nextAvailable = pickPrefillLane(dpRank);
            long startNanos;
            long finishNanos;
            while (true) {
                long previous = nextAvailable.get();
                startNanos = Math.max(now, previous);
                finishNanos = startNanos + executionNanos;
                if (nextAvailable.compareAndSet(previous, finishNanos)) {
                    break;
                }
            }

            stats.recordPrefillBatch(shapes.size(), executionMs);
            long startDelayNanos = Math.max(0, startNanos - now);
            if (startDelayNanos == 0) {
                startPrefillBatch(shapes, batchId, dpRank);
            } else {
                scheduler.schedule(() -> startPrefillBatch(shapes, batchId, dpRank),
                        startDelayNanos, TimeUnit.NANOSECONDS);
            }

            long delayNanos = Math.max(0, finishNanos - now);
            scheduler.schedule(() -> {
                int activeCount = 0;
                for (MockPerformanceModel.RequestShape shape : shapes) {
                    long requestId = shape.input().getRequestId();
                    boolean alreadyCancelled = cancelledRequests.containsKey(requestId);
                    EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
                    // Only count non-cancelled requests toward pendingRequests
                    // decrement. A cancelled member was re-put to RUNNING by
                    // startPrefillBatch (which loops all shapes), so removed!=null
                    // alone would double-decrement pendingRequests (cancel already
                    // decremented it). The !alreadyCancelled guard fixes this.
                    if (removed != null && !alreadyCancelled) {
                        activeCount++;
                    }
                    recordCompletion(shape, batchId, executionMs, dpRank);
                    // Python marks the prefill-side lifecycle entry finished when the
                    // prefill phase ends, even though decode may continue elsewhere.
                    recordLifecycleEnd(requestId, alreadyCancelled);
                    // Python compat (_run_prefill_batch): an engine
                    // with inject_config["no_respond"] completes its own work but
                    // never queues responses nor hands off to decode, so the client
                    // stream hangs until it times out.
                    boolean decodeStarted = false;
                    if (!alreadyCancelled && !faultConfig.isNoRespond()) {
                        decodeStarted = startDecode(shape, batchId);
                    }
                    if (!decodeStarted) {
                        if (!alreadyCancelled) {
                            completedCount.incrementAndGet();
                            requestStates.put(requestId, "completed");
                        }
                        if (!faultConfig.isNoRespond()) {
                            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                                    responseQueues.get(requestId);
                            if (queue != null && !alreadyCancelled) {
                                queue.offer(buildOutput(shape, true));
                            }
                        }
                        // Clean up per-request state to prevent unbounded map growth
                        responseQueues.remove(requestId);
                        cancelledRequests.remove(requestId);
                    }
                    if (performance.shouldAdmitCache() && cache.admit(shape.blockKeys())) {
                        cacheVersion.incrementAndGet();
                    }
                }
                activePrefillBatches.decrementAndGet();
                pendingRequests.addAndGet(-activeCount);
                // Drain one pending batch under the same lock that guards admission,
                // handing this completion's freed slot to a queued batch atomically.
                PrefillPendingBatch nextBatch = null;
                synchronized (prefillQueueLock) {
                    while (!prefillPendingQueue.isEmpty()) {
                        PrefillPendingBatch candidate = prefillPendingQueue.peekFirst();
                        // Skip batches whose every member was cancelled while queued
                        // (cancel removed their runningTasks entries and already
                        // decremented pendingRequests). Drop them without reserving.
                        boolean anyAlive = false;
                        for (MockPerformanceModel.RequestShape s : candidate.shapes()) {
                            if (runningTasks.containsKey(s.input().getRequestId())) {
                                anyAlive = true;
                                break;
                            }
                        }
                        if (!anyAlive) {
                            prefillPendingQueue.pollFirst();
                            waitingPrefillRequests.addAndGet(-candidate.shapes().size());
                            continue;
                        }
                        activePrefillBatches.incrementAndGet();
                        waitingPrefillRequests.addAndGet(-candidate.shapes().size());
                        nextBatch = prefillPendingQueue.pollFirst();
                        break;
                    }
                }
                if (nextBatch != null) {
                    runPrefillBatch(nextBatch.shapes(), nextBatch.batchId(), nextBatch.dpRank());
                }
            }, delayNanos, TimeUnit.NANOSECONDS);
        }

        private void startPrefillBatch(List<MockPerformanceModel.RequestShape> shapes,
                                       long batchId,
                                       int dpRank) {
            // activePrefillBatches is reserved at admission (schedulePrefillCompletion)
            // and drain time, not here, so maxPrefillConcurrency acts as a real hard
            // gate rather than a report-only value.
            for (MockPerformanceModel.RequestShape shape : shapes) {
                runningTasks.put(shape.input().getRequestId(),
                        task(shape, batchId, dpRank, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
            }
        }

        private boolean startDecode(MockPerformanceModel.RequestShape shape, long batchId) {
            EngineRpcService.GenerateInputPB input = shape.input();
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    responseQueues.get(input.getRequestId());
            for (EngineRpcService.RoleAddrPB addr : input.getGenerateConfig().getRoleAddrsList()) {
                if (addr.getRoleType() != EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                    continue;
                }
                FastRpcService decode = services.get(addr.getGrpcPort());
                if (decode != null && decode.grpcPort != grpcPort) {
                    // Propagate the decode admission result: true = admitted/queued,
                    // false = backpressure (decode pending queue full). On false the
                    // caller treats decodeStarted=false and delivers the finished
                    // output from the prefill side (degraded but no request lost).
                    return decode.scheduleDecodeCompletion(shape, batchId, queue);
                }
                return false;
            }
            return false;
        }

        /**
         * Admission point for a decode request. Returns true if the request was
         * accepted (scheduled immediately, queued behind the concurrency gate, or
         * already scheduled previously); false if rejected due to pending-queue
         * backpressure (RESOURCE_EXHAUSTED) so the caller can degrade.
         *
         * <p>The putIfAbsent guard (ConcurrentDoubleSchedulingTest) is preserved at
         * the entry to prevent double-scheduling the same requestId.
         */
        private boolean scheduleDecodeCompletion(MockPerformanceModel.RequestShape shape, long batchId,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue) {
            long requestId = shape.input().getRequestId();
            // Guard: never schedule the same requestId twice on this engine.
            EngineRpcService.TaskInfoPB existing = runningTasks.putIfAbsent(
                    requestId,
                    task(shape, batchId, 0, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
            if (existing != null) {
                return true; // already accepted/scheduled on this engine
            }

            int pendingCap = effectiveDecodePendingCap();
            synchronized (decodeQueueLock) {
                if (activeDecodeRequests.get() < decodeMaxConcurrency) {
                    // Free slot — reserve it now (under the lock) so a concurrent
                    // completion-callback drain cannot admit another request into
                    // the same slot. runDecode does the actual scheduling outside
                    // the lock; activeDecodeRequests is already accounted here.
                    // pendingRequests is incremented here to match the unconditional
                    // decrement in the completion callback (wasRunning path). The
                    // queued path increments it at enqueue; both must balance so
                    // checkLeakDrain / periodicCleanup see net zero.
                    activeDecodeRequests.incrementAndGet();
                    pendingRequests.incrementAndGet();
                } else if (pendingCap <= 0 || decodePendingQueue.size() < pendingCap) {
                    // Concurrency gate hit — park the request in the pending queue.
                    // It will be drained when a running request completes.
                    decodePendingQueue.addLast(new DecodePendingTask(shape, batchId, responseQueue));
                    pendingRequests.incrementAndGet();
                    recordLifecycleStart(requestId, batchId,
                            batchId >= 0 ? "enqueue_batch" : "generate_stream");
                    lastEnqueueTime.set(System.nanoTime());
                    return true; // queued (accepted, will run when a slot frees)
                } else {
                    // Backpressure: pending queue full — reject so the caller
                    // (master / prefill hand-off) perceives decode overload.
                    // Undo the runningTasks claim made by putIfAbsent above.
                    runningTasks.remove(requestId);
                    return false;
                }
            }
            // Admitted immediately — record lifecycle and run (outside the lock so
            // scheduler.schedule never blocks the admission/drain critical section).
            recordLifecycleStart(requestId, batchId,
                    batchId >= 0 ? "enqueue_batch" : "generate_stream");
            lastEnqueueTime.set(System.nanoTime());
            runDecode(shape, batchId, responseQueue);
            return true;
        }

        /**
         * Actually schedule a decode completion. The concurrency slot is already
         * reserved by the caller (scheduleDecodeCompletion admission or the
         * pending-queue drain), so this method must NOT touch activeDecodeRequests
         * — only activeKvTokens (modelled when the request starts running).
         */
        private void runDecode(MockPerformanceModel.RequestShape shape, long batchId,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue) {
            activeKvTokens.addAndGet(shape.inputLen());
            int activeBatch = activeDecodeRequests.get();
            long executionMs = performance.decodeMs(shape.outputLen(), activeBatch);
            scheduler.schedule(() -> {
                long requestId = shape.input().getRequestId();
                EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
                boolean wasRunning = removed != null;
                // Free the slot and drain one pending request under the same lock
                // that guards admission, so the freed slot is handed to a queued
                // request atomically (no lost slot, no over-admission). Skip
                // queued requests cancelled while queued (cancel removed their
                // runningTasks entry and already decremented pendingRequests).
                DecodePendingTask nextPending = null;
                synchronized (decodeQueueLock) {
                    if (wasRunning) {
                        activeDecodeRequests.decrementAndGet();
                        activeKvTokens.addAndGet(-shape.inputLen());
                        pendingRequests.decrementAndGet();
                        // Only drain when this completion actually freed a slot.
                        // If wasRunning=false the request was cancelled before the
                        // completion fired; cancel() already released the slot and
                        // drained the queue, so draining here would over-admit
                        // beyond decodeMaxConcurrency.
                        while (!decodePendingQueue.isEmpty()) {
                            DecodePendingTask candidate = decodePendingQueue.peekFirst();
                            if (!runningTasks.containsKey(candidate.shape().input().getRequestId())) {
                                decodePendingQueue.pollFirst();
                                continue;
                            }
                            activeDecodeRequests.incrementAndGet();
                            nextPending = decodePendingQueue.pollFirst();
                            break;
                        }
                    }
                }
                recordCompletion(shape, batchId, executionMs, 0);
                boolean alreadyCancelled = cancelledRequests.containsKey(requestId);
                recordLifecycleEnd(requestId, alreadyCancelled);
                if (!alreadyCancelled) {
                    completedCount.incrementAndGet();
                    requestStates.put(requestId, "completed");
                }
                // Python compat (_run_decode): no_respond on the
                // decode engine only suppresses the intermediate first-step output;
                // the finished output is still delivered, so keep this unconditional.
                if (responseQueue != null && !alreadyCancelled) {
                    responseQueue.offer(buildOutput(shape, true));
                }
                // Clean up per-request state to prevent unbounded map growth
                responseQueues.remove(requestId);
                cancelledRequests.remove(requestId);
                if (performance.shouldAdmitCache() && cache.admit(shape.blockKeys())) {
                    cacheVersion.incrementAndGet();
                }
                // Run the next queued decode request (slot already reserved).
                if (nextPending != null) {
                    lastEnqueueTime.set(System.nanoTime());
                    runDecode(nextPending.shape(), nextPending.batchId(),
                            nextPending.responseQueue());
                }
            }, executionMs, TimeUnit.MILLISECONDS);
        }

        /**
         * Effective pending-queue capacity for decode backpressure. Reuses the
         * fault-injection {@code queue_depth_limit} when configured (>0); otherwise
         * falls back to a sane default so the queue acts as a real backpressure
         * gate (2× decode concurrency, at least 256). /set_queue_depth can still
         * override at runtime.
         */
        private int effectiveDecodePendingCap() {
            int qdl = faultConfig.getQueueDepthLimit();
            return qdl > 0 ? qdl : Math.max(256, decodeMaxConcurrency * 2);
        }

        /** Snapshot size of the decode pending queue (for waitingQueryLen reporting). */
        private int decodePendingQueueSize() {
            synchronized (decodeQueueLock) {
                return decodePendingQueue.size();
            }
        }

        private EngineRpcService.TaskInfoPB task(MockPerformanceModel.RequestShape shape,
                                                 long batchId,
                                                 int dpRank,
                                                 EngineRpcService.TaskPhase phase) {
            return EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(shape.input().getRequestId())
                    .setInputLength(shape.inputLen())
                    .setPrefixLength(shape.hitTokens())
                    .setBatchId(batchId)
                    .setPhase(phase)
                    .setDpRank(dpRank)
                    .build();
        }

        private void recordCompletion(MockPerformanceModel.RequestShape shape,
                                      long batchId,
                                      long executionMs,
                                      int dpRank) {
            recordRecentExecutionTime(executionMs);
            long version = completionVersion.incrementAndGet();
            EngineRpcService.TaskInfoPB task = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(shape.input().getRequestId())
                    .setInputLength(shape.inputLen())
                    .setPrefixLength(shape.hitTokens())
                    .setBatchId(batchId)
                    .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .setEndTimeMs(System.currentTimeMillis())
                    .setExecutionTimeMs(executionMs)
                    .setIterateCount(1)
                    .setDpRank(dpRank)
                    .build();
            completions.add(new VersionedTask(version, task));
        }

        private EngineRpcService.GenerateOutputsPB buildOutput(MockPerformanceModel.RequestShape shape,
                                                               boolean finished) {
            return EngineRpcService.GenerateOutputsPB.newBuilder()
                    .setRequestId(shape.input().getRequestId())
                    .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder()
                            .addFinished(finished)
                            .addAuxInfo(EngineRpcService.AuxInfoPB.newBuilder()
                                    .setInputLen(shape.inputLen())
                                    .setPrefixLen((int) shape.hitTokens())
                                    .setOutputLen(shape.outputLen())
                                    .setIterCount(1)
                                    .setStepOutputLen(shape.outputLen())
                                    .build())
                            .build())
                    .build();
        }

        @Override
        public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                   StreamObserver<EngineRpcService.CacheStatusPB> observer) {
            stats.cacheRpcs.increment();
            long usedKv = Math.min(totalKvTokens, activeKvTokens.get());
            EngineRpcService.CacheStatusPB.Builder status = EngineRpcService.CacheStatusPB.newBuilder()
                    .setAvailableKvCache(totalKvTokens - usedKv)
                    .setTotalKvCache(totalKvTokens)
                    .setBlockSize(performance.blockSize())
                    .setVersion(cacheVersion.get());
            if (request.getNeedCacheKeys()) {
                for (Long key : cache.snapshotKeys()) {
                    status.putCacheKeys(key, true);
                }
            }
            observer.onNext(status.build());
            observer.onCompleted();
        }

        @Override
        public void checkHealth(EngineRpcService.EmptyPB request,
                                StreamObserver<EngineRpcService.CheckHealthResponsePB> observer) {
            observer.onNext(EngineRpcService.CheckHealthResponsePB.newBuilder().setHealth("OK").build());
            observer.onCompleted();
        }

        void checkLeakDrain(long graceWindowNanos) {
            long timeSinceLastEnqueue = System.nanoTime() - lastEnqueueTime.get();
            if (timeSinceLastEnqueue < graceWindowNanos) {
                return;
            }
            int pending = pendingRequests.get();
            int running = runningTasks.size();
            int activeDecode = activeDecodeRequests.get();
            if (pending != 0 || running != 0 || activeDecode != 0) {
                leakDetected.set(true);
                System.err.printf("LEAK DETECTED on engine %s (port %d): pending=%d running=%d activeDecode=%d%n",
                        roleName, grpcPort, pending, running, activeDecode);
            }
        }

        /**
         * Remove orphaned entries from responseQueues, requestStates, and cancelledRequests
         * for requestIds that are no longer in runningTasks. This is a safety net for
         * entries that were not cleaned up by the completion or cancel callbacks.
         */
        void periodicCleanup() {
            Set<Long> activeIds = runningTasks.keySet();
            // Only prune responseQueues when there are no pending requests. A queue
            // may still be awaiting decode output from another engine even after the
            // local runningTasks entry has been removed (prefill completed, decode in
            // flight). Using pendingRequests > 0 as the guard prevents premature
            // removal of queues that are still being polled by fetchResponse or
            // generateStreamCall.
            if (pendingRequests.get() == 0) {
                responseQueues.keySet().retainAll(activeIds);
            }
            requestStates.keySet().retainAll(activeIds);
            // NOTE: cancelledRequests must NOT be pruned against runningTasks —
            // cancel() removes the runningTasks entry first, so retainAll would
            // drop the marker before the completion callback observes it and a
            // cancelled request could still be forwarded to decode / counted as
            // completed. Completion callbacks remove their own entries; TTL is
            // only a safety net for orphaned markers.
            long ttlDeadline = System.nanoTime()
                    - TimeUnit.SECONDS.toNanos(CANCELLED_MARKER_TTL_SECONDS);
            cancelledRequests.entrySet().removeIf(e -> e.getValue() < ttlDeadline);
        }

        /**
         * Shut down the dedicated response-polling executor.
         */
        void shutdown() {
            responseExecutor.shutdownNow();
        }

        // ──────────── Getters and setters for MockControlServer ────────────

        FaultInjectionConfig getFaultConfig() { return faultConfig; }
        void setFaultConfig(FaultInjectionConfig config) { this.faultConfig = config; }
        void clearFaultConfig() { this.faultConfig = FaultInjectionConfig.builder().build(); }
        void resetEnqueueCount() { this.enqueueCount.set(0); }
        void setStopped(boolean s) { this.stopped = s; }
        boolean isStopped() { return stopped; }
        int getGrpcPort() { return grpcPort; }
        String getRoleName() { return roleName; }
        String getEngineName() { return engineName; }
        String getHost() { return host; }
        long getTotalKvTokens() { return totalKvTokens; }
        int getMaxPrefillConcurrency() { return maxPrefillConcurrency; }
        MockPerformanceModel getPerformance() { return performance; }
        int getRunningCount() { return runningTasks.size(); }
        int getWaitingCount() { return waitingPrefillRequests.get(); }
        long getAcceptedCount() { return acceptedCount.get(); }
        long getCompletedCount() { return completedCount.get(); }
        long getCancelledCount() { return cancelledCount.get(); }
        long getActiveKvTokens() { return activeKvTokens.get(); }
        long getCacheKeyCount() { return cache.snapshotKeys().size(); }
        long getCacheEvictions() { return cache.evictions(); }
        boolean isLeakDetected() { return leakDetected.get(); }
        Map<Long, String> getRequestStates() { return requestStates; }

        /**
         * Python /set_kv_pressure uses ABSOLUTE active_kv_tokens semantics
         * (Python semantics: state._active_kv_tokens = value). The Java engine models
         * pressure as an additive fault-config term on top of live decode tokens, so
         * convert the requested absolute value into the equivalent additive pressure.
         */
        void setAbsoluteActiveKvTokens(long absoluteTokens) {
            long pressure = Math.max(0, absoluteTokens - activeKvTokens.get());
            faultConfig = faultConfig.toBuilder().kvPressureTokens(pressure).build();
            statusVersion.incrementAndGet();
        }

        /**
         * Python /set_perf {@code max_prefill_concurrency}: replaces the prefill
         * semaphore (Python-compat /set_perf). Activates a global lane pool of
         * {@code n} lanes modelling concurrent prefill execution, and updates the
         * reported available_concurrency accordingly.
         */
        void setMaxPrefillConcurrency(int n) {
            int lanes = Math.max(1, n);
            AtomicLong[] pool = new AtomicLong[lanes];
            for (int i = 0; i < lanes; i++) {
                pool[i] = new AtomicLong(0);
            }
            this.prefillLanes = pool;
            this.maxPrefillConcurrency = lanes;
            statusVersion.incrementAndGet();
        }

        private AtomicLong pickPrefillLane(int dpRank) {
            AtomicLong[] lanes = this.prefillLanes;
            if (lanes == null) {
                return nextPrefillAvailableNanosByDp.computeIfAbsent(
                        dpRank, ignored -> new AtomicLong());
            }
            synchronized (lanes) {
                AtomicLong best = lanes[0];
                for (AtomicLong lane : lanes) {
                    if (lane.get() < best.get()) {
                        best = lane;
                    }
                }
                return best;
            }
        }

        private void addCancelledRid(long requestId) {
            synchronized (cancelledRidHistory) {
                if (cancelledRidHistory.add(requestId) && cancelledRidHistory.size() > CANCELLED_RID_CAP) {
                    cancelledRidHistory.remove(cancelledRidHistory.iterator().next());
                }
            }
        }

        private void recordLifecycleStart(long requestId, long batchId, String method) {
            long arrivedMs = System.currentTimeMillis();
            Map<String, Object> lifecycle = new LinkedHashMap<>();
            lifecycle.put("rid", requestId);
            lifecycle.put("method", method);
            lifecycle.put("batch_id", batchId);
            lifecycle.put("arrived_ms", arrivedMs);
            lifecycle.put("running_ms", arrivedMs);
            lifecycle.put("end_ms", 0L);
            lifecycle.put("end_state", "running");
            synchronized (requestLifecycles) {
                requestLifecycles.put(requestId, lifecycle);
                // Bounded like Python _prune_lifecycle: evict the oldest entries once over cap.
                while (requestLifecycles.size() > LIFECYCLE_CAP) {
                    requestLifecycles.remove(requestLifecycles.keySet().iterator().next());
                }
            }
        }

        private void recordLifecycleEnd(long requestId, boolean cancelled) {
            synchronized (requestLifecycles) {
                Map<String, Object> lifecycle = requestLifecycles.get(requestId);
                if (lifecycle != null && "running".equals(lifecycle.get("end_state"))) {
                    lifecycle.put("end_ms", System.currentTimeMillis());
                    lifecycle.put("end_state", cancelled ? "cancelled" : "completed");
                }
            }
        }

        private void recordRecentExecutionTime(long executionMs) {
            if (executionMs <= 0) {
                return;
            }
            ArrayDeque<Double> target = roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    ? recentDecodeTimes : recentPrefillTimes;
            synchronized (target) {
                target.addLast((double) executionMs);
                while (target.size() > RECENT_TIME_CAP) {
                    target.removeFirst();
                }
            }
        }

        /** Snapshot of the bounded request lifecycle map with string keys (Python /requests). */
        Map<String, Map<String, Object>> getRequestLifecycleSnapshot() {
            Map<String, Map<String, Object>> result = new LinkedHashMap<>();
            synchronized (requestLifecycles) {
                for (Map.Entry<Long, Map<String, Object>> entry : requestLifecycles.entrySet()) {
                    result.put(String.valueOf(entry.getKey()), new LinkedHashMap<>(entry.getValue()));
                }
            }
            return result;
        }

        private static double avg(ArrayDeque<Double> values) {
            synchronized (values) {
                if (values.isEmpty()) {
                    return 0.0;
                }
                double sum = 0.0;
                for (double v : values) {
                    sum += v;
                }
                return sum / values.size();
            }
        }

        private static double p99(ArrayDeque<Double> values) {
            synchronized (values) {
                if (values.isEmpty()) {
                    return 0.0;
                }
                List<Double> sorted = new ArrayList<>(values);
                sorted.sort(Double::compareTo);
                int idx = Math.max(0, Math.min(sorted.size() - 1,
                        (int) Math.ceil(0.99 * sorted.size()) - 1));
                return sorted.get(idx);
            }
        }

        int getInflightCount() {
            // pendingRequests already counts both prefill and decode requests
            // (incremented in schedulePrefillCompletion and scheduleDecodeCompletion).
            // activeDecodeRequests is a subset for decode-specific reporting only;
            // adding it would double-count decode requests.
            return pendingRequests.get();
        }

        /**
         * Python-compatible snapshot (field names per the legacy MockEngineState.snapshot,
         * ~L327-358), followed by the pre-existing Java-only fields. Python field names and
         * nesting must not be renamed.
         */
        Map<String, Object> getSnapshot() {
            Map<String, Object> snap = new LinkedHashMap<>();
            long effectiveActiveKv = activeKvTokens.get() + faultConfig.getKvPressureTokens();
            snap.put("name", engineName);
            snap.put("role", roleName.toLowerCase());
            snap.put("grpc_addr", host + ":" + grpcPort);
            snap.put("http_addr", host + ":" + (grpcPort - 1));
            snap.put("running", runningTasks.size());
            // Python: max(_injected_queue_depth, _prefill_waiting). Java has no fake injected
            // depth (see /set_queue_depth note), so this is the real waiting count.
            // For decode engines, report the decode pending queue depth (consistent
            // with getWorkerStatus waitingQueryLen) instead of waitingPrefillRequests
            // which is always 0 for decode engines.
            snap.put("waiting", roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    ? decodePendingQueueSize() : waitingPrefillRequests.get());
            snap.put("accepted", acceptedCount.get());
            snap.put("completed", completedCount.get());
            snap.put("cache_keys", cache.snapshotKeys().size());
            snap.put("cache_evictions", cache.evictions());
            snap.put("active_kv_tokens", effectiveActiveKv);
            snap.put("available_kv_tokens", Math.max(0, totalKvTokens - effectiveActiveKv));
            snap.put("status_version", statusVersion.get());
            snap.put("cache_version", cacheVersion.get());
            Map<String, Object> injectConfig = new LinkedHashMap<>();
            injectConfig.put("enqueue_error", faultConfig.isFailOnEnqueue());
            injectConfig.put("fetch_error", faultConfig.isFetchError());
            injectConfig.put("generate_error", faultConfig.isGenerateError());
            injectConfig.put("no_respond", faultConfig.isNoRespond());
            snap.put("inject_config", injectConfig);
            Map<String, Object> rpcCounts = new LinkedHashMap<>();
            rpcCounts.put("enqueue_batch", rpcEnqueueBatch.get());
            rpcCounts.put("generate_stream", rpcGenerateStream.get());
            rpcCounts.put("fetch_response", rpcFetchResponse.get());
            rpcCounts.put("cancel", rpcCancel.get());
            snap.put("rpc_counts", rpcCounts);
            snap.put("cancelled_count", cancelledCount.get());
            List<Long> cancelledRids;
            synchronized (cancelledRidHistory) {
                cancelledRids = new ArrayList<>(cancelledRidHistory);
            }
            cancelledRids.sort(Long::compareTo);
            snap.put("cancelled_rids", cancelledRids);
            snap.put("request_lifecycle", getRequestLifecycleSnapshot());
            snap.put("prefill_ms_avg", avg(recentPrefillTimes));
            snap.put("prefill_ms_p99", p99(recentPrefillTimes));
            synchronized (recentPrefillTimes) {
                snap.put("prefill_ms_count", recentPrefillTimes.size());
            }
            snap.put("decode_ms_avg", avg(recentDecodeTimes));
            snap.put("decode_ms_p99", p99(recentDecodeTimes));
            synchronized (recentDecodeTimes) {
                snap.put("decode_ms_count", recentDecodeTimes.size());
            }
            // Python cluster.snapshot() adds "stopped" per engine.
            snap.put("stopped", stopped);
            // Java-only fields retained (do not rename Python fields above).
            snap.put("port", grpcPort);
            snap.put("inflight", getInflightCount());
            snap.put("leak_detected", leakDetected.get());
            snap.put("kv_tokens_used", effectiveActiveKv);
            return snap;
        }

        private record VersionedTask(long version, EngineRpcService.TaskInfoPB task) {
        }

        /** A decode request parked in the pending queue waiting for a concurrency slot. */
        private record DecodePendingTask(
                MockPerformanceModel.RequestShape shape,
                long batchId,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue) {
        }

        /** A prefill batch parked in the pending queue waiting for a concurrency slot. */
        private record PrefillPendingBatch(
                List<MockPerformanceModel.RequestShape> shapes,
                long batchId,
                int dpRank) {
        }
    }

    static final class ClusterStats {
        private final LongAdder enqueueRpcs = new LongAdder();
        private final LongAdder enqueuedRequests = new LongAdder();
        private final LongAdder statusRpcs = new LongAdder();
        private final LongAdder cacheRpcs = new LongAdder();
        private final LongAdder generateStreamRpcs = new LongAdder();
        private final LongAdder fetchResponseRpcs = new LongAdder();
        private final LongAdder cancelRpcs = new LongAdder();
        private final LongAdder prefillBatches = new LongAdder();
        private final LongAdder prefillBatchRequests = new LongAdder();
        private final LongAdder prefillBatchExecutionMs = new LongAdder();
        private final AtomicInteger maxPrefillBatchSize = new AtomicInteger();
        private final AtomicLong maxPrefillBatchExecutionMs = new AtomicLong();

        private void recordPrefillBatch(int batchSize, long executionMs) {
            prefillBatches.increment();
            prefillBatchRequests.add(batchSize);
            prefillBatchExecutionMs.add(executionMs);
            maxPrefillBatchSize.accumulateAndGet(batchSize, Math::max);
            maxPrefillBatchExecutionMs.accumulateAndGet(executionMs, Math::max);
        }
    }

    static final class Config {
        // Package-private for direct assertions in ClusterConfigParamTest.
        int nPrefill = 2;
        int nDecode = 4;
        int baseGrpcPort = 61_000;
        int eventLoopThreads = 32;
        int completionThreads = 8;
        int prefillCacheBlocks = 6_000;
        int decodeCacheBlocks = 3_000;
        String host = "127.0.0.1";
        String prefillDomain = "mock.prefill.hosts.address";
        String decodeDomain = "mock.decode.hosts.address";
        String endpointFile;
        String envFile;
        String performanceFile;
        String masterConfigFile;
        long totalKvTokens = DEFAULT_TOTAL_KV_TOKENS;
        int blockSize = 0;
        int decodeMaxConcurrency = DEFAULT_DECODE_MAX_CONCURRENCY;

        static Config parse(String[] args) {
            Config config = new Config();
            for (int i = 0; i < args.length; i++) {
                String key = args[i];
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("Missing value for " + key);
                }
                String value = args[++i];
                switch (key) {
                    case "--n-prefill" -> config.nPrefill = Integer.parseInt(value);
                    case "--n-decode" -> config.nDecode = Integer.parseInt(value);
                    case "--base-grpc-port" -> config.baseGrpcPort = Integer.parseInt(value);
                    case "--event-loop-threads" -> config.eventLoopThreads = Integer.parseInt(value);
                    case "--completion-threads" -> config.completionThreads = Integer.parseInt(value);
                    case "--prefill-cache-blocks" -> config.prefillCacheBlocks = Integer.parseInt(value);
                    case "--decode-cache-blocks" -> config.decodeCacheBlocks = Integer.parseInt(value);
                    case "--host" -> config.host = value;
                    case "--prefill-domain" -> config.prefillDomain = value;
                    case "--decode-domain" -> config.decodeDomain = value;
                    case "--endpoint-file" -> config.endpointFile = value;
                    case "--env-file" -> config.envFile = value;
                    case "--performance" -> config.performanceFile = value;
                    case "--master-config" -> config.masterConfigFile = value;
                    case "--total-kv-tokens" -> config.totalKvTokens = Long.parseLong(value);
                    case "--block-size" -> config.blockSize = Integer.parseInt(value);
                    case "--decode-max-concurrency" -> config.decodeMaxConcurrency = Integer.parseInt(value);
                    default -> throw new IllegalArgumentException("Unknown argument: " + key);
                }
            }
            if (config.endpointFile == null || config.performanceFile == null
                    || config.masterConfigFile == null) {
                throw new IllegalArgumentException(
                        "--endpoint-file, --performance, and --master-config are required");
            }
            // Single-role clusters are allowed (e.g. engine_kill_restart_test victim JVMs
            // hosting only prefill or only decode engines), but at least one engine is required.
            if (config.nPrefill < 0 || config.nDecode < 0
                    || config.nPrefill + config.nDecode < 1) {
                throw new IllegalArgumentException(
                        "n-prefill/n-decode must be >= 0 with at least one engine in total");
            }
            if (config.eventLoopThreads < 1 || config.completionThreads < 1) {
                throw new IllegalArgumentException("thread counts must be positive");
            }
            if (config.decodeMaxConcurrency < 1) {
                throw new IllegalArgumentException("--decode-max-concurrency must be >= 1");
            }
            return config;
        }
    }
}
