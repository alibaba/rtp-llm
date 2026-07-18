package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Formula-driven Engine gRPC cluster for FlexLB control-plane and capacity tests.
 *
 * <p>Requests are queued using their input/output token shape, configured prefill formula,
 * cache hits, and decode batch curve. It models service timing and queue pressure rather
 * than GPU kernels. All listening ports share Netty event loops.
 */
public final class JavaMockEngineCluster {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final long TOTAL_KV_TOKENS = 6_291_456L;

    private JavaMockEngineCluster() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        MockPerformanceModel performance = MockPerformanceModel.load(
                config.performanceFile, config.masterConfigFile);
        ClusterStats stats = new ClusterStats();
        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup(config.eventLoopThreads);
        List<Server> servers = new ArrayList<>(config.nPrefill + config.nDecode);
        Map<Integer, FastRpcService> services = new ConcurrentHashMap<>();
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "java-mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });

        try {
            startRole(config, performance, servers, bossGroup, workerGroup, services, scheduler, stats,
                    0, config.nPrefill, "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL);
            startRole(config, performance, servers, bossGroup, workerGroup, services, scheduler, stats,
                    config.nPrefill, config.nDecode, "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE);
            writeDiscoveryFiles(config);
        } catch (Throwable error) {
            scheduler.shutdownNow();
            shutdown(servers, bossGroup, workerGroup);
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
                            + "max_prefill_pending=%d decode_running=%d heap_used_mb=%d heap_max_mb=%d%n",
                    stats.enqueueRpcs.sum(), stats.enqueuedRequests.sum(),
                    stats.statusRpcs.sum(), stats.cacheRpcs.sum(),
                    prefillBatches, avgBatchSize, stats.maxPrefillBatchSize.get(),
                    avgBatchMs, stats.maxPrefillBatchExecutionMs.get(),
                    prefillPending, maxPrefillPending, decodeRunning, heapUsedMb, heapMaxMb);
        },
                5, 5, TimeUnit.SECONDS);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            scheduler.shutdownNow();
            shutdown(servers, bossGroup, workerGroup);
        }, "java-mock-engine-shutdown"));

        System.out.printf("Java mock engine ready: prefill=%d decode=%d ports=%d-%d eventLoops=%d performance=%s%n",
                config.nPrefill, config.nDecode, config.baseGrpcPort,
                config.baseGrpcPort + config.nPrefill + config.nDecode - 1,
                config.eventLoopThreads, config.performanceFile);
        new CountDownLatch(1).await();
    }

    private static void startRole(Config config,
                                  MockPerformanceModel performance,
                                  List<Server> servers,
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
                    roleName, roleType, grpcPort, services, scheduler,
                    performance, cacheCapacity, stats);
            services.put(grpcPort, service);
            Server server = NettyServerBuilder.forPort(grpcPort)
                    .bossEventLoopGroup(bossGroup)
                    .workerEventLoopGroup(workerGroup)
                    .channelType(NioServerSocketChannel.class)
                    .directExecutor()
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .addService(service)
                    .build()
                    .start();
            servers.add(server);
        }
    }

    private static void shutdown(List<Server> servers,
                                 EventLoopGroup bossGroup,
                                 EventLoopGroup workerGroup) {
        for (Server server : servers) {
            server.shutdownNow();
        }
        bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
    }

    private static void writeDiscoveryFiles(Config config) throws IOException {
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
        private final String roleName;
        private final EngineRpcService.RoleTypePB roleType;
        private final int grpcPort;
        private final Map<Integer, FastRpcService> services;
        private final ScheduledExecutorService scheduler;
        private final MockPerformanceModel performance;
        private final MockLruBlockCache cache;
        private final ClusterStats stats;
        private final AtomicLong statusVersion = new AtomicLong();
        private final AtomicLong completionVersion = new AtomicLong();
        private final AtomicLong cacheVersion = new AtomicLong(1);
        private final Map<Integer, AtomicLong> nextPrefillAvailableNanosByDp = new ConcurrentHashMap<>();
        private final AtomicLong activeKvTokens = new AtomicLong();
        private final AtomicInteger pendingRequests = new AtomicInteger();
        private final AtomicInteger waitingPrefillRequests = new AtomicInteger();
        private final AtomicInteger activePrefillBatches = new AtomicInteger();
        private final AtomicInteger activeDecodeRequests = new AtomicInteger();
        private final ConcurrentLinkedQueue<VersionedTask> completions = new ConcurrentLinkedQueue<>();
        private final Map<Long, EngineRpcService.TaskInfoPB> runningTasks = new ConcurrentHashMap<>();

        FastRpcService(String roleName,
                       EngineRpcService.RoleTypePB roleType,
                       int grpcPort,
                       Map<Integer, FastRpcService> services,
                       ScheduledExecutorService scheduler,
                       MockPerformanceModel performance,
                       int cacheCapacity,
                       ClusterStats stats) {
            this.roleName = roleName.toUpperCase();
            this.roleType = roleType;
            this.grpcPort = grpcPort;
            this.services = services;
            this.scheduler = scheduler;
            this.performance = performance;
            this.cache = new MockLruBlockCache(cacheCapacity);
            this.stats = stats;
        }

        @Override
        public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                 StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
            stats.enqueueRpcs.increment();
            EngineRpcService.EnqueueBatchResponsePB.Builder response =
                    EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                List<MockPerformanceModel.RequestShape> shapes = new ArrayList<>(slot.getRequestsCount());
                for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                    stats.enqueuedRequests.increment();
                    response.addSuccessesBuilder().setRequestId(input.getInput().getRequestId());
                    shapes.add(performance.shape(input.getInput(), cache));
                }
                schedulePrefillCompletion(shapes, request.getBatchId(), slot.getDpRank());
            }
            observer.onNext(response.build());
            observer.onCompleted();
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
            long usedKv = Math.min(TOTAL_KV_TOKENS, activeKvTokens.get());
            EngineRpcService.WorkerStatusPB.Builder status = EngineRpcService.WorkerStatusPB.newBuilder()
                    .setAlive(true)
                    .setRole(roleName)
                    .setRoleType(roleType)
                    .setAvailableConcurrency(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                            ? Math.max(0, 1 - activePrefillBatches.get())
                            : Math.max(0, 132 - (int) runningCount))
                    .setWaitingQueryLen(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                            ? waitingPrefillRequests.get() : 0)
                    .setRunningQueryLen((int) runningCount)
                    .setAvailableKvCache(TOTAL_KV_TOKENS - usedKv)
                    .setTotalKvCache(TOTAL_KV_TOKENS)
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

        private void schedulePrefillCompletion(List<MockPerformanceModel.RequestShape> shapes,
                                               long batchId,
                                               int dpRank) {
            if (shapes.isEmpty()) {
                return;
            }
            long executionMs = performance.prefillMs(shapes);
            long now = System.nanoTime();
            long executionNanos = TimeUnit.MILLISECONDS.toNanos(executionMs);
            AtomicLong nextAvailable = nextPrefillAvailableNanosByDp.computeIfAbsent(
                    dpRank, ignored -> new AtomicLong());
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
            pendingRequests.addAndGet(shapes.size());
            for (MockPerformanceModel.RequestShape shape : shapes) {
                runningTasks.put(shape.input().getRequestId(),
                        task(shape, batchId, dpRank, EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
            }
            long startDelayNanos = Math.max(0, startNanos - now);
            if (startDelayNanos == 0) {
                startPrefillBatch(shapes, batchId, dpRank);
            } else {
                waitingPrefillRequests.addAndGet(shapes.size());
                scheduler.schedule(() -> {
                    waitingPrefillRequests.addAndGet(-shapes.size());
                    startPrefillBatch(shapes, batchId, dpRank);
                }, startDelayNanos, TimeUnit.NANOSECONDS);
            }

            long delayNanos = Math.max(0, finishNanos - now);
            scheduler.schedule(() -> {
                for (MockPerformanceModel.RequestShape shape : shapes) {
                    runningTasks.remove(shape.input().getRequestId());
                    recordCompletion(shape, batchId, executionMs, dpRank);
                    startDecode(shape, batchId);
                    if (cache.admit(shape.blockKeys())) {
                        cacheVersion.incrementAndGet();
                    }
                }
                activePrefillBatches.decrementAndGet();
                pendingRequests.addAndGet(-shapes.size());
            }, delayNanos, TimeUnit.NANOSECONDS);
        }

        private void startPrefillBatch(List<MockPerformanceModel.RequestShape> shapes,
                                       long batchId,
                                       int dpRank) {
            activePrefillBatches.incrementAndGet();
            for (MockPerformanceModel.RequestShape shape : shapes) {
                runningTasks.put(shape.input().getRequestId(),
                        task(shape, batchId, dpRank, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
            }
        }

        private void startDecode(MockPerformanceModel.RequestShape shape, long batchId) {
            EngineRpcService.GenerateInputPB input = shape.input();
            for (EngineRpcService.RoleAddrPB addr : input.getGenerateConfig().getRoleAddrsList()) {
                if (addr.getRoleType() != EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                    continue;
                }
                FastRpcService decode = services.get(addr.getGrpcPort());
                if (decode != null && decode.grpcPort != grpcPort) {
                    decode.scheduleDecodeCompletion(shape, batchId);
                }
                return;
            }
        }

        private void scheduleDecodeCompletion(MockPerformanceModel.RequestShape shape, long batchId) {
            int activeBatch = activeDecodeRequests.incrementAndGet();
            activeKvTokens.addAndGet(shape.inputLen());
            pendingRequests.incrementAndGet();
            runningTasks.put(shape.input().getRequestId(),
                    task(shape, batchId, 0, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
            long executionMs = performance.decodeMs(shape.outputLen(), activeBatch);
            scheduler.schedule(() -> {
                runningTasks.remove(shape.input().getRequestId());
                activeDecodeRequests.decrementAndGet();
                activeKvTokens.addAndGet(-shape.inputLen());
                pendingRequests.decrementAndGet();
                recordCompletion(shape, batchId, executionMs, 0);
                if (cache.admit(shape.blockKeys())) {
                    cacheVersion.incrementAndGet();
                }
            }, executionMs, TimeUnit.MILLISECONDS);
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

        @Override
        public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                   StreamObserver<EngineRpcService.CacheStatusPB> observer) {
            stats.cacheRpcs.increment();
            long usedKv = Math.min(TOTAL_KV_TOKENS, activeKvTokens.get());
            EngineRpcService.CacheStatusPB.Builder status = EngineRpcService.CacheStatusPB.newBuilder()
                    .setAvailableKvCache(TOTAL_KV_TOKENS - usedKv)
                    .setTotalKvCache(TOTAL_KV_TOKENS)
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

        @Override
        public void cancel(EngineRpcService.CancelRequestPB request,
                           StreamObserver<EngineRpcService.EmptyPB> observer) {
            observer.onNext(EngineRpcService.EmptyPB.getDefaultInstance());
            observer.onCompleted();
        }

        private record VersionedTask(long version, EngineRpcService.TaskInfoPB task) {
        }
    }

    static final class ClusterStats {
        private final LongAdder enqueueRpcs = new LongAdder();
        private final LongAdder enqueuedRequests = new LongAdder();
        private final LongAdder statusRpcs = new LongAdder();
        private final LongAdder cacheRpcs = new LongAdder();
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

    private static final class Config {
        private int nPrefill = 2;
        private int nDecode = 4;
        private int baseGrpcPort = 61_000;
        private int eventLoopThreads = 32;
        private int prefillCacheBlocks = 6_000;
        private int decodeCacheBlocks = 3_000;
        private String host = "127.0.0.1";
        private String prefillDomain = "mock.prefill.hosts.address";
        private String decodeDomain = "mock.decode.hosts.address";
        private String endpointFile;
        private String envFile;
        private String performanceFile;
        private String masterConfigFile;

        private static Config parse(String[] args) {
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
                    case "--prefill-cache-blocks" -> config.prefillCacheBlocks = Integer.parseInt(value);
                    case "--decode-cache-blocks" -> config.decodeCacheBlocks = Integer.parseInt(value);
                    case "--host" -> config.host = value;
                    case "--prefill-domain" -> config.prefillDomain = value;
                    case "--decode-domain" -> config.decodeDomain = value;
                    case "--endpoint-file" -> config.endpointFile = value;
                    case "--env-file" -> config.envFile = value;
                    case "--performance" -> config.performanceFile = value;
                    case "--master-config" -> config.masterConfigFile = value;
                    default -> throw new IllegalArgumentException("Unknown argument: " + key);
                }
            }
            if (config.endpointFile == null || config.performanceFile == null
                    || config.masterConfigFile == null) {
                throw new IllegalArgumentException(
                        "--endpoint-file, --performance, and --master-config are required");
            }
            if (config.nPrefill < 1 || config.nDecode < 1 || config.eventLoopThreads < 1) {
                throw new IllegalArgumentException("worker counts and event loops must be positive");
            }
            return config;
        }
    }
}
