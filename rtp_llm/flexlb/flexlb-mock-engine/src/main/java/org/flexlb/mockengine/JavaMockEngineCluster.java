package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.grpc.Context;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.HashMap;
import java.util.HashSet;
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
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
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
    static final int[] PREFILL_BATCH_SIZE_BUCKETS = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128};

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    /** Default PREFILL pool token capacity per engine (Python --prefill/--decode-total-kv-tokens default). */
    static final long DEFAULT_TOTAL_KV_TOKENS = 6_291_456L;
    /**
     * Default DECODE pool token capacity per engine — deliberately heterogeneous
     * (2/3 of the prefill pool): decode engines hold each request's KV for its
     * whole life (input + growing output), so a smaller decode pool lets the
     * master's cross-engine comparisons (min kvCacheUsed / KV% gates) see real
     * per-role capacity divergence instead of a uniform static constant.
     */
    static final long DEFAULT_DECODE_TOTAL_KV_TOKENS = 4_194_304L;
    /**
     * Engine-side KV admission failure code (production C++ ErrorCode::MALLOC_FAILED
     * = 602; the master's own 8431 RESOURCE_EXHAUSTED is NOT valid from an engine).
     * Surfaced synchronously in the EnqueueBatch error list so the master's
     * DefaultBatchDispatcher raises EngineRejectedException on the dispatch path.
     */
    static final long LACK_MEM_ERROR_CODE = 602L;
    /** Resolved concurrency_limit default from server_args/concurrent_group_args.py. */
    static final int DEFAULT_DECODE_MAX_CONCURRENCY = 32;
    /** CLI flag for unique per-engine loopback advertisement IPs (default on). */
    static final String UNIQUE_ENGINE_IPS_FLAG = "--unique-engine-ips";
    /**
     * Default per-frame poll timeout for the client-facing response pump
     * (generate_stream / fetch_response). Must cover the LONGEST possible
     * gap between consecutive frames on one stream. The dominant gap is the
     * decode execution between the prefill first-token frame and the decode
     * terminal frame: with the production trace's max output_len (20 000
     * tokens) and the slowest decode step in the performance curve (25.1 ms
     * at batch 256), that gap reaches ~502 s — the legacy hard-coded 60 s
     * truncated e2e for long-output requests. 600 s leaves headroom over
     * the 502 s worst case.
     */
    static final long DEFAULT_RESPONSE_POLL_TIMEOUT_MS = 600_000L;

    private JavaMockEngineCluster() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        WhaleMockMonitor whaleMonitor = config.kmonitor ? WhaleMockMonitor.create() : null;
        MockPerformanceModel performance = MockPerformanceModel.loadCalibration(
                config.calibrationFile);
        config.engineInitializer = service -> configureWhaleMetrics(service, whaleMonitor);
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
        // Per-request JSONL event stream (engine_events.jsonl): opened BEFORE
        // any engine starts so every FastRpcService (initial and dynamically
        // added) receives the same log instance via setEngineEventLog.
        EngineEventLog engineEventLog = EngineEventLog.open(config.eventsFile);
        try {
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    0, config.nPrefill, "prefill");
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    config.nPrefill, config.nDecode, "decode");
            for (FastRpcService service : services.values()) {
                service.setEngineEventLog(engineEventLog);
            }
            if (!config.whale || config.whaleBundle) writeDiscoveryFiles(config);
            // File-based discovery mode (--discovery-file): maintain the dynamic
            // domain→hosts mapping consumed by LocalServiceDiscovery on the master,
            // kept in sync by /add_engine + /remove_engine at runtime.
            DiscoveryFileStore discoveryFileStore = config.discoveryFile != null
                    ? new DiscoveryFileStore(config.discoveryFile, config.prefillDomain, config.decodeDomain)
                    : null;
            if (discoveryFileStore != null) {
                discoveryFileStore.rewrite(services);
            }
            DynamicEngineManager engineManager = new DynamicEngineManager(
                    config, performance, services, serversByPort, bossGroup, workerGroup,
                    scheduler, stats, discoveryFileStore, engineEventLog);
            controlServer = new MockControlServer(
                    services, serversByPort, bossGroup, workerGroup, config.host, config.baseGrpcPort - 1,
                    engineManager);
            controlServer.start();
        } catch (Throwable error) {
            scheduler.shutdownNow();
            for (FastRpcService s : services.values()) s.shutdown();
            shutdown(serversByPort, bossGroup, workerGroup, config);
            throw error;
        }

        if (config.statsStdout) {
            // Debug surface (default OFF): consolidate_run_outputs.py parses
            // the java_mock_stats lines out of mock_engine.log to build the
            // mock.json stats timeline, so run_online_eval.sh passes
            // --stats-stdout explicitly to keep that chain alive; ad-hoc
            // launches get a quiet stdout.
            scheduler.scheduleAtFixedRate(() -> System.out.print(buildStatsLine(services.values(), stats)),
                    config.statsIntervalMs, config.statsIntervalMs, TimeUnit.MILLISECONDS);
        }

        if (whaleMonitor != null) {
            scheduler.scheduleAtFixedRate(() -> {
                try {
                    services.values().forEach(whaleMonitor::sample);
                } catch (RuntimeException error) {
                    System.err.println("Whale mock metric reporting failed: " + error);
                }
            }, 1000, 1000, TimeUnit.MILLISECONDS);
        }

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
            List<Runnable> close = new ArrayList<>();
            close.add(controlServer::stop);
            // Drain while completion scheduling still exists, then close each
            // independent owner even when another close operation fails.
            for (FastRpcService service : services.values()) close.add(service::drainAndShutdown);
            close.add(scheduler::shutdownNow);
            for (FastRpcService service : services.values()) close.add(service::shutdown);
            close.add(() -> shutdown(serversByPort, bossGroup, workerGroup, config));
            if (whaleMonitor != null) close.add(whaleMonitor::close);
            if (engineEventLog != null) close.add(engineEventLog::close);
            ShutdownTasks.runAll(close);
        }, "java-mock-engine-shutdown"));

        System.out.printf("Java mock engine ready: prefill=%d decode=%d ports=%d-%d eventLoops=%d performance=%s completionThreads=%d statsIntervalMs=%d%n",
                config.nPrefill, config.nDecode, config.baseGrpcPort,
                config.baseGrpcPort + config.nPrefill + config.nDecode - 1,
                config.eventLoopThreads, config.calibrationFile, config.completionThreads,
                config.statsIntervalMs);
        if (config.eventsFile != null) {
            System.out.printf("Engine event stream enabled: %s%n", config.eventsFile);
        }
        System.out.printf("HTTP control server listening on port %d%n", config.baseGrpcPort - 1);
        if (config.discoveryFile != null) {
            System.out.printf("File service discovery enabled: %s (add/remove_engine keep it in sync)%n",
                    config.discoveryFile);
        }
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
                                  String roleName) throws IOException {
        for (int i = 0; i < count; i++) {
            int grpcPort = config.baseGrpcPort + portOffset + i;
            startEngine(config, performance, serversByPort, bossGroup, workerGroup,
                    services, scheduler, stats, roleName, roleName + "-" + i, grpcPort,
                    portOffset + i);
        }
    }

    private static void configureWhaleMetrics(FastRpcService service, WhaleMockMonitor whaleMonitor) {
        if (whaleMonitor != null) service.schedulerMetricReporter =
                values -> whaleMonitor.reportScheduler(values, service.whaleMetricTags());
        if (whaleMonitor != null) {
            service.statusMetricReporter = latency -> whaleMonitor.reportWorkerStatus(service.whaleMetricTags(), latency);
            service.cacheMetricReporter = latency -> whaleMonitor.reportCacheStatus(service.whaleMetricTags(), latency);
        }
        if (whaleMonitor != null) service.eventMetricReporter =
                values -> whaleMonitor.reportEvent(values, service.whaleMetricTags());
        if (whaleMonitor != null) service.cacheEvictionReporter = (scope, ms) -> {
            var tags = new HashMap<>(service.whaleMetricTags());
            tags.put("scope", scope);
            tags.put("backing", scope.equals("gpu") ? "device" : "memory");
            tags.put("kind", scope.equals("gpu") ? "chain" : "complete");
            whaleMonitor.reportEvent(Map.of("mock_cache_evicted_entry_age_ms", ms), tags);
        };
    }

    /**
     * Create, register, and start ONE engine gRPC server. Extracted from
     * {@link #startRole} so dynamic scale-out ({@link DynamicEngineManager}
     * /add_engine) can create engines at runtime with the same construction
     * rules: shared scheduler/stats/eventLoopGroups, per-role cache capacity,
     * SO_REUSEADDR, cluster-owned RPC executors.
     *
     * <p>Registration is rolled back if the port cannot be bound, so a failed
     * dynamic add leaves no services-map residue.
     *
     * <p>{@code engineIndex} is the engine's GLOBAL index (initial engines:
     * prefill first, decode after; dynamic engines get freshly allocated
     * indices from {@link DynamicEngineManager}). It feeds
     * {@link #declaredHost} so unique advertisement IPs
     * ({@code --unique-engine-ips}) stay unique across initial and dynamically
     * added engines — the gRPC server bind below stays wildcard (forPort),
     * only the advertised address changes.
     *
     * @return the started service
     */
    static FastRpcService startEngine(Config config,
                                      MockPerformanceModel performance,
                                      Map<Integer, Server> serversByPort,
                                      EventLoopGroup bossGroup,
                                      EventLoopGroup workerGroup,
                                      Map<Integer, FastRpcService> services,
                                      ScheduledExecutorService scheduler,
                                      ClusterStats stats,
                                      String roleName,
                                      String engineName,
                                      int grpcPort,
                                      int engineIndex) throws IOException {
        EngineRpcService.RoleTypePB roleType = "decode".equalsIgnoreCase(roleName)
                ? EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                : EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
        if (config.engineConfig == null) throw new IllegalArgumentException("engine config is required");
        performance = config.engineConfig.configure(roleName, performance);
        int totalBlocks = config.engineConfig.blocks(roleName);
        long poolTotalKvTokens = (long) totalBlocks * performance.blockSize();
        FastRpcService service = new FastRpcService(
                engineName, declaredHost(config, engineIndex), roleName, roleType, grpcPort,
                services, scheduler, performance, totalBlocks, stats,
                poolTotalKvTokens, config.engineConfig.concurrency(roleName));
        service.effectiveConfig = config.engineConfig.snapshot();
        service.setResponsePollTimeoutMs(DEFAULT_RESPONSE_POLL_TIMEOUT_MS);
        service.setAutoFetch(config.autoFetch);
        service.cacheStatusCompactEncoding = config.cacheStatusCompactEncoding;
        // Bundled P/D share one JVM and event loop group. Use local ownership
        // callbacks so P cannot block those event loops awaiting D RPC admission.
        service.setWhaleRemote(config.whale && !config.whaleBundle);
        service.whaleBundle = config.whaleBundle;
        service.performance.nativeTokenCacheKeys = true;
        service.whalePodIp = config.host;
        service.setFetchAttachTimeoutMs(config.fetchAttachTimeoutMs);
        if (config.engineInitializer != null) config.engineInitializer.accept(service);
        services.put(grpcPort, service);
        try {
            MockRpcExecutors runtime = config.rpcExecutors(workerGroup);
            MockRpcExecutors.Engine rpcEngine = runtime.engine();
            service.rpcEngine = rpcEngine;
            Server server = NettyServerBuilder.forAddress(new java.net.InetSocketAddress(config.bindHost, grpcPort))
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
                    .executor(runtime.controlExecutor())
                    .callExecutor(rpcEngine)
                    .intercept(rpcEngine)
                    .maxInboundMessageSize(config.grpcMaxInboundMessageBytes)
                    .addService(CacheFixtureWire.bind(service))
                    .build()
                    .start();
            serversByPort.put(grpcPort, server);
            // Let the service kill its own port on crash_after (true-crash
            // semantics) — the FastRpcService has no other handle on the
            // server it backs.
            service.setGrpcServer(server);
        } catch (IOException e) {
            services.remove(grpcPort);
            throw e;
        }
        return service;
    }

    private static void shutdown(Map<Integer, Server> serversByPort,
                                 EventLoopGroup bossGroup,
                                 EventLoopGroup workerGroup, Config config) {
        List<Runnable> close = new ArrayList<>();
        serversByPort.values().forEach(server -> close.add(server::shutdownNow));
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        serversByPort.values().forEach(server -> close.add(() -> {
            try {
                if (!server.awaitTermination(Math.max(0, end - System.nanoTime()), TimeUnit.NANOSECONDS))
                    throw new IllegalStateException("mock RPC server did not terminate");
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted stopping mock RPC server", error);
            }
        }));
        close.add(() -> {
            if (!bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS).awaitUninterruptibly(3, TimeUnit.SECONDS))
                throw new IllegalStateException("mock RPC boss transport did not terminate");
        });
        close.add(() -> {
            if (!workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS).awaitUninterruptibly(3, TimeUnit.SECONDS))
                throw new IllegalStateException("mock RPC worker transport did not terminate");
        });
        close.add(() -> { if (config.rpcExecutors != null) config.rpcExecutors.close(); });
        ShutdownTasks.runAll(close);
    }

    /**
     * Builds one java_mock_stats telemetry line. Extracted from the stats
     * scheduler lambda so the emission surface (which keys the line carries)
     * is unit-testable without booting the cluster
     * ({@code TelemetryEmissionSurfaceTest}).
     */
    static String buildStatsLine(Collection<FastRpcService> services, ClusterStats stats) {
        // Symmetric P/D four-state queue metrics. Units: prefill_waiting /
        // decode_waiting count queued REQUESTS (not yet running);
        // prefill_running counts running BATCHES (a batch may hold several
        // requests — prefill_running_reqs counts those); decode_running counts
        // running requests. The old prefill_pending (pendingRequests =
        // waiting + running mixed) was misleading and is gone.
        int prefillWaiting = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .mapToInt(service -> service.waitingPrefillRequests.get()).sum();
        int maxPrefillWaiting = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .mapToInt(service -> service.waitingPrefillRequests.get()).max().orElse(0);
        int prefillRunning = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .mapToInt(service -> service.activePrefillBatches.get()).sum();
        int prefillRunningReqs = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .mapToInt(service -> service.activePrefillRequests.get()).sum();
        int decodeWaiting = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .mapToInt(FastRpcService::decodePendingQueueSize).sum();
        int decodeRunning = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .mapToInt(service -> service.activeDecodeRequests.get()).sum();
        // Decode balance summary: per-engine running spread (min/max) plus the
        // deepest per-engine pending queue — imbalance and single-engine
        // backlog are invisible in the cluster-wide sums above.
        int decodeRunMin = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .mapToInt(service -> service.activeDecodeRequests.get()).min().orElse(0);
        int decodeRunMax = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .mapToInt(service -> service.activeDecodeRequests.get()).max().orElse(0);
        int maxDecodeWaiting = services.stream()
                .filter(service -> service.roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .mapToInt(FastRpcService::decodePendingQueueSize).max().orElse(0);
        // Per-tick decode completion window: count + execution-time summary
        // since the previous stats sample (drained, so each tick is disjoint).
        ClusterStats.DecodeWindow decodeWindow = stats.drainDecodeWindow();
        ClusterStats.PrefillWindow prefillWindow = stats.drainPrefillWindow();
        long prefillBatches = stats.prefillBatches.sum();
        double avgBatchSize = prefillBatches == 0
                ? 0.0 : stats.prefillBatchRequests.sum() / (double) prefillBatches;
        double avgBatchMs = prefillBatches == 0
                ? 0.0 : stats.prefillBatchExecutionMs.sum() / (double) prefillBatches;
        Runtime runtime = Runtime.getRuntime();
        long heapUsedMb = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        long heapMaxMb = runtime.maxMemory() / (1024 * 1024);
        return String.format(
                "java_mock_stats ts_epoch_ms=%d enqueue_rpcs=%d enqueued_requests=%d status_rpcs=%d cache_rpcs=%d "
                        + "prefill_batches=%d avg_batch_size=%.2f max_batch_size=%d "
                        + "avg_batch_ms=%.2f max_batch_ms=%d prefill_exec_p50=%d prefill_exec_p95=%d "
                        + "prefill_waiting=%d prefill_running=%d "
                        + "prefill_running_reqs=%d max_prefill_waiting=%d decode_waiting=%d decode_running=%d "
                        + "decode_run_min=%d decode_run_max=%d max_decode_waiting=%d "
                        + "decode_admitted=%d decode_done=%d decode_exec_p50=%d decode_exec_p95=%d decode_exec_max=%d "
                        + "heap_used_mb=%d heap_max_mb=%d "
                        + "generate_stream_rpcs=%d fetch_response_rpcs=%d cancel_rpcs=%d "
                        + "cancel_census_tracked=%d cancel_census_finished=%d cancel_census_unknown=%d cancel_census_already_cancelled=%d "
                        + "cancel_census_injected=%d cancel_census_client_gone=%d%n",
                System.currentTimeMillis(),
                stats.enqueueRpcs.sum(), stats.enqueuedRequests.sum(),
                stats.statusRpcs.sum(), stats.cacheRpcs.sum(),
                prefillBatches, avgBatchSize, stats.maxPrefillBatchSize.get(),
                avgBatchMs, stats.maxPrefillBatchExecutionMs.get(),
                prefillWindow.p50Ms(), prefillWindow.p95Ms(),
                prefillWaiting, prefillRunning, prefillRunningReqs, maxPrefillWaiting,
                decodeWaiting, decodeRunning, decodeRunMin, decodeRunMax, maxDecodeWaiting,
                stats.decodeAdmitted.sum(),
                decodeWindow.count(), decodeWindow.p50Ms(), decodeWindow.p95Ms(), decodeWindow.maxMs(),
                heapUsedMb, heapMaxMb,
                stats.generateStreamRpcs.sum(), stats.fetchResponseRpcs.sum(), stats.cancelRpcs.sum(),
                stats.cancelCensusTracked.sum(), stats.cancelCensusAlreadyFinished.sum(),
                stats.cancelCensusUnknown.sum(), stats.cancelCensusAlreadyCancelled.sum(),
                stats.cancelCensusInjected.sum(), stats.cancelCensusClientGone.sum());
    }

    static void writeDiscoveryFiles(Config config) throws IOException {
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

        serviceConfig.put("discovery_file", config.discoveryFile);
        Map<String, String> env = Map.of(
                "MODEL_SERVICE_CONFIG", OBJECT_MAPPER.writeValueAsString(serviceConfig));

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

    /**
     * Declared host for engine {@code engineIndex} (global index: prefill
     * engines first, then decode): a unique 127.x.y.z loopback address when
     * {@code --unique-engine-ips} is on (default), else {@link Config#host}.
     *
     * <p>Motivation: when every engine advertises host 127.0.0.1, the master's
     * Prometheus {@code engineIp} label has a single variant and per-engine
     * gauge series (batcher queue / KV / inflight) overwrite each other. Unique
     * advertisement IPs make the labels distinct. The gRPC server bind is
     * unchanged (wildcard {@code forPort}); Linux routes all of 127.0.0.0/8 to
     * loopback, so master-to-engine connections to these addresses work on the
     * remote eval hosts.
     */
    static String declaredHost(Config config, int engineIndex) {
        if (config.whale && !config.whaleBundle) return config.host;
        return config.uniqueEngineIps ? derivedLoopbackIp(engineIndex) : config.host;
    }

    /**
     * Derives the unique loopback advertisement IP for engine index
     * {@code engineIndex}: {@code 127.(idx/250 + 1).(idx%250)} — the third
     * octet starts at 1 to stay out of the real 127.0.0.x range, and 250
     * slots per third octet mean 1250 engines (750P + 500D) fit inside
     * 127.1.0.0-127.5.249. Valid for engineIndex in [0, 63749].
     */
    static String derivedLoopbackIp(int engineIndex) {
        if (engineIndex < 0) {
            throw new IllegalArgumentException("engine index must be >= 0");
        }
        int thirdOctet = engineIndex / 250 + 1;
        if (thirdOctet > 255) {
            throw new IllegalArgumentException(
                    "engine index " + engineIndex + " exceeds the unique loopback IP space (max 63749)");
        }
        return "127." + thirdOctet + "." + (engineIndex % 250) + ".1";
    }

    /** Parses a strict true/false CLI value for {@code flag}. */
    static boolean parseBooleanFlag(String value, String flag) {
        if ("true".equalsIgnoreCase(value)) {
            return true;
        }
        if ("false".equalsIgnoreCase(value)) {
            return false;
        }
        throw new IllegalArgumentException(
                "Invalid boolean value for " + flag + ": " + value + " (expected true|false)");
    }

    private static void addEngineRecords(List<Map<String, Object>> engines,
                                         Config config,
                                         int portOffset,
                                         int count,
                                         String role) {
        for (int i = 0; i < count; i++) {
            int grpcPort = config.baseGrpcPort + portOffset + i;
            String host = declaredHost(config, portOffset + i);
            Map<String, Object> engine = new LinkedHashMap<>();
            engine.put("name", role + "-" + i);
            engine.put("role", role);
            engine.put("ip", host);
            engine.put("grpc_port", grpcPort);
            engine.put("http_port", grpcPort - 1);
            engine.put("grpc_addr", host + ":" + grpcPort);
            engine.put("http_addr", host + ":" + (grpcPort - 1));
            engines.add(engine);
        }
    }

    static final class FastRpcService extends RpcServiceGrpc.RpcServiceImplBase {
        volatile MockCacheDiagnostics cacheDiagnostics;
        private final MockVitClient vitClient = new MockVitClient();

        boolean isDiagnosticPrefill() { return roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL; }

        Set<Long> diagnosticResidentKeys() {
            Set<Long> keys = new HashSet<>(cache.snapshotKeys());
            if (memoryCache != null) keys.addAll(memoryCache.keys());
            return keys;
        }

        private static Map<Integer, Integer> multimodalLengths(EngineRpcService.GenerateInputPB input) {
            Map<Integer, Integer> lengths = new java.util.HashMap<>();
            if (input.hasV41Inputs()) {
                int rows = input.getV41Inputs().getImagesList().stream().mapToInt(EngineRpcService.V41ImageInputPB::getTypesCount).sum();
                if (rows > 0) lengths.put(1, rows);
                return lengths;
            }
            int count = Math.min(input.getMultimodalInputsCount(), input.getMultimodalTokenLayout().getSpansCount());
            for (int i = 0; i < count; i++) lengths.merge(input.getMultimodalInputs(i).getMultimodalType(),
                    input.getMultimodalTokenLayout().getSpans(i).getLength(), Integer::sum);
            return lengths;
        }

        private void observeCacheDiagnostics(MockPerformanceModel.RequestShape shape) {
            MockCacheDiagnostics diag = cacheDiagnostics;
            if (diag == null || !isDiagnosticPrefill() || !diag.select()) return;
            int best = 0;
            String bestName = "";
            for (FastRpcService peer : services.values()) {
                if (!peer.isDiagnosticPrefill() || peer.stopped || peer.seqSizePerBlock != seqSizePerBlock) continue;
                int hit = peer.cache.peekPrefixHitBlocks(shape.blockKeys());
                if (peer.memoryCache != null) hit += peer.memoryCache.peekMatch(shape.blockKeys(), hit);
                if (hit > best) { best = hit; bestName = peer.engineName; }
            }
            diag.observe(engineName, shape.blockKeys(), seqSizePerBlock, shape.inputLen(),
                    shape.hitTokens(), best, bestName);
        }

        private volatile boolean whaleRemote;
        private boolean whaleBundle;
        private MockMemoryBlockCache memoryCache;
        private MockHostTransfer hostTransfer;
        private final Map<Long, MockMemoryBlockCache.ReadLease> memoryReads = new ConcurrentHashMap<>();
        private final Map<Long, MemoryWrite> memoryWrites = new ConcurrentHashMap<>();
        private final Map<Long, HostLoad> hostLoads = new ConcurrentHashMap<>();
        private record HostLoad(MockPerformanceModel.RequestShape shape, java.util.concurrent.CompletableFuture<Void> done) {}
        private final java.util.Set<MemoryWrite> watermarkWrites = ConcurrentHashMap.newKeySet();
        private final java.util.concurrent.atomic.AtomicBoolean watermarkQueued = new java.util.concurrent.atomic.AtomicBoolean();
        private int pendingWatermarkBlocks;
        private boolean deviceWatermarkTriggered, memoryWatermarkTriggered;
        private void scheduleWatermarks() {
            if (performance.cacheWatermarks == null || shuttingDown || stopped || !watermarkQueued.compareAndSet(false, true)) return;
            try { scheduler.execute(() -> {
                watermarkQueued.set(false);
                synchronized (completionLock) { applyWatermarks(); }
            }); } catch (java.util.concurrent.RejectedExecutionException ignored) { watermarkQueued.set(false); }
        }
        private void applyWatermarks() {
            if (shuttingDown || stopped) return;
            var policy = performance.cacheWatermarks;
            if (memoryCache != null && policy.memory().enabled()) {
                int used = memoryCache.usedBlocks();
                memoryWatermarkTriggered |= used >= policy.memory().highBlocks(memoryCache.totalBlocks());
                if (memoryWatermarkTriggered) memoryCache.evictTo(policy.memory().lowBlocks(memoryCache.totalBlocks()));
                memoryWatermarkTriggered &= memoryCache.usedBlocks() > policy.memory().lowBlocks(memoryCache.totalBlocks());
            }
            if (!policy.device().enabled() || !(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    ? performance.decodeDeviceCache : performance.prefillDeviceCache)) return;
            int total = cache.totalBlocks();
            int used = cache.usedBlocks() - pendingWatermarkBlocks;
            deviceWatermarkTriggered |= used >= policy.device().highBlocks(total);
            if (!deviceWatermarkTriggered) return;
            int needed = used - policy.device().lowBlocks(total);
            if (needed <= 0) { deviceWatermarkTriggered = false; return; }
            if (memoryCache == null) {
                for (var keys : cache.watermarkCandidates(needed))
                    cache.evict(keys);
                return;
            }
            // BlockTreeEvictor submits one asynchronous descriptor batch, then
            // rechecks the watermark after settlement; do not reserve the whole round.
            var plans = new ArrayList<List<Long>>();
            var copies = new ArrayList<MemoryWrite>();
            int descriptors = 0, sourceBlocks = 0;
            for (var keys : cache.watermarkCandidates(Math.min(needed, performance.hostTransferPolicy.descriptors()))) {
                var device = cache.pinExisting(keys);
                if (device == null) continue;
                var host = memoryCache.beginWrite(keys);
                if (host == null) {
                    cache.release(device); copies.forEach(copy -> copy.finish(false));
                    return;
                }
                plans.add(keys); copies.add(new MemoryWrite(host, device));
                descriptors += host.blocks(); sourceBlocks += keys.size();
            }
            if (copies.isEmpty()) return;
            watermarkWrites.addAll(copies); pendingWatermarkBlocks += sourceBlocks;
            hostTransfer.submit(false, descriptors, 0).whenComplete((ignored, error) -> {
                synchronized (completionLock) {
                    for (int index = 0; index < copies.size(); index++) {
                        var copy = copies.get(index); var keys = plans.get(index);
                        if (!watermarkWrites.remove(copy)) continue;
                        pendingWatermarkBlocks -= keys.size();
                        copy.finish(error == null);
                        if (error == null && copy.epoch == crashEpoch.get()) cache.evict(keys);
                    }
                }
                if (error == null) scheduleWatermarks();
            });
        }
        private void abortWatermarks() {
            synchronized (completionLock) {
                watermarkWrites.forEach(copy -> copy.finish(false)); watermarkWrites.clear();
                pendingWatermarkBlocks = 0; deviceWatermarkTriggered = memoryWatermarkTriggered = false;
            }
        }
        private java.util.function.LongFunction<java.util.concurrent.CompletableFuture<Void>> hostLoadBackend = this::startHostLoad;
        private java.util.concurrent.CompletableFuture<Void> startHostLoad(long id) {
            if (hostTransfer != null) {
                var read = memoryReads.get(id);
                return read == null ? java.util.concurrent.CompletableFuture.failedFuture(new java.util.concurrent.CancellationException("host load ownership ended"))
                        : hostTransfer.submit(true, read.blocks(), performance.hostLoadMs);
            }
            var done = new java.util.concurrent.CompletableFuture<Void>();
            scheduler.schedule(() -> done.complete(null), performance.hostLoadMs, TimeUnit.MILLISECONDS);
            return done;
        }
        void setHostLoadBackend(java.util.function.LongFunction<java.util.concurrent.CompletableFuture<Void>> backend) { hostLoadBackend = backend; }
        private java.util.function.LongFunction<java.util.concurrent.CompletableFuture<Void>> remoteLoadBackend = id -> java.util.concurrent.CompletableFuture.completedFuture(null);
        void setRemoteLoadBackend(java.util.function.LongFunction<java.util.concurrent.CompletableFuture<Void>> backend) { remoteLoadBackend = backend; }

        private final class MemoryWrite {
            final MockMemoryBlockCache.WriteLease host;
            final MockLruBlockCache.BlockLease device;
            final long epoch;
            boolean closed;
            MockLruBlockCache.BlockLease deferredRelease;
            synchronized boolean deferRelease(MockLruBlockCache.BlockLease lease) {
                if (closed || device != null) return false;
                deferredRelease = lease;
                return true;
            }
            MemoryWrite(MockMemoryBlockCache.WriteLease host, MockLruBlockCache.BlockLease device) {
                this.host = host; this.device = device; this.epoch = crashEpoch.get();
            }
            synchronized void finish(boolean success) {
                if (closed) return;
                closed = true;
                if (success && epoch == crashEpoch.get()) {
                    host.commit();
                } else host.close();
                // A crash has already reset GPU reference accounting.
                if (epoch == crashEpoch.get()) {
                    if (device != null) cache.release(device);
                    if (deferredRelease != null) cache.release(deferredRelease);
                }
                wakeFifoOnCapacityRelease();
            }
        }

        private void releaseMemoryRead(long id) {
            var read = memoryReads.remove(id);
            if (read != null) read.close();
        }

        private void abortMemoryCopies(long id) {
            var load = hostLoads.remove(id);
            if (load != null) load.done().cancel(false);
            releaseMemoryRead(id);
            var write = memoryWrites.remove(id);
            if (write != null) write.finish(false);
        }

        private List<BatchMember> prepareMemoryReads(List<BatchMember> members) {
            if (memoryCache == null) return members;
            List<BatchMember> prepared = new ArrayList<>();
            for (var member : members) {
                var old = member.shape();
                long id = old.input().getRequestId();
                synchronized (completionLock) {
                    int gpu = old.hitBlocks() - old.memoryHitBlocks();
                    int host = 0;
                    if (!cancelledRequests.containsKey(id) && !shuttingDown) {
                        var lease = memoryCache.pinRead(reusableKeys(old, false), gpu);
                        var previous = memoryReads.put(id, lease);
                        if (previous != null) previous.close();
                        host = lease.blocks();
                    }
                    var fresh = new MockPerformanceModel.RequestShape(old.input(), old.inputLen(), old.outputLen(),
                            old.blockKeys(), Math.min((long) (gpu + host) * seqSizePerBlock, old.inputLen()),
                            gpu + host, old.nativeKeys(), host);
                    prepared.add(new BatchMember(fresh, member.batchId(), member.dpRank()));
                }
            }
            return prepared;
        }

        private boolean prefillCacheEnabled(MockPerformanceModel.RequestShape shape, String flag) {
            return switch (flag) {
                case "reuse_cache" -> MockCachePolicy.reuse(shape.input(), shape.nativeKeys(), performance.prefillReuseCache);
                case "enable_device_cache" -> performance.prefillDeviceCache;
                case "enable_memory_cache" -> memoryCache != null;
                default -> throw new IllegalArgumentException("unknown cache policy: " + flag);
            };
        }

        private List<Long> reusableKeys(MockPerformanceModel.RequestShape shape, boolean decode) {
            boolean enabled = decode ? performance.decodeReuseCache && !performance.disableDecodeFirstReuse : performance.prefillReuseCache;
            return MockCachePolicy.reuse(shape.input(), shape.nativeKeys(), enabled)
                    ? MockCachePolicy.matchKeys(shape.blockKeys(), shape.inputLen(), seqSizePerBlock, shape.nativeKeys())
                    : List.of();
        }

        private boolean storePrefillDevice(MockPerformanceModel.RequestShape shape) {
            return prefillCacheEnabled(shape, "reuse_cache") && performance.prefillDeviceCache;
        }

        /** Successful FINISHED resource handover, under the same lock as cancellation. */
        private void finishPrefillCache(MockPerformanceModel.RequestShape shape) {
            long id = shape.input().getRequestId();
            if (cancelledRequests.containsKey(id) || shuttingDown || !activeBlockLeases.containsKey(id)) return;
            boolean deviceCache = storePrefillDevice(shape);
            if (deviceCache) {
                // Publish once at FINISHED. Retain the connector's ownership for P->D.
                activeBlockLeases.computeIfPresent(id, (ignored, lease) -> cache.retainComputed(lease, shape.blockKeys()));
            }
            if (memoryCache == null || !prefillCacheEnabled(shape, "reuse_cache")
                    || !prefillCacheEnabled(shape, "enable_memory_cache")) return;
            var host = memoryCache.beginWrite(shape.blockKeys());
            if (host == null) return; // optional cache write failure never fails the request
            // With device indexing disabled, the active request's naked allocation
            // remains the source owner until this synchronous copy has committed.
            var device = deviceCache ? cache.pinExisting(host.keys()) : null;
            if (deviceCache && device == null) { host.close(); return; }
            var copy = new MemoryWrite(host, device);
            var previous = memoryWrites.put(id, copy);
            if (previous != null) previous.finish(false);
            if (hostTransfer == null) {
                if (memoryWrites.remove(id, copy)) copy.finish(true);
            } else {
                hostTransfer.submit(false, host.blocks(), 0).whenComplete((ignored, error) -> {
                    copy.finish(error == null);
                    memoryWrites.remove(id, copy);
                    scheduleWatermarks();
                });
            }
        }

        private final java.util.concurrent.atomic.LongAdder memoryReadBlocks = new java.util.concurrent.atomic.LongAdder();
        private java.util.function.BiConsumer<String, Double> cacheEvictionReporter;

        private void reportCacheEviction(String scope, double ms) {
            try {
                if (cacheEvictionReporter != null) cacheEvictionReporter.accept(scope, ms);
                else reportMetricEvent(Map.of(scope.equals("memory")
                        ? "mock_memory_evicted_block_lifetime_ms" : "mock_cache_evicted_entry_age_ms", ms));
            } catch (RuntimeException error) {
                System.err.println("Whale cache metric reporting failed: " + error);
            }
        }

        MockPerformanceModel.RequestShape matchPrefillMemory(MockPerformanceModel.RequestShape shape) {
            if (memoryCache == null) return shape;
            var keys = reusableKeys(shape, false);
            int gpu = performance.prefillDeviceCache ? cache.prefixHitBlocks(keys) : 0;
            int host = memoryCache.match(keys, gpu);
            int total = gpu + host;
            return new MockPerformanceModel.RequestShape(shape.input(), shape.inputLen(), shape.outputLen(),
                    shape.blockKeys(), Math.min((long) total * seqSizePerBlock, shape.inputLen()),
                    total, shape.nativeKeys(), host);
        }

        private String whalePodIp;
        private final Map<Long, Object> remoteDecodeLeaseOwners = new ConcurrentHashMap<>();
        private final Map<Long, Runnable> remoteDecodeStops = new ConcurrentHashMap<>();
        private final Map<String, io.grpc.ManagedChannel> remoteChannels = new ConcurrentHashMap<>();
        private final Map<Long, RemotePrefillOwner> remotePrefillOwners = new ConcurrentHashMap<>();
        private final java.util.concurrent.ThreadPoolExecutor prepareExecutor;
        private final ThreadLocal<Throwable> remotePreparationError = new ThreadLocal<>();
        private volatile String processGeneration = java.util.UUID.randomUUID().toString();

        private final class RemotePrefillOwner {
            final MockPrefillSession session;
            volatile MockRemoteDecodeStream stream;
            boolean admitted;
            boolean closed;

            RemotePrefillOwner(MockPrefillSession session) { this.session = session; }

            synchronized void close() {
                if (closed) return;
                closed = true;
                if (stream != null) stream.close();
            }

            void failed(Throwable error) {
                long id = session.shape.input().getRequestId();
                synchronized (this) {
                    if (closed || !admitted || !remotePrefillOwners.remove(id, this)) return;
                    closed = true;
                }
                // Do not take scheduler/completion locks while holding the owner
                // monitor: scheduler teardown closes this owner in the reverse order.
                if (stream != null) stream.close();
                var details = MockRpcErrors.downstream(error, 8209);
                requestErrors.putIfAbsent(id, details);
                details = requestErrors.get(id);
                synchronized (completionLock) {
                    if (fifoOwned.contains(id)) { deferBoundaryCancellation(id, false); wakeFifoOnCapacityRelease(); }
                }
                EngineRpcService.GenerateOutputsPB failure = EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(id).setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                .setErrorCodeValue((int) details.getErrorCode()).setErrorMessage(details.getErrorMessage())).build();
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue = requests.output(id);
                if (queue != null) queue.offer(failure);
                // Keep the failure in an unattached Fetch context, so a late
                // Fetch observes the same typed terminal rather than success.
                session.fail(failure);
                requests.state(id, "failed");
                if (session.hasContinued()) {
                    prefillSessions.remove(id, session);
                    requests.removeOutput(id, queue);
                    session.close();
                } else {
                    continuePrefillSession(session);
                }
                if (!fifoOwned.contains(id)) releaseBlockLease(id);
            }
        }

        private io.grpc.ManagedChannel decodeChannel(EngineRpcService.GenerateInputPB input) {
            EngineRpcService.RoleAddrPB target = input.getGenerateConfig().getRoleAddrsList().stream()
                    .filter(addr -> RoleTypeProtoConverter.fromRoleAddr(addr) == RoleType.DECODE)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Whale prefill requires a Decode route"));
            if (target.getIp().isBlank() || target.getGrpcPort() < 1 || target.getGrpcPort() > 65535) {
                throw new IllegalArgumentException("Whale Decode route requires IP and valid gRPC port");
            }
            String endpoint = target.getIp() + ":" + target.getGrpcPort();
            return remoteChannels.computeIfAbsent(endpoint,
                    ignored -> io.grpc.ManagedChannelBuilder.forAddress(target.getIp(), target.getGrpcPort())
                            .usePlaintext().disableRetry().build());
        }

        private boolean prepareRemoteDecode(MockPerformanceModel.RequestShape shape, long batchId,
                                            int dpRank, boolean automaticContinuation) {
            var channel = decodeChannel(shape.input());
            long id = shape.input().getRequestId();
            MockPrefillSession session = new MockPrefillSession(shape, batchId, dpRank, automaticContinuation, null);
            session.remoteDecode = true;
            RemotePrefillOwner owner = new RemotePrefillOwner(session);
            if (remotePrefillOwners.putIfAbsent(id, owner) != null) return false;
            try {
                synchronized (owner) {
                    if (owner.closed) throw io.grpc.Status.CANCELLED.asRuntimeException();
                    owner.stream = new MockRemoteDecodeStream(channel, shape.input(), processGeneration,
                            performance.pdPolicy.rpcMs(beginRequest(shape.input()).remainingMs()),
                            value -> {
                                if (remotePrefillOwners.get(id) != owner) return;
                                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue = requests.output(id);
                                if (queue != null) queue.offer(value);
                            }, error -> responseExecutor.execute(() -> owner.failed(error)),
                            () -> responseExecutor.execute(() -> {
                                if (remotePrefillOwners.remove(id, owner)) {
                                    requests.removeOutput(id);
                                    finishRequest(id);
                                }
                            }));
                    if (allocationCancelled(id, () -> hasFencedRequestId(id))) owner.stream.close();
                }
                Context preparation = Context.current();
                Context.CancellationListener stop = ignored -> owner.stream.close();
                preparation.addListener(stop, Runnable::run);
                try {
                    long remaining = performance.pdPolicy.rpcMs(beginRequest(shape.input()).remainingMs());
                    if (remaining == Long.MAX_VALUE) owner.stream.allocated().get();
                    else owner.stream.allocated().get(Math.max(1, remaining), TimeUnit.MILLISECONDS);
                } finally { preparation.removeListener(stop); }
                synchronized (owner) {
                    if (owner.closed || allocationCancelled(id, () -> hasFencedRequestId(id)))
                        throw io.grpc.Status.CANCELLED.withDescription("Prefill cancelled during ALLOCATE").asRuntimeException();
                    prefillSessions.put(id, session);
                    owner.admitted = true;
                    return true;
                }
            } catch (Exception error) {
                if (error instanceof InterruptedException) Thread.currentThread().interrupt();
                remotePreparationError.set(error instanceof java.util.concurrent.ExecutionException
                        && error.getCause() != null ? error.getCause() : error);
                remotePrefillOwners.remove(id, owner);
                owner.close();
                return false;
            }
        }

        private void awaitDecodeChannel(io.grpc.ManagedChannel channel, long id, EngineRpcService.GenerateInputPB input) throws Exception {
            long budget = performance.pdPolicy.readyMs(beginRequest(input).remainingMs());
            long deadline = budget > 0 ? MockPdPolicy.deadline(System.nanoTime(), budget) : System.nanoTime();
            while (channel.getState(true) != io.grpc.ConnectivityState.READY) {
                if (allocationCancelled(id, () -> hasFencedRequestId(id)))
                    throw MockRpcErrors.error(io.grpc.Status.CANCELLED, 8100, "channel readiness cancelled");
                long remaining = MockPdPolicy.remaining(deadline, System.nanoTime());
                if (remaining == 0) throw MockRpcErrors.error(io.grpc.Status.UNAVAILABLE, 8201, "get ready grpc connection failed");
                var changed = new java.util.concurrent.CompletableFuture<Void>();
                var state = channel.getState(true);
                if (state == io.grpc.ConnectivityState.READY) return;
                channel.notifyWhenStateChanged(state, () -> changed.complete(null));
                try { changed.get(Math.min(50, remaining), TimeUnit.MILLISECONDS); }
                catch (java.util.concurrent.TimeoutException ignored) { }
            }
        }

        private void continueRemoteDecode(MockPrefillSession session) {
            long id = session.shape.input().getRequestId();
            RemotePrefillOwner owner = remotePrefillOwners.get(id);
            if (owner == null) return;
            var queue = requests.output(id);
            owner.stream.load().orTimeout(performance.pdPolicy.loadMs(beginRequest(session.shape.input()).remainingMs()),
                    TimeUnit.MILLISECONDS).whenCompleteAsync((ignored, error) -> {
                if (error != null) {
                    if (error instanceof java.util.concurrent.TimeoutException)
                        error = MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 8300, "KV load deadline exhausted");
                    owner.failed(error); owner.close(); return;
                }
                synchronized (owner) {
                    if (owner.closed || cancelledRequests.containsKey(id)) return;
                    try {
                        if (session.shape.outputLen() == 1) {
                            owner.stream.completePrefill().whenCompleteAsync((done, failure) -> {
                                if (failure != null) owner.failed(failure);
                                else if (queue != null) queue.offer(buildOutput(session.shape, true));
                            }, responseExecutor);
                        } else owner.stream.generate(0);
                        releaseBlockLease(id);
                        prefillSessions.remove(id, session);
                        session.close();
                    } catch (RuntimeException failure) { owner.failed(failure); }
                }
            }, responseExecutor);
        }

        void setWhaleRemote(boolean enabled) {
            whaleRemote = enabled;
        }

        boolean isWhaleRemote() { return whaleRemote; }

        @Override
        public StreamObserver<EngineRpcService.GenerateRequestPB> remoteGenerate(
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            if (!admitRpc(observer)) {
                return new StreamObserver<>() {
                    public void onNext(EngineRpcService.GenerateRequestPB ignored) { }
                    public void onError(Throwable ignored) { }
                    public void onCompleted() { }
                };
            }
            if (!whaleRemote || roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                return super.remoteGenerate(observer);
            }
            return new MockRemoteDecodeCall(this::allocateRemoteDecode, observer, responseExecutor);
        }

        private MockRemoteDecodeCall.Lease allocateRemoteDecode(
                EngineRpcService.GenerateInputPB input, String clientId, Runnable stoppedCallback) {
            if (!supportsSingleSequence(input))
                throw io.grpc.Status.UNIMPLEMENTED.withDescription("mock supports only single-sequence output")
                        .asRuntimeException();
            input = MockRequestRules.normalize(input, true);
            long id = input.getRequestId();
            Object owner = new Object();
            beginRequest(input);
            MockPerformanceModel.RequestShape shape = performance.shape(input, cache);
            var invalidInput = fifoInputError(shape);
            if (invalidInput != null) {
                finishRequest(id);
                throw MockRpcErrors.error(MockRpcErrors.status(invalidInput), invalidInput.getErrorCode(), invalidInput.getErrorMessage());
            }
            synchronized (decodeQueueLock) {
                if (stopped || shuttingDown || runningTasks.containsKey(id)
                        || cancelledRequests.containsKey(id)
                        || remoteDecodeLeaseOwners.putIfAbsent(id, owner) != null) {
                    throw io.grpc.Status.ALREADY_EXISTS.withDescription("Decode request is already owned or closed")
                            .asRuntimeException();
                }
                decodeAllocationInProgress.add(id);
            }
            boolean allocated = false;
            try {
                if (!reserveDecodeLease(id, shape)) {
                    throw allocationError.get();
                }
                synchronized (decodeQueueLock) {
                    if (stopped || shuttingDown || cancelledRequests.containsKey(id)) {
                        throw io.grpc.Status.CANCELLED.withDescription("Decode stopped during ALLOCATE")
                                .asRuntimeException();
                    }
                    decodeWaitingForKv.add(id);
                    remoteDecodeStops.put(id, stoppedCallback);
                    runningTasks.put(id, task(shape, -1L, 0, EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED));
                    pendingRequests.incrementAndGet();
                    acceptedCount.incrementAndGet();
                    requests.state(id, "waiting_for_kv");
                    recordLifecycleStart(id, -1L, "remote_allocate");
                    recordEventArrival(id);
                    lastEnqueueTime.set(System.nanoTime());
                    allocated = true;
                }
            } finally {
                synchronized (decodeQueueLock) {
                    decodeAllocationInProgress.remove(id);
                    if (!allocated) {
                        remoteDecodeLeaseOwners.remove(id, owner);
                        releaseBlockLease(id);
                        finishRequest(id);
                    }
                }
            }
            return new MockRemoteDecodeCall.Lease() {
                private java.util.concurrent.CompletableFuture<Void> loading;
                @Override public synchronized java.util.concurrent.CompletableFuture<Void> load() {
                    if (remoteDecodeLeaseOwners.get(id) != owner)
                        return java.util.concurrent.CompletableFuture.failedFuture(io.grpc.Status.CANCELLED.asRuntimeException());
                    loading = remoteLoadBackend.apply(id);
                    return loading.copy().orTimeout(performance.pdPolicy.loadMs(beginRequest(shape.input()).remainingMs()), TimeUnit.MILLISECONDS)
                            .exceptionallyCompose(error -> java.util.concurrent.CompletableFuture.failedFuture(
                                    error instanceof java.util.concurrent.TimeoutException
                                            ? MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 8300, "load cache timeout")
                                            : error));
                }
                @Override
                public boolean start(LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> outputs) {
                    synchronized (decodeQueueLock) {
                        if (remoteDecodeLeaseOwners.get(id) != owner) {
                            return false;
                        }
                        requests.output(id, outputs);
                        return scheduleDecodeCompletion(shape, -1L, outputs);
                    }
                }

                @Override
                public synchronized void cancel() {
                    if (loading != null) loading.cancel(false);
                    synchronized (decodeQueueLock) {
                        if (remoteDecodeLeaseOwners.remove(id, owner)) {
                            FastRpcService.this.cancel(id, false, false);
                        }
                    }
                }

                @Override
                public void completePrefill() {
                    if (!completeAllocatedPrefill(id, shape)) throw io.grpc.Status.FAILED_PRECONDITION.asRuntimeException();
                    completed();
                }

                @Override
                public void completed() {
                    synchronized (decodeQueueLock) {
                        if (remoteDecodeLeaseOwners.remove(id, owner)) remoteDecodeStops.remove(id);
                    }
                }
            };
        }
        /** DecodeRpcServer::PREFILL_COMPLETE settles the reservation without a model step. */
        private boolean completeAllocatedPrefill(long id, MockPerformanceModel.RequestShape shape) {
            synchronized (decodeQueueLock) {
                if (!decodeWaitingForKv.remove(id)) return false;
                var tracked = runningTasks.remove(id);
                pendingRequests.decrementAndGet();
                admitBlockLease(id, shape);
                remoteDecodeStops.remove(id);
                finishRequest(id);
                requests.state(id, "completed");
                recordLifecycleEnd(id, false);
                completedCount.incrementAndGet();
                var terminal = tracked == null ? task(shape, -1L, 0, EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED) : tracked;
                publishCompletion(terminal.toBuilder().setEndTimeMs(System.currentTimeMillis())
                        .setExecutionTimeMs(0).setIterateCount(0).build());
                clearUpstreamOwnership(id);
                return true;
            }
        }

        /** Bound for the per-engine request_lifecycle map (Python _prune_lifecycle cap). */
        private static final int LIFECYCLE_CAP = 10_000;
        /** Bound for the cancelled_rids history exposed in the Python snapshot schema. */
        private static final int CANCELLED_RID_CAP = 10_000;
        /** Window of recent execution times used for prefill_ms/decode_ms avg+p99 (Python keeps 100). */
        private static final int RECENT_TIME_CAP = 100;
        /** Grace between a crash_after trigger and the gRPC port kill: the
         *  crash-triggering EMPTY ack must flush to the wire before the socket
         *  dies (the master's BATCH_ACK_UNCERTAIN fence contract reads an empty
         *  ack, not a connection reset); 200ms is far inside the master's
         *  3-strike retire scale. */
        static final long CRASH_PORT_KILL_DELAY_MS = 200L;
        /** Real RpcServerRuntimeMeta retains the newest 1000 finished records. */
        static final int DEFAULT_COMPLETION_RETAIN_WINDOW = 1000;

        private static long currentTimeUs() {
            var now = java.time.Instant.now();
            return now.getEpochSecond() * 1_000_000L + now.getNano() / 1000;
        }

        private final String engineName;
        // Mock process identity, deliberately distinct from master endpoint generation.
        private volatile String engineIncarnation = java.util.UUID.randomUUID().toString();
        private final String host;
        private final String roleName;
        private final EngineRpcService.RoleTypePB roleType;
        private final int grpcPort;
        /**
         * Cluster-shared JSONL event log (engine_events.jsonl), injected via
         * {@link #setEngineEventLog} after construction (main wires the file
         * opened from {@code --events-file}; tests inject per-service logs).
         * Null = event streaming disabled — the per-request terminal rows
         * (prefill_done / decode_done) are then simply not written, exactly
         * like the stdout trace lines they replaced.
         */
        private volatile EngineEventLog engineEventLog;
        /** Live event/metric stamps share one identity and are consumed together.
         * This table is separate from the bounded diagnostic lifecycle history. */
        private record EventTiming(long arrivalMs, long startMs) {
            private static final EventTiming EMPTY = new EventTiming(0L, 0L);
        }
        private final ConcurrentHashMap<Long, EventTiming> eventTimings = new ConcurrentHashMap<>();
        private final Map<Integer, FastRpcService> services;
        private final ScheduledExecutorService scheduler;
        private final MockPerformanceModel performance;
        private final MockLruBlockCache cache;
        private final ClusterStats stats;
        /** Reported token capacity (per-role; pool is ceil(total/spb) blocks). */
        private final long totalKvTokens;
        /** Blocks per cache block (spb) — the pool's token<->block conversion factor. */
        private final int seqSizePerBlock;
        private JsonNode effectiveConfig;
        private final int decodeMaxConcurrency;
        // Per-method RPC counters (Python _rpc_counts, snapshot "rpc_counts").
        private final AtomicLong rpcEnqueueBatch = new AtomicLong();
        private final AtomicLong rpcGenerateStream = new AtomicLong();
        private final AtomicLong rpcFetchResponse = new AtomicLong();
        private final AtomicLong rpcCancel = new AtomicLong();
        // Bounded request lifecycle map keyed by request id (Python _request_lifecycle).
        private final LinkedHashMap<Long, RequestLifecycle> requestLifecycles = new LinkedHashMap<>();
        // Bounded cancelled rid history (Python _cancelled / snapshot "cancelled_rids").
        private final LinkedHashSet<Long> cancelledRidHistory = new LinkedHashSet<>();
        // Priority-Cancel terminal records mirror the C++ Prefill contract: once a
        // request accepted priority cancellation, retries stay ACCEPTED even
        // after the live ownership entry has been removed.  Keep this separate
        // from client-cancel history so a normal terminal remains NOT_FOUND.
        private final ExpiringRequestIds priorityCancelledRequestIds = new ExpiringRequestIds();
        // Absent-fence terminal records mirror the C++ Prefill ABSENT_FENCE
        // contract: a Cancel for a rid this engine NEVER saw installs a
        // fence; any racing later Enqueue of the same rid is rejected with
        // the typed 8429 (PRIORITY_PREEMPTED) error before it reaches the
        // scheduler.  Kept separate from priorityCancelledRequestIds (the
        // ACTIVE_CANCEL marker, whose cancel retries stay ACCEPTED) and from
        // cancelledRidHistory (terminal history, whose late cancels answer
        // NOT_FOUND — seen-but-terminal, the production recently-seen set).
        private final ExpiringRequestIds fencedRequestIds = new ExpiringRequestIds();
        private final ExpiringRequestIds recentlySeenRequests = new ExpiringRequestIds();
        // Recent execution times for snapshot prefill_ms_*/decode_ms_* fields.
        private final ArrayDeque<Double> recentPrefillTimes = new ArrayDeque<>();
        private final ArrayDeque<Double> recentDecodeTimes = new ArrayDeque<>();
        private final AtomicLong statusVersion = new AtomicLong();
        private final AtomicLong completionVersion = new AtomicLong(currentTimeUs());
        private volatile CacheSyncFixture cacheSyncFixture;
        private volatile boolean naturalCacheSeeded;
        private final MockCacheKeyIndex cacheKeyIndex = new MockCacheKeyIndex();
        private final Map<Integer, AtomicLong> nextPrefillAvailableNanosByDp = new ConcurrentHashMap<>();
        // ── KV capacity model v2: single block-pool truth ──
        // In-flight requests pin blocks via per-request leases (acquired at
        // prefill enqueue / decode run-start, grown per decode step, handed to
        // the LRU on completion, returned to free on cancel). Occupied tokens
        // are derived from the pool — never tracked as an independent counter.
        private final Map<Long, MockLruBlockCache.BlockLease> activeBlockLeases = new ConcurrentHashMap<>();
        /** Terminal scheduler admission/growth failures; temporary ALLOCATE
         * backpressure does not increment a failed-request counter. */
        private final LongAdder kvAdmissionFails = new LongAdder();
        /** Permanent physical-capacity failures, shared by P and D (domain 602). */
        private final LongAdder prefillLackMemRejects = new LongAdder();
        /** Decode prefix-reuse blocks accumulated (KV v2 fix #5): every
         *  acquireWithReuse hit key — the net-demand deduction the decode
         *  engine makes against its OWN LRU. Cumulative counter (never
         *  drained): /metrics carries it as mock_engine_decode_reuse_blocks_total,
         *  /snapshot as decode_reuse_blocks. */
        private final LongAdder decodeReuseBlocks = new LongAdder();
        /** Mock key-level cache-hit observability: cumulative counters recorded at the prefill
         *  admission hit computation (MockPerformanceModel.shape's prefixHitBlocks
         *  call — BOTH the enqueue-batch path and the direct generate_stream
         *  path). cacheKeyHits = Σ raw prefix-match run lengths (keys),
         *  cacheKeysRequested = Σ request blockKeys sizes; an empty-bh request
         *  adds 0/0 by construction and never contributes. Cumulative counters
         *  (never drained, reset only by crash-restart fresh-process semantics):
         *  /metrics carries them as mock_engine_cache_key_hits_total /
         *  mock_engine_cache_keys_requested_total, /snapshot as cache_key_hits /
         *  cache_keys_requested. */
        private final LongAdder cacheKeyHits = new LongAdder();
        private final LongAdder cacheKeysRequested = new LongAdder();
        /** prefill_async_partial_fail members terminated at the prefill
         *  completion callback (execution-phase injected failures): one bump
         *  per failed member, cumulative (never drained). /snapshot carries
         *  it as prefill_async_partial_fails. */
        private final LongAdder prefillAsyncPartialFails = new LongAdder();
        // Per-engine busy time. Prefill engines accumulate batch execution ms
        // (one forward at a time -> busy == wall-clock occupancy; utilization =
        // busy/elapsed). Decode engines accumulate per-request execution ms under
        // soft concurrency, so busy/elapsed reads as the average concurrent
        // request count, not a <=100% utilization. Exported via /snapshot busy_ms
        // (never per-engine keys on the per-second stats line: 1250 engines would
        // bloat every tick line).
        private final AtomicLong busyMs = new AtomicLong();
        // Per-engine PREFILL batch composition counters (20260903 #8): the
        // engine-internal regroup makes the EXECUTION batch the unit worth
        // observing — these count the regrouped batches this engine actually
        // ran (not the master-composed batches that arrived). Mirrors the
        // cluster-level ClusterStats batch counters, but per-engine so
        // /snapshot consumers (flexlb_test_framework cases, canvas batch-caliber checks)
        // see one engine's own batch composition without cross-engine
        // contamination in shared-stats topologies.
        private final LongAdder prefillBatchesExecuted = new LongAdder();
        private final LongAdder prefillBatchRequestsExecuted = new LongAdder();
        private final AtomicLongArray prefillBatchSizeCounts =
                new AtomicLongArray(PREFILL_BATCH_SIZE_BUCKETS.length + 1);
        private final AtomicInteger maxPrefillBatchSizeExecuted = new AtomicInteger();
        private final AtomicInteger pendingRequests = new AtomicInteger();
        private final AtomicInteger waitingPrefillRequests = new AtomicInteger();
        private final AtomicInteger activePrefillBatches = new AtomicInteger();
        // Requests inside RUNNING prefill batches (a batch may hold several
        // requests). Incremented by shapes.size() when a batch reserves a running
        // slot (admission or pending-queue drain), decremented by the same amount
        // in the batch completion callback — NOT derivable as pendingRequests −
        // waitingPrefillRequests because cancel() decrements pendingRequests for
        // a queued request immediately while waitingPrefillRequests is only
        // adjusted at drain time. Feeds java_mock_stats prefill_running_reqs.
        private final AtomicInteger activePrefillRequests = new AtomicInteger();
        private final AtomicInteger activeDecodeRequests = new AtomicInteger();
        // ── Decode wait queue + unconditional hard concurrency gate ──
        // Pending decode requests waiting for a concurrency slot. Drained by the
        // decode completion callback after a running request finishes.
        // decodeMaxConcurrency is an UNCONDITIONAL hard gate (production
        // semantics: running is capped and excess requests park in an unbounded
        // engine-side waiting queue, mirroring waiting_streams_). The slot is
        // reserved under decodeQueueLock at admission/drain time so
        // activeDecodeRequests can never exceed the cap, and a completion hands
        // its freed slot to one queued request atomically (no lost slot, no
        // over-admission).
        private final ArrayDeque<DecodePendingTask> decodePendingQueue = new ArrayDeque<>();
        private final Object decodeQueueLock = new Object();
        // ── Per-step continuous batching decode engine (production FIFOScheduler
        // alignment, task #69 per-step + MTP fold caliber) ──
        // A decode request admitted into a running slot becomes a DecodeStream.
        // The engine advances ALL running streams one step at a time on a single
        // chained scheduler task: each step emits performance.tokensPerStep()
        // tokens per stream (MTP acceptance fold, production DSv4 ≈ 2.6) and
        // takes decodeStepDelayMs(currentRunningCount) — the step latency
        // (explicit step_ms_by_batch curve or the linear production fit
        // 19.5 + 0.175 × running) re-read at every step boundary, so a
        // batch-size change (completion, top-up, cancel) re-prices the NEXT
        // step. A stream whose step budget (ceil(outputLen / tokensPerStep)
        // steps, pre-computed at admission) is exhausted completes at that step
        // boundary (terminal ownership claimed under decodeQueueLock,
        // publishing outside the lock) and the waiting-queue head is admitted
        // immediately (production top-up semantics). This replaces the former
        // one-shot sleep: decodeMs(outputLen, batchSizeAtAdmission) computed
        // once at admission, immune to later concurrency changes.
        // decode exec statistics now measure the SUM OF ACTUAL STEP DURATIONS
        // (old caliber: one-shot estimate at admission — numbers shift when the
        // running batch size changes mid-flight; both are ms per request).
        private final LinkedHashMap<Long, DecodeStream> decodeRunning = new LinkedHashMap<>();
        /** True while a step tick is pending on the shared scheduler (decodeQueueLock-guarded); at most one per engine. */
        private boolean decodeStepScheduled = false;
        /** Duration of the currently pending step, locked in when the step was
         * armed (decodeQueueLock-guarded): the batch size at THAT boundary
         * prices the whole step, exactly like production where a step's duration
         * is fixed by the batch that entered it. The tick consumes this value
         * for exec accounting, so booked time always matches elapsed time. */
        private long pendingStepDelayMs = 0;

        /** One decode stream occupying a running slot in the per-step loop. */
        private static final class DecodeStream {
            final MockPerformanceModel.RequestShape shape;
            final long batchId;
            final LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue;
            /** Steps left until finished: ceil(outputLen / tokensPerStep) at
             * admission (MTP fold — every step emits tokensPerStep tokens).
             * Decremented once per step. */
            int remainingSteps;
            int emittedOutputTokens;
            double generatedTokens;
            /** Total step budget at admission — with remainingSteps it yields
             * tokens generated so far (tokensPerStep x (total - remaining)),
             * which drives the per-step KV block growth (production incrMalloc). */
            final int totalSteps;
            /** Σ actual step durations (the per-step exec caliber). */
            double accumulatedExecMs;
            long firstTokenTimeMs;
            long initialReuseTokens;
            /** Set under decodeQueueLock when this step's terminal ownership is claimed (cancel may win it first). */
            boolean owned;
            /**
             * Decode batch size (running streams INCLUDING this one) at the
             * terminal claim — the batch this request's last step executed in.
             * Set under decodeQueueLock by claimDecodeTerminalLocked (after the
             * stream's removal from decodeRunning, before the top-up), read by
             * the engine_events.jsonl decode_done row.
             */
            int terminalBatchSize;

            DecodeStream(MockPerformanceModel.RequestShape shape, long batchId,
                         LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue,
                         int totalSteps) {
                this.shape = shape;
                this.batchId = batchId;
                this.responseQueue = responseQueue;
                this.totalSteps = totalSteps;
                this.remainingSteps = totalSteps;
            }
        }
        private final Object prefillQueueLock = new Object();
        private final Set<Long> prefillExecuting = ConcurrentHashMap.newKeySet();
        private final Set<Long> fifoOwned = ConcurrentHashMap.newKeySet();
        private final ThreadLocal<Boolean> schedulerTransition = ThreadLocal.withInitial(() -> false);
        private int admittedContextBatch;
        private long admittedContextTokens;
        private final Map<Long, Boolean> boundaryCancellations = new ConcurrentHashMap<>();
        private boolean decodeBoundaryMaintenance;
        private record DownstreamCancellation(FastRpcService prefill, boolean priority) {}
        private final Map<Long, DownstreamCancellation> boundaryDownstreamCancellations = new ConcurrentHashMap<>();
        private final Map<Long, EngineRpcService.TaskInfoPB> cancelingOverlays = new ConcurrentHashMap<>();
        private final MockFifoPrefillQueue<BatchMember> fifoPrefillQueue = new MockFifoPrefillQueue<>();
        /**
         * Finished-task backlog keyed by monotonically increasing completion
         * version. Multi-consumer delivery protocol (dual-master HA polling):
         * readers only FILTER by their own cursor (version > cursor) and never
         * remove entries — each flexlb instance keeps an independent
         * per-worker cursor, so the earliest poller must not destroy records
         * a slower consumer has not read yet. Bounded by the retain window
         * ({@link #DEFAULT_COMPLETION_RETAIN_WINDOW}) enforced on every insertion.
         */
        private final ArrayDeque<VersionedTask> completions = new ArrayDeque<>();
        /**
         * Publishes completion records and their cursor as one ordered operation.
         * A status reader must never observe a latest version whose record has
         * not yet been inserted, otherwise advancing its cursor loses that
         * completion permanently.
         */
        // All map mutations take the same monitor as completion publication
        // and status snapshots. Queue locks may precede this monitor; never
        // call queue/resource operations while a status snapshot holds it.
        private final Map<Long, EngineRpcService.TaskInfoPB> runningTasks =
                Collections.synchronizedMap(new LinkedHashMap<>());
        private final Object completionLock = runningTasks;
        private final StatusDeliveryFaults statusDeliveryFaults = new StatusDeliveryFaults();
        private volatile boolean statusDeliveryFaultsActive;
        private final boolean statusSnapshotLog = Boolean.parseBoolean(
                System.getenv().getOrDefault("MOCK_STATUS_SNAPSHOT_LOG", "false"));
        private long statusSnapshotSequence;
        private long completionRetentionDrops;
        final MockRequestTable requests = new MockRequestTable();
        private final ThreadLocal<RuntimeException> allocationError = new ThreadLocal<>();

        private MockRequestDeadline beginRequest(EngineRpcService.GenerateInputPB input) {
            long id = input.getRequestId();
            var candidate = new MockRequestDeadline(input.getGenerateConfig().getTimeoutMs());
            var existing = requests.inheritDeadline(id, candidate);
            if (existing != null) return existing;
            long epoch = crashEpoch.get();
            candidate.arm(scheduler, () -> responseExecutor.execute(() -> {
                if (crashEpoch.get() == epoch && requests.deadline(id) == candidate) {
                    var admission = batchAdmissions.get(id);
                    if (admission != null && !admission.published) return;
                    if (runningTasks.containsKey(id) || (requests.output(id) != null) || prefillSessions.containsKey(id))
                        cancel(id, false, false);
                    else cache.capacityChanged();
                }
            }));
            return candidate;
        }

        private void finishRequest(long id) {
            finishPrefillWait(id);
            requests.finish(id);
        }

        // GenerateStateMachine::reportEvent retains the first error, even if
        // another event arrives before the scheduler can commit FINISHED.
        private final Map<Long, EngineRpcService.ErrorDetailsPB> requestErrors = new ConcurrentHashMap<>();
        private EngineRpcService.ErrorDetailsPB cancellationError(long id) {
            return requestErrors.computeIfAbsent(id, this::newCancellationError);
        }
        private EngineRpcService.ErrorDetailsPB newCancellationError(long id) {
            var deadline = requests.deadline(id);
            boolean timeout = deadline != null && deadline.expired();
            return EngineRpcService.ErrorDetailsPB.newBuilder()
                    .setErrorCode(timeout ? 603 : 8100)
                    .setErrorMessage(timeout ? "request deadline exhausted" : "cancelled by client").build();
        }

        private EngineRpcService.GenerateOutputsPB cancellationOutput(long id) {
            EngineRpcService.ErrorDetailsPB error = cancellationError(id);
            return EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(id)
                    .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                            .setErrorCodeValue((int) error.getErrorCode())
                            .setErrorMessage(error.getErrorMessage())).build();
        }

        private MockResponseQueue responseQueue(long id) {
            return new MockResponseQueue(() -> {
                if (!cancelledRequests.containsKey(id)) finishRequest(id);
            });
        }
        // Explicit P->D ownership used by the test-only Cancel channel.  A Prefill
        // may cancel only the Decode selected from that request's role_addrs; it
        // must never scan the cluster for a matching request id.
        private final Map<Long, FastRpcService> downstreamDecodeOwners = new ConcurrentHashMap<>();
        private final Map<Long, FastRpcService> upstreamPrefillOwners = new ConcurrentHashMap<>();
        /** Safety-net TTL for cancelled markers never consumed by a completion callback. */
        private static final long CANCELLED_MARKER_TTL_SECONDS = 600;
        /** Public error contract for a victim canceled by priority preemption. */
        private static final long PRIORITY_PREEMPTED_ERROR_CODE = 8429;

        /** rid -> insertion time (System.nanoTime); consumed by completion callbacks. */
        private final Map<Long, Long> cancelledRequests = new ConcurrentHashMap<>();
        /** Decode-side KV reservation ownership (P-enqueue pre-alignment,
         *  20260903): on THIS prefill engine, rid -> the DECODE engine whose
         *  pool already holds the request's decode block lease, provisioned at
         *  EnqueueBatch Phase 1 (the mock counterpart of the production prepare
         *  stage sending the ALLOCATE RPC to the FlexLB-selected D while the
         *  prefill itself is still executing). The entry is removed when the
         *  reservation is CONSUMED by startDecode (the lease becomes the
         *  running stream's lease, owned by the decode lifecycle) or RELEASED
         *  by cancel / prefill-completion alreadyCancelled / a rejected
         *  hand-off. Absent entry = no live reservation (idempotent release). */
        private final Map<Long, FastRpcService> decodeReservationOwners = new ConcurrentHashMap<>();
        private final Map<Long, MockPrefillSession> prefillSessions = new ConcurrentHashMap<>();
        /** Per-context arbitration, matching DeferredPrefillContext's terminal cause and ACK publication. */
        private final class BatchAdmission {
            final EngineRpcService.GenerateInputPB input;
            MockPerformanceModel.RequestShape shape;
            EngineRpcService.ErrorDetailsPB error, cancellation, streamError;
            volatile boolean published, schedulerOwned;
            boolean preparing = true, groupActive, answered, prepareStarted;
            volatile java.util.concurrent.CompletableFuture<Void> preparation;
            BatchAdmission(EngineRpcService.GenerateInputPB input) { this.input = input; }
            synchronized boolean cancel(boolean priority) {
                if (error != null) return cancellation != null && (!priority || cancellation.getErrorCode() == 8429);
                if (!published && (cancellation == null || priority)) cancellation = priority
                        ? batchError(8429, "preempted during batch admission") : cancellationError(input.getRequestId());
                return true;
            }
            synchronized EngineRpcService.ErrorDetailsPB reject(EngineRpcService.ErrorDetailsPB failure) {
                if (error == null) error = cancellation != null ? cancellation : failure;
                return error;
            }
            synchronized EngineRpcService.ErrorDetailsPB liveError() { return error != null ? error : cancellation; }
            synchronized EngineRpcService.ErrorDetailsPB publish() {
                if (liveError() != null) return reject(liveError());
                groupActive = false; published = true; return null;
            }
        }
        private final Map<Long, BatchAdmission> batchAdmissions = new ConcurrentHashMap<>();
        // Package-local observation points allow deterministic races without a public fault/config API.
        java.util.function.Consumer<String> batchAdmissionObserver;
        private void batchPhase(String phase) {
            var observer = batchAdmissionObserver;
            if (observer != null) observer.accept(phase);
        }
        private static EngineRpcService.ErrorDetailsPB batchError(long code, String message) {
            return EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(message).build();
        }

        private void rejectBatchAdmission(BatchAdmission slot, EngineRpcService.ErrorDetailsPB failure) {
            var error = slot.reject(failure);
            synchronized (slot) { slot.preparing = slot.groupActive = false; }
            long id = slot.input.getRequestId();
            recentlySeenRequests.add(id);
            if (slot.schedulerOwned) {
                requestErrors.putIfAbsent(id, error);
                cancel(id, slot.cancellation != null && slot.cancellation.getErrorCode() == 8429, false); // retain scheduler ownership until its boundary
            } else {
                closePrefillSession(id); releaseReservedDecode(id); releaseBlockLease(id);
                requests.removeOutput(id); finishRequest(id);
                if (slot.cancellation != null) cancel(id, slot.cancellation.getErrorCode() == 8429, false);
            }
            if (!slot.schedulerOwned) requests.state(id, "rejected");
        }

        private void answerBatchError(BatchAdmission slot, EngineRpcService.ErrorDetailsPB failure,
                EngineRpcService.EnqueueBatchResponsePB.Builder response) {
            if (slot.answered) return;
            rejectBatchAdmission(slot, failure);
            response.addErrorsBuilder().setRequestId(slot.input.getRequestId()).setErrorInfo(slot.error);
            slot.answered = true;
        }

        private boolean storeBatchAdmission(BatchAdmission slot) {
            synchronized (slot) {
                if (slot.liveError() != null) return false;
                slot.groupActive = true;
                long id = slot.input.getRequestId();
                if (!autoFetch || !prefillSessions.containsKey(id)) requests.ensureOutput(id, () -> responseQueue(id));
                requests.state(id, "waiting");
                return true;
            }
        }

        private void prepareBatchAdmission(BatchAdmission slot, long batchId) {
            long id = slot.input.getRequestId();
            try {
                var input = MockRequestRules.normalizeQuery(slot.input);
                slot.shape = matchPrefillMemory(performance.shape(input, cache));
                observeCacheDiagnostics(slot.shape);
                cacheKeyHits.add(slot.shape.hitBlocks()); cacheKeysRequested.add(slot.shape.blockKeys().size());
                var decode = findDecodeEngine(input);
                if (!prepareDecodeSession(decode, slot.shape, batchId, 0, autoFetch))
                    rejectBatchAdmission(slot, decodePreparationError(decode, slot.shape));
            } catch (Throwable failure) {
                rejectBatchAdmission(slot, MockRpcErrors.downstream(failure));
            } finally {
                EngineRpcService.ErrorDetailsPB cancelled;
                synchronized (slot) { slot.preparing = false; cancelled = slot.cancellation; }
                if (cancelled != null) cancel(id, cancelled.getErrorCode() == 8429, false);
            }
        }

        private void acceptBatchGroup(EngineRpcService.EnqueueBatchRequestPB request,
                EngineRpcService.EnqueueBatchResponsePB.Builder response) {
            if (stopped || shuttingDown) {
                for (var dp : request.getDpSlotsList()) for (var item : dp.getRequestsList())
                    response.addErrorsBuilder().setRequestId(item.hasInput() ? item.getInput().getRequestId() : 0)
                            .setErrorInfo(batchError(dp.getDpRank() == 0 ? 14 : 3, "Prefill batch server is shutting down"));
                return;
            }
            var inputs = new ArrayList<EngineRpcService.GenerateInputPB>();
            for (var dp : request.getDpSlotsList()) for (var item : dp.getRequestsList()) {
                if (dp.getDpRank() != 0 || !item.hasInput()) response.addErrorsBuilder()
                        .setRequestId(item.hasInput() ? item.getInput().getRequestId() : 0)
                        .setErrorInfo(batchError(3, dp.getDpRank() != 0 ? "EnqueueBatch dp_rank mismatch" : "missing input"));
                else inputs.add(item.getInput());
            }
            var slots = new ArrayList<BatchAdmission>();
            var faults = faultConfig;
            int rejectCount = faults.getEnqueueAckPartialFail() > 0 ? faults.getEnqueueAckPartialFail()
                    : faults.getEnqueueAckErrorCode() != 0 ? Integer.MAX_VALUE : 0;
            try {
                for (int index = 0; index < inputs.size(); index++) {
                    var input = inputs.get(index).toBuilder().setGroupSize(inputs.size())
                            .setGroupId(com.google.protobuf.Int64Value.of(request.getBatchId())).build();
                    long id = input.getRequestId();
                    if (!supportsSingleSequence(input)) {
                        response.addErrorsBuilder().setRequestId(id).setErrorInfo(batchError(12, "mock supports only single-sequence output"));
                        continue;
                    }
                    var slot = new BatchAdmission(input);
                    if (batchAdmissions.putIfAbsent(id, slot) != null) {
                        response.addErrorsBuilder().setRequestId(id).setErrorInfo(batchError(6, "request already exists in active context map"));
                        continue;
                    }
                    if (prefillSessions.containsKey(id) || downstreamDecodeOwners.containsKey(id)
                            || (requests.output(id) != null) || runningTasks.containsKey(id)) {
                        batchAdmissions.remove(id, slot);
                        response.addErrorsBuilder().setRequestId(id).setErrorInfo(batchError(6, "request already exists in active context map"));
                        continue;
                    }
                    slots.add(slot); beginRequest(input); recordEventArrival(id);
                    if (stopped || shuttingDown) rejectBatchAdmission(slot, batchError(14, "Prefill batch server is shutting down"));
                    else if (hasFencedRequestId(id)) rejectBatchAdmission(slot, batchError(8429, "absent fence: request cancelled before admission"));
                    else if (index < rejectCount) rejectBatchAdmission(slot, batchError(
                            faults.getEnqueueAckErrorCode() != 0 ? faults.getEnqueueAckErrorCode() : 13,
                            faults.getEnqueueAckPartialFail() > 0 ? "injected enqueue_ack_partial_fail" : "injected enqueue_ack_error_code"));
                }
                // Register every context first, submit every prepare next, and only then join.
                for (var slot : slots) {
                    if (slot.liveError() != null) { rejectBatchAdmission(slot, slot.liveError()); continue; }
                    slot.preparation = new java.util.concurrent.CompletableFuture<>();
                    try { prepareExecutor.execute(() -> {
                        synchronized (slot) {
                            if (slot.preparation.isDone()) return;
                            slot.prepareStarted = true;
                        }
                        try { prepareBatchAdmission(slot, request.getBatchId()); }
                        finally { slot.preparation.complete(null); }
                    }); } catch (java.util.concurrent.RejectedExecutionException failure) {
                        rejectBatchAdmission(slot, batchError(stopped || shuttingDown ? 14 : 13, "submit prepare task exception: " + failure));
                        slot.preparation.complete(null);
                    }
                }
                for (var slot : slots) if (slot.preparation != null) slot.preparation.join();
                batchPhase("prepared");
                var ready = new ArrayList<BatchAdmission>();
                for (var slot : slots) {
                    if (storeBatchAdmission(slot)) ready.add(slot);
                    else answerBatchError(slot, slot.liveError(), response);
                }
                batchPhase("before_enqueue");
                ready.removeIf(slot -> {
                    var error = slot.liveError();
                    if (error == null) return false;
                    answerBatchError(slot, error, response); return true;
                }); // R1: reject and finish each canceled context before engine stream creation
                var engineSlots = List.copyOf(ready);
                batchPhase("enqueueing");
                // createStream errors are excluded from the mixed-mode check; input-length errors are checked after it.
                for (var slot : ready) {
                    try { MockRequestRules.normalize(slot.shape.input(), false); }
                    catch (RuntimeException failure) { slot.streamError = MockRpcErrors.downstream(failure); }
                }
                ready.removeIf(slot -> slot.streamError != null);
                boolean prefillOnly = ready.stream().anyMatch(slot -> slot.shape.input().getGenerateConfig().getMaxNewTokens() == 0);
                boolean generation = ready.stream().anyMatch(slot -> slot.shape.input().getGenerateConfig().getMaxNewTokens() != 0);
                if (prefillOnly && generation) {
                    for (var slot : ready) slot.streamError = batchError(605, "group cannot mix prefill-only and generation streams");
                } else for (var slot : ready) {
                    var invalid = fifoInputError(slot.shape);
                    if (invalid != null) slot.streamError = invalid;
                }
                ready.removeIf(slot -> slot.streamError != null);
                if (!ready.isEmpty()) {
                    boolean admitted;
                    try { admitted = schedulePrefillCompletion(ready.stream().map(slot -> slot.shape).toList(), request.getBatchId(), 0); }
                    catch (RuntimeException failure) {
                        admitted = false;
                        for (var slot : ready) {
                            slot.schedulerOwned = fifoOwned.contains(slot.input.getRequestId());
                            rejectBatchAdmission(slot, batchError(13, "enqueueMultiple exception: " + failure));
                        }
                    }
                    for (var slot : ready) {
                        slot.schedulerOwned |= admitted;
                        if (!admitted && slot.error == null) slot.streamError = batchError(14, "Prefill batch server is shutting down");
                    }
                }
                batchPhase("after_enqueue");
                // Engine stream errors are candidates until this context-local R2 terminal claim.
                for (var slot : engineSlots) {
                    var failure = slot.streamError != null ? slot.streamError : slot.liveError();
                    if (failure != null) answerBatchError(slot, failure, response);
                }
                for (var slot : engineSlots) {
                    if (slot.answered) continue;
                    long id = slot.input.getRequestId();
                    batchPhase("before_publish");
                    var error = slot.liveError();
                    var deadline = requests.deadline(id);
                    if (error == null && deadline != null && deadline.remainingMs() == 0)
                        error = batchError(4, "request deadline expired before FetchResponse became available");
                    if (error == null) error = armFetchExpiry(id, request.getFetchAttachTimeoutMs());
                    if (error == null) error = slot.publish(); // R2 / publication vs Cancel linearization
                    if (error != null) {
                        answerBatchError(slot, error, response);
                    } else {
                        stats.enqueuedRequests.increment(); acceptedCount.incrementAndGet();
                        response.addSuccessesBuilder().setRequestId(id);
                        slot.answered = true;
                        recordLifecycleStart(id, request.getBatchId(), "enqueue_batch");
                    }
                }
            } catch (RuntimeException failure) {
                for (var slot : slots) if (!slot.answered) {
                    slot.schedulerOwned |= fifoOwned.contains(slot.input.getRequestId());
                    answerBatchError(slot, batchError(13, "batch admission exception: " + failure), response);
                }
            } finally {
                for (var slot : slots) batchAdmissions.remove(slot.input.getRequestId(), slot);
            }
        }
        // Decode ALLOCATE has succeeded; KV is owned but no decode execution
        // slot has been claimed. Guarded by decodeQueueLock with runningTasks.
        private final Set<Long> decodeWaitingForKv = ConcurrentHashMap.newKeySet();
        private final Set<Long> decodeAllocationInProgress = ConcurrentHashMap.newKeySet();
        private volatile boolean autoFetch = false;
        private volatile long fetchAttachTimeoutMs = 600_000L;
        private final AtomicLong fetchAttachExpirations = new AtomicLong();

        private volatile FaultInjectionConfig faultConfig = FaultInjectionConfig.builder().build();
        private final AtomicInteger enqueueCount = new AtomicInteger();
        private volatile boolean admissionClosed;
        private long admissionClosedEpochMs;
        private long admittedRpcs;
        private long admittedRpcsAtClose;
        private long rejectedRpcs;

        synchronized void closeAdmission() {
            if (!admissionClosed) {
                admissionClosed = true;
                admissionClosedEpochMs = System.currentTimeMillis();
                admittedRpcsAtClose = admittedRpcs;
            }
        }

        synchronized void reopenAdmissionAfterFailedRemoval() {
            admissionClosed = false;
            admissionClosedEpochMs = 0;
        }

        synchronized Map<String, Object> admissionSnapshot() {
            return Map.of("admission_open", admissionClosed ? 0 : 1,
                    "admitted_rpcs_total", admittedRpcs,
                    "rejected_rpcs_total", rejectedRpcs,
                    "admission_closed_epoch_ms", admissionClosedEpochMs,
                    "admitted_rpcs_at_close", admittedRpcsAtClose);
        }

        private boolean admitRpc(StreamObserver<?> observer) {
            synchronized (this) {
                if (!admissionClosed) {
                    admittedRpcs++;
                    return true;
                }
                rejectedRpcs++;
            }
            observer.onError(io.grpc.Status.UNAVAILABLE
                    .withDescription("engine admission closed for removal").asRuntimeException());
            return false;
        }

        private volatile boolean stopped = false;
        // ── crash_after true-crash semantics ──
        // Epoch fence against in-flight scheduler callbacks: prefill batch
        // start / completion callbacks, decode step ticks and delayed enqueue
        // processing already queued on the shared scheduler when a crash hits
        // CANNOT be unscheduled — they capture the epoch at trigger time and
        // drop out once crashNow() bumps it, so a late callback can never
        // resurrect state on the memory-wiped engine.
        private final AtomicLong crashEpoch = new AtomicLong();
        // This engine's own gRPC server (set at startEngine / control-plane
        // start_engine time). crash_after shuts it down so the master's health
        // poller hits a dead port and walks the same 3-strike retire path as
        // stop_engine — the difference is what survives: stop_engine keeps
        // every pool for in-place continuation, a crash wipes them (recovery
        // == a reboot from zero).
        private volatile Server grpcServer;
        private MockRpcExecutors.Engine rpcEngine;
        // Set once by drainAndShutdown(): rejects new admissions and disables
        // checkLeakDrain so shutdown-time in-flight requests are not misreported
        // as leaks. Never reset (the process is exiting).
        private volatile boolean shuttingDown = false;
        private final AtomicBoolean leakDetected = new AtomicBoolean(false);
        private final AtomicLong lastEnqueueTime = new AtomicLong(System.nanoTime());
        private final AtomicLong acceptedCount = new AtomicLong();
        private final AtomicLong completedCount = new AtomicLong();
        private final AtomicLong cancelledCount = new AtomicLong();
        // Successful completion counters are kept separate from execution TPS:
        // cancellation after execution starts does not undo work done by a batch.
        private final PrefillTpsMetrics prefillTps = new PrefillTpsMetrics();
        private final PrefillTpsMetrics.Reader prometheusPrefillTps = new PrefillTpsMetrics.Reader();
        private final LongAdder lifetimeContextComputeTokens = new LongAdder();
        private final LongAdder lifetimeContextTokens = new LongAdder();
        private final LongAdder lifetimeContextRequests = new LongAdder();
        private final LongAdder lifetimeGenerateTokens = new LongAdder();
        private final CounterRateMetrics workerStatusRates = new CounterRateMetrics();
        private final CounterRateMetrics cacheStatusRates = new CounterRateMetrics();
        private final CounterRateMetrics.Reader prometheusCacheStatus = new CounterRateMetrics.Reader();
        private final AtomicLong cacheStatusRtUs = new AtomicLong();
        private final CounterRateMetrics.Reader prometheusWorkerStatus = new CounterRateMetrics.Reader();
        private final AtomicLong workerStatusRtUs = new AtomicLong();
        java.util.function.LongConsumer statusMetricReporter = ignored -> {};
        java.util.function.LongConsumer cacheMetricReporter = ignored -> {};
        private final CounterRateMetrics decodeTps = new CounterRateMetrics();
        private final CounterRateMetrics.Reader prometheusDecodeTps = new CounterRateMetrics.Reader();
        private final AtomicLong hitTokensTotal = new AtomicLong();
        private final ExecutorService responseExecutor;
        /**
         * Per-frame poll timeout for this engine's response pump. Overrides via
         * {@link #setResponsePollTimeoutMs(long)} (set from Config at startEngine
         * time; tests inject short timeouts to exercise the timeout path).
         */
        private volatile long responsePollTimeoutMs = DEFAULT_RESPONSE_POLL_TIMEOUT_MS;
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
                       int totalBlocks,
                       ClusterStats stats,
                       long totalKvTokens,
                       int decodeMaxConcurrency) {
            this.engineName = engineName;
            this.host = host;
            this.totalKvTokens = totalKvTokens;
            this.seqSizePerBlock = Math.max(1, performance.blockSize());
            this.decodeMaxConcurrency = decodeMaxConcurrency;
            this.roleName = roleName.toUpperCase();
            this.roleType = roleType;
            this.grpcPort = grpcPort;
            this.services = services;
            this.scheduler = scheduler;
            this.performance = performance.forEngine();
            if (performance.hostTransferPolicy != null && roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                hostTransfer = new MockHostTransfer(performance.hostTransferPolicy);
            int prepareThreads = performance.pdPolicy.prepareThreads();
            this.prepareExecutor = new java.util.concurrent.ThreadPoolExecutor(prepareThreads, prepareThreads,
                    0, TimeUnit.MILLISECONDS, new java.util.concurrent.ArrayBlockingQueue<>(prepareThreads),
                    task -> Thread.ofVirtual().name("mock-prepare-" + grpcPort).unstarted(task),
                    new java.util.concurrent.ThreadPoolExecutor.AbortPolicy());
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL && performance.memoryCacheBlocks > 0)
                this.memoryCache = new MockMemoryBlockCache(performance.memoryCacheBlocks,
                        performance.memoryPrefixTree, ms -> reportCacheEviction("memory", ms), System::nanoTime);
            this.cache = roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    && performance.decodeReserveBlockRatio != null
                    ? new MockLruBlockCache(totalBlocks, performance.decodeReserveBlockRatio / 100.0, true)
                    : roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                    && performance.prefillReserveBlockRatio != null
                    ? new MockLruBlockCache(totalBlocks, performance.prefillReserveBlockRatio / 100.0)
                    : new MockLruBlockCache(totalBlocks);
            if (performance.observedReserveBlocks != null) cache.setObservedReserveBlocks(performance.observedReserveBlocks);
            this.cache.setKeyListener(cacheKeyIndex::update);
            if (memoryCache != null) memoryCache.setKeyListener(cacheKeyIndex::update);
            // Cache invalidation must not depend on event-log/monitor wiring.
            this.cache.setEvictionListener(this::onCacheEviction);
            this.cache.setPrefixTreeEnabled(roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    ? performance.decodeGpuPrefixTree : performance.prefillGpuPrefixTree);
            this.responseExecutor = Executors.newCachedThreadPool(r ->
                    Thread.ofVirtual().name("mock-response-poller-" + grpcPort).unstarted(r));
            this.stats = stats;
        }

        @Override
        public void enqueueBatch(EngineRpcService.EnqueueBatchRequestPB request,
                                 StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
            if (!admitRpc(observer)) return;
            stats.enqueueRpcs.increment();
            rpcEnqueueBatch.incrementAndGet();
            EngineRpcService.EnqueueBatchResponsePB.Builder response =
                    EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());

            var batchIds = new HashSet<Long>();
            boolean duplicate = request.getDpSlotsList().stream().flatMap(slot -> slot.getRequestsList().stream())
                    .filter(EngineRpcService.EnqueueBatchExternalInputPB::hasInput)
                    .anyMatch(input -> !batchIds.add(input.getInput().getRequestId()));
            if (duplicate) {
                request.getDpSlotsList().forEach(slot -> slot.getRequestsList().forEach(input -> response.addErrorsBuilder()
                        .setRequestId(input.hasInput() ? input.getInput().getRequestId() : 0)
                        .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                .setErrorCode(input.hasInput() ? 6 : 3)
                                .setErrorMessage(input.hasInput() ? "duplicate request_id in EnqueueBatch" : "missing input"))));
                observer.onNext(response.build()); observer.onCompleted(); return;
            }
            if (stopped || shuttingDown) {
                for (var dp : request.getDpSlotsList()) for (var item : dp.getRequestsList())
                    response.addErrorsBuilder().setRequestId(item.hasInput() ? item.getInput().getRequestId() : 0)
                            .setErrorInfo(batchError(dp.getDpRank() == 0 ? 14 : 3,
                                    dp.getDpRank() == 0 ? "Prefill batch server is shutting down" : "EnqueueBatch dp_rank mismatch"));
                observer.onNext(response.build()); observer.onCompleted(); return;
            }
            if (faultConfig.isNoRespond()) {
                return; // explicit RPC blackhole; not a missing client Fetch
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
                // True-crash semantics (process death, not a graceful stop):
                // ack THIS batch first — the crash-triggering enqueue must
                // surface as an EMPTY ack (neither successes nor errors) to
                // preserve the master-side BATCH_ACK_UNCERTAIN fence contract —
                // then die: wipe ALL per-engine memory and kill the gRPC port.
                // The port-kill itself runs a beat later (CRASH_PORT_KILL_DELAY_MS)
                // so the ack flushes to the wire while the socket is still open.
                stopped = true;
                observer.onNext(response.build());
                observer.onCompleted();
                crashNow();
                return;
            }

            Runnable process = Context.ROOT.wrap(() -> {
                acceptBatchGroup(request, response);
                if (faultConfig.isEnqueueAckDrop()) {
                    observer.onNext(EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId()).build());
                } else {
                    long count = request.getDpSlotsList().stream().mapToLong(EngineRpcService.EnqueueBatchDpSlotPB::getRequestsCount).sum();
                    if (response.getSuccessesCount() + response.getErrorsCount() != count)
                        throw new IllegalStateException("EnqueueBatch result size mismatch");
                    observer.onNext(response.build());
                }
                observer.onCompleted();
            });

            lastEnqueueTime.set(System.nanoTime());

            if (faultConfig.getEnqueueDelayMs() > 0) {
                // Crash fence on the delayed-process path (enqueue_delay and
                // crash_after can be co-injected): if the engine crashed while
                // this process sat in the scheduler queue, it must not admit
                // anything into the wiped state.
                final long epoch = crashEpoch.get();
                scheduler.schedule(() -> {
                    if (crashEpoch.get() == epoch) {
                        process.run();
                    }
                }, faultConfig.getEnqueueDelayMs(), TimeUnit.MILLISECONDS);
            } else {
                process.run();
            }
        }

        @Override
        public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                    StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
            long statusStarted = System.nanoTime();
            stats.statusRpcs.increment();
            // status_no_respond: hang the RPC — no onNext/onCompleted ever,
            // mirroring generateStreamCall's noRespond handling.
            if (faultConfig.isStatusNoRespond()) {
                return;
            }
            long requestedVersion = request.getLatestFinishedVersion();
            long latestVersion;
            List<VersionedTask> visibleCompletions = new ArrayList<>();
            List<EngineRpcService.TaskInfoPB> visibleRunning;
            ObjectNode snapshotEvent = null;
            // Readers filter their own cursor without consuming shared records.
            // A lagging reader permanently loses records evicted from the fixed
            // retention window; the engine does not compensate for that loss.
            synchronized (completionLock) {
                if (!stopped && statusDeliveryFaultsActive) {
                    statusDeliveryFaults.releaseReady(System.nanoTime(), this::publishCompletionNow);
                }
                latestVersion = requestedVersion;
                visibleRunning = new ArrayList<>(runningTasks.values());
                cancelingOverlays.forEach((id, task) -> { if (!runningTasks.containsKey(id)) visibleRunning.add(task); });
                for (VersionedTask completion : completions) {
                    if (completion.version > requestedVersion) {
                        visibleCompletions.add(completion);
                        latestVersion = Math.max(latestVersion, completion.version);
                    }
                }
                if (statusSnapshotLog && engineEventLog != null) {
                    snapshotEvent = OBJECT_MAPPER.createObjectNode();
                    snapshotEvent.put("event", "worker_status_snapshot");
                    snapshotEvent.put("engine", engineName);
                    snapshotEvent.put("port", grpcPort);
                    snapshotEvent.put("role", roleName);
                    snapshotEvent.put("sequence", ++statusSnapshotSequence);
                    snapshotEvent.put("requested_version", requestedVersion);
                    snapshotEvent.put("latest_version", latestVersion);
                    snapshotEvent.put("retention_drops", completionRetentionDrops);
                    snapshotEvent.put("alive", !stopped);
                    var running = snapshotEvent.putArray("running");
                    visibleRunning.forEach(t -> running.add(t.getRequestId()));
                    var retained = snapshotEvent.putArray("retained");
                    completions.forEach(t -> retained.addObject()
                            .put("rid", t.task.getRequestId()).put("version", t.version));
                    var returned = snapshotEvent.putArray("returned");
                    visibleCompletions.forEach(t -> returned.add(t.task.getRequestId()));
                }
            }
            long statusReadUs = Math.max(0, (System.nanoTime() - statusStarted) / 1000);
            // Append the captured snapshot only: no extra RPC or debug HTTP
            // observer, and no file I/O inside the status critical section.
            if (snapshotEvent != null) {
                engineEventLog.write(snapshotEvent);
            }
            // Capacity model v2: used/available both derive from the block pool
            // (occupied = held + referenced-key blocks) plus injected pressure —
            // the same caliber getCacheStatus and /snapshot report, so the master
            // sees one consistent number on every surface.
            EngineRpcService.WorkerStatusPB.Builder status = EngineRpcService.WorkerStatusPB.newBuilder()
                    .setAlive(!stopped)
                    .setRole("RoleType." + roleName)
                    .setRoleType(roleType)
                    .setAvailableKvCache(availableKvTokens())
                    .setTotalKvCache(totalKvTokens)
                    .setStatusVersion(statusVersion.updateAndGet(previous ->
                            faultConfig.isStatusVersionRegress() ? previous - 1 : currentTimeUs()))
                    .setLatestFinishedVersion(latestVersion)
                    .setDpSize(1)
                    .setTpSize(effectiveConfig == null ? 1 : effectiveConfig.path("roles").path(roleName.toLowerCase()).path("model").path("tp_size").intValue())
                    .setDpRank(0)
                    // Static engine limits. Production engines publish max_seq_len /
                    // max_batch_tokens_size so the master can clamp decision-group
                    // token capacity (BatcherContext); reporting them keeps the mock's
                    // admission semantics aligned with production instead of the
                    // implicit unlimited fallback.
                    .setMaxSeqLen(performance.prefillBatchPolicy().maxSeqLen())
                    .setMaxBatchTokensSize(performance.prefillBatchPolicy().maxTokens());

            status.addAllRunningTaskInfo(visibleRunning.stream()
                    .map(FastRpcService::withLegacyTaskState)
                    .toList());
            for (VersionedTask completion : visibleCompletions) {
                status.addFinishedTaskList(withLegacyTaskState(completion.task));
            }
            // ── Status-report fault injections: pure output-layer filters on
            // the assembled status. The completion backlog, its cursor
            // filtering and the version bookkeeping above are untouched — a
            // completion suppressed here is permanently lost once the (real)
            // cursor advances past it, which is exactly the fault under
            // test. ──
            applyStatusReportFaults(status);
            ObjectNode deliveryEvent = null;
            if (statusDeliveryFaultsActive) {
                synchronized (completionLock) {
                    List<Long> hidden = statusDeliveryFaults.filter(status);
                    if (!statusDeliveryFaults.rules.isEmpty() && engineEventLog != null) {
                        deliveryEvent = OBJECT_MAPPER.createObjectNode();
                        deliveryEvent.put("event", "worker_status_delivery_fault");
                        deliveryEvent.put("time_ms", System.currentTimeMillis());
                        deliveryEvent.put("engine", engineName);
                        deliveryEvent.put("port", grpcPort);
                        deliveryEvent.put("role", roleName);
                        deliveryEvent.put("epoch", crashEpoch.get());
                        deliveryEvent.put("status_version", status.getStatusVersion());
                        deliveryEvent.put("requested_version", requestedVersion);
                        deliveryEvent.put("latest_version", status.getLatestFinishedVersion());
                        var targets = deliveryEvent.putArray("targets");
                        statusDeliveryFaults.rules.forEach((rid, rule) -> {
                            var target = targets.addObject().put("rid", rid)
                                    .put("hidden", hidden.contains(rid))
                                    .put("missing_remaining", rule.missingRounds)
                                    .put("pending_completions", rule.pending.size())
                                    .put("released_version", rule.releasedVersion);
                            var running = target.putArray("running");
                            status.getRunningTaskInfoList().stream().filter(t -> t.getRequestId() == rid)
                                    .forEach(t -> running.add(t.getPhase().name()));
                            var finished = target.putArray("finished");
                            status.getFinishedTaskListList().stream().filter(t -> t.getRequestId() == rid)
                                    .forEach(t -> finished.addObject().put("batch_id", t.getBatchId())
                                            .put("error_code", t.getErrorInfo().getErrorCode())
                                            .put("phase", t.getPhase().name()));
                        });
                    }
                }
            }
            // This event describes the post-filter response sent to the client.
            // It is emitted only for explicitly targeted fault-injection requests.
            if (deliveryEvent != null) engineEventLog.write(deliveryEvent);
            workerStatusRates.add(1);
            workerStatusRtUs.set(statusReadUs);
            statusMetricReporter.accept(statusReadUs);
            observer.onNext(status.build());
            observer.onCompleted();
        }

        /**
         * Apply the status-report fault family to an assembled WorkerStatusPB
         * (output layer only; the completion queue and version protocol are
         * never touched here):
         * <ul>
         *   <li>status_suppress_finished / status_suppress_running — clear the
         *       finished/running task lists while scalars (latestFinishedVersion,
         *       runningQueryLen, ...) stay REAL, so the master observes a
         *       self-inconsistent report;</li>
         *   <li>status_suppress_rids — drop specific rids from BOTH lists
         *       (a request the engine selectively stops reporting);</li>
         *   <li>status_fake_task — append synthetic tasks that never existed
         *       (running-form TaskInfoPB / finished-form completion with
         *       optional errorCode) on EVERY poll until cleared;</li>
         *   <li>status_cursor_regress — report latestFinishedVersion n below
         *       reality (replaying an already-consumed interval);</li>
         *   <li>status_version_regress — report a statusVersion that DECREASES
         *       by one per poll: addAndGet(-2) undoes the incrementAndGet in
         *       the builder chain above and steps one further down (engine
         *       restart with version reset).</li>
         * </ul>
         */
        private void applyStatusReportFaults(EngineRpcService.WorkerStatusPB.Builder status) {
            if (faultConfig.isStatusSuppressRunning()) {
                status.clearRunningTaskInfo();
            }
            if (faultConfig.isStatusSuppressFinished()) {
                status.clearFinishedTaskList();
            }
            List<Long> suppressRids = faultConfig.getStatusSuppressRids();
            if (!suppressRids.isEmpty()) {
                for (long rid : suppressRids) {
                    for (int i = status.getRunningTaskInfoCount() - 1; i >= 0; i--) {
                        if (status.getRunningTaskInfo(i).getRequestId() == rid) {
                            status.removeRunningTaskInfo(i);
                        }
                    }
                    for (int i = status.getFinishedTaskListCount() - 1; i >= 0; i--) {
                        if (status.getFinishedTaskList(i).getRequestId() == rid) {
                            status.removeFinishedTaskList(i);
                        }
                    }
                }
            }
            int cursorRegress = faultConfig.getStatusCursorRegress();
            if (cursorRegress > 0) {
                status.setLatestFinishedVersion(
                        Math.max(0L, status.getLatestFinishedVersion() - cursorRegress));
            }
            for (FaultInjectionConfig.StatusFakeTask fake : faultConfig.getStatusFakeTasks()) {
                if (fake.isFinishedForm()) {
                    EngineRpcService.TaskInfoPB.Builder finished =
                            EngineRpcService.TaskInfoPB.newBuilder()
                                    .setRequestId(fake.requestId())
                                    .setInputLength(1)
                                    .setPrefixLength(0)
                                    .setBatchId(fake.batchId())
                                    .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                                    .setEndTimeMs(System.currentTimeMillis())
                                    .setExecutionTimeMs(0)
                                    .setIterateCount(1)
                                    .setDpRank(0);
                    if (fake.errorCode() != 0) {
                        finished.setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                .setErrorCode(fake.errorCode())
                                .setErrorMessage("injected status_fake_task")
                                .build());
                    }
                    status.addFinishedTaskList(finished.build());
                } else {
                    status.addRunningTaskInfo(withLegacyTaskState(
                            EngineRpcService.TaskInfoPB.newBuilder()
                                    .setRequestId(fake.requestId())
                                    .setInputLength(1)
                                    .setPrefixLength(0)
                                    .setBatchId(fake.batchId())
                                    .setPhase(parseFakeTaskPhase(fake.phase()))
                                    .setDpRank(0)
                                    .build()));
                }
            }
        }

        /** Map a status_fake_task phase string to the TaskPhase enum (default RUNNING). */
        private static EngineRpcService.TaskPhase parseFakeTaskPhase(String phase) {
            if (phase == null) {
                return EngineRpcService.TaskPhase.TASK_PHASE_RUNNING;
            }
            return switch (phase.toUpperCase()) {
                case "KV_ALLOCATED" -> EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED;
                case "RECEIVED" -> EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED;
                default -> EngineRpcService.TaskPhase.TASK_PHASE_RUNNING;
            };
        }

        private static io.grpc.StatusRuntimeException capacityError(int code, String message) {
            io.grpc.Metadata trailers = new io.grpc.Metadata();
            trailers.put(io.grpc.Metadata.Key.of("grpc-status-details-bin",
                    io.grpc.Metadata.BINARY_BYTE_MARSHALLER),
                    EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(code).setErrorMessage(message).build().toByteArray());
            return io.grpc.Status.RESOURCE_EXHAUSTED.withDescription(message)
                    .asRuntimeException(trailers);
        }

        @Override
        public void generateStreamCall(EngineRpcService.GenerateInputPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            if (!admitRpc(observer)) return;
            final EngineRpcService.GenerateInputPB normalized;
            try { normalized = MockRequestRules.normalize(request, roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE); }
            catch (io.grpc.StatusRuntimeException error) { observer.onError(error); return; }
            if (!supportsSingleSequence(request)) {
                observer.onError(io.grpc.Status.UNIMPLEMENTED
                        .withDescription("mock supports only single-sequence output; beam/multi-return unsupported")
                        .asRuntimeException());
                return;
            }
            stats.generateStreamRpcs.increment();
            rpcGenerateStream.incrementAndGet();
            // Per-RPC gRPC context, captured on the handler thread (the only
            // place where Context.current() is this request's cancellable
            // context). The response pump holds the reference and checks
            // isCancelled() cross-thread (supported grpc-java usage), while a
            // registered CancellationListener delivers millisecond-level
            // notification: the pump may sit in a 600 s queue.poll, so polling
            // alone cannot meet the detection-latency budget.
            Context rpcContext = Context.current();

            if (faultConfig.isGenerateError()) {
                observer.onError(io.grpc.Status.INTERNAL
                            .withDescription("injected generate_error").asRuntimeException());
                return;
            }
            // Python compat: inject_config["enqueue_error"] also makes generate_stream
            // raise (the Python mock checked enqueue_error in generate_stream too).
            if (faultConfig.isNoRespond()) {
                return; // explicit RPC blackhole; not a missing client Fetch
            }
            if (faultConfig.isFailOnEnqueue()) {
                observer.onError(io.grpc.Status.INTERNAL
                            .withDescription("injected enqueue_error").asRuntimeException());
                return;
            }
            if (stopped) {
                observer.onError(io.grpc.Status.UNAVAILABLE
                            .withDescription("engine stopped").asRuntimeException());
                return;
            }

            long requestId = request.getRequestId();
            beginRequest(request);
            MockPerformanceModel.RequestShape shape = matchPrefillMemory(performance.shape(normalized, cache));
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                var invalidInput = fifoInputError(shape);
                if (invalidInput != null) {
                    finishRequest(requestId);
                    observer.onError(MockRpcErrors.error(MockRpcErrors.status(invalidInput), invalidInput.getErrorCode(), invalidInput.getErrorMessage()));
                    return;
                }
            }
            observeCacheDiagnostics(shape);
            // Key-level cache-hit accounting (direct path, same admission hit
            // computation point as the enqueue-batch path above).
            cacheKeyHits.add(shape.hitBlocks());
            cacheKeysRequested.add(shape.blockKeys().size());
            acceptedCount.incrementAndGet();
            lastEnqueueTime.set(System.nanoTime());
            requests.state(requestId, "running");
            recordLifecycleStart(requestId, -1, "generate_stream");
            recordEventArrival(requestId);

            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    requests.ensureOutput(requestId, () -> responseQueue(requestId));

            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                if (!scheduleDecodeCompletion(shape, -1, queue)) {
                    // A stopping scheduler cannot accept new work.
                    requests.removeOutput(requestId);
                    requests.state(requestId, "rejected");
                    finishRequest(requestId);
                    observer.onError(io.grpc.Status.UNAVAILABLE
                            .withDescription("decode scheduler is stopping").asRuntimeException());
                    return;
                }
            } else {
                // Reject impossible requests before preparation. Physical P KV is
                // initialized by scheduler candidate selection; transient shortage waits.
                var invalidInput = fifoInputError(shape);
                if (invalidInput != null) {
                    requests.removeOutput(requestId); finishRequest(requestId);
                    requests.state(requestId, "rejected");
                    observer.onError(MockRpcErrors.error(MockRpcErrors.status(invalidInput), invalidInput.getErrorCode(), invalidInput.getErrorMessage()));
                    return;
                }

                // Prepare D before P starts; preserve typed allocation errors.
                FastRpcService decodeEngine = findDecodeEngine(request);
                if (!prepareDecodeSession(decodeEngine, shape, -1L, 0, true)) {
                    releaseBlockLease(requestId);
                    requests.removeOutput(requestId);
                    requests.state(requestId, "rejected");
                    EngineRpcService.ErrorDetailsPB error = decodePreparationError(decodeEngine, shape);
                    io.grpc.Metadata trailers = new io.grpc.Metadata();
                    trailers.put(io.grpc.Metadata.Key.of("grpc-status-details-bin", io.grpc.Metadata.BINARY_BYTE_MARSHALLER),
                            error.toByteArray());
                    observer.onError(MockRpcErrors.status(error)
                            .withDescription(error.getErrorMessage()).asRuntimeException(trailers));
                    return;
                }
                if (!admitDirectPrefill(shape)) {
                    // Admission closed while preparation was in flight; release its owners.
                    closePrefillSession(requestId);
                    releaseReservedDecode(requestId);
                    requests.removeOutput(requestId);
                    requests.state(requestId, "rejected");
                    finishRequest(requestId);
                    observer.onError(io.grpc.Status.UNAVAILABLE
                            .withDescription(String.format(
                            "prefill scheduler is stopping: waiting=%d",
                            waitingPrefillRequests.get())).asRuntimeException());
                    return;
                }
            }

            // Admitted: arm the autonomous client-gone detector for this
            // stream (production C++ engine IsCancelled semantics).
            registerClientGoneListener(requestId, rpcContext);

            pumpResponses(queue, rpcContext, observer);
        }

        @Override
        public void fetchResponse(EngineRpcService.FetchRequestPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            if (!admitRpc(observer)) return;
            stats.fetchResponseRpcs.increment();
            rpcFetchResponse.incrementAndGet();
            // Same client-gone capture as generateStreamCall: under the BATCH
            // dispatcher the client's FetchResponse stream is glued to the
            // ORIGINAL PREFILL engine, so a broken fetch must drive that
            // prefill's autonomous cancel (and the P->D propagation when the
            // hand-off already happened).
            Context rpcContext = Context.current();

            long requestId = request.getRequestId();

            if (faultConfig.isNoRespond()) {
                return;
            }
            if (faultConfig.isFetchError()) {
                // A failed Fetch owns the deferred context just like a failed
                // finishStream in the real engine. Do not retain P's connector
                // or D's ALLOCATE until the missing-Fetch TTL after returning.
                MockPrefillSession failedFetch = prefillSessions.get(requestId);
                if (failedFetch != null && failedFetch.attach()) {
                    cancel(requestId, false, false);
                }
                observer.onNext(EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(requestId)
                        .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder()
                                .addFinished(false)
                                .build())
                        .build());
                observer.onError(io.grpc.Status.INTERNAL
                            .withDescription("injected fetch_error").asRuntimeException());
                return;
            }

            MockPrefillSession session = prefillSessions.get(requestId);
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue;
            if (session != null) {
                if (!session.attach()) {
                    observer.onError(io.grpc.Status.NOT_FOUND
                            .withDescription("Fetch context already attached or expired")
                            .asRuntimeException());
                    return;
                }
                queue = requests.ensureOutput(requestId, () -> responseQueue(requestId));
            } else {
                queue = requests.output(requestId);
                if (queue == null || downstreamDecodeOwners.containsKey(requestId)
                        || remotePrefillOwners.containsKey(requestId)) {
                    observer.onError(io.grpc.Status.NOT_FOUND
                            .withDescription("No unconsumed Fetch context")
                            .asRuntimeException());
                    return;
                }
            }

            // Arm the autonomous client-gone detector for this fetch stream.
            registerClientGoneListener(requestId, rpcContext);
            if (session != null) {
                armPrefillWait(requestId);
                continuePrefillSession(session);
            }

            pumpResponses(queue, rpcContext, observer);
        }

        /**
         * Generate and Fetch consume the same output protocol. A frame-gap timeout
         * completes even an empty stream; 602/603 become gRPC errors, other error
         * frames and finished outputs terminate after delivery. Cancellation exits
         * without completing a dead observer. Poll outside the scheduler so blocked
         * readers cannot starve the tasks producing their responses.
         */
        private void pumpResponses(LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue,
                                   Context rpcContext,
                                   StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            responseExecutor.execute(() -> {
                try {
                    while (true) {
                        if (rpcContext.isCancelled()) {
                            return;
                        }
                        EngineRpcService.GenerateOutputsPB output =
                                queue.poll(responsePollTimeoutMs, TimeUnit.MILLISECONDS);
                        if (output == null) {
                            break;
                        }
                        if (output.hasErrorInfo()) {
                            var details = EngineRpcService.ErrorDetailsPB.newBuilder()
                                    .setErrorCode(output.getErrorInfo().getErrorCodeValue() == EngineRpcService.ErrorCodePB.CANCELLED_VALUE
                                            ? 8100 : output.getErrorInfo().getErrorCodeValue())
                                    .setErrorMessage(output.getErrorInfo().getErrorMessage()).build();
                            observer.onError(MockRpcErrors.error(MockRpcErrors.status(details), details.getErrorCode(), details.getErrorMessage()));
                            return;
                        }
                        observer.onNext(output);
                        boolean terminal = false;
                        {
                            EngineRpcService.FlattenOutputPB flatten = output.getFlattenOutput();
                            for (int j = 0; j < flatten.getFinishedCount(); j++) {
                                if (flatten.getFinished(j)) {
                                    terminal = true;
                                    break;
                                }
                            }
                        }
                        if (terminal) {
                            break;
                        }
                    }
                    observer.onCompleted();
                } catch (InterruptedException e) {
                    observer.onError(e);
                } catch (RuntimeException e) {
                    if (!rpcContext.isCancelled()) {
                        throw e;
                    }
                    // Delivering to an already-cancelled fetch call is
                    // expected; swallow so the pump thread exits cleanly.
                }
            });
        }

        /**
         * Override the response-pump per-frame poll timeout (mainly for tests
         * exercising the timeout path; production keeps the
         * {@link #DEFAULT_RESPONSE_POLL_TIMEOUT_MS} default).
         */
        void setResponsePollTimeoutMs(long responsePollTimeoutMs) {
            if (responsePollTimeoutMs < 1) {
                throw new IllegalArgumentException("response poll timeout must be >= 1 ms");
            }
            this.responsePollTimeoutMs = responsePollTimeoutMs;
        }

        /** Wire the cluster-shared engine_events.jsonl writer (null disables). */
        void setEngineEventLog(EngineEventLog engineEventLog) {
            this.engineEventLog = engineEventLog;
            cache.setEvictionLifetimeListener(ms -> reportCacheEviction("gpu", ms));
        }

        private void onCacheEviction(MockLruBlockCache.EvictionEvent event) {
            reportMetricEvent(Map.of("mock_cache_direct_evicted_blocks", event.blocksFreed()));
            if (engineEventLog == null) return;
            ObjectNode row = OBJECT_MAPPER.createObjectNode();
            row.put("event", "evict_chain");
            row.put("engine_name", engineName);
            row.put("engine_incarnation", engineIncarnation);
            row.put("engine_address", host + ":" + grpcPort);
            row.put("timestamp_ms", System.currentTimeMillis());
            row.put("leaf_key", event.leafKey());
            for (Long key : event.chainKeys()) row.withArray("chain_keys").add(key);
            row.put("blocks_freed", event.blocksFreed());
            row.put("reason", event.reason());
            engineEventLog.write(row);
        }

        /**
         * @return the phase the cancelled request was in when its runningTasks
         *         entry was removed (snapshotted under decodeQueueLock for
         *         decode requests, so a concurrent drain cannot flip
         *         KV_ALLOCATED→RUNNING between snapshot and removal), or
         *         {@code null} when no entry was found (already terminal).
         */
        EngineRpcService.TaskPhase cancel(long requestId) {
            return cancel(requestId, false, true);
        }

        private EngineRpcService.TaskPhase cancel(long requestId,
                                                  boolean priorityPreemption,
                                                  boolean countRpc) {
            var admission = batchAdmissions.get(requestId);
            if (admission != null) synchronized (admission) {
                if (!admission.cancel(priorityPreemption) && !admission.schedulerOwned) return null;
                // A context operation owns its downstream resources until it returns.
                if (admission.preparing || admission.groupActive) {
                    // TryCancelDownstream interrupts ALLOCATE, without finishing this context's operation early.
                    var remote = remotePrefillOwners.get(requestId);
                    var stream = remote == null ? null : remote.stream;
                    if (stream != null) stream.close();
                    var decode = findDecodeEngine(admission.input);
                    if (decode != null) decode.cache.capacityChanged();
                    if (priorityPreemption) {
                        var overlay = runningTasks.get(requestId);
                        if (overlay == null) overlay = EngineRpcService.TaskInfoPB.newBuilder().setRequestId(requestId)
                                .setBatchId(admission.input.getGroupId().getValue()).setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING).build();
                        overlay = overlay.toBuilder().setPriorityPreemptionProgress(
                                EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELING).build();
                        cancelingOverlays.put(requestId, overlay);
                        var visible = overlay;
                        runningTasks.computeIfPresent(requestId, (ignored, task) -> visible);
                    }
                    return EngineRpcService.TaskPhase.TASK_PHASE_RUNNING;
                }
            }
            Runnable remoteStop = remoteDecodeStops.remove(requestId);
            if (remoteStop != null) remoteStop.run();
            if (countRpc) {
                stats.cancelRpcs.increment();
                rpcCancel.incrementAndGet();
            }
            cancellationError(requestId);
            if (!shuttingDown && !schedulerTransition.get()) {
                if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                    synchronized (decodeQueueLock) {
                        if (!decodeBoundaryMaintenance && (decodeRunning.containsKey(requestId)
                                || decodePendingQueue.stream().anyMatch(task -> task.shape().input().getRequestId() == requestId))) {
                            var phase = deferBoundaryCancellation(requestId, priorityPreemption);
                            if (phase != null) {
                                if (!decodeStepScheduled) scheduler.execute(() -> {
                                    synchronized (decodeQueueLock) { topUpDecodeRunningLocked(); scheduleDecodeStepLocked(); }
                                });
                                return phase;
                            }
                        }
                    }
                } else synchronized (completionLock) {
                    if (fifoOwned.contains(requestId)) {
                        var phase = deferBoundaryCancellation(requestId, priorityPreemption);
                        if (phase != null) { wakeFifoOnCapacityRelease(); return phase; }
                    }
                }
            }
            // Terminal-ownership claim — the claimDecodeTerminalLocked pattern
            // generalized to every cancel entry point: the FIRST cancel for a
            // rid arms the cancelled marker and owns the terminal bookkeeping;
            // a second cancel (an autonomous client-gone cancellation racing
            // the explicit Cancel RPC, a duplicate channel cancel, ...) finds
            // the marker armed and no-ops, so the typed CANCELLED terminal and
            // cancelledCount are published exactly once per engine. The claim
            // is also the marker the completion callbacks and
            // scheduleDecodeCompletion check, so its ordering with the
            // remove/release section below is unchanged.
            if (cancelledRequests.putIfAbsent(requestId, System.nanoTime()) != null) {
                return null;
            }
            abortMemoryCopies(requestId);
            addCancelledRid(requestId);
            if (priorityPreemption) {
                addPriorityCancelledRequestId(requestId);
            }
            MockPrefillSession session = prefillSessions.remove(requestId);
            if (session != null) {
                session.close();
            }
            // Queued-vs-running discrimination, the runningTasks removal, the
            // slot/KV release, and the freed-slot drain all run under
            // decodeQueueLock in ONE atomic section, mirroring the completion
            // path in the per-step decode loop (runDecodeStep). Splitting them
            // (removeIf in one locked section,
            // release outside any lock, drain in a second locked section) opened
            // a transient over-admission: a concurrent admission could observe
            // the freed slot before our unconditional drain re-consumed it,
            // pushing activeDecodeRequests to cap+1.
            EngineRpcService.TaskPhase cancelledPhase = null;
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                synchronized (decodeQueueLock) {
                    // Waiting requests have no running-slot claim. Their KV may
                    // already belong to an ALLOCATE reservation and is released
                    // independently of running-slot accounting.
                    boolean wasWaitingForKv = decodeWaitingForKv.remove(requestId);
                    boolean wasQueuedDecode = decodePendingQueue.removeIf(
                            t -> t.shape().input().getRequestId() == requestId);
                    EngineRpcService.TaskInfoPB removed = cancelReportedTask(requestId, priorityPreemption);
                    if (removed != null) {
                        cancelledPhase = removed.getPhase();
                        pendingRequests.decrementAndGet();
                        if (!wasQueuedDecode && !wasWaitingForKv) {
                            activeDecodeRequests.decrementAndGet();
                            // Capacity model v2: the running stream's block lease
                            // goes back to the pool (no LRU handover — a
                            // cancelled request leaves no cache).
                            releaseBlockLease(requestId);
                            // Drop the stream from the per-step loop (no further
                            // step advances it) and hand the freed slot to queued
                            // requests in the SAME locked section — release +
                            // top-up stay atomic so the cap is never transiently
                            // exceeded and no slot is stranded until the next
                            // step boundary.
                            decodeRunning.remove(requestId);
                            topUpDecodeRunningLocked();
                            scheduleDecodeStepLocked();
                        } else {
                            // A waiting request may already own an ALLOCATE
                            // reservation. Release idempotently before draining.
                            releaseBlockLease(requestId);
                        }
                    } else {
                        // P-enqueue decode-KV pre-alignment (20260903): a cancel
                        // arriving BEFORE this engine accepted the request can
                        // still hold a reservation lease provisioned by the
                        // upstream prefill (the hand-off never ran, so no
                        // runningTasks entry exists). Release it here — the
                        // queued-vs-running discrimination above only covers
                        // accepted requests. Idempotent: no lease = no-op.
                        releaseBlockLease(requestId);
                    }
                }
            } else {
                // For prefill, partial batches are left in the queue (other
                // requests in the batch may still be alive); the drain's anyAlive
                // check drops fully-cancelled batches and the completion's
                // !alreadyCancelled guard prevents double-decrement of
                // pendingRequests for cancelled members. The block lease is
                // released here for BOTH queued and running members (the
                // completion callback's alreadyCancelled release is idempotent —
                // activeBlockLeases.remove() wins exactly once).
                EngineRpcService.TaskInfoPB removed = cancelReportedTask(requestId, priorityPreemption);
                if (removed != null) {
                    cancelledPhase = removed.getPhase();
                    pendingRequests.decrementAndGet();
                }
            }
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                // Also covers cancellation before the task reached runningTasks.
                // Both release helpers are idempotent across racing completions.
                releaseBlockLease(requestId);
                releaseReservedDecode(requestId);
            }
            // Determine ownership from the previous ledger state. Marking
            // terminal before cancelReportedTask would misclassify a live
            // failed stream as a deliberately retained zombie and leak census.
            recordLifecycleEnd(requestId, true);
            requests.state(requestId, "cancelled");
            cancelledCount.incrementAndGet();
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue = requests.output(requestId);
            if (queue != null) {
                queue.offer(cancellationOutput(requestId));
                // The poller already holds a reference to the queue, so it is safe
                // to remove it from the map after offering the cancel response.
                requests.removeOutput(requestId);
            }
            // (A cancelled running slot's top-up ran under decodeQueueLock inside
            // the decode branch above — nothing to schedule outside the lock.)
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                synchronized (decodeQueueLock) {
                    FastRpcService prefill = upstreamPrefillOwners.get(requestId);
                    if (prefill != null) {
                        if (prefill.prefillSessions.containsKey(requestId)) {
                            prefill.cancel(requestId, false, false);
                        } else {
                            // After handoff the client queue still belongs to P.
                            // A cancel winning against link death must close it
                            // before removing the last propagation route.
                            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> upstreamQueue =
                                    prefill.requests.output(requestId);
                            if (upstreamQueue != null) {
                                upstreamQueue.offer(cancellationOutput(requestId));
                                prefill.requests.removeOutput(requestId, upstreamQueue);
                            }
                        }
                    }
                    clearUpstreamOwnership(requestId);
                }
            }
            fifoOwned.remove(requestId);
            finishRequest(requestId);
            cache.capacityChanged();
            return cancelledPhase;
        }

        private EngineRpcService.TaskPhase commitSchedulerCancellation(long id, boolean priority) {
            boolean previous = schedulerTransition.get();
            schedulerTransition.set(true);
            try { return cancel(id, priority, false); }
            finally { schedulerTransition.set(previous); }
        }

        private EngineRpcService.TaskPhase deferBoundaryCancellation(long id, boolean priority) {
            var tracked = runningTasks.get(id);
            if (tracked == null) return null;
            boundaryCancellations.merge(id, priority, (before, next) -> before || next);
            if (!priority) {
                // NormalGenerateStream wakes its consumer on Error before
                // scheduler-owned FINISHED/resource release has committed.
                var error = cancellationError(id);
                var queue = requests.output(id);
                if (queue != null) queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(id)
                        .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder().setErrorCodeValue((int)error.getErrorCode())
                                .setErrorMessage(error.getErrorMessage())).build());
            }
            if (priority) runningTasks.computeIfPresent(id, (ignored, task) -> task.toBuilder()
                    .setPriorityPreemptionProgress(EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELING).build());
            return tracked.getPhase();
        }

        private EngineRpcService.TaskInfoPB cancellationTask(long requestId,
                EngineRpcService.TaskPhase phase, boolean priorityPreemption) {
            EngineRpcService.ErrorDetailsPB error = priorityPreemption
                    ? EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(PRIORITY_PREEMPTED_ERROR_CODE)
                            .setErrorMessage("preempted by higher-priority request").build()
                    : cancellationError(requestId);
            // Preserve the actual cancelled phase; unknown tasks retain RUNNING.
            EngineRpcService.TaskInfoPB.Builder taskBuilder = terminalTaskBuilder(requestId,
                    phase != null ? phase : EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, error);
            if (priorityPreemption) {
                taskBuilder.setPriorityPreemptionProgress(
                        EngineRpcService.PriorityPreemptionProgressPB
                                .PRIORITY_PREEMPTION_CANCELED);
            }
            return taskBuilder.build();
        }

        private EngineRpcService.TaskInfoPB cancelReportedTask(long requestId,
                boolean priorityPreemption) {
            synchronized (completionLock) {
                EngineRpcService.TaskInfoPB tracked = runningTasks.get(requestId);
                if (tracked != null && "failed".equals(requests.state(requestId))
                        && hasTerminalLifecycle(requestId)) {
                    // Zombie diagnostics can retain a task after its ledger settled.
                    runningTasks.remove(requestId, tracked);
                    return null;
                }
                publishCompletion(cancellationTask(requestId,
                        tracked == null ? null : tracked.getPhase(), priorityPreemption));
                return runningTasks.remove(requestId);
            }
        }

        /**
         * Cancel-RPC arrival bookkeeping + fault-injection gate, shared by
         * all three cancel surfaces (gRPC Cancel handler, HTTP
         * {@code /cancel_request}, in-process {@link MockEngineCancelChannel}).
         * Counts the arrival (stats.cancelRpcs + rpcCancel — exactly like
         * statusNoRespond counts statusRpcs before hanging), then returns
         * the armed injection kind or {@code null} for the normal path.
         *
         * <p>An injected cancel is an RPC-LAYER failure simulated BEFORE the
         * engine cancel state machine is touched: no fences, no terminal records,
         * no census branch — production semantics "RPC failed = engine state
         * unchanged". The master's one-shot cancel contract (never retries,
         * 50ms ack timeout) turns every injected kind into a failed future
         * on the master side while the engine keeps its exact prior state.
         */
        CancelFaultKind arriveCancelRpc() {
            stats.cancelRpcs.increment();
            rpcCancel.incrementAndGet();
            if (faultConfig.isCancelNoRespond()) {
                stats.cancelCensusInjected.increment();
                return CancelFaultKind.NO_RESPOND;
            }
            if (faultConfig.isCancelError()) {
                stats.cancelCensusInjected.increment();
                return CancelFaultKind.ERROR;
            }
            if (faultConfig.isCancelUnexpectedStatus()) {
                stats.cancelCensusInjected.increment();
                return CancelFaultKind.UNEXPECTED_STATUS;
            }
            return null;
        }

        /**
         * Three-branch cancel used by {@link MockEngineCancelChannel} and the
         * gRPC Cancel handler, mapped to the production C++ Prefill contract:
         * a live request (or an accepted-cancel terminal record retry) is
         * {@code found} → ACCEPTED; a request this engine has seen but which
         * already finished (completed or previously cancelled) is
         * {@code alreadyFinished} → NOT_FOUND — seen-but-terminal; the
         * completion record stays deliverable from the retain-window
         * backlog (bounded to 1000 FIFO records); a rid the engine NEVER saw is unknown → REQUEST_FENCED with the
         * ABSENT_FENCE record installed here (later Enqueues of that rid
         * are rejected with 8429). The mock behaviour on the found branch is
         * identical to {@link #cancel(long)}: the request is removed and a
         * CANCELLED completion is surfaced in the next WorkerStatus finished
         * list.
         */
        CancelResult cancelRequest(long requestId) {
            // Arrival counting (stats.cancelRpcs + rpcCancel) lives in
            // arriveCancelRpc() so EVERY cancel surface (gRPC handler, HTTP
            // /cancel_request, in-process MockEngineCancelChannel) counts the
            // arrival exactly once — including the injected short-circuit
            // paths, mirroring statusNoRespond counting statusRpcs before
            // hanging. Direct unit-test calls of cancelRequest skip the
            // arrival counters; the census BRANCH counters below stay here.
            if (roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                throw new UnsupportedOperationException(
                        "priority Cancel is only implemented by the original Prefill");
            }
            var admission = batchAdmissions.get(requestId);
            if (admission != null) {
                if (!admission.cancel(true)) {
                    stats.cancelCensusAlreadyFinished.increment();
                    return new CancelResult(false, null, true);
                }
                stats.cancelCensusTracked.increment();
                cancel(requestId, true, false);
                return new CancelResult(true, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, false);
            }
            if (hasPriorityCancelledRequestId(requestId)) {
                stats.cancelCensusAlreadyCancelled.increment();
                return new CancelResult(true, null, true);
            }
            if (hasFencedRequestId(requestId)) {
                // ABSENT_FENCE retry: the rid is still never-seen — the
                // production handler answers REQUEST_FENCED again (idempotent),
                // it must NOT flip onto the ACCEPTED priority-terminal record
                // branch.
                stats.cancelCensusUnknown.increment();
                return new CancelResult(false, null, false);
            }
            EngineRpcService.TaskInfoPB tracked = runningTasks.get(requestId);
            if (tracked == null && !hasTerminalLifecycle(requestId) && (prefillSessions.containsKey(requestId) || remotePrefillOwners.containsKey(requestId)
                    || batchAdmissions.containsKey(requestId))
                    && !downstreamDecodeOwners.containsKey(requestId)) {
                stats.cancelCensusTracked.increment();
                cancel(requestId, true, false);
                return new CancelResult(true, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, false);
            }
            if (tracked != null) {
                // A1 leak-attribution census: cancel landed on a request this
                // engine still actively tracks (the branch that publishes a
                // typed CANCELLED terminal the master must reconcile).
                stats.cancelCensusTracked.increment();
                // P2-2: the authoritative phase is the one cancel() snapshots
                // under decodeQueueLock when it removes the entry — the pre-lock
                // read above may be stale (the pending-queue drain can flip
                // KV_ALLOCATED→RUNNING in between). Fall back to the pre-read
                // only when a racing terminal already emptied the entry.
                EngineRpcService.TaskPhase phase = cancel(
                        requestId, true, false);
                return new CancelResult(true, phase != null ? phase : tracked.getPhase(), false);
            }
            FastRpcService decode = downstreamDecodeOwners.get(requestId);
            if (decode != null) {
                CancelResult downstream = decode.cancelFromPrefill(requestId, this);
                if (downstream.found()) {
                    return downstream;
                }
                // The Decode won a terminal race and already removed its
                // reverse ownership. Drop the stale Prefill entry as well.
                downstreamDecodeOwners.remove(requestId, decode);
            }
            boolean alreadyFinished = recentlySeenRequests.contains(requestId);
            if (alreadyFinished) {
                // A2 census: cancel raced a terminal that already finished — the
                // master's reconcile should see the original terminal instead.
                stats.cancelCensusAlreadyFinished.increment();
                return new CancelResult(false, null, true);
            }
            // A2 census: cancel addressed a request this engine never knew —
            // stale master bookkeeping or a cancelled generation. Production
            // behaviour: install the ABSENT_FENCE record and answer
            // REQUEST_FENCED so the caller's map (alreadyFinished→NOT_FOUND,
            // unknown→REQUEST_FENCED) fences any racing later Enqueue with 8429.
            stats.cancelCensusUnknown.increment();
            addFencedRequestId(requestId);
            return new CancelResult(false, null, false);
        }

        /**
         * Register the exact Decode selected for one request. Package-private so
         * E2E tests that inject a Decode task directly can model the otherwise
         * normal Prefill hand-off without teaching the cancel channel to scan.
         */
        void registerDecodeOwnership(long requestId, FastRpcService decode) {
            if (roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                    || decode == null
                    || decode.roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                throw new IllegalArgumentException("decode ownership requires Prefill -> Decode");
            }
            synchronized (decode.decodeQueueLock) {
                if (decode.stopped || decode.shuttingDown) {
                    decode.deliverLinkBreak(requestId, this);
                    return;
                }
                FastRpcService previousDecode = downstreamDecodeOwners.put(requestId, decode);
                if (previousDecode != null && previousDecode != decode) {
                    previousDecode.upstreamPrefillOwners.remove(requestId, this);
                }
                FastRpcService previousPrefill = decode.upstreamPrefillOwners.put(requestId, this);
                if (previousPrefill != null && previousPrefill != this) {
                    previousPrefill.downstreamDecodeOwners.remove(requestId, decode);
                }
            }
        }

        void clearDecodeOwnership(long requestId, FastRpcService decode) {
            if (decode == null) {
                return;
            }
            synchronized (decode.decodeQueueLock) {
                if (downstreamDecodeOwners.remove(requestId, decode)) {
                    decode.upstreamPrefillOwners.remove(requestId, this);
                }
            }
        }

        private CancelResult cancelFromPrefill(long requestId, FastRpcService expectedPrefill) {
            return cancelDownstream(requestId, expectedPrefill, true);
        }

        /**
         * Client-gone propagation variant used by the autonomous cancellation
         * path: the client-facing stream glued to the ORIGINAL PREFILL broke
         * while the Decode selected by role_addrs is executing the request.
         * Same Decode-side cleanup and terminal discipline as the explicit
         * priority-cancel propagation, except the Prefill publishes an
         * ordinary typed CANCELLED terminal ("cancelled by client"), matching
         * the production tryCancelDownstream contract for a broken P context.
         */
        private CancelResult cancelFromClientGone(long requestId, FastRpcService expectedPrefill) {
            return cancelDownstream(requestId, expectedPrefill, false);
        }

        /**
         * Shared P->D cancel propagation (explicit priority Cancel and
         * autonomous client-gone). Runs on the DECODE side: ownership removal
         * is the one-shot guard, cancel() cleans the decode bookkeeping and
         * publishes the decode-local CANCELLED terminal, the prefill-side
         * queue is terminated so its pump exits, and the PREFILL publishes the
         * typed terminal (priority-preemption form for the explicit path,
         * ordinary CANCELLED for the client-gone path).
         */
        private CancelResult cancelDownstream(long requestId,
                                              FastRpcService expectedPrefill,
                                              boolean priorityPreemption) {
            synchronized (decodeQueueLock) {
                if (!shuttingDown
                        && !decodeBoundaryMaintenance && (decodeRunning.containsKey(requestId)
                            || decodePendingQueue.stream().anyMatch(task -> task.shape.input().getRequestId() == requestId))
                        && upstreamPrefillOwners.get(requestId) == expectedPrefill) {
                    boundaryDownstreamCancellations.merge(requestId, new DownstreamCancellation(expectedPrefill, priorityPreemption),
                            (before, next) -> new DownstreamCancellation(before.prefill(), before.priority() || next.priority()));
                    if (priorityPreemption) expectedPrefill.cancelingOverlays.put(requestId,
                            runningTasks.get(requestId).toBuilder()
                                    .setPriorityPreemptionProgress(EngineRpcService.PriorityPreemptionProgressPB.PRIORITY_PREEMPTION_CANCELING).build());
                    if (!decodeStepScheduled) scheduler.execute(() -> {
                        synchronized (decodeQueueLock) { topUpDecodeRunningLocked(); scheduleDecodeStepLocked(); }
                    });
                    return new CancelResult(true, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, false);
                }
                if (!upstreamPrefillOwners.remove(requestId, expectedPrefill)) {
                    return new CancelResult(false, null, false);
                }
                if (priorityPreemption) {
                    // A1 census (decode-side): a forwarded cancel landed on a
                    // request this Decode actively tracks via ownership.
                    stats.cancelCensusTracked.increment();
                } else {
                    // Census: a client-gone propagation landed on a request
                    // this Decode actively tracks via ownership.
                    stats.cancelCensusClientGone.increment();
                }
                expectedPrefill.downstreamDecodeOwners.remove(requestId, this);
                expectedPrefill.closePrefillSession(requestId);
                expectedPrefill.decodeReservationOwners.remove(requestId, this);
                // This is downstream stream cancellation, not a Decode Cancel
                // RPC.  Preserve ordinary Decode accounting/terminal behavior
                // without incrementing the Decode RPC counter.
                EngineRpcService.TaskPhase phase = commitSchedulerCancellation(requestId, false);
                EngineRpcService.TaskPhase observedPhase = phase != null
                        ? phase : EngineRpcService.TaskPhase.TASK_PHASE_RUNNING;
                // Terminate the client-facing stream: after the P->D hand-off the
                // response queue the client's FetchResponse / GenerateStreamCall
                // poller hangs on lives on the ORIGINAL PREFILL (startDecode passed
                // it to this Decode). cancel() above only offers into this
                // Decode's own outputs, which never contained the request,
                // so without this delivery the cancelled request's stream hangs
                // until the 60s poll timeout (Python mock terminates the stream
                // from its async cancel finalizer instead).
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> prefillQueue =
                        expectedPrefill.requests.output(requestId);
                if (prefillQueue != null) {
                    prefillQueue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                            .setRequestId(requestId)
                            .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                    .setErrorCode(EngineRpcService.ErrorCodePB.CANCELLED)
                                    .setErrorMessage("cancelled by client")
                                    .build())
                            .build());
                    expectedPrefill.requests.removeOutput(requestId);
                }
                if (priorityPreemption) {
                    // Decode retains its ordinary CANCELLED terminal for local
                    // accounting. The original Prefill is the authoritative
                    // producer of the typed priority-preemption completion.
                    expectedPrefill.recordPriorityPreemptionCanceled(requestId, observedPhase);
                } else {
                    expectedPrefill.recordClientGoneCanceled(requestId, observedPhase);
                }
                // Ownership is itself the admission proof. A cancel may win the
                // narrow hand-off race before Decode publishes runningTasks; the
                // existing cancelled marker then prevents later scheduling.
                return new CancelResult(true, observedPhase, false);
            }
        }

        // ──────────── Autonomous client-gone cancellation (production alignment) ────────────

        /**
         * Arm the client-gone detector for one client-facing stream
         * (generateStreamCall / fetchResponse). The per-RPC gRPC context is
         * captured on the handler thread — the only place where
         * Context.current() is the request's cancellable context — and the
         * registered listener fires on the response-pump executor within
         * milliseconds of the client cancelling the call, because the pump
         * itself may be blocked in a 600 s queue.poll and the per-iteration
         * isCancelled() check alone cannot meet the detection-latency budget.
         *
         * <p>grpc-java supports holding the Context reference and checking
         * isCancelled() from other threads; the listener callback re-checks
         * isCancelled() because a CancellableContext also notifies listeners
         * on normal close (stream completed fine), where isCancelled() stays
         * false.
         */
        private void registerClientGoneListener(long requestId, Context rpcContext) {
            if (rpcContext == Context.ROOT || shuttingDown) {
                // In-process invocation (unit tests call the handler directly
                // with no transport): there is no stream to break, and arming
                // listeners on the process-wide ROOT context would never fire.
                // During shutdown drain the executor is gone — the drain
                // sweep already cancels everything in flight.
                return;
            }
            rpcContext.addListener(context -> {
                if (context.isCancelled()) {
                    handleClientGone(requestId);
                }
            }, responseExecutor);
        }

        /**
         * Autonomous cancellation driven by the CLIENT's stream breaking —
         * the mock's counterpart of the production C++ engine checking
         * IsCancelled in its per-token loop: a broken
         * FetchResponse/GenerateStream context makes the engine itself clean
         * the request up, record the cancel, publish the typed CANCELLED
         * terminal, and propagate to the downstream decode once the P->D
         * hand-off has happened.
         *
         * <p>Branching mirrors the production roles:
         * <ul>
         *   <li>this engine still tracks the request (runningTasks holds
         *       it — the prefill is executing/queueing it, or a NON_BATCH
         *       decode is directly serving it): cancel() here. For a prefill,
         *       the batch completion callback's alreadyCancelled guard then
         *       suppresses the P->D hand-off, so no propagation is needed —
         *       exactly the explicit-cancel contract. For a decode, the
         *       CANCELLED terminal is reported IMMEDIATELY (production's
         *       early terminal for a decode-side broken stream: no stale
         *       inflight TTL wait).</li>
         *   <li>this engine is the prefill and the hand-off already happened
         *       (downstreamDecodeOwners holds the decode selected from
         *       role_addrs): propagate via cancelFromClientGone — the decode
         *       cleans its own slot/KV bookkeeping and records its CANCELLED
         *       terminal, and this prefill publishes the typed CANCELLED
         *       terminal carrying the EnqueueBatch identity so the master
         *       reconcile converges on both engines.</li>
         *   <li>request already terminal (or unknown) on this engine — the
         *       client dropped the stream after/without a live request;
         *       no-op.</li>
         * </ul>
         *
         * <p>Every race resolves through the existing terminal-ownership
         * claims: cancel()'s cancelledRequests.putIfAbsent (vs an explicit
         * Cancel RPC or a duplicate signal), the ownership remove in
         * cancelDownstream (vs the decode's own completion), and
         * runningTasks.remove (vs the normal completion callbacks) — first
         * one to claim owns the terminal, later arrivals no-op.
         */
        private boolean hasTerminalLifecycle(long requestId) {
            synchronized (requestLifecycles) {
                var row = requestLifecycles.get(requestId);
                return row != null && !row.running();
            }
        }

        boolean cancellationPending(long id) {
            return boundaryCancellations.containsKey(id) || boundaryDownstreamCancellations.containsKey(id);
        }

        private void handleClientGone(long requestId) {
            if (cancellationPending(requestId)) return;
            if (!runningTasks.containsKey(requestId) && !downstreamDecodeOwners.containsKey(requestId)
                    && !remotePrefillOwners.containsKey(requestId) && hasTerminalLifecycle(requestId)) return;
            if (runningTasks.containsKey(requestId) || prefillSessions.containsKey(requestId)
                    || remotePrefillOwners.containsKey(requestId)) {
                stats.cancelCensusClientGone.increment();
                cancel(requestId, false, false);
                return;
            }
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                FastRpcService decode = downstreamDecodeOwners.get(requestId);
                if (decode != null) {
                    decode.cancelFromClientGone(requestId, this);
                }
            }
            // else: request already finished or unknown to this engine —
            // nothing to clean (census counts only effective cancellations).
        }

        /**
         * Typed CANCELLED terminal published by a PREFILL whose client stream
         * broke after the P->D hand-off (autonomous client-gone propagation).
         * Same batch-identity discipline as the explicit-cancel terminal —
         * the master's reconcile must correlate the cancel with the batch it
         * displaced or the inflight slot leaks.
         */
        private void recordClientGoneCanceled(long requestId,
                                              EngineRpcService.TaskPhase phase) {
            EngineRpcService.TaskInfoPB.Builder task = terminalTaskBuilder(requestId, phase,
                    EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(8100)
                            .setErrorMessage("cancelled by client (stream gone)").build());
            publishCompletion(task.build());
        }

        private void recordPriorityPreemptionCanceled(long requestId,
                                                      EngineRpcService.TaskPhase phase) {
            addPriorityCancelledRequestId(requestId);
            EngineRpcService.TaskInfoPB.Builder task = terminalTaskBuilder(requestId, phase,
                    EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(PRIORITY_PREEMPTED_ERROR_CODE)
                            .setErrorMessage("preempted by higher-priority request").build())
                    .setPriorityPreemptionProgress(EngineRpcService.PriorityPreemptionProgressPB
                            .PRIORITY_PREEMPTION_CANCELED);
            publishCompletion(task.build());
        }

        /** All cancellation terminals carry the same EnqueueBatch identity. */
        private EngineRpcService.TaskInfoPB.Builder terminalTaskBuilder(long requestId,
                EngineRpcService.TaskPhase phase, EngineRpcService.ErrorDetailsPB error) {
            var original = runningTasks.get(requestId);
            if (original == null) original = cancelingOverlays.get(requestId);
            EngineRpcService.TaskInfoPB.Builder task = (original == null
                    ? EngineRpcService.TaskInfoPB.newBuilder() : original.toBuilder())
                    .setRequestId(requestId).setPhase(phase).setErrorInfo(error)
                    .setEndTimeMs(System.currentTimeMillis());
            var admission = batchAdmissions.get(requestId);
            if (admission != null) task.setBatchId(admission.input.getGroupId().getValue());
            else if (original == null) {
                long batchId = positiveLifecycleBatchId(requestId);
                if (batchId > 0L) task.setBatchId(batchId);
            }
            return task;
        }

        /** Preserve the exact EnqueueBatch identity on typed Prefill terminals. */
        private long positiveLifecycleBatchId(long requestId) {
            synchronized (requestLifecycles) {
                RequestLifecycle lifecycle = requestLifecycles.get(requestId);
                if (lifecycle != null && lifecycle.batchId > 0L) return lifecycle.batchId;
            }
            // Direct generate/test-only ownership has no positive batch identity;
            // leave the proto field unset instead of fabricating one.
            return 0L;
        }

        private void clearUpstreamOwnership(long requestId) {
            synchronized (decodeQueueLock) {
                FastRpcService prefill = upstreamPrefillOwners.remove(requestId);
                if (prefill != null) {
                    prefill.downstreamDecodeOwners.remove(requestId, this);
                    prefill.decodeReservationOwners.remove(requestId, this);
                    prefill.closePrefillSession(requestId);
                    if (prefill.requests.output(requestId) == null) prefill.finishRequest(requestId);
                    // An attached response pump owns its queue directly. No
                    // terminal request may retain a map entry indefinitely.
                    if (!cancelledRequests.containsKey(requestId)) {
                        prefill.requests.removeOutput(requestId);
                    }
                }
            }
        }

        private boolean schedulePrefillCompletion(List<MockPerformanceModel.RequestShape> shapes,
                                                  long batchId, int dpRank) {
            return admitFifoPrefill(batchMembers(shapes, batchId, dpRank), true);
        }

        /**
         * One request's slot in an engine execution batch, carrying its
         * ORIGIN master-batch identity. Regrouped execution batches may span
         * several master batches; per-request ledger attribution (the
         * TaskInfoPB batchId the master reconciles fetch_response against,
         * the engine_events prefill_done batch_id column, the decode
         * hand-off) must stay keyed to the batch the request actually
         * ARRIVED in — never to the engine's internal execution grouping.
         */
        private record BatchMember(
                MockPerformanceModel.RequestShape shape,
                long batchId,
                int dpRank) {
        }

        private final AtomicBoolean fifoDrainScheduled = new AtomicBoolean();

        /** Wake on a real P lease release, not on a batching timeout. */
        private void wakeFifoOnCapacityRelease() {
            if (roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                    || shuttingDown || stopped || !fifoDrainScheduled.compareAndSet(false, true)) return;
            try {
                scheduler.execute(() -> {
                    List<BatchMember> selected = List.of();
                    synchronized (prefillQueueLock) {
                        fifoDrainScheduled.set(false);
                        if (!shuttingDown && !stopped && activePrefillBatches.get() == 0) {
                            selected = pollFifoBatchLocked();
                            if (!selected.isEmpty()) {
                                activePrefillBatches.incrementAndGet();
                                activePrefillRequests.addAndGet(selected.size());
                            }
                        }
                    }
                    if (!selected.isEmpty()) runPrefillBatch(selected);
                });
            } catch (java.util.concurrent.RejectedExecutionException shuttingDownExecutor) {
                fifoDrainScheduled.set(false);
            }
        }

        private long initedKvSchedulerStreams() {
            return activeBlockLeases.entrySet().stream().filter(entry -> entry.getValue().totalBlocks() > 0
                    && runningTasks.containsKey(entry.getKey()) && !decodeWaitingForKv.contains(entry.getKey())
                    && !decodeAllocationInProgress.contains(entry.getKey())).count();
        }

        private boolean decodeMayInitialize(long id) {
            var policy = performance.prefillBatchPolicy();
            return policy.allowsKvInitialization(initedKvSchedulerStreams(),
                    activeBlockLeases.containsKey(id));
        }

        private final Map<Long, BatchMember> resumedPrefillShapes = new ConcurrentHashMap<>();
        private final Map<Long, EngineRpcService.ErrorDetailsPB> prefillLoadFailures = new ConcurrentHashMap<>();
        private final Set<Long> permanentPrefillFailures = ConcurrentHashMap.newKeySet();
        private EngineRpcService.ErrorDetailsPB fifoInputError(MockPerformanceModel.RequestShape shape) {
            var policy = performance.prefillBatchPolicy();
            long reserve = shape.input().getGenerateConfig().getMaxNewTokens() == 0 ? 0 : performance.decodeReserveStep();
            return MockRequestRules.inputError(shape.inputLen(), policy.maxSeqLen(), reserve,
                    (long) cache.totalBlocks() * seqSizePerBlock);
        }

        private void failPrefillBeforeExecution(BatchMember member, long code, String message) {
            long id = member.shape().input().getRequestId();
            synchronized (completionLock) {
                Boolean cancellation = boundaryCancellations.remove(id);
                if (cancellation != null) {
                    commitSchedulerCancellation(id, cancellation);
                    return;
                }
                var task = runningTasks.remove(id);
                if (task == null) return;
                var error = EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(code).setErrorMessage(message).build();
                requestErrors.putIfAbsent(id, error);
                error = requestErrors.get(id);
                publishCompletion(task.toBuilder().setEndTimeMs(System.currentTimeMillis()).setErrorInfo(requestErrors.get(id)).build());
                pendingRequests.decrementAndGet();
                prefillLackMemRejects.increment();
                fifoOwned.remove(id);
                abortMemoryCopies(id); releaseBlockLease(id); releaseReservedDecode(id);
                requests.state(id, "failed"); recordLifecycleEnd(id, "failed");
                var output = EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(id)
                        .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder().setErrorCodeValue((int)error.getErrorCode()).setErrorMessage(error.getErrorMessage())).build();
                var queue = requests.output(id);
                if (queue != null) queue.offer(output);
                var session = prefillSessions.get(id);
                if (session != null) { session.fail(output); continuePrefillSession(session); }
                finishRequest(id);
            }
        }

        private MockPerformanceModel.RequestShape selectFifoCandidate(
                MockPerformanceModel.RequestShape queued, MockPrefillBatchPolicy.Budget budget) {
            long id = queued.input().getRequestId();
            synchronized (completionLock) {
                if (!runningTasks.containsKey(id) || cancelledRequests.containsKey(id)) return null;
                var policy = performance.prefillBatchPolicy();
                var existingLease = activeBlockLeases.get(id);
                boolean alreadyInitialized = existingLease != null && existingLease.totalBlocks() > 0;
                long initializedStreams = policy.maxInitedKvStreams() > 0
                        ? initedKvSchedulerStreams() : 0;
                if (!policy.allowsKvInitialization(initializedStreams, alreadyInitialized)) return null;
                var loaded = hostLoads.get(id);
                if (loaded != null) {
                    var current = loaded.shape();
                    return budget.fits(current.inputLen(), current.hitTokens(), current.prefillSequenceCount(),
                            current.input().getGenerateConfig().getMaxNewTokens() == 0) ? current : null;
                }
                synchronized (cache) {
                    // A preceding batch may have populated or evicted this prefix
                    // since enqueue. Price the batch using the execution-time hit.
                    int hits = performance.prefillDeviceCache ? cache.prefixHitBlocks(reusableKeys(queued, false)) : 0;
                    var current = matchPrefillMemory(new MockPerformanceModel.RequestShape(queued.input(), queued.inputLen(),
                            queued.outputLen(), queued.blockKeys(),
                            Math.min((long) hits * performance.blockSize(), queued.inputLen()), hits, queued.nativeKeys()));
                    if (!budget.fits(current.inputLen(), current.hitTokens(), current.prefillSequenceCount(),
                            current.input().getGenerateConfig().getMaxNewTokens() == 0)) return null;
                    // An initialized waiter owns its existing blocks. Do not
                    // allocate a second lease and overwrite the first one.
                    if (existingLease == null) {
                        var allocation = cache.acquireDetailed(needBlocks(current), reusableKeys(current, false));
                        if (!allocation.success()) {
                            if (allocation.failure() == MockLruBlockCache.AllocationFailure.PERMANENT)
                                permanentPrefillFailures.add(id);
                            return null;
                        }
                        activeBlockLeases.put(id, allocation.lease());
                    }
                    // Admission hit observations are cumulative; execution repricing
                    // must not subtract from a counter already exported to Prometheus.
                    return current;
                }
            }
        }

        /** Queue mutation and task/counter ownership are one admission transaction. */
        private boolean admitFifoPrefill(List<BatchMember> members, boolean group) {
            List<BatchMember> selected = List.of();
            synchronized (prefillQueueLock) {
                if (stopped || shuttingDown) return false;
                synchronized (completionLock) {
                    members = members.stream().filter(member ->
                            !cancelledRequests.containsKey(member.shape().input().getRequestId())).toList();
                    if (members.isEmpty()) return true;
                    var policy = performance.prefillBatchPolicy();
                    for (var member : members) {
                        var shape = member.shape();
                        long id = shape.input().getRequestId();
                        runningTasks.put(id, task(shape, member.batchId(), member.dpRank(), EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                        fifoOwned.add(id);
                    }
                    pendingRequests.addAndGet(members.size());
                    waitingPrefillRequests.addAndGet(members.size());
                    fifoPrefillQueue.enqueue(members, group, policy.maxRequests(), policy.maxInitedKvStreams());
                    for (var member : members) {
                        long id = member.shape().input().getRequestId();
                        var session = prefillSessions.get(id);
                        if (session != null && session.autoFetch) armPrefillWait(id);
                    }
                }
                if (activePrefillBatches.get() == 0) {
                    selected = pollFifoBatchLocked();
                    if (!selected.isEmpty()) {
                        activePrefillBatches.incrementAndGet();
                        activePrefillRequests.addAndGet(selected.size());
                    }
                }
            }
            if (!selected.isEmpty()) runPrefillBatch(selected);
            return true;
        }

        private MockFifoPrefillQueue.State advancePrefill(BatchMember member, MockFifoPrefillQueue.State state) {
            long id = member.shape().input().getRequestId();
            var failure = prefillLoadFailures.remove(id);
            if (failure != null) {
                failPrefillBeforeExecution(member, failure.getErrorCode(), failure.getErrorMessage());
                return MockFifoPrefillQueue.State.FINISHED;
            }
            Boolean priority = boundaryCancellations.remove(id);
            if (priority != null) {
                commitSchedulerCancellation(id, priority);
                return MockFifoPrefillQueue.State.FINISHED;
            }
            if (!runningTasks.containsKey(id)) return MockFifoPrefillQueue.State.FINISHED;
            var load = hostLoads.get(id);
            if (state == MockFifoPrefillQueue.State.LOADING_CACHE) {
                if (load == null || !load.done().isDone()) return state;
                if (load.done().isCompletedExceptionally()) {
                    MockLoadFailure materialization;
                    try { load.done().join(); materialization = null; }
                    catch (java.util.concurrent.CompletionException | java.util.concurrent.CancellationException error) { materialization = MockLoadFailure.from(error); }
                    if (materialization != null) {
                        hostLoads.remove(id); releaseMemoryRead(id);
                        if (materialization.kind == MockLoadFailure.Kind.RETRYABLE) {
                            releaseBlockLease(id);
                            runningTasks.computeIfPresent(id, (ignored, task) -> task.toBuilder()
                                    .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED).setIsWaiting(true).build());
                            requests.state(id, "waiting");
                        } else {
                            var error = EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(602)
                                    .setErrorMessage(materialization.getMessage()).build();
                            requestErrors.putIfAbsent(id, error); prefillLoadFailures.put(id, requestErrors.get(id));
                            var queue = requests.output(id);
                            if (queue != null) queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(id)
                                    .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder().setErrorCodeValue(602)
                                            .setErrorMessage(error.getErrorMessage())).build());
                        }
                        return MockFifoPrefillQueue.State.WAITING;
                    }
                    // PREFILL finalizeAllocatorLoad falls back to recomputation
                    // on a copy failure with no materialization allocation error.
                    releaseMemoryRead(id);
                    var old = load.shape();
                    int gpu = old.hitBlocks() - old.memoryHitBlocks();
                    var recompute = new MockPerformanceModel.RequestShape(old.input(), old.inputLen(), old.outputLen(),
                            old.blockKeys(), Math.min((long) gpu * seqSizePerBlock, old.inputLen()), gpu, old.nativeKeys(), 0);
                    hostLoads.put(id, new HostLoad(recompute, java.util.concurrent.CompletableFuture.completedFuture(null)));
                    return MockFifoPrefillQueue.State.WAITING;
                }
                synchronized (completionLock) {
                    var read = memoryReads.remove(id);
                    if (read != null) {
                        if (performance.prefillDeviceCache) {
                            var keys = load.shape().blockKeys();
                            int prefix = Math.min(keys.size(), load.shape().hitBlocks());
                            activeBlockLeases.computeIfPresent(id, (ignored, lease) -> cache.retainComputed(lease, keys.subList(0, prefix)));
                        }
                        read.consume();
                    }
                }
                requests.state(id, "waiting");
                return MockFifoPrefillQueue.State.WAITING;
            }
            if (state == MockFifoPrefillQueue.State.WAITING) {
                if (load != null) return MockFifoPrefillQueue.State.RUNNING;
                // A loading group's retry retains CanRun in real. Normal loads
                // return to ordinary waiting and are initialized by its scan.
                var admission = initializePrefill(member, performance.prefillBatchPolicy().newBudget(false), true);
                if (admission == null) return state;
                resumedPrefillShapes.put(id, admission.value());
                return admission.state();
            }
            return state;
        }

        private MockFifoPrefillQueue.Admission<BatchMember> initializePrefill(BatchMember member,
                MockPrefillBatchPolicy.Budget budget, boolean group) {
            var shape = selectFifoCandidate(member.shape(), budget);
            if (shape == null) {
                long id = member.shape().input().getRequestId();
                if (permanentPrefillFailures.remove(id)) {
                    failPrefillBeforeExecution(member, 602, "P KV initialization permanently exceeds usable capacity");
                    return new MockFifoPrefillQueue.Admission<>(member, MockFifoPrefillQueue.State.FINISHED);
                }
                return null;
            }
            long id = shape.input().getRequestId();
            boolean resumed = hostLoads.containsKey(id);
            if (!resumed && memoryCache != null && shape.memoryHitBlocks() > 0) {
                var prepared = prepareMemoryReads(List.of(new BatchMember(shape, member.batchId(), member.dpRank()))).getFirst();
                shape = prepared.shape();
                if (shape.memoryHitBlocks() > 0) {
                    memoryReadBlocks.add(shape.memoryHitBlocks());
                    hostLoads.put(id, new HostLoad(shape, hostLoadBackend.apply(id)));
                    requests.state(id, "loading_cache");
                }
            }
            finishPrefillWait(id);
            var state = !resumed && hostLoads.containsKey(id) ? MockFifoPrefillQueue.State.LOADING_CACHE : MockFifoPrefillQueue.State.RUNNING;
            if (state == MockFifoPrefillQueue.State.RUNNING) hostLoads.remove(id);
            budget.add(shape.inputLen(), shape.hitTokens(), shape.prefillSequenceCount(), shape.input().getGenerateConfig().getMaxNewTokens() == 0);
            if (!group) { admittedContextBatch++; admittedContextTokens += Math.max(0, shape.inputLen() - shape.hitTokens()); }
            runningTasks.computeIfPresent(id, (ignored, task) -> task.toBuilder().setPhase(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED).setIsWaiting(true).build());
            return new MockFifoPrefillQueue.Admission<>(new BatchMember(shape, member.batchId(), member.dpRank()), state);
        }

        private List<BatchMember> pollFifoBatchLocked() {
            admittedContextBatch = 0; admittedContextTokens = 0;
            var selected = fifoPrefillQueue.poll(new MockFifoPrefillQueue.Driver<>() {
                public boolean hasError(BatchMember member) {
                    long id = member.shape().input().getRequestId();
                    return boundaryCancellations.containsKey(id) || prefillLoadFailures.containsKey(id) || !runningTasks.containsKey(id);
                }
                public MockFifoPrefillQueue.State advance(BatchMember member, MockFifoPrefillQueue.State state) {
                    return advancePrefill(member, state);
                }
                public BatchMember refreshed(BatchMember member) {
                    long id = member.shape().input().getRequestId();
                    var resumed = resumedPrefillShapes.remove(id);
                    if (resumed != null) member = resumed;
                    var load = hostLoads.get(id);
                    return load == null ? member : new BatchMember(load.shape(), member.batchId(), member.dpRank());
                }
                public MockFifoPrefillQueue.Round<BatchMember> round(boolean group, List<BatchMember> alreadyRunning) {
                    var budget = performance.prefillBatchPolicy().newBudget(!group);
                    for (var member : alreadyRunning) {
                        var shape = member.shape();
                        budget.add(shape.inputLen(), shape.hitTokens(), shape.prefillSequenceCount(), shape.input().getGenerateConfig().getMaxNewTokens() == 0);
                        if (!group) { admittedContextBatch++; admittedContextTokens += Math.max(0, shape.inputLen() - shape.hitTokens()); }
                    }
                    return member -> initializePrefill(member, budget, group);
                }
            });
            waitingPrefillRequests.set(fifoPrefillQueue.size());
            reportFifoScheduler(selected.size());
            scheduleFifoMaintenance();
            return selected;
        }

        private void scheduleFifoMaintenance() {
            if (!fifoPrefillQueue.hasWork() || shuttingDown || stopped || !fifoDrainScheduled.compareAndSet(false, true)) return;
            long epoch = crashEpoch.get();
            long delay = fifoPrefillQueue.loadingSize() > 0 || fifoPrefillQueue.groupCount() > 0 ? 10 : 100;
            scheduler.schedule(() -> {
                if (epoch != crashEpoch.get()) return;
                List<BatchMember> selected = List.of();
                synchronized (prefillQueueLock) {
                    fifoDrainScheduled.set(false);
                    if (!shuttingDown && !stopped && activePrefillBatches.get() == 0) {
                        selected = pollFifoBatchLocked();
                        if (!selected.isEmpty()) {
                            activePrefillBatches.incrementAndGet(); activePrefillRequests.addAndGet(selected.size());
                        }
                    }
                }
                if (!selected.isEmpty()) runPrefillBatch(selected);
            }, delay, TimeUnit.MILLISECONDS);
        }

        private void reportFifoScheduler(int newlyRunning) {
            var values = new LinkedHashMap<String, Number>();
            values.put("rtp_llm_wait_stream_size", fifoPrefillQueue.size());
            values.put("rtp_llm_loading_cache_stream_size", fifoPrefillQueue.loadingSize());
            values.put("rtp_llm_running_stream_size", executingPrefillRequests.get() + newlyRunning);
            values.put("rtp_llm_scheduler_admitted_context_batch_size", admittedContextBatch);
            values.put("rtp_llm_scheduler_admitted_context_token_size", admittedContextTokens);
            values.put("rtp_llm_scheduler_waiting_oldest_age_us", fifoPrefillQueue.oldestAgeUs());
            values.put("rtp_llm_remote_running_stream_size", 0);
            values.put("rtp_llm_pending_decode_stream_size", 0);
            values.put("rtp_llm_decode_since_prefill", 0);
            long fallbacks = fifoPrefillQueue.takeGroupFallbacks();
            if (fallbacks > 0) reportMetricEvent(Map.of("rtp_llm_scheduler_group_fallback_acc", fallbacks));
            var reporter = schedulerMetricReporter;
            if (reporter != null) {
                try { reporter.accept(values); }
                catch (RuntimeException error) { System.err.println("Whale scheduler metric reporting failed: " + error); }
            }
        }

        private static List<MockPerformanceModel.RequestShape> shapesOf(List<BatchMember> members) {
            List<MockPerformanceModel.RequestShape> shapes = new ArrayList<>(members.size());
            for (BatchMember member : members) {
                shapes.add(member.shape());
            }
            return shapes;
        }

        /** Preserve each request's origin identity when constructing a scheduler group. */
        private static List<BatchMember> batchMembers(
                List<MockPerformanceModel.RequestShape> shapes, long batchId, int dpRank) {
            List<BatchMember> members = new ArrayList<>(shapes.size());
            for (MockPerformanceModel.RequestShape shape : shapes) {
                members.add(new BatchMember(shape, batchId, dpRank));
            }
            return members;
        }

        /** Direct requests enter the ordinary FIFO queue. */
        private boolean admitDirectPrefill(MockPerformanceModel.RequestShape shape) {
            return admitFifoPrefill(List.of(new BatchMember(shape, -1L, 0)), false);
        }

        private void runPrefillBatch(List<BatchMember> originalMembers) {
            List<BatchMember> members = originalMembers;
            List<MockPerformanceModel.RequestShape> shapes = shapesOf(members);
            long executionMs = performance.prefillMs(shapes);

            long generateDelayMs = faultConfig.getGenerateDelayMs();
            long now = System.nanoTime();
            long executionNanos = TimeUnit.MILLISECONDS.toNanos(executionMs + generateDelayMs);
            // Each rank has one forward execution timeline.
            AtomicLong nextAvailable = pickPrefillLane(members.get(0).dpRank());
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
            // Per-engine mirror of the same observation (see field comment).
            prefillBatchesExecuted.increment();
            prefillBatchRequestsExecuted.add(shapes.size());
            int sizeBucket = 0;
            while (sizeBucket < PREFILL_BATCH_SIZE_BUCKETS.length
                    && shapes.size() > PREFILL_BATCH_SIZE_BUCKETS[sizeBucket]) sizeBucket++;
            prefillBatchSizeCounts.incrementAndGet(sizeBucket);
            maxPrefillBatchSizeExecuted.accumulateAndGet(shapes.size(), Math::max);
            // Per-engine prefill busy: one executionMs per scheduled batch (the
            // execution duration is known at schedule time in the mock model).
            busyMs.addAndGet(executionMs);
            // Crash fence: callbacks queued before a crash_after must never
            // touch the wiped state — capture the epoch at schedule time and
            // drop out on mismatch (the late callback cannot be unscheduled).
            final long epoch = crashEpoch.get();
            long startDelayNanos = Math.max(0, startNanos - now);
            var tpsBatch = new java.util.concurrent.atomic.AtomicReference<PrefillTpsMetrics.Batch>();
            Runnable start = () -> {
                if (crashEpoch.get() != epoch) return;
                long compute = 0, input = 0;
                synchronized (completionLock) {
                    for (var member : members) {
                        var shape = member.shape();
                        if (cancelledRequests.containsKey(shape.input().getRequestId())) continue;
                        compute += Math.max(0L, shape.inputLen() - shape.hitTokens());
                        input += shape.inputLen();
                    }
                    // Freeze executed membership at start, before cancellations
                    // can turn a completed forward into a failed client request.
                    tpsBatch.set(prefillTps.begin(compute, input, System.nanoTime()));
                }
                startPrefillBatch(members);
            };
            if (startDelayNanos == 0) start.run();
            else scheduler.schedule(start, startDelayNanos, TimeUnit.NANOSECONDS);

            if (memoryCache != null) {
                for (var member : members) {
                    long id = member.shape().input().getRequestId();
                    var read = memoryReads.get(id);
                    // Pins last until the existing execution lane starts; copying adds no delay.
                    long readDelay = startDelayNanos;
                    Runnable release = () -> {
                        if (read != null && memoryReads.remove(id, read)) {
                            read.consume();
                        }
                    };
                    if (readDelay == 0) release.run();
                    else {
                        try { scheduler.schedule(release, readDelay, TimeUnit.NANOSECONDS); }
                        catch (java.util.concurrent.RejectedExecutionException stopped) { release.run(); }
                    }
                }
            }

            long delayNanos = Math.max(0, finishNanos - now);
            scheduler.schedule(() -> {
                if (crashEpoch.get() != epoch) {
                    return; // crashed mid-flight: this batch died with the process
                }
                for (var member : members) {
                    long id = member.shape().input().getRequestId();
                    Boolean priority;
                    synchronized (completionLock) {
                        prefillExecuting.remove(id);
                        priority = boundaryCancellations.remove(id);
                    }
                    if (priority != null) commitSchedulerCancellation(id, priority);
                }
                // MtpExecutor times the execution step after prepareStreams; queue/lane waiting is excluded.
                // Model compute and execution stalls are separate observations; native bookkeeping is not simulated.
                reportMetricEvent(Map.of("rtp_llm_model_forward_us", executionMs * 1000.0,
                        "rtp_llm_sp_step_latency_us", (executionMs + generateDelayMs) * 1000.0));
                int activeCount = 0;
                // ── prefill_async_partial_fail (execution-phase partial
                // failure): snapshot the volatile config ONCE for the whole
                // batch, then select the first k NON-cancelled members in
                // shapes order (the enqueue-ack fault family's cursor
                // pattern — cancelled members already own a terminal and
                // stay untouched). Production semantics: the batch ran, the
                // ack was honest, the failure surfaces per-request through
                // the status channel only — no batch-level propagation.
                // Dual-budget caliber: the cursor is local to this EXECUTION
                // batch (one runPrefillBatch call — under regroup that may
                // span several origin master batches), so k counts per
                // execution batch, matching production where the failure
                // rides each request's own stream into FIFOScheduler
                // regardless of how admission regrouped the members. ──
                FaultInjectionConfig asyncFaultSnapshot = faultConfig;
                int asyncFaultBudget = asyncFaultSnapshot.getPrefillAsyncPartialFail();
                long asyncFaultCode = asyncFaultSnapshot.getPrefillAsyncFailCode();
                final int[] asyncFaultCursor = {0};
                // One wall-clock stamp for the whole batch: every member shares
                // the same completion instant (the batch is the execution unit).
                long doneTsMs = System.currentTimeMillis();
                for (BatchMember member : members) {
                    MockPerformanceModel.RequestShape shape = member.shape();
                    long requestId = shape.input().getRequestId();
                    boolean alreadyCancelled;
                    boolean asyncFail;
                    synchronized (completionLock) {
                        alreadyCancelled = cancelledRequests.containsKey(requestId);
                        asyncFail = !alreadyCancelled && asyncFaultBudget > 0
                                && asyncFaultCursor[0]++ < asyncFaultBudget;
                        EngineRpcService.TaskInfoPB tracked = runningTasks.get(requestId);
                        if (tracked != null && !alreadyCancelled) {
                            if (asyncFail) {
                                recordCompletion(shape, member.batchId(), executionMs,
                                        member.dpRank(), asyncFaultCode);
                                prefillAsyncPartialFails.increment();
                            } else {
                                recordCompletion(shape, member.batchId(), executionMs,
                                        member.dpRank());
                                recordLifecycleEnd(requestId, false);
                                finishPrefillCache(shape);
                            }
                            // Keep the deliberate zombie fault as a report fault.
                            if (!faultConfig.isStatusZombieRunning()) {
                                runningTasks.remove(requestId);
                            }
                            activeCount++;
                        }
                    }
                    // Per-rid prefill terminal event row (engine_events.jsonl,
                    // replaces the former mock_prefill_done stdout trace line —
                    // per-request data now streams as structured JSONL, not
                    // stdout grep). aggregate_canvas_run.py joins prefill_done_ms
                    // / exec_ms against the load client's send_start_epoch_ms by
                    // request_id to bucket prefill exec percentiles on the
                    // request-BIRTH axis (same axis as e2e/full_e2e); exec_ms is
                    // the BATCH execution duration — prefill runs whole batches,
                    // so every member of one batch logs the same value.
                    if (!alreadyCancelled && !asyncFail && shape.outputLen() > 0) {
                        long requestStartMs = requestStartEpochMs(shape.input(), doneTsMs);
                        if (requestStartMs != 0) {
                            reportMetricEvent(Map.of("mock_backend_ttft_us",
                                    (doneTsMs - requestStartMs) * 1000L));
                        }
                    }
                    long prefillArrivedMs = eventArrivalMs(requestId);
                    writePrefillDoneEvent(shape, requestId, member.batchId(), doneTsMs,
                            executionMs, shapes.size(), alreadyCancelled);
                    if (!alreadyCancelled) {
                        // Successful completion/cache accounting is separate
                        // from executed-batch TPS (which includes cancelled work).
                        MockCacheDiagnostics diag = cacheDiagnostics;
                        if (!asyncFail && diag != null) diag.completed(engineName, shape.blockKeys());
                        long inputLen = shape.inputLen();
                        long hitTokens = shape.hitTokens();
                        lifetimeContextComputeTokens.add(Math.max(0L, inputLen - hitTokens));
                        lifetimeContextTokens.add(inputLen);
                        lifetimeContextRequests.increment();
                        if (!asyncFail) {
                            long residenceUs = prefillArrivedMs == 0L ? 0L
                                    : Math.max(0L, doneTsMs - prefillArrivedMs) * 1000L;
                            // The P stream ends after producing its first token, not the D output length.
                            reportMetricEvent(MockStreamMetrics.completed(inputLen,
                                    shape.outputLen() > 0 ? 1 : 0, hitTokens, residenceUs, residenceUs));
                        }
                        hitTokensTotal.addAndGet(hitTokens);
                    }
                    // Python marks the prefill-side lifecycle entry finished when the
                    // prefill phase ends, even though decode may continue elsewhere.
                    // (recordLifecycleEnd is first-terminal-wins — the asyncFail
                    // member must record "failed", not "completed".)
                    if (asyncFail) {
                        recordLifecycleEnd(requestId, "failed");
                    } else {
                        recordLifecycleEnd(requestId, alreadyCancelled);
                    }
                    fifoOwned.remove(requestId);
                    MockPrefillSession session = prefillSessions.get(requestId);
                    if (alreadyCancelled) {
                        closePrefillSession(requestId);
                        releaseReservedDecode(requestId);
                    } else if (asyncFail) {
                        // Preserve the deferred error for a Fetch that arrives
                        // after local failure, while freeing both KV pools now.
                        releaseBlockLease(requestId);
                        releaseReservedDecode(requestId);
                        requests.state(requestId, "failed");
                        if (session != null) {
                            session.fail(EngineRpcService.GenerateOutputsPB.newBuilder()
                                    .setRequestId(requestId)
                                    .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                            .setErrorCodeValue((int) asyncFaultCode)
                                            .setErrorMessage("injected prefill_async_partial_fail code="
                                                    + asyncFaultCode))
                                    .build());
                            continuePrefillSession(session);
                        }
                    } else if (session != null) {
                        // P's scheduler slot is finished; the RPC context still
                        // owns connector KV until Fetch/auto-fetch transfers it.
                        requests.state(requestId, "waiting_fetch");
                        session.prefillDone();
                        continuePrefillSession(session);
                    } else {
                        // A cancellation may remove the context after this
                        // callback took its initial cancellation snapshot.
                        if (cancelledRequests.containsKey(requestId)) {
                            releaseBlockLease(requestId);
                            continue;
                        }
                        // The deferred context was already claimed by a
                        // terminal path. Never manufacture a P-only success.
                        releaseBlockLease(requestId);
                        releaseReservedDecode(requestId);
                    }
                }
                // Real TPS includes scheduler/execution/dispatch elapsed time,
                // once per batch; never divide by the polling or GPU-model time.
                prefillTps.finish(tpsBatch.get(), System.nanoTime());
                activePrefillBatches.decrementAndGet();
                // Mirror the addAndGet(shapes.size()) made when this batch reserved
                // its running slot (admission or drain). Cancelled members stay
                // counted until the batch finishes — the batch keeps executing.
                activePrefillRequests.addAndGet(-shapes.size());
                executingPrefillRequests.addAndGet(-shapes.size());

                pendingRequests.addAndGet(-activeCount);
                List<BatchMember> nextMembers = List.of();
                synchronized (prefillQueueLock) {
                    fifoPrefillQueue.executionFinished();
                    if (activePrefillBatches.get() == 0) {
                        nextMembers = pollFifoBatchLocked();
                        if (!nextMembers.isEmpty()) {
                            activePrefillBatches.incrementAndGet();
                            activePrefillRequests.addAndGet(nextMembers.size());
                        }
                    }
                }
                if (!nextMembers.isEmpty()) runPrefillBatch(nextMembers);
            }, delayNanos, TimeUnit.NANOSECONDS);
        }

        // Optional Whale sink: local test mode has no monitoring dependency.
        private volatile java.util.function.Consumer<Map<String, Number>> schedulerMetricReporter;
        private volatile java.util.function.Consumer<Map<String, Number>> eventMetricReporter;
        private final WhalePrefillMatchMetrics prefillMatchMetrics =
                new WhalePrefillMatchMetrics(WhalePrefillMatchMetrics.configuredWindow());

        private final CacheHitMetrics cacheHitMetrics = new CacheHitMetrics();

        CacheHitMetrics.Snapshot cacheHitSnapshot() { return cacheHitMetrics.snapshot(); }

        private void reportPrefillMatch(MockPerformanceModel.RequestShape shape) {
            long hostTokens = memoryCache == null ? 0L
                    : Math.min(shape.hitTokens(), (long) shape.memoryHitBlocks() * seqSizePerBlock);
            if (prefillCacheEnabled(shape, "reuse_cache")) {
                cacheHitMetrics.record(shape.inputLen(), shape.hitTokens());
                reportMetricEvent(Map.of("rtp_llm_kv_cache_reuse_length", shape.hitTokens(),
                        "rtp_llm_stream_cache_device_reuse_length", shape.hitTokens() - hostTokens,
                        "rtp_llm_stream_cache_memory_reuse_length", hostTokens));
            }
            if (eventMetricReporter != null && roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                reportMetricEvent(prefillMatchMetrics.record(shape.blockKeys(), shape.inputLen(),
                        shape.hitTokens(), seqSizePerBlock, System.currentTimeMillis()));
        }

        private void reportMetricEvent(Map<String, Number> metrics) {
            var reporter = eventMetricReporter;
            if (reporter != null) {
                try { reporter.accept(metrics); }
                catch (RuntimeException error) { System.err.println("Whale event metric reporting failed: " + error); }
            }
        }
        private final AtomicInteger executingPrefillRequests = new AtomicInteger();

        private void reportSchedulerStep(int prefill, int decode) {
            var reporter = schedulerMetricReporter;
            if (reporter == null) return;
            try {
                var values = new LinkedHashMap<String, Number>();
                values.put("rtp_llm_running_stream_size", prefill + decode);
                values.put("rtp_llm_generate_batch_size", decode);
                if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                    values.put("rtp_llm_wait_stream_size", decodePendingQueue.size());
                    values.put("rtp_llm_loading_cache_stream_size", 0);
                    values.put("rtp_llm_remote_running_stream_size", 0);
                    values.put("rtp_llm_pending_decode_stream_size", 0);
                    values.put("rtp_llm_decode_since_prefill", 0);
                    values.put("rtp_llm_scheduler_admitted_context_batch_size", 0);
                    values.put("rtp_llm_scheduler_admitted_context_token_size", 0);
                    long oldest = decodePendingQueue.stream().mapToLong(task -> eventTimings.getOrDefault(task.shape().input().getRequestId(),
                            new EventTiming(System.currentTimeMillis(), 0)).arrivalMs()).min().orElse(System.currentTimeMillis());
                    values.put("rtp_llm_scheduler_waiting_oldest_age_us", Math.max(0, System.currentTimeMillis() - oldest) * 1000);
                }
                reporter.accept(values);
            } catch (RuntimeException error) {
                // Observability must never prevent execution or strand a lease.
                System.err.println("Whale scheduler metric reporting failed: " + error);
            }
        }

        private void startPrefillBatch(List<BatchMember> members) {
            reportMetricEvent(Map.of("rtp_llm_context_batch_size", members.size()));
            int executing = executingPrefillRequests.addAndGet(members.size());

            members.forEach(member -> hostLoads.remove(member.shape().input().getRequestId()));
            // activePrefillBatches is reserved at admission (schedulePrefillCompletion)
            for (BatchMember member : members) {
                // Per-member origin identity: a regrouped batch writes RUNNING
                // with each member's OWN batchId/dpRank (the master reconciles
                // fetch_response against the delivery batch, not the engine's
                // internal execution grouping).
                synchronized (completionLock) {
                    if (cancelledRequests.containsKey(member.shape().input().getRequestId())) {
                        continue; // a delayed batch start must not revive a cancelled report
                    }
                    runningTasks.put(member.shape().input().getRequestId(),
                            task(member.shape(), member.batchId(), member.dpRank(),
                                    EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
                    {
                        prefillExecuting.add(member.shape().input().getRequestId());
                        requests.state(member.shape().input().getRequestId(), "running");
                    }
                }
                reportPrefillMatch(member.shape());
                // engine_events.jsonl: execution-start stamp (lane serialization
                // actually begins for this batch member).
                recordEventStart(member.shape().input().getRequestId());
            }
        }

        /**
         * Locate the DECODE engine a request's role_addrs route to (the same
         * routing source the EnqueueBatch Phase-1.6 reservation uses — one
         * resolver, one answer, so the reserved D is by construction the D the
         * hand-off targets). Returns null when no DECODE addr is present, the
         * addr's service is unknown to the cluster, or it resolves to THIS
         * engine (P/D co-located topologies route decode in-process and get no
         * cross-engine reservation).
         */
        private EngineRpcService.ErrorDetailsPB decodePreparationError(
                FastRpcService decode, MockPerformanceModel.RequestShape shape) {
            Throwable error = decode == null ? remotePreparationError.get() : decode.allocationError.get();
            if (decode == null) remotePreparationError.remove(); else decode.allocationError.remove();
            if (hasPriorityCancelledRequestId(shape.input().getRequestId()))
                return EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(PRIORITY_PREEMPTED_ERROR_CODE)
                        .setErrorMessage("preempted during ALLOCATE").build();
            var details = MockRpcErrors.downstream(error);
            var deadline = requests.deadline(shape.input().getRequestId());
            if (details.getErrorCode() == 8207 && deadline != null && deadline.remainingMs() == 0)
                return EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(603)
                        .setErrorMessage("request deadline exhausted during ALLOCATE").build();
            return details;
        }

        private FastRpcService findDecodeEngine(EngineRpcService.GenerateInputPB input) {
            if (whaleRemote) return null;
            for (EngineRpcService.RoleAddrPB addr : input.getGenerateConfig().getRoleAddrsList()) {
                if (RoleTypeProtoConverter.fromRoleAddr(addr)
                        != RoleType.DECODE) {
                    continue;
                }
                FastRpcService decode = services.get(addr.getGrpcPort());
                if (decode != null && decode.grpcPort != grpcPort) {
                    return decode;
                }
                return null;
            }
            return null;
        }

        /**
         * Release this engine's bookkeeping for a request's decode-side KV
         * reservation, returning the reserved blocks to the owning D pool
         * (idempotent: no recorded reservation = no-op). Called from the
         * prefill cancel branch, the completion callback's alreadyCancelled
         * branch, a rejected hand-off, and the direct-path backpressure
         * reject — the full release-loop closure for pre-reservations that
         * never became running streams.
         */
        private void releaseReservedDecode(long requestId) {
            RemotePrefillOwner remote = remotePrefillOwners.remove(requestId);
            if (remote != null) remote.close();
            FastRpcService decode = decodeReservationOwners.remove(requestId);
            if (decode != null) {
                clearDecodeOwnership(requestId, decode);
                decode.cancel(requestId, false, false);
            }
        }

        void setAutoFetch(boolean enabled) {
            autoFetch = enabled;
        }

        boolean autoFetchEnabled() { return autoFetch; }

        void setFetchAttachTimeoutMs(long timeoutMs) {
            if (timeoutMs < 1) {
                throw new IllegalArgumentException("fetch attach timeout must be positive");
            }
            fetchAttachTimeoutMs = timeoutMs;
        }

        private boolean prepareDecodeSession(FastRpcService decode, MockPerformanceModel.RequestShape shape,
                                             long batchId, int dpRank, boolean automaticContinuation) {
            long id = shape.input().getRequestId();
            var lifetime = beginRequest(shape.input());
            var mmProcessed = new java.util.concurrent.atomic.AtomicBoolean();
            try {
                return performance.pdPolicy.prepare(
                        () -> {
                            boolean localOnly = (shape.input().getGenerateConfig().getMaxNewTokens() == 0
                                    || shape.input().getGenerateConfig().getReturnPromptLogits());
                            if (whaleRemote && !localOnly) {
                                try { awaitDecodeChannel(decodeChannel(shape.input()), id, shape.input()); }
                                catch (Exception failure) {
                                    remotePreparationError.set(failure);
                                    if (io.grpc.Status.fromThrowable(failure).getCode() == io.grpc.Status.Code.CANCELLED) throw failure;
                                    return false;
                                }
                            }
                            if (!mmProcessed.get() && isDiagnosticPrefill()
                                    && (shape.input().getMultimodalInputsCount() > 0 || shape.input().hasV41Inputs())) {
                                try { vitClient.prepare(shape.input(), lifetime.remainingMs()); mmProcessed.set(true); }
                                catch (RuntimeException failure) {
                                    if (decode == null) remotePreparationError.set(failure); else decode.allocationError.set(failure);
                                    if (!MockVitClient.retryable(failure)) throw failure;
                                    return false;
                                }
                            }
                            return prepareDecodeSessionAttempt(decode, shape, batchId, dpRank, automaticContinuation);
                        },
                        () -> allocationCancelled(id, () -> hasFencedRequestId(id)),
                        lifetime::remainingMs);
            } catch (Exception error) {
                if (error instanceof InterruptedException) Thread.currentThread().interrupt();
                var first = requestErrors.get(id);
                if (first != null) error = MockRpcErrors.error(MockRpcErrors.status(first), first.getErrorCode(), first.getErrorMessage());
                closePrefillSession(id);
                releaseReservedDecode(id);
                if (decode == null) remotePreparationError.set(error); else decode.allocationError.set(error instanceof RuntimeException runtime ? runtime : new RuntimeException(error));
                return false;
            }
        }

        private boolean prepareDecodeSessionAttempt(FastRpcService decode,
                MockPerformanceModel.RequestShape shape, long batchId, int dpRank,
                boolean automaticContinuation) {
            long requestId = shape.input().getRequestId();
            if (allocationCancelled(requestId, () -> hasFencedRequestId(requestId)))
                throw MockRpcErrors.error(io.grpc.Status.CANCELLED, 8100, "cancelled before ALLOCATE");
            if (beginRequest(shape.input()).remainingMs() == 0)
                throw MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 603, "request deadline exhausted before ALLOCATE");
            boolean localOnly = (shape.input().getGenerateConfig().getMaxNewTokens() == 0
                    || shape.input().getGenerateConfig().getReturnPromptLogits());
            if (whaleRemote && !localOnly) return prepareRemoteDecode(shape, batchId, dpRank, automaticContinuation);
            if (localOnly) decode = null;
            long id = shape.input().getRequestId();
            MockPrefillSession session = new MockPrefillSession(shape, batchId, dpRank,
                    automaticContinuation, decode);
            if (decode == null) {
                prefillSessions.put(id, session);
                return true;
            }
            synchronized (decode.decodeQueueLock) {
                if (decode.runningTasks.containsKey(id) || !decode.decodeAllocationInProgress.add(id)) {
                    return false; // never overwrite or release another live D request
                }
            }
            try {
                return allocateDecodeSession(decode, session);
            } finally {
                decode.decodeAllocationInProgress.remove(id);
            }
        }

        private MockPerformanceModel.RequestShape shapeForDecode(
                FastRpcService decode, MockPerformanceModel.RequestShape shape) {
            var input = MockRequestRules.decodeInput(shape.input());
            // P groups are scheduler-local; D receives independent streams.
            if (!shape.nativeKeys() || seqSizePerBlock == decode.seqSizePerBlock)
                return new MockPerformanceModel.RequestShape(input, shape.inputLen(), shape.outputLen(),
                        shape.blockKeys(), shape.hitTokens(), shape.hitBlocks(), shape.nativeKeys(), shape.memoryHitBlocks());
            var local = decode.performance.shape(input, decode.cache);
            return new MockPerformanceModel.RequestShape(input, shape.inputLen(),
                    shape.outputLen(), local.blockKeys(), local.hitTokens(), local.hitBlocks(), true);
        }

        private boolean allocateDecodeSession(FastRpcService decode, MockPrefillSession session) {
            MockPerformanceModel.RequestShape shape = session.shape;
            long batchId = session.batchId;
            long id = shape.input().getRequestId();
            // Retry without decodeQueueLock: existing completions must be able
            // to release KV while ALLOCATE waits for capacity.
            var deadline = requests.deadline(id);
            if (deadline != null) decode.requests.inheritDeadline(id, deadline);
            if (decode.admissionClosed || decode.stopped || decode.shuttingDown || !decode.reserveDecodeLease(id, shapeForDecode(decode, shape),
                    () -> allocationCancelled(id, () -> hasFencedRequestId(id)))) {
                if (deadline != null) decode.requests.removeDeadline(id, deadline);
                return false;
            }
            synchronized (decode.decodeQueueLock) {
                if (decode.admissionClosed || decode.stopped || decode.shuttingDown
                        || allocationCancelled(id, () -> hasFencedRequestId(id)) || decode.cancelledRequests.containsKey(id)) {
                    decode.releaseBlockLease(id);
                    if (deadline != null) decode.requests.removeDeadline(id, deadline);
                    return false;
                }
                prefillSessions.put(id, session);
                decodeReservationOwners.put(id, decode);
                registerDecodeOwnership(id, decode);
                decode.decodeWaitingForKv.add(id);
                decode.runningTasks.put(id, task(shape, batchId, 0,
                        EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED));
                decode.pendingRequests.incrementAndGet();
                decode.acceptedCount.incrementAndGet();
                decode.requests.state(id, "waiting_for_kv");
                decode.recordLifecycleStart(id, batchId, "remote_allocate");
                synchronized (decode.requestLifecycles) {
                    decode.requestLifecycles.get(id).runningMs = 0L;
                }
                decode.recordEventArrival(id);
                decode.lastEnqueueTime.set(System.nanoTime());
                return true;
            }
        }

        private final Map<Long, java.util.concurrent.ScheduledFuture<?>> prefillWaitTimers = new ConcurrentHashMap<>();
        private void finishPrefillWait(long id) {
            var timer = prefillWaitTimers.remove(id);
            if (timer != null) timer.cancel(false);
        }
        private void armPrefillWait(long id) {
            if (!fifoOwned.contains(id)
                    || prefillExecuting.contains(id) || hostLoads.containsKey(id)) return;
            prefillWaitTimers.computeIfAbsent(id, ignored -> scheduler.schedule(() -> {
                prefillWaitTimers.remove(id);
                if (fifoOwned.contains(id) && !prefillExecuting.contains(id) && !hostLoads.containsKey(id)) {
                    requestErrors.putIfAbsent(id, EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(8213)
                            .setErrorMessage("wait to run timeout").build());
                    cancel(id, false, false);
                }
            }, Math.max(0, performance.pdPolicy.prefillMaxWaitTimeoutMs()), TimeUnit.MILLISECONDS));
        }

        private EngineRpcService.ErrorDetailsPB armFetchExpiry(long requestId, long requestedTimeoutMs) {
            MockPrefillSession session = prefillSessions.get(requestId);
            if (session == null || session.autoFetch) {
                return null;
            }
            long timeoutMs = requestedTimeoutMs > 0 ? requestedTimeoutMs : fetchAttachTimeoutMs;
            long requestTimeoutMs = session.shape.input().getGenerateConfig().getTimeoutMs();
            if (requestTimeoutMs > 0) {
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(
                        System.nanoTime() - session.preparationStartedNanos);
                long remaining = requestTimeoutMs - elapsedMs;
                var absolute = requests.deadline(requestId);
                if (absolute != null) remaining = Math.min(remaining, absolute.remainingMs());
                if (remaining <= 0) {
                    var error = EngineRpcService.ErrorDetailsPB.newBuilder().setErrorCode(603)
                            .setErrorMessage("request deadline expired before FetchResponse became available").build();
                    requestErrors.putIfAbsent(requestId, error);
                    cancel(requestId, false, false);
                    return error;
                }
                timeoutMs = Math.min(timeoutMs, remaining);
            }
            session.setExpiry(scheduler.schedule(() -> {
                if (session.expireUnattached() && prefillSessions.remove(requestId, session)) {
                    fetchAttachExpirations.incrementAndGet();
                    // The local P compute may already be FINISHED. Its execution
                    // slot is separate from the retained RPC/KV context; cancel
                    // must still release P's connector pin and D's ALLOCATE.
                    cancel(requestId, false, false);
                }
            }, timeoutMs, TimeUnit.MILLISECONDS));
            return null;
        }

        private void closePrefillSession(long requestId) {
            MockPrefillSession session = prefillSessions.remove(requestId);
            if (session != null) {
                session.close();
            }
            releaseBlockLease(requestId);
        }

        private void continuePrefillSession(MockPrefillSession session) {
            long id = session.shape.input().getRequestId();
            if (!session.claimContinuation()) {
                return;
            }
            // Fetch can attach before compute ends, or after it. Both paths
            // converge here, once, after the local first token is available.
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue = requests.output(id);
            if (session.failure() != null) {
                if (queue != null) {
                    queue.offer(session.failure());
                }
                requests.removeOutput(id);
                prefillSessions.remove(id, session);
                session.close();
                if (!fifoOwned.contains(id)) releaseBlockLease(id);
                return;
            }
            if (queue != null && (session.decode != null || session.remoteDecode)
                    && (!streamsTokenPayloads() || session.shape.outputLen() > 1)) {
                queue.offer(buildOutput(session.shape, false));
            }
            if (!session.isClosed() && !cancelledRequests.containsKey(id)) {
                if (session.remoteDecode) {
                    continueRemoteDecode(session);
                    return;
                } else if (session.decode != null && session.shape.outputLen() == 1
                        && session.decode.completeAllocatedPrefill(id, shapeForDecode(session.decode, session.shape))) {
                    decodeReservationOwners.remove(id, session.decode);
                    clearDecodeOwnership(id, session.decode);
                    if (queue != null) queue.offer(buildOutput(session.shape, true));
                    requests.removeOutput(id, queue);
                } else if (session.decode == null) {
                    completedCount.incrementAndGet();
                    requests.state(id, "completed");
                    if (queue != null) {
                        queue.offer(buildOutput(session.shape, true));
                    }
                    requests.removeOutput(id);
                } else if (!startDecode(session.shape, session.batchId)) {
                    requests.state(id, "failed");
                    if (queue != null) {
                        queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                                .setRequestId(id)
                                .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                        .setErrorCode(EngineRpcService.ErrorCodePB.UNKNOWN_ERROR)
                                        .setErrorMessage("prepared decode became unavailable"))
                                .build());
                    }
                    requests.removeOutput(id, queue);
                }
            }
            prefillSessions.remove(id, session);
            session.close();
            // KV transfer is instantaneous in the mock; the same transition
            // that starts Decode releases the P connector's extra reference.
            releaseBlockLease(id);
        }

        private boolean startDecode(MockPerformanceModel.RequestShape shape, long batchId) {
            EngineRpcService.GenerateInputPB input = shape.input();
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    requests.output(input.getRequestId());
            FastRpcService decode = decodeReservationOwners.get(input.getRequestId());
            if (decode == null) {
                // No routable decode engine — the reservation phase (if any)
                // found no D either, so nothing to release here.
                return false;
            }
            // Propagate the decode admission result: true = admitted/queued,
            // false = Decode is shutting down. The caller emits a failure;
            // losing a prepared Decode cannot become a successful P-only result.
            registerDecodeOwnership(input.getRequestId(), decode);
            boolean accepted = decode.scheduleDecodeCompletion(shapeForDecode(decode, shape), batchId, queue);
            if (!accepted) {
                clearDecodeOwnership(input.getRequestId(), decode);
                releaseReservedDecode(input.getRequestId());
            } else {
                // Reservation consumed: the pre-allocated lease (if this
                // request carried one) transfers into the decode lifecycle —
                // scheduleDecodeCompletion's admission found it in
                // activeBlockLeases and adopted it without re-charging the D
                // pool. Drop the P-side bookkeeping; release paths from here
                // on are the decode engine's own cancel/terminal flows.
                decodeReservationOwners.remove(input.getRequestId());
            }
            return accepted;
        }

        /**
         * Admission point for a decode request. Returns true if the request was
         * accepted (scheduled immediately, queued behind the concurrency gate,
         * already scheduled previously, or already cancelled); false only while
         * a shutdown drain is in progress so the caller can fail the request.
         *
         * <p>The hard concurrency gate is UNCONDITIONAL (production semantics:
         * decodeMaxConcurrency caps running requests; once full, new requests
         * park in the unbounded decodePendingQueue — the engine-side analogue
         * of waiting_streams_ — and drain as completions free slots; nothing is
         * ever rejected on the decode side for queue pressure).
         *
         * <p>The putIfAbsent guard (ConcurrentDoubleSchedulingTest) is preserved
         * to prevent double-scheduling the same requestId. It runs INSIDE
         * decodeQueueLock, together with the cancelled-marker check and the
         * counter claims: cancel() publishes its cancelledRequests marker before
         * taking the lock and does its own removal/release under the same lock,
         * so a cancel can never observe the runningTasks entry while the matching
         * counters are still unclaimed. (Previously putIfAbsent ran before the
         * lock; a cancel landing in that window over-decremented counters that
         * were never incremented — a permanent slot/pendingRequests/KV leak.)
         */
        private boolean scheduleDecodeCompletion(MockPerformanceModel.RequestShape shape, long batchId,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue) {
            long requestId = shape.input().getRequestId();
            // engine_events.jsonl arrival stamp: decode-engine arrival is the
            // hand-off moment (covers BOTH the immediate-admission and the
            // parked-in-waiting-queue paths below).
            recordEventArrival(requestId);
            // Shutdown drain in progress — reject before any claim so a
            // cross-engine prefill hand-off racing the drain cannot re-populate
            // runningTasks after the cancel sweep. The caller degrades exactly
            // like pending-queue backpressure (no residue on this engine).
            if (shuttingDown) {
                return false;
            }
            synchronized (decodeQueueLock) {
                if (stopped || shuttingDown) {
                    return false;
                }
                // Cancel raced ahead of scheduling: bail out before claiming
                // anything. The cancel path has already surfaced the CANCELLED
                // completion/response, so treat the request as accepted-and-
                // cancelled rather than as backpressure.
                if (cancelledRequests.containsKey(requestId)) {
                    return true;
                }
                if (decodeAllocationInProgress.contains(requestId)) {
                    return false; // a duplicate direct call raced remote ALLOCATE
                }
                boolean prepared = decodeWaitingForKv.remove(requestId);
                if (prepared) {
                    pendingRequests.decrementAndGet(); // existing ALLOCATE claim transfers below
                }
                // Guard: never schedule the same requestId twice on this engine.
                EngineRpcService.TaskInfoPB nextTask =
                        task(shape, batchId, 0, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING);
                EngineRpcService.TaskInfoPB existing;
                if (prepared) {
                    runningTasks.put(requestId, nextTask);
                    existing = null;
                } else {
                    existing = runningTasks.putIfAbsent(requestId, nextTask);
                }
                if (existing != null) {
                    return true; // already accepted/scheduled on this engine
                }
                if (activeDecodeRequests.get() < decodeMaxConcurrency
                        && (!decodeStepScheduled)
                        && decodeMayInitialize(requestId)) {
                    // A slot is free — admit immediately as a running DecodeStream
                    // in the per-step loop. ALL run-start bookkeeping (slot + KV +
                    // the stream itself) is claimed under the lock so the step
                    // loop can never advance a half-admitted stream and a
                    // concurrent completion/cancel drain cannot admit another
                    // request into the same slot. pendingRequests is incremented
                    // here to match the unconditional decrement in the
                    // step-completion terminal path (owned). The queued path
                    // increments it at enqueue; both must balance so
                    // checkLeakDrain / periodicCleanup see net zero.
                    activeDecodeRequests.incrementAndGet();
                    pendingRequests.incrementAndGet();
                    // Adopt an existing ALLOCATE lease or provision KV at run
                    // start. Allocation failure is a request terminal failure.
                    // Fix #5: the decode engine re-matches the request's keys
                    // against its OWN LRU here — net demand
                    // ceil(il/spb) − hitBlocks, reused blocks referenced not
                    // re-allocated (production reuse_block_size semantics).
                    // A P-enqueue reservation (if the request carried one) is
                    // adopted here without re-charging the pool.
                    MockLruBlockCache.AllocationOutcome admission =
                            acquireDecodeBlockLeaseDetailed(requestId, shape);
                    if (!admission.success()) {
                        countDecodeKvFailure(admission.failure());
                        // Roll back every claim this branch made (slot,
                        // pendingRequests, the runningTasks admission entry and
                        // the P->D ownership) and publish the terminal failure.
                        // The request IS accepted-and-terminal from the caller's
                        // perspective: production rejects at ALLOCATE and the
                        // master may retry — the mock terminal is final (README
                        // documents the retry-semantics divergence).
                        activeDecodeRequests.decrementAndGet();
                        pendingRequests.decrementAndGet();
                        clearUpstreamOwnership(requestId);
                        publishDecodeKvFailure(requestId, shape, batchId,
                                0, responseQueue, "admission");
                        return true;
                    }
                    // New streams join only before the next forward is armed.
                    decodeRunning.put(requestId,
                            newDecodeStream(shape, batchId, responseQueue));
                    stats.decodeAdmitted.increment();
                    recordLifecycleRunning(requestId);
                    recordEventStart(requestId);
                    scheduleDecodeStepLocked();
                } else {
                    // Concurrency gate hit — park the request in the unbounded
                    // pending queue (production waiting_streams_ semantics:
                    // engine waits, never rejects). It will be drained when a
                    // running request completes.
                    // PD preparation owns KV before execution; ordinary waiting streams do not.
                    if (prepared) {
                        runningTasks.put(requestId, task(shape, batchId, 0,
                                EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED));
                        MockLruBlockCache.AllocationOutcome queuedClaim =
                                acquireDecodeBlockLeaseDetailed(requestId, shape);
                        if (!queuedClaim.success()) {
                            countDecodeKvFailure(queuedClaim.failure());
                            clearUpstreamOwnership(requestId);
                            publishDecodeKvFailure(requestId, shape, batchId,
                                    0, responseQueue, "admission");
                            return true;
                        }
                    }
                    if (!prepared)
                        runningTasks.put(requestId, task(shape, batchId, 0, EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    decodePendingQueue.addLast(new DecodePendingTask(shape, batchId, responseQueue));
                    pendingRequests.incrementAndGet();
                    recordLifecycleStart(requestId, batchId,
                            batchId >= 0 ? "enqueue_batch" : "generate_stream");
                    lastEnqueueTime.set(System.nanoTime());
                    return true; // queued (accepted, will run when a slot frees)
                }
            }
            // Admitted immediately — record lifecycle outside the lock (the
            // stream itself is already registered and the step loop armed
            // under the lock above).
            recordLifecycleStart(requestId, batchId,
                    batchId >= 0 ? "enqueue_batch" : "generate_stream");
            lastEnqueueTime.set(System.nanoTime());
            return true;
        }

        /**
         * Per-step decode loop tick. Runs on the SHARED completion scheduler (one
         * chained task per engine, guarded by decodeStepScheduled, so 1250 engines
         * cost 1250 lightweight pending timers on the completionThreads pool —
         * no per-engine thread). Advances every running stream by one step
         * (tokensPerStep tokens per stream, MTP fold); the step's duration was
         * locked in when it was ARMED (the batch size at that boundary prices
         * the whole step — see pendingStepDelayMs), and a batch-size change
         * (completion, top-up, cancel) re-prices the NEXT step. The executing
         * batch stays fixed until that boundary. Completed streams release their
         * ownership here; the
         * waiting batch is topped up immediately (production top-up semantics),
         * then the next step is chained. Terminal bookkeeping is claimed under
         * decodeQueueLock (racing cancel); the completion frame, stats and
         * cleanup publish outside the lock, mirroring the former one-shot
         * completion callback's split.
         */
        private void runDecodeStep() {
            long forwardMs = -1;
            int acceptedTokens = 0;
            int participatingStreams = 0;
            List<DecodeStream> finished = new ArrayList<>();
            List<DecodeStream> kvFailed = new ArrayList<>();
            synchronized (decodeQueueLock) {
                decodeStepScheduled = false;
                if (shuttingDown || decodeRunning.isEmpty()) {
                    return;
                }
                // Consume the duration locked in when this step was ARMED: the
                // batch size at that boundary priced the whole step, so booked
                // exec time always matches elapsed time. The count read here
                // only decides WHO advances, not how long the step was.
                long stepDelayMs = pendingStepDelayMs;
                forwardMs = stepDelayMs;
                if (!boundaryCancellations.isEmpty() || !boundaryDownstreamCancellations.isEmpty()) {
                    decodeBoundaryMaintenance = true;
                    try {
                        for (long id : List.copyOf(decodeRunning.keySet())) {
                            var downstream = boundaryDownstreamCancellations.remove(id);
                            if (downstream != null) cancelDownstream(id, downstream.prefill(), downstream.priority());
                            Boolean priority = boundaryCancellations.remove(id);
                            if (priority != null) commitSchedulerCancellation(id, priority);
                        }
                    } finally { decodeBoundaryMaintenance = false; }
                }
                for (Iterator<DecodeStream> it = decodeRunning.values().iterator(); it.hasNext(); ) {
                    DecodeStream stream = it.next();
                    // One step = tokensPerStep tokens (MTP fold): the step budget
                    // was pre-computed as ceil(outputLen / tokensPerStep) at
                    // admission, so the tick only decrements whole steps.
                    int previousGenerated = (int) Math.ceil(stream.generatedTokens);
                    stream.generatedTokens = Math.min(stream.shape.outputLen(),
                            stream.generatedTokens + performance.tokensPerStep());
                    int generated = (int) Math.ceil(stream.generatedTokens);
                    if (stream.firstTokenTimeMs == 0 && generated > previousGenerated)
                        stream.firstTokenTimeMs = System.currentTimeMillis();
                    stream.remainingSteps = (int) Math.ceil((stream.shape.outputLen()
                            - stream.generatedTokens) / performance.tokensPerStep());
                    stream.accumulatedExecMs += stepDelayMs;
                    // Per-step KV growth (production incrMalloc): extend the
                    // lease toward ceil((inputLen + generated)/spb) blocks — a
                    // no-op except at spb/tokensPerStep step boundaries. Runs
                    // under decodeQueueLock (lock order: queue -> cache).
                    // Production-aligned failure semantics (20260903): a growth
                    // failure is a REQUEST TERMINAL FAILURE (production
                    // incrKVBlock LACK_MEM terminates the request) — the former
                    // stall-and-continue degradation is retired.
                    MockLruBlockCache.AllocationFailure growthFailure =
                            growDecodeLeaseLocked(stream.shape.input().getRequestId(),
                                    stream.shape.inputLen() + generated);
                    if (growthFailure != null) {
                        countDecodeKvFailure(growthFailure);
                        it.remove();
                        kvFailed.add(stream);
                        continue;
                    }
                    decodeTps.add(Math.max(0, generated - previousGenerated));
                    // Token one was produced by P; count only tokens generated by this D step.
                    int accepted = Math.max(0, generated - Math.max(1, previousGenerated));
                    if (accepted > 0) {
                        acceptedTokens += accepted;
                        participatingStreams++;
                    }
                    if (stream.remainingSteps <= 0) {
                        it.remove();
                        finished.add(stream);
                    } else if (streamsTokenPayloads() && stream.responseQueue != null) {
                        // P already emitted token one. Publish D progress while
                        // it runs so Fetch sees activity before its idle timeout.
                        int previous = Math.max(1, stream.emittedOutputTokens);
                        int streamed = Math.min(stream.shape.outputLen() - 1, generated);
                        if (streamed > previous) {
                            stream.responseQueue.offer(buildOutput(
                                    stream.shape, false, streamed, streamed - previous));
                            stream.emittedOutputTokens = streamed;
                        }
                    }
                }
                for (DecodeStream stream : finished) {
                    claimDecodeTerminalLocked(stream, false);
                }
                // KV-failure streams claim the SAME terminal machinery (slot /
                // pendingRequests / runningTasks / ownership) as exhausted
                // streams — only the post-lock publish differs.
                for (DecodeStream stream : kvFailed) {
                    claimDecodeTerminalLocked(stream, true);
                }
                topUpDecodeRunningLocked();
                scheduleDecodeStepLocked();
            }
            if (forwardMs >= 0) {
                reportMetricEvent(MockDecodeStepMetrics.values(forwardMs, acceptedTokens, participatingStreams));
            }
            for (DecodeStream stream : finished) {
                if (stream.owned) {
                    publishDecodeCompletion(stream);
                }
            }
            for (DecodeStream stream : kvFailed) {
                if (stream.owned) {
                    finishDecodeKvFailure(
                            stream.shape.input().getRequestId(), stream.shape,
                            stream.batchId, stream.terminalBatchSize,
                            stream.responseQueue, "growth");
                }
            }
        }

        /**
         * Arm the next step tick. MUST be called holding decodeQueueLock.
         * At most one pending tick per engine: admission, top-up and the tick
         * itself all funnel through here, so the loop is a single chained task
         * while streams are running and stops naturally when the engine idles.
         */
        private void scheduleDecodeStepLocked() {
            if (decodeStepScheduled || decodeBoundaryMaintenance || decodeRunning.isEmpty() || shuttingDown) {
                return;
            }
            reportSchedulerStep(0, decodeRunning.size());
            long delayMs = performance.decodeStepDelayMs(decodeRunning.size());
            pendingStepDelayMs = delayMs; // lock in this step's price at arm time
            decodeStepScheduled = true;
            // Crash fence: a tick armed before a crash_after must not advance
            // the wiped engine — the late callback drops out on epoch mismatch.
            final long epoch = crashEpoch.get();
            scheduler.schedule(() -> {
                if (crashEpoch.get() == epoch) {
                    runDecodeStep();
                }
            }, delayMs, TimeUnit.MILLISECONDS);
        }

        /**
         * Admit waiting-queue heads into free running slots (production top-up:
         * freed slots are handed to queued requests immediately at the step
         * boundary where the slot was released). MUST be called holding
         * decodeQueueLock. Reuses the skip-cancelled pattern of the former
         * completion drain: a queued request cancelled while queued has no
         * runningTasks entry (cancel already released its pendingRequests) and
         * is dropped here.
         */
        private void topUpDecodeRunningLocked() {
            if (decodeBoundaryMaintenance || (decodeStepScheduled)) return;
            {
                decodeBoundaryMaintenance = true;
                try {
                    for (var queued : List.copyOf(decodePendingQueue)) {
                        long id = queued.shape().input().getRequestId();
                        var downstream = boundaryDownstreamCancellations.remove(id);
                        if (downstream != null) cancelDownstream(id, downstream.prefill(), downstream.priority());
                        Boolean priority = boundaryCancellations.remove(id);
                        if (priority != null) commitSchedulerCancellation(id, priority);
                    }
                } finally { decodeBoundaryMaintenance = false; }
            }
            var waiting = decodePendingQueue.iterator();
            while (waiting.hasNext() && activeDecodeRequests.get() < decodeMaxConcurrency) {
                DecodePendingTask candidate = waiting.next();
                long candidateId = candidate.shape().input().getRequestId();
                if (!runningTasks.containsKey(candidateId)) {
                    waiting.remove();
                    continue;
                }
                if (!decodeMayInitialize(candidateId)) continue;
                waiting.remove();
                activeDecodeRequests.incrementAndGet();
                // Adopt an existing ALLOCATE lease without charging twice.
                // Otherwise re-match this engine's LRU before allocating.
                if (activeBlockLeases.get(candidateId) == null) {
                    MockLruBlockCache.AllocationOutcome runStart =
                            acquireDecodeBlockLeaseDetailed(candidateId, candidate.shape());
                    if (!runStart.success()) {
                        countDecodeKvFailure(runStart.failure());
                        // Roll back the slot claim, drop the admission entry and
                        // the enqueue-time pendingRequests claim, release the
                        // P->D ownership, and publish the terminal failure. The
                        // freed slot loops to the NEXT queued candidate.
                        activeDecodeRequests.decrementAndGet();
                        pendingRequests.decrementAndGet();
                        clearUpstreamOwnership(candidateId);
                        publishDecodeKvFailure(candidateId, candidate.shape(),
                                candidate.batchId(), 0, candidate.responseQueue(), "admission");
                        continue;
                    }
                }
                {
                    runningTasks.computeIfPresent(candidateId, (id, tracked) ->
                            tracked.toBuilder()
                                            .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                                            .setIsWaiting(false).build());
                }
                decodeRunning.put(candidateId,
                        newDecodeStream(candidate.shape(), candidate.batchId(), candidate.responseQueue()));
                stats.decodeAdmitted.increment();
                recordLifecycleRunning(candidateId);
                recordEventStart(candidateId);
                lastEnqueueTime.set(System.nanoTime());
            }
        }

        /**
         * Claim the terminal transition for a stream that just exhausted its
         * step budget (outputLen tokens via the MTP fold).
         * MUST be called holding decodeQueueLock. Whichever of this path or
         * cancel() removes the runningTasks entry first owns the terminal: the
         * winner releases the slot/KV/pendingRequests counters; the loser does
         * nothing (stream.owned stays false and no completion publishes).
         */
        private void claimDecodeTerminalLocked(DecodeStream stream, boolean kvFailure) {
            long requestId = stream.shape.input().getRequestId();
            finishRequest(requestId);
            synchronized (completionLock) {
                if (!runningTasks.containsKey(requestId)) {
                    return; // cancellation already published and removed it
                }
                if (kvFailure) {
                    publishCompletion(decodeKvFailureTask(requestId, "growth"));
                } else {
                    recordCompletion(stream.shape, stream.batchId,
                            Math.round(stream.accumulatedExecMs), 0);
                }
                if (!faultConfig.isStatusZombieRunning()) {
                    runningTasks.remove(requestId);
                }
            }
            clearUpstreamOwnership(requestId);
            stream.owned = true;
            stream.terminalBatchSize = decodeRunning.size() + 1;
            activeDecodeRequests.decrementAndGet();
            // Capacity model v2: the lease is handed to the LRU OUTSIDE this
            // lock by publishDecodeCompletion (admitBlockLease) — release !=
            // delete — so cancelling cannot double-release it.
            pendingRequests.decrementAndGet();
        }

        private DecodeStream newDecodeStream(MockPerformanceModel.RequestShape shape, long batchId,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue) {
            var stream = new DecodeStream(shape, batchId, responseQueue,
                    performance.decodeSteps(shape.outputLen()));
            var lease = activeBlockLeases.get(shape.input().getRequestId());
            stream.initialReuseTokens = lease == null ? 0L
                    : Math.min(shape.inputLen(), (long) lease.hitKeys.size() * seqSizePerBlock);
            return stream;
        }

        /**
         * Publish the normal completion of a stream whose terminal ownership was
         * claimed at a step boundary. Caliber note: executionMs is the SUM OF
         * ACTUAL STEP DURATIONS the stream experienced (per-step continuous
         * batching), replacing the former one-shot estimate
         * decodeMs(outputLen, batchSizeAtAdmission) — the two diverge whenever
         * the running batch size changes mid-flight, which is exactly the
         * production-shaped behaviour this model now reproduces.
         */
        private void publishDecodeCompletion(DecodeStream stream) {
            MockPerformanceModel.RequestShape shape = stream.shape;
            long requestId = shape.input().getRequestId();
            long executionMs = Math.round(stream.accumulatedExecMs);
            boolean alreadyCancelled = cancelledRequests.containsKey(requestId);
            long arrivedMs = eventArrivalMs(requestId);
            if (!alreadyCancelled) {
                // D owns its own admission-to-first-token and terminal observations.
                long firstUs = arrivedMs == 0L || stream.firstTokenTimeMs == 0 ? 0L
                        : Math.max(0L, stream.firstTokenTimeMs - arrivedMs) * 1000L;
                reportMetricEvent(MockStreamMetrics.completed(shape.inputLen(), shape.outputLen(),
                        stream.initialReuseTokens, firstUs,
                        arrivedMs == 0L ? 0L : Math.max(0L, System.currentTimeMillis() - arrivedMs) * 1000L));
            }
            if (!alreadyCancelled) {
                long doneMs = System.currentTimeMillis();
                long requestStartMs = requestStartEpochMs(shape.input(), doneMs);
                if (requestStartMs != 0) {
                    reportMetricEvent(Map.of("mock_backend_latency_us",
                            (doneMs - requestStartMs) * 1000L));
                }
            }
            // The status terminal was published by claimDecodeTerminalLocked.
            // Feed the per-sample decode completion window (java_mock_stats
            // decode_done / decode_exec_*): Σ actual step durations.
            stats.recordDecodeDone(executionMs);
            // Per-rid decode terminal event row (engine_events.jsonl, replaces
            // the former mock_decode_done stdout trace line). aggregate_canvas_run.py
            // joins decode_done_ms / exec_ms against the load client's
            // per-request send_start_epoch_ms by request_id to build the
            // schedule-only full-e2e metric (client send → decode end); without
            // this row the engine side has no per-rid persisted terminal time
            // (completions queue / requestLifecycles are in-memory only).
            writeDecodeDoneEvent(stream,
                    cancelledRequests.containsKey(requestId));
            // Per-engine decode busy: one executionMs per completed request.
            busyMs.addAndGet(executionMs);
            recordLifecycleEnd(requestId, alreadyCancelled);
            if (!alreadyCancelled) {
                completedCount.incrementAndGet();
                requests.state(requestId, "completed");
                // Successful output totals are business completions; execution
                // TPS comes from decodeTps, including work later cancelled.
                lifetimeGenerateTokens.add(shape.outputLen());
                // Stream dimensions were emitted with the terminal timing above.
            }
            // Completion does not depend on a client Fetch in auto-fetch mode;
            // strict mode reaches this point only after the client attached.
            if (stream.responseQueue != null && !alreadyCancelled) {
                if (streamsTokenPayloads()) {
                    int previous = shape.outputLen() > 1
                            ? Math.max(1, stream.emittedOutputTokens) : 0;
                    stream.responseQueue.offer(buildOutput(
                            shape, true, shape.outputLen(), shape.outputLen() - previous));
                } else {
                    stream.responseQueue.offer(buildOutput(shape, true));
                }
            }
            requests.removeOutput(requestId);
            cancelledRequests.remove(requestId);
            // Capacity model v2 + fix #5 (cancelled streams never admit): the
            // stream's block lease hands its key blocks to the LRU on a NORMAL
            // completion (release != delete: cache_keys grows, availability is
            // restored because pure-LRU blocks count as available). A stream
            // whose cancel landed AFTER the terminal claim (alreadyCancelled)
            // instead releases its blocks to the pool WITHOUT LRU handover —
            // production cancel runs free() and leaves no cache, mirroring the
            // prefill-side completion's alreadyCancelled branch.
            if (alreadyCancelled) {
                releaseBlockLease(requestId);
            } else {
                admitBlockLease(requestId, shape);
            }
        }

        /** Terminal scheduler KV failure retains domain MALLOC_FAILED (602).
         * Queue/lease ownership is claimed before publication; failed leases never enter LRU. */
        private void publishDecodeKvFailure(
                long requestId,
                MockPerformanceModel.RequestShape shape,
                long batchId,
                int terminalBatchSize,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue,
                String stage) {
            synchronized (completionLock) {
                publishCompletion(decodeKvFailureTask(requestId, stage));
                if (!faultConfig.isStatusZombieRunning()) {
                    runningTasks.remove(requestId);
                }
            }
            finishDecodeKvFailure(requestId, shape, batchId, terminalBatchSize, responseQueue, stage);
        }

        private EngineRpcService.TaskInfoPB decodeKvFailureTask(long requestId, String stage) {
            EngineRpcService.TaskInfoPB.Builder task = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(requestId)
                    .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(LACK_MEM_ERROR_CODE)
                            .setErrorMessage(String.format(
                                    "decode KV %s failure: MALLOC_FAILED (602; "
                                            + "request terminated)", stage))
                            .build())
                    .setEndTimeMs(System.currentTimeMillis())
                    .setDpRank(0);
            // Same batch-identity discipline as the cancel terminals: the
            // master's reconcile correlates the failure with the EnqueueBatch
            // it dispatched or the inflight slot leaks.
            long lifecycleBatchId = positiveLifecycleBatchId(requestId);
            if (lifecycleBatchId > 0L) {
                task.setBatchId(lifecycleBatchId);
            }
            return task.build();
        }

        private void finishDecodeKvFailure(long requestId, MockPerformanceModel.RequestShape shape,
                long batchId, int terminalBatchSize,
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue, String stage) {
            // engine_events.jsonl failure row (error_code key present →
            // aggregate/canvas joins exclude it from latency calibers).
            writeDecodeDoneEventCore(shape, batchId, terminalBatchSize, false,
                    LACK_MEM_ERROR_CODE);
            // Lifecycle: the failure may predate the admission-path lifecycle
            // entry (immediate-admission failures return before
            // recordLifecycleStart) — backfill an arrival row in that case so
            // the "failed" terminal is observable in /requests; queued/topUp
            // paths already hold an entry (only the terminal state is set).
            synchronized (requestLifecycles) {
                if (!requestLifecycles.containsKey(requestId)) {
                    recordLifecycleStart(requestId, batchId,
                            batchId >= 0 ? "enqueue_batch" : "generate_stream");
                }
            }
            recordLifecycleEnd(requestId, "failed");
            requests.state(requestId, "failed");
            if (responseQueue != null) {
                responseQueue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(requestId)
                        .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                .setErrorCodeValue(602)
                                .setErrorMessage(String.format(
                                        "LACK_MEM (602): decode KV %s failed — request terminated",
                                        stage))
                                .build())
                        .build());
            }
            requests.removeOutput(requestId);
            cancelledRequests.remove(requestId);
            // Growth failures hold a (partial) lease — return it to the pool
            // without LRU handover (a failed request leaves no cache, same as
            // cancel). Admission failures hold nothing: idempotent no-op.
            releaseBlockLease(requestId);
        }

        /** Snapshot size of the decode pending queue (for waitingQueryLen reporting). */
        private int decodePendingQueueSize() {
            synchronized (decodeQueueLock) {
                return decodePendingQueue.size();
            }
        }

        /** Scheduler WAITING streams only; a D ALLOCATE waiting for P/Fetch is not enqueued yet. */
        private int schedulerWaitingStreamSize() {
            return roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    ? decodePendingQueueSize() : waitingPrefillRequests.get();
        }

        private int schedulerLoadingStreamSize() {
            synchronized (prefillQueueLock) { return fifoPrefillQueue.loadingSize(); }
        }

        /** Snapshot size of the prefill pending queue in BATCHES (cap accounting unit). */
        private int prefillPendingQueueSize() {
            synchronized (prefillQueueLock) {
                return fifoPrefillQueue.groupCount();
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
                    .setIsWaiting(phase != EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .setDpRank(dpRank)
                    .build();
        }

        private static EngineRpcService.TaskInfoPB withLegacyTaskState(
                EngineRpcService.TaskInfoPB task) {
            return task.toBuilder()
                    .setIsWaiting(task.getPhase()
                            != EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .build();
        }

        private void recordCompletion(MockPerformanceModel.RequestShape shape,
                                      long batchId,
                                      long executionMs,
                                      int dpRank) {
            recordRecentExecutionTime(executionMs);
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
            publishCompletion(task);
        }

        /**
         * prefill_async_partial_fail variant: the SAME normal versioned
         * completion path (publishCompletion — the zombie / duplicate
         * output-layer faults compose on top), but the TaskInfo carries
         * error_info(code, "injected prefill_async_partial_fail"):
         * production "stream->reportError → dequeue fills
         * task_info.error_code / error_message → finished_task_list"
         * semantics. batch_id rides along unchanged (stream group_id →
         * TaskInfo) so the master's reconcile correlates the typed failure
         * terminal with the dispatched batch.
         */
        private void recordCompletion(MockPerformanceModel.RequestShape shape,
                                      long batchId,
                                      long executionMs,
                                      int dpRank,
                                      long errorCode) {
            recordRecentExecutionTime(executionMs);
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
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(errorCode)
                            .setErrorMessage("injected prefill_async_partial_fail")
                            .build())
                    .build();
            publishCompletion(task);
        }

        private void publishCompletion(EngineRpcService.TaskInfoPB task) {
            synchronized (completionLock) {
                // status_zombie_running: drop the completion record entirely —
                // the request finished internally but is never reported
                // finished (paired with the runningTasks re-insert at the
                // completion points).
                if (faultConfig.isStatusZombieRunning()) {
                    return;
                }
                if (statusDeliveryFaultsActive && statusDeliveryFaults.defer(task, System.nanoTime())) {
                    return;
                }
                publishCompletionNow(task);
            }
        }

        /** Caller holds completionLock. Fault-delayed records get their cursor here. */
        private void publishCompletionNow(EngineRpcService.TaskInfoPB task) {
            cancelingOverlays.remove(task.getRequestId());
            long version = completionVersion.getAndIncrement();
            StatusDeliveryFaults.Rule delivery = statusDeliveryFaults.rules.get(task.getRequestId());
            if (delivery != null) delivery.releasedVersion = version;
            completions.add(new VersionedTask(version, task));
            // status_duplicate_finished: enqueue the SAME completion twice
            // under the SAME version — one poll reports the rid twice, and
            // advancing the cursor past that version consumes both copies.
            if (faultConfig.isStatusDuplicateFinished()) {
                completions.add(new VersionedTask(version, task));
            }
            trimCompletionsLocked();
        }

        private void trimCompletionsLocked() {
            while (completions.size() > DEFAULT_COMPLETION_RETAIN_WINDOW) {
                completions.removeFirst();
                completionRetentionDrops++;
            }
        }

        private Set<Long> runningTaskIds() {
            synchronized (completionLock) {
                return new LinkedHashSet<>(runningTasks.keySet());
            }
        }

        private boolean streamsTokenPayloads() {
            return whaleRemote || whaleBundle;
        }

        static boolean supportsSingleSequence(EngineRpcService.GenerateInputPB input) {
            var config = input.getGenerateConfig();
            return config.getNumReturnSequences() <= 1 && config.getNumBeams() <= 1
                    && config.getVariableNumBeamsList().stream().allMatch(width -> width <= 1);
        }

        private EngineRpcService.GenerateOutputsPB buildOutput(MockPerformanceModel.RequestShape shape,
                                                               boolean finished) {
            int outputLen = streamsTokenPayloads() && !finished ? Math.min(1, shape.outputLen()) : shape.outputLen();
            // Frontend concatenates frames. P already sent the first token, so
            // a remote D terminal carries only the remaining tokens. A one-token
            // request has no preliminary frame: Python rejects an empty tensor.
            int stepOutputLen = streamsTokenPayloads() && finished
                    && roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE && outputLen > 1
                    ? outputLen - 1 : outputLen;
            return buildOutput(shape, finished, outputLen, stepOutputLen);
        }

        private EngineRpcService.GenerateOutputsPB buildOutput(MockPerformanceModel.RequestShape shape,
                                                               boolean finished, int outputLen,
                                                               int stepOutputLen) {
            var flatten = EngineRpcService.FlattenOutputPB.newBuilder()
                    .addFinished(finished)
                    .addAuxInfo(EngineRpcService.AuxInfoPB.newBuilder()
                            .setInputLen(shape.inputLen())
                            .putAllMultimodalLengths(multimodalLengths(shape.input()))
                            .setPrefixLen((int) shape.hitTokens())
                            .setOutputLen(outputLen)
                            .setIterCount(1)
                            .setStepOutputLen(stepOutputLen)
                            .build());
            if (streamsTokenPayloads()) {
                // QueryConverter / Python trans_output require [outputs, beams, tokens]
                // and little-endian INT32 bytes. Repeat an input token, not model inference.
                int tokenCount = shape.input().getTokenIdsCount();
                int token = tokenCount == 0 ? 0 : shape.input().getTokenIds(tokenCount - 1);
                ByteBuffer ids = ByteBuffer.allocate(Math.multiplyExact(stepOutputLen, Integer.BYTES))
                        .order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < stepOutputLen; i++) ids.putInt(token);
                flatten.setOutputIds(EngineRpcService.TensorPB.newBuilder()
                        .setDataType(EngineRpcService.TensorPB.DataType.INT32)
                        .addShape(1).addShape(1).addShape(stepOutputLen)
                        .setInt32Data(com.google.protobuf.ByteString.copyFrom(ids.array())));
            }
            return EngineRpcService.GenerateOutputsPB.newBuilder()
                    .setRequestId(shape.input().getRequestId())
                    .setFlattenOutput(flatten)
                    .build();
        }

        void getCacheStatusWire(EngineRpcService.CacheVersionPB request,
                                StreamObserver<CacheFixtureWire.Frame> observer) {
            observer.onNext(cacheStatusFrame(request));
            observer.onCompleted();
        }

        @Override
        public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                   StreamObserver<EngineRpcService.CacheStatusPB> observer) {
            observer.onNext(cacheStatusFrame(request).toProto());
            observer.onCompleted();
        }

        // Real LocalRpcServer::GetCacheStatus still builds a protobuf map per RPC.
        // This extra Mock allocation optimization is deliberately opt-in.
        boolean cacheStatusCompactEncoding;

        private java.util.Set<Long> naturalCacheKeys() {
            return cacheKeyIndex.keys();
        }

        private CacheFixtureWire.Frame cacheStatusFrame(EngineRpcService.CacheVersionPB request) {
            long started = System.nanoTime();
            stats.cacheRpcs.increment();
            MockCacheKeyIndex.Snapshot snapshot;
            synchronized (cacheKeyIndex) {
                snapshot = cacheKeyIndex.snapshot(cacheSyncFixture, request.getLatestCacheVersion(), request.getNeedCacheKeys());
            }
            var status = EngineRpcService.CacheStatusPB.newBuilder()
                    .setAvailableKvCache(availableKvTokens()).setTotalKvCache(totalKvTokens)
                    .setBlockSize(performance.blockSize()).setVersion(snapshot.version());
            // Real reports metadata read time before protobuf key-map serialization.
            long readUs = Math.max(0L, (System.nanoTime() - started) / 1000);
            cacheStatusRates.add(1);
            cacheStatusRtUs.set(readUs);
            cacheMetricReporter.accept(readUs);
            if (cacheStatusCompactEncoding) {
                var keys = CacheFixtureWire.NaturalKeys.takeSnapshots(snapshot.natural(), new long[0], snapshot.external());
                return new CacheFixtureWire.Frame(status.build(), snapshot.external(), keys);
            }
            for (long key : snapshot.natural())
                if (snapshot.external() == null || !snapshot.external().contains(key)) status.putCacheKeys(key, true);
            return new CacheFixtureWire.Frame(status.build(), snapshot.external());
        }

        /**
         * Force-evict block keys from this engine's LRU (control-plane POST
         * /cache_evict — the flexlb_test_framework KV family's forced-eviction hook).
         * Idempotent: evicting keys that are not present is a no-op. When
         * the key set changes the engine's cacheVersion is bumped, so the
         * master's next cache-status poll re-pulls the key set and its
         * global key→holder index converges on the eviction (the test
         * family's sync premise).
         *
         * @return true when at least one key was actually removed
         */
        boolean evictCacheKeys(List<Long> keys) {
            return cache.evict(keys);
        }

        // ──────────── KV capacity model v2: block-pool helpers ────────────

        /** Physical demand is derived from token length, independently of cache-key count. */
        private int needBlocks(MockPerformanceModel.RequestShape shape) {
            return (shape.inputLen() + seqSizePerBlock - 1) / seqSizePerBlock;
        }

        /**
         * KV v2 fix #5 — decode admission with local reuse deduction: at
         * hand-off the decode engine re-matches the request's block keys
         * against its OWN LRU (production DecodeRpcServerNew:
         * {@code reuse_block_size = generate_stream->reuseBlockSize()}; the
         * engine receives the PD transfer from [reuse, total) only) and admits
         * on the NET demand ceil(inputLen/spb) − hitBlocks (floor 0). Reused
         * blocks are referenced, never re-allocated.
         *
         * <p>Token caliber: the demand is ceil(inputLen/spb) — the FULL input
         * including the hash-channel-uncovered suffix (decode holds the
         * request's complete KV for its lifetime; prefill-side admission keeps
         * its hash-key-count caliber). The decode re-match also gives the
         * decode LRU its READ path — the hit count is independent of the
         * prefill engine's shape.hitTokens (which stays untouched for TPS
         * accounting).
         *
         * <p>Growth semantics: the lease's totalBlocks() starts at
         * hitBlocks + netNew = the priced demand, so the per-step growth
         * only ever allocates the GENERATED increment — the reuse deduction
         * persists for the stream's whole life (the old keys.size-caliber
         * lease started below the token demand and back-filled the uncovered
         * suffix at the first grow boundary, diluting the deduction whenever
         * the pool ran dry).
         *
         * <p>Reserve-step window (production {@code reserve_step_}, JSON
         * "decode.reserve_step", default 0 = non-speculative): when > 0 the
         * demand prices {@code ceil((inputLen + reserve_step)/spb)} — a
         * CONSTANT look-ahead front-loaded at admission and CONSUMED as
         * seq_len grows (see {@link #decodeDemandBlocks}); 0 keeps the exact
         * ceil(inputLen/spb) fit bit-identical.
         *
         * @return the outcome ({@code lease} registered in
         *         {@code activeBlockLeases} on success; {@code failure} = the
         *         PERMANENT/RETRYABLE family on LACK_MEM — callers terminate
         *         the request either way, production-aligned semantics)
         */
        private MockLruBlockCache.AllocationOutcome acquireDecodeBlockLeaseDetailed(
                long requestId, MockPerformanceModel.RequestShape shape) {
            // P-enqueue decode-KV pre-alignment (20260903): a request whose
            // decode blocks were RESERVED on this engine at prefill ENQUEUE
            // time already holds its lease in activeBlockLeases — consume it
            // (adoption): the pool is NOT re-charged and decodeReuseBlocks was
            // already booked at reservation time, exactly once.
            MockLruBlockCache.BlockLease reserved = activeBlockLeases.get(requestId);
            if (reserved != null) {
                return new MockLruBlockCache.AllocationOutcome(reserved, null);
            }
            int totalDemand = decodeDemandBlocks(shape.inputLen());
            MockLruBlockCache.AllocationOutcome outcome =
                    cache.acquireWithReuseDetailed(totalDemand,
                            reusableKeys(shape, true));
            if (!outcome.success()) {
                return outcome;
            }
            // Reuse observability (KV v2 fix #5): the hit keys are exactly the
            // blocks this decode admission did NOT re-allocate — the
            // net-demand deduction, accumulated for the /metrics counter.
            decodeReuseBlocks.add(outcome.lease().hitKeys.size());
            if (MockCachePolicy.reuse(shape.input(), shape.nativeKeys(), performance.decodeReuseCache)) {
                long reuseTokens = Math.min(shape.inputLen(), (long) outcome.lease().hitKeys.size() * seqSizePerBlock);
                cacheHitMetrics.record(shape.inputLen(), reuseTokens);
                reportMetricEvent(Map.of("rtp_llm_kv_cache_reuse_length", reuseTokens,
                        "rtp_llm_stream_cache_device_reuse_length", reuseTokens,
                        "rtp_llm_stream_cache_memory_reuse_length", 0L));
            }
            activeBlockLeases.put(requestId, outcome.lease());
            return outcome;
        }

        /**
         * Single decode block-demand caliber: {@code ceil((tokens +
         * reserve_step)/spb)} — the INITIAL admission, every per-step growth
         * target and the P-side reservation rejection message all price
         * through here, so the reserve-step window (production
         * {@code reserve_step_ = gen_num_per_circle + 1} on speculative
         * configs, 0 otherwise — CONSTANT front-load consumed by growth,
         * never an accumulating append) can never disagree across surfaces.
         */
        int decodeDemandBlocks(int totalTokens) {
            return (totalTokens + performance.decodeReserveStep() + seqSizePerBlock - 1)
                    / seqSizePerBlock;
        }

        /** DecodeRpcServer::allocateResource: permanent errors fail immediately;
         * temporary pressure waits for a resource generation change or request stop. */
        boolean reserveDecodeLease(long requestId, MockPerformanceModel.RequestShape shape) {
            return reserveDecodeLease(requestId, shape, () -> false);
        }

        private boolean allocationCancelled(long id, java.util.function.BooleanSupplier ownerCancelled) {
            var admission = batchAdmissions.get(id);
            return stopped || shuttingDown || cancelledRequests.containsKey(id) || ownerCancelled.getAsBoolean()
                    || (admission != null && admission.liveError() != null)
                    || Context.current().isCancelled() || Thread.currentThread().isInterrupted();
        }

        private boolean reserveDecodeLease(long requestId, MockPerformanceModel.RequestShape shape,
                                           java.util.function.BooleanSupplier ownerCancelled) {
            allocationError.remove();
            var lifetime = requests.deadline(requestId);
            if (lifetime == null) lifetime = new MockRequestDeadline(shape.input().getGenerateConfig().getTimeoutMs());
            while (true) {
                if (lifetime.remainingMs() == 0) {
                    allocationError.set(MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 603, "decode allocation exceeded request timeout"));
                    return false;
                }
                if (allocationCancelled(requestId, ownerCancelled)) {
                    allocationError.set(io.grpc.Status.CANCELLED.withDescription("decode allocation cancelled").asRuntimeException());
                    return false;
                }
                long generation = cache.allocationGeneration();
                var outcome = acquireDecodeBlockLeaseDetailed(requestId, shape);
                if (outcome.success()) return true;
                if (outcome.failure() == MockLruBlockCache.AllocationFailure.PERMANENT) {
                    countDecodeKvFailure(outcome.failure());
                    allocationError.set(capacityError(602, "decode ALLOCATE permanently exceeds KV capacity"));
                    return false;
                }
                try {
                    while (cache.allocationGeneration() == generation && lifetime.remainingMs() > 0
                            && !allocationCancelled(requestId, ownerCancelled)) {
                        cache.awaitCapacityChange(generation, Math.min(Math.max(performance.pdPolicy.decodeRetryIntervalMs(), 50), lifetime.remainingMs()));
                    }
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        /**
         * Terminal decode-KV failure accounting, split by production family
         * (KVCacheAllocator.cc:100-110): RETRYABLE (pool total sufficient,
         * temporarily short) → {@link #kvAdmissionFails}
         * (mock_engine_kv_admission_fails_total); PERMANENT (required +
         * reserve > physical total) → {@link #prefillLackMemRejects}
         * (mock_engine_lack_mem_rejects_total — the same counter the P-POOL
         * 602 surface feeds, by design: the request can never fit).
         */
        private void countDecodeKvFailure(MockLruBlockCache.AllocationFailure family) {
            if (family == MockLruBlockCache.AllocationFailure.PERMANENT) {
                prefillLackMemRejects.increment();
            } else {
                kvAdmissionFails.increment();
            }
        }

        /**
         * Normal completion: release the request's lease and hand its cache-keyed
         * blocks to the LRU (release != delete: the key set grows, availability is
         * restored because pure-LRU blocks count as available). Keyless held
         * blocks return to free.
         */
        private boolean admitBlockLease(long requestId, MockPerformanceModel.RequestShape shape) {
            MockLruBlockCache.BlockLease lease = activeBlockLeases.remove(requestId);
            if (lease == null) {
                return false;
            }
            if ((roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                    && (!performance.decodeDeviceCache || !MockCachePolicy.reuse(shape.input(), shape.nativeKeys(), performance.decodeReuseCache)))
                    || (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL && !storePrefillDevice(shape))) {
                releaseAfterMemoryCopy(requestId, lease);
                return false;
            }
            boolean changed = cache.admit(lease, shape.blockKeys());
            scheduleWatermarks();
            wakeFifoOnCapacityRelease();
            return changed;
        }

        private void releaseAfterMemoryCopy(long id, MockLruBlockCache.BlockLease lease) {
            var write = memoryWrites.get(id);
            if (write == null || !write.deferRelease(lease)) cache.release(lease);
        }

        /** Cancel path: return the lease's blocks to the pool without LRU handover. */
        private void releaseBlockLease(long requestId) {
            releaseMemoryRead(requestId);
            MockLruBlockCache.BlockLease lease = activeBlockLeases.remove(requestId);
            if (lease != null) {
                releaseAfterMemoryCopy(requestId, lease);
                scheduleWatermarks();
                wakeFifoOnCapacityRelease();
            }
        }

        /**
         * Per-step decode growth (production incrMalloc): extend the running
         * request's allocation toward {@code decodeDemandBlocks(inputLen +
         * grown)} blocks — the SAME caliber as the initial admission, so a
         * reserve_step window (when configured) is consumed by growth instead
         * of being re-appended every step. Free blocks first, LRU-tail
         * eviction second; on exhaustion the growth FAILS (family-classified
         * and counted by the caller via countDecodeKvFailure) — production-
         * aligned semantics (20260903): the request is TERMINATED with
         * LACK_MEM (production incrKVBlock failure aborts the stream), never
         * left running with a stalled allocation.
         *
         * @return the failure family when the pool cannot satisfy the growth
         *         (caller terminates the request); {@code null} on success
         */
        private MockLruBlockCache.AllocationFailure growDecodeLeaseLocked(
                long requestId, int totalTokensSoFar) {
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            if (lease == null) {
                // Defensive: with the un-pooled degradation retired every
                // running stream holds a lease (immediate admission, top-up and
                // reservation-consumption all provision one). A missing lease
                // here is a bookkeeping bug — keep the stream running rather
                // than failing it on a defensive path.
                return null;
            }
            return cache.growTo(lease, decodeDemandBlocks(totalTokensSoFar)).failure();
        }

        /** Tokens pinned by in-flight requests: (held + referenced key blocks) x spb. */
        private long occupiedKvTokens() {
            return (long) (cache.heldBlocks() + cache.referencedKeyBlocks()) * seqSizePerBlock;
        }

        long getOccupiedKvTokens() { return occupiedKvTokens(); }
        long getCrashEpoch() { return crashEpoch.get(); }

        /** Tokens available per the pool (free + pure-LRU blocks count) — LRU included. */
        private long poolAvailableKvTokens() {
            return (long) cache.availableBlocks() * seqSizePerBlock;
        }

        /**
         * Master-facing available tokens: pool availability clamped to the reported
         * total, minus injected KV pressure.  The total is pool-aligned at
         * construction (totalBlocks x spb), so the clamp is a structural guard
         * (pool availability never exceeds the pool) that keeps the
         * available <= total invariant explicit on every reporting surface.
         */
        private long availableKvTokens() {
            return Math.max(0L,
                    Math.min(totalKvTokens, poolAvailableKvTokens()) - faultConfig.getKvPressureTokens());
        }

        /** Master-facing used tokens (occupied + pressure, clamped to total). */
        private long usedKvTokens() {
            return Math.min(totalKvTokens,
                    occupiedKvTokens() + faultConfig.getKvPressureTokens());
        }

        /** Current cache version (gRPC getCacheStatus / /cache_evict echo). */
        long getCacheVersion() {
            synchronized (cacheKeyIndex) { return cacheKeyIndex.version(cacheSyncFixture); }
        }

        /** One-time warm start of the ordinary cache; subsequent requests own all changes. */
        synchronized Map<String, Object> seedNaturalCache(List<Long> keys) {
            if (naturalCacheSeeded)
                throw new IllegalArgumentException("natural cache already seeded");
            if (acceptedCount.get() != 0 || cache.heldBlocks() != 0 || !cache.snapshotKeys().isEmpty())
                throw new IllegalArgumentException("natural seed requires a fresh idle engine");
            if (keys.isEmpty() || keys.size() > Math.min(cache.totalBlocks(), cache.retentionBlocks())
                    || keys.size() > 1_000_000 || new java.util.HashSet<>(keys).size() != keys.size())
                throw new IllegalArgumentException("seed requires unique keys within physical cache capacity");
            cache.admit(keys);
            naturalCacheSeeded = true;
            return Map.of("mode", "natural", "keys", cache.snapshotKeys().size(), "version", getCacheVersion());
        }

        synchronized Map<String, Object> cacheFixture(String operation, int worker, int workers,
                int unique, int shared, int delta, int intervalMs) {
            if (!"PREFILL".equals(getRoleName()))
                throw new IllegalArgumentException("cache fixture requires PREFILL");
            // Serialize source selection with snapshots and process reset.
            synchronized (cacheKeyIndex) {
                if ("configure".equals(operation)) {
                    if (cacheSyncFixture != null)
                        throw new IllegalArgumentException("external keys already configured");
                    var candidate = new CacheSyncFixture(worker, workers, unique, shared, 1);
                    candidate.churn(delta, intervalMs); // Validate everything before selecting the source.
                    cacheSyncFixture = candidate;
                } else {
                    if (cacheSyncFixture == null) throw new IllegalArgumentException("controlled fixture not configured");
                    if ("churn".equals(operation)) cacheSyncFixture.churn(delta, intervalMs);
                    else if (!"inspect".equals(operation)) throw new IllegalArgumentException("unknown fixture operation");
                }
                synchronized (cacheSyncFixture) {
                    var result = new java.util.LinkedHashMap<>(cacheSyncFixture.snapshot());
                    var external = cacheSyncFixture.keys();
                    result.put("fixture_version", external.version());
                    result.put("version", getCacheVersion());
                    result.put("reported_keys", external.count()
                            + naturalCacheKeys().stream().filter(key -> !external.contains(key)).count());
                    return result;
                }
            }
        }

        @Override
        public void checkHealth(EngineRpcService.EmptyPB request,
                                StreamObserver<EngineRpcService.CheckHealthResponsePB> observer) {
            observer.onNext(EngineRpcService.CheckHealthResponsePB.newBuilder().setHealth("OK").build());
            observer.onCompleted();
        }

        /**
         * gRPC Cancel (proto {@code RpcService/Cancel}, the priority-preemption
         * engine contract) with the production-faithful C++ Prefill mapping:
         * a live request and its accepted-cancel terminal record both return
         * ACCEPTED (idempotent retry, the ACTIVE_CANCEL branch); a request
         * this addressed Prefill has SEEN but which already finished returns
         * NOT_FOUND — seen-but-terminal, the production recently-seen
         * behaviour (the completion record stays deliverable from the
         * retain-window backlog); a rid the engine NEVER saw returns REQUEST_FENCED —
         * {@link #cancelRequest} installs the ABSENT_FENCE record, so any
         * racing later Enqueue of the same rid is rejected with the typed
         * 8429 before reaching the scheduler. The production 10-minute recently-seen TTL is retained. Decode rejects the RPC with
         * UNIMPLEMENTED (production role contract). The HTTP control-plane
         * {@code /cancel_request} (diagnostics surface, not the master's
         * live path) keeps its own JSON shape with the
         * {@code already_finished} flag.
         */
        @Override
        public void cancel(EngineRpcService.CancelRequestPB request,
                           StreamObserver<EngineRpcService.CancelResponsePB> observer) {
            // Cancel-RPC fault injections (RPC-layer failures; the HTTP
            // /cancel_request and in-process channel surfaces run the same
            // arriveCancelRpc gate). Everything below short-circuits BEFORE
            // cancelRequest so the engine state machine is never touched —
            // production semantics "RPC failed = engine state unchanged"
            // (no fences, no terminal records), and the master's one-shot cancel
            // contract (never retries, 50ms ack timeout) is what turns
            // these into failed futures master-side.
            CancelFaultKind fault = arriveCancelRpc();
            if (fault == CancelFaultKind.NO_RESPOND) {
                // cancel_no_respond: hang the RPC — no onNext / onError /
                // onCompleted ever, mirroring statusNoRespond's hang. The
                // arrival was counted; the master's cancel-ack timeout
                // fails the future with zero engine-side state change.
                return;
            }
            try {
                if (fault == CancelFaultKind.ERROR) {
                    // cancel_error: transport-layer failure (gRPC INTERNAL)
                    // — the master's future fails; no fence is installed.
                    observer.onError(io.grpc.Status.INTERNAL
                            .withDescription("mock cancel error injection")
                            .asException());
                    return;
                }
                if (fault == CancelFaultKind.UNEXPECTED_STATUS) {
                    // cancel_unexpected_status: the RPC succeeds but the ack
                    // status sits outside the cancel contract (UNSPECIFIED)
                    // — the master's response mapping must fail it, never
                    // accept it.
                    observer.onNext(EngineRpcService.CancelResponsePB.newBuilder()
                            .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_UNSPECIFIED)
                            .build());
                    observer.onCompleted();
                    return;
                }
                CancelResult result = cancelRequest(request.getRequestId());
                EngineRpcService.CancelStatusPB status;
                if (result.found()) {
                    status = EngineRpcService.CancelStatusPB.CANCEL_STATUS_ACCEPTED;
                } else if (result.alreadyFinished()) {
                    // Production-faithful branch (C++ Cancel handler):
                    // seen-but-terminal answers NOT_FOUND; the completion
                    // record stays deliverable from the retain-window
                    // backlog via GetWorkerStatus.
                    status = EngineRpcService.CancelStatusPB.CANCEL_STATUS_NOT_FOUND;
                } else {
                    // Never-seen rid: REQUEST_FENCED — cancelRequest already
                    // installed the ABSENT_FENCE record, and any racing
                    // later Enqueue of this rid is rejected with the typed
                    // 8429 before reaching the scheduler (production
                    // ABSENT_FENCE contract).
                    status = EngineRpcService.CancelStatusPB.CANCEL_STATUS_TOMBSTONED;
                }
                observer.onNext(EngineRpcService.CancelResponsePB.newBuilder()
                        .setStatus(status)
                        .build());
                observer.onCompleted();
            } catch (UnsupportedOperationException e) {
                observer.onError(io.grpc.Status.UNIMPLEMENTED
                        .withDescription(e.getMessage())
                        .asException());
            }
        }

        void checkLeakDrain(long graceWindowNanos) {
            // Shutdown drain in progress: remaining in-flight requests are being
            // cancelled deliberately, not leaking — a non-zero count here is
            // teardown noise, so never set leak_detected. Runtime leak detection
            // (below) is unchanged while the engine is live.
            if (shuttingDown) {
                return;
            }
            long timeSinceLastEnqueue = System.nanoTime() - lastEnqueueTime.get();
            if (timeSinceLastEnqueue < graceWindowNanos) {
                return;
            }
            int pending = pendingRequests.get();
            int running = runningTasks.size();
            int activeDecode = activeDecodeRequests.get();
            // ALLOCATE may legitimately outlive the drain grace while its P
            // context waits for Fetch. Only discount reservations with a live
            // owner; orphan KV must still trigger the leak detector.
            long ownedWaiting = decodeWaitingForKv.stream().filter(id -> {
                FastRpcService p = upstreamPrefillOwners.get(id);
                MockPrefillSession context = p == null ? null : p.prefillSessions.get(id);
                return context != null && !context.isClosed();
            }).count();
            if (pending != ownedWaiting || running != ownedWaiting || activeDecode != 0) {
                leakDetected.set(true);
                System.err.printf("LEAK DETECTED on engine %s (port %d): pending=%d running=%d activeDecode=%d%n",
                        roleName, grpcPort, pending, running, activeDecode);
            }
        }

        /**
         * Remove orphaned request outputs, terminal states and cancelled markers
         * for requestIds that are no longer in runningTasks. This is a safety net for
         * entries that were not cleaned up by the completion or cancel callbacks.
         */
        void periodicCleanup() {
            long now = System.nanoTime();
            requests.removeOutputsIf((id, output) -> {
                if (!(output instanceof MockResponseQueue queue)
                        || !queue.expiredTerminal(now, fetchAttachTimeoutMs)) return false;
                return !runningTasks.containsKey(id) && !prefillSessions.containsKey(id)
                        && !downstreamDecodeOwners.containsKey(id)
                        && !remotePrefillOwners.containsKey(id)
                        && !remoteDecodeLeaseOwners.containsKey(id);
            });
            // A fresh request may register after housekeeping begins. Never prune
            // unowned nonterminal state from a time-point snapshot of active IDs.
            requests.removeStatesIf((id, state) ->
                    ("completed".equals(state) || "cancelled".equals(state)
                            || "failed".equals(state) || "rejected".equals(state))
                    && requests.output(id) == null
                    && !runningTasks.containsKey(id)
                    && !prefillSessions.containsKey(id)
                    && !remotePrefillOwners.containsKey(id));
            // NOTE: cancelledRequests must NOT be pruned against runningTasks —
            // cancel() removes the runningTasks entry first, so retainAll would
            // drop the marker before the completion callback observes it and a
            // cancelled request could still be forwarded to decode / counted as
            // completed. Completion callbacks remove their own entries; TTL is
            // only a safety net for orphaned markers.
            long ttlDeadline = System.nanoTime()
                    - TimeUnit.SECONDS.toNanos(CANCELLED_MARKER_TTL_SECONDS);
            cancelledRequests.entrySet().removeIf(e -> e.getValue() < ttlDeadline);
            requestErrors.keySet().removeIf(id -> !cancelledRequests.containsKey(id)
                    && !boundaryCancellations.containsKey(id) && !runningTasks.containsKey(id)
                    && !(requests.output(id) != null));
            // engine_events.jsonl stamp safety net: a cancelled-in-queue request
            // that never reached a terminal callback leaves its arrival/start
            // stamps behind. A rid is truly gone once it is neither a running
            // task nor has a live response queue (queued prefill/decode members
            // always hold a runningTasks entry).
            eventTimings.keySet().removeIf(id ->
                    !runningTasks.containsKey(id) && requests.output(id) == null);
            // Retention is enforced at insertion; housekeeping is idempotent.
            synchronized (completionLock) {
                trimCompletionsLocked();
            }
        }

        /**
         * Graceful-stop drain for process shutdown: reject new admissions and
         * cancel every in-flight request through the existing {@link #cancel}
         * bookkeeping (verified idempotent against racing completion callbacks
         * via the runningTasks remove-guard), so pendingRequests /
         * activeDecodeRequests / activeBlockLeases / waitingPrefillRequests and
         * both pending queues net to zero without waiting for the simulated
         * completions (up to ~90s for long decodes) to fire. Called from the
         * JVM shutdown hook; completes in milliseconds.
         *
         * <p>Counters owned by RUNNING prefill batch completions
         * (activePrefillBatches / activePrefillRequests) are left to their
         * already-scheduled ms-scale callbacks — cancel() intentionally never
         * touches them, and with {@link #shuttingDown} set they can no longer
         * trip checkLeakDrain.
         */
        void drainAndShutdown() {
            shuttingDown = true;
            abortWatermarks();
            cache.capacityChanged();
            memoryReads.keySet().forEach(this::releaseMemoryRead);
            memoryWrites.keySet().forEach(this::abortMemoryCopies);
            setStopped(true);
            requests.deadlineIds().forEach(id -> cancel(id, false, false));
            for (Long requestId : List.copyOf(prefillSessions.keySet())) {
                cancel(requestId, false, false);
            } // same rejection semantics as the control-plane /stop_engine
            // Cancel sweep. A pass can promote queued decode tasks into running
            // slots (cancel's slot hand-off) and a racing cross-engine hand-off
            // may slip in before the shuttingDown guard was observed, so retry
            // a bounded number of passes until runningTasks is empty.
            for (int pass = 0; pass < 3 && !runningTasks.isEmpty(); pass++) {
                for (Long requestId : runningTaskIds()) {
                    cancel(requestId);
                }
            }
            // Direct-path parked requests were cancelled above (their
            // runningTasks entries are gone); drop the shapes themselves so a
            // post-shutdown snapshot sees an empty queue.
            synchronized (prefillQueueLock) {
                {
                    waitingPrefillRequests.addAndGet(-fifoPrefillQueue.size());
                    fifoPrefillQueue.clear();
                }
                prefillExecuting.clear(); fifoOwned.clear(); permanentPrefillFailures.clear(); prefillLoadFailures.clear(); resumedPrefillShapes.clear();
                prefillWaitTimers.values().forEach(timer -> timer.cancel(false)); prefillWaitTimers.clear();
                hostLoads.values().forEach(load -> load.done().cancel(false)); hostLoads.clear();
                fifoDrainScheduled.set(false);
                boundaryCancellations.clear();
                boundaryDownstreamCancellations.clear();
                cancelingOverlays.clear();
            }
            // Decode queue is emptied by the cancel sweep (queued tasks always
            // have a runningTasks entry); clear any dead leftovers, mirroring
            // the drain's skip-cancelled check. Dead entries carry no counters.
            synchronized (decodeQueueLock) {
                decodePendingQueue.removeIf(
                        t -> !runningTasks.containsKey(t.shape().input().getRequestId()));
            }
            for (Map.Entry<Long, FastRpcService> entry : downstreamDecodeOwners.entrySet()) {
                clearDecodeOwnership(entry.getKey(), entry.getValue());
            }
            for (Long requestId : List.copyOf(upstreamPrefillOwners.keySet())) {
                clearUpstreamOwnership(requestId);
            }
        }

        /**
         * crash_after true-crash semantics — the engine process "dies":
         * <ol>
         *   <li>{@code stopped = true} — control-plane parity flag (requests
         *       that still reach the in-process service see the stopped
         *       rejection while the port is gone).</li>
         *   <li>Bump {@link #crashEpoch} — every in-flight scheduler callback
         *       (delayed enqueue process, prefill batch start, prefill
         *       completion, decode step tick) becomes a no-op: they cannot be
         *       unscheduled, so they check the epoch they captured at trigger
         *       time and drop out when it no longer matches.</li>
         *   <li>Wipe ALL per-engine memory — running tasks, queues, response
         *       streams, per-request states, KV block pool (held + LRU +
         *       eviction history), leases, cross-engine P->D ownership
         *       (bidirectional), bounded observability histories, and the
         *       un-acked completion backlog. Recovery therefore means a
         *       reboot from zero, exactly like a real engine process
         *       restarting.</li>
         *   <li>Reset the admission gauges, observability counters and
         *       {@code enqueueCount} (a fresh process has served zero
         *       requests — a crash_after N armed before the restart must not
         *       re-fire instantly on the new incarnation).</li>
         *   <li>Kill the gRPC port — same network-level death as stop_engine
         *       (the master walks the 3-strike retire path either way); the
         *       port-kill is scheduled a beat ahead so the crash-triggering
         *       empty ack flushes first. The MockControlServer HTTP plane is
         *       a separate server and stays up, so /start_engine can rebuild
         *       the gRPC server on clean state (it also resets the fault
         *       config and enqueue count).</li>
         * </ol>
         *
         * <p>Contrast: {@code stop_engine} (MockControlServer.handleStopEngine)
         * closes the port and its P->D streams, releasing their request resources,
         * but retains the engine incarnation and cache for an in-place restart
         * — a network-level outage, not a process death. {@link #drainAndShutdown()} is the graceful process
         * EXIT path and cancels everything through the normal terminal
         * machinery instead of discarding it.
         */
        void crashNow() {
            synchronized (decodeQueueLock) {
                setStopped(true);
                crashEpoch.incrementAndGet();
                // Lease callbacks use owner identity under this same lock.
                // Drop old owners before any queued stop callback can reenter.
                remoteDecodeLeaseOwners.clear();
                remoteDecodeStops.clear();
            }
            abortWatermarks();
            processGeneration = java.util.UUID.randomUUID().toString();
            engineIncarnation = java.util.UUID.randomUUID().toString();

            for (Long requestId : List.copyOf(prefillSessions.keySet())) {
                closePrefillSession(requestId);
                releaseReservedDecode(requestId);
            }
            decodeWaitingForKv.clear();
            synchronized (completionLock) {
                runningTasks.clear();
                completions.clear();
                completionVersion.set(currentTimeUs());
                statusDeliveryFaults.rules.clear();
                statusDeliveryFaultsActive = false;
            }
            requests.clearOutputs();
            requests.clearStates();
            activeBlockLeases.clear();
            requests.closeDeadlines();
            cancelledRequests.clear();
            requestErrors.clear();
            // Queued work dies with the process: pending prefill batches,
            // direct-stream parks, decode wait queue and running streams.
            synchronized (prefillQueueLock) {
                {
                    waitingPrefillRequests.addAndGet(-fifoPrefillQueue.size());
                    fifoPrefillQueue.clear();
                }
                prefillExecuting.clear(); fifoOwned.clear(); permanentPrefillFailures.clear(); prefillLoadFailures.clear(); resumedPrefillShapes.clear();
                prefillWaitTimers.values().forEach(timer -> timer.cancel(false)); prefillWaitTimers.clear();
                hostLoads.values().forEach(load -> load.done().cancel(false)); hostLoads.clear();
                fifoDrainScheduled.set(false);
                boundaryCancellations.clear();
                boundaryDownstreamCancellations.clear();
                cancelingOverlays.clear();
            }
            synchronized (decodeQueueLock) {
                decodePendingQueue.clear();
                decodeRunning.clear();
                decodeStepScheduled = false;
                pendingStepDelayMs = 0;
            }
            memoryReads.keySet().forEach(this::releaseMemoryRead);
            memoryWrites.keySet().forEach(this::abortMemoryCopies);
            // KV memory: every held block and LRU entry is gone with the process.
            cache.clear();
            if (memoryCache != null) memoryCache.clear();
            synchronized (cacheKeyIndex) {
                cacheSyncFixture = null;
                naturalCacheSeeded = false;
                cacheKeyIndex.reset();
            }
            // Observability histories are process memory as well.
            synchronized (requestLifecycles) {
                requestLifecycles.clear();
            }
            recentlySeenRequests.clear();
            synchronized (cancelledRidHistory) {
                cancelledRidHistory.clear();
            }
            synchronized (priorityCancelledRequestIds) {
                priorityCancelledRequestIds.clear();
            }
            synchronized (fencedRequestIds) {
                fencedRequestIds.clear();
            }
            synchronized (recentPrefillTimes) {
                recentPrefillTimes.clear();
            }
            synchronized (recentDecodeTimes) {
                recentDecodeTimes.clear();
            }
            // Cross-engine P->D ownership is bidirectional — clean both sides
            // so neither this engine nor its peers keep dangling entries.
            for (Map.Entry<Long, FastRpcService> entry : downstreamDecodeOwners.entrySet()) {
                clearDecodeOwnership(entry.getKey(), entry.getValue());
            }
            for (Long requestId : List.copyOf(upstreamPrefillOwners.keySet())) {
                clearUpstreamOwnership(requestId);
            }
            downstreamDecodeOwners.clear();
            upstreamPrefillOwners.clear();
            // P-enqueue decode-KV pre-alignment (20260903): every reservation
            // THIS prefill placed on peer decode engines is returned before the
            // simulated death — a real process death would strand them until
            // the D-side TTL/restart, but the mock's crash tests keep engines
            // serving afterwards and a stranded reservation would poison
            // subsequent pool assertions.
            for (Map.Entry<Long, FastRpcService> entry : decodeReservationOwners.entrySet()) {
                entry.getValue().cancel(entry.getKey(), false, false);
            }
            decodeReservationOwners.clear();
            // Admission gauges: nothing is queued or running on a dead process.
            pendingRequests.set(0);
            waitingPrefillRequests.set(0);
            activePrefillBatches.set(0);
            activePrefillRequests.set(0);
            executingPrefillRequests.set(0);
            activeDecodeRequests.set(0);
            // Fresh process: lane time axes start from "available now" and the
            // observability counters restart from zero.
            nextPrefillAvailableNanosByDp.clear();
            enqueueCount.set(0);
            rpcEnqueueBatch.set(0);
            rpcGenerateStream.set(0);
            rpcFetchResponse.set(0);
            rpcCancel.set(0);
            busyMs.set(0);
            kvAdmissionFails.reset();
            cacheKeyHits.reset();
            cacheKeysRequested.reset();
            acceptedCount.set(0);
            completedCount.set(0);
            cancelledCount.set(0);
            prefillTps.reset(System.nanoTime());
            cacheHitMetrics.reset(System.nanoTime());
            decodeTps.reset(System.nanoTime());
            prometheusDecodeTps.reset();
            workerStatusRates.reset(System.nanoTime());
            cacheStatusRates.reset(System.nanoTime());
            prometheusWorkerStatus.reset();
            prometheusCacheStatus.reset();
            workerStatusRtUs.set(0);
            cacheStatusRtUs.set(0);
            hitTokensTotal.set(0);
            lifetimeContextComputeTokens.reset();
            lifetimeContextTokens.reset();
            lifetimeContextRequests.reset();
            lifetimeGenerateTokens.reset();
            leakDetected.set(false);
            lastEnqueueTime.set(System.nanoTime());
            // Kill the port (captured reference: a racing /start_engine that
            // already rebuilt a new server must keep it; re-shutting the dead
            // victim is a harmless no-op).
            Server victim = this.grpcServer;
            if (victim != null) {
                scheduler.schedule(victim::shutdownNow,
                        CRASH_PORT_KILL_DELAY_MS, TimeUnit.MILLISECONDS);
            }
        }

        /**
         * Shut down the dedicated response-polling executor.
         */
        void shutdown() {
            shuttingDown = true;
            batchAdmissions.values().forEach(slot -> slot.reject(batchError(14, "Prefill batch server is shutting down")));
            vitClient.close();
            abortWatermarks();
            long expires = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            requests.closeDeadlines();
            List<Runnable> close = new ArrayList<>();
            if (rpcEngine != null) close.add(rpcEngine::stop);
            remotePrefillOwners.forEach((id, owner) -> close.add(() -> {
                owner.close();
                remotePrefillOwners.remove(id, owner);
            }));
            remoteChannels.values().forEach(channel -> close.add(channel::shutdownNow));
            if (hostTransfer != null) close.add(hostTransfer::close);
            close.add(() -> {
                prepareExecutor.shutdownNow();
                // Removed queue entries still owe their joining RPC a terminal result.
                batchAdmissions.values().forEach(slot -> {
                    synchronized (slot) {
                        if (!slot.prepareStarted && slot.preparation != null) {
                            slot.preparing = false; slot.preparation.complete(null);
                        }
                    }
                });
            });
            close.add(responseExecutor::shutdownNow);
            remoteChannels.forEach((target, channel) -> close.add(() -> {
                try {
                    if (!channel.awaitTermination(Math.max(0, expires-System.nanoTime()), TimeUnit.NANOSECONDS))
                        throw new IllegalStateException("remote channel did not terminate: " + target);
                    remoteChannels.remove(target, channel);
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted awaiting remote channel", error);
                }
            }));
            close.add(() -> {
                try {
                    if (!responseExecutor.awaitTermination(Math.max(0, expires-System.nanoTime()), TimeUnit.NANOSECONDS))
                        throw new IllegalStateException("response executor did not terminate");
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted awaiting response executor", error);
                }
            });
            if (rpcEngine != null) close.add(() -> rpcEngine.awaitStopped(expires));
            ShutdownTasks.runAll(close);
        }

        // ──────────── Getters and setters for MockControlServer ────────────

        FaultInjectionConfig getFaultConfig() { return faultConfig; }
        void setFaultConfig(FaultInjectionConfig config) { this.faultConfig = config; }
        void clearFaultConfig() {
            synchronized (completionLock) {
                this.faultConfig = FaultInjectionConfig.builder().build();
                statusDeliveryFaults.clear(this::publishCompletionNow);
                statusDeliveryFaultsActive = false;
            }
        }
        void configureStatusDelivery(long rid, int missingRounds, long delayMs) {
            synchronized (completionLock) {
                statusDeliveryFaults.configure(rid, missingRounds, delayMs);
                statusDeliveryFaultsActive = true;
            }
        }
        void resetEnqueueCount() { this.enqueueCount.set(0); }
        void setStopped(boolean s) {
            synchronized (decodeQueueLock) {
                this.stopped = s;
                if (s && roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                    remoteDecodeStops.values().forEach(Runnable::run);
                    // The mock has no P->D socket: explicitly model the broken
                    // RemoteGenerate stream before teardown discards its owners.
                    // Normal completion and downstream cancellation claim the
                    // same ownership under this lock, so only the winner emits.
                    for (Long requestId : List.copyOf(upstreamPrefillOwners.keySet())) {
                        FastRpcService prefill = upstreamPrefillOwners.get(requestId);
                        if (prefill != null) {
                            deliverLinkBreak(requestId, prefill);
                            // Closing the stream also releases a prepared D
                            // allocation: otherwise closing P's deferred context
                            // strands waiting_for_kv until its lease timeout.
                            cancel(requestId, false, false);
                            clearUpstreamOwnership(requestId);
                        }
                    }
                }
            }
        }

        private void deliverLinkBreak(long requestId, FastRpcService prefill) {
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    prefill.requests.output(requestId);
            if (queue != null) {
                queue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                        .setRequestId(requestId)
                        .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                .setErrorCodeValue(8209) // REMOTE_GENERATE_FAILED (C++ ErrorCode)
                                .setErrorMessage("P->D link closed: decode engine " + grpcPort + " stopped"))
                        .build());
            }
        }
        void setGrpcServer(Server server) { this.grpcServer = server; }
        boolean isStopped() { return stopped; }
        Map<String, String> whaleMetricTags() {
            Map<String, String> tags = WhaleMockMonitor.engineTags(
                    System.getenv(), whaleBundle ? whalePodIp : host, roleType.name());
            // The reachable engine address and the monitoring identity are
            // separate in both Whale layouts. Bundle engines share one Pod IP;
            // standalone engines have distinct Pod IPs. Never overwrite the
            // physical host_ip/container_ip tags with a virtual loopback.
            tags.put("engine_port", Integer.toString(grpcPort));
            tags.put("engine_ip", host);
            tags.putAll(Map.of("engine", engineName, "role", roleType.name(),
                    "generation", processGeneration, "backend", "mock"));
            return tags;
        }

        Map<String, Number> whaleMetrics() {
            var metrics = new HashMap<String, Number>(Map.ofEntries(
                    Map.entry("mock_context_compute_tokens_total", lifetimeContextComputeTokens.sum()),
                    Map.entry("mock_context_tokens_total", lifetimeContextTokens.sum()),
                    Map.entry("mock_hit_tokens_total", hitTokensTotal.get()),
                    Map.entry("mock_context_requests_total", lifetimeContextRequests.sum()),
                    Map.entry("mock_generate_tokens_total", lifetimeGenerateTokens.sum()),
                    Map.entry("mock_decode_step_tokens_total", decodeTps.snapshot().tokens()),
                    Map.entry("mock_kv_total_tokens", getTotalKvTokens()),
                    Map.entry("mock_kv_available_tokens", getAvailableKvTokens()),
                    Map.entry("mock_kv_occupied_tokens", occupiedKvTokens()),
                    Map.entry("mock_prefill_admitted_requests", activePrefillRequests.get()),
                    Map.entry("mock_decode_waiting_requests", decodePendingQueueSize() + decodeWaitingForKv.size()),
                    Map.entry("mock_decode_reserved_requests", decodeWaitingForKv.size()),
                    Map.entry("mock_engine_completed_total", completedCount.get()),
                    Map.entry("mock_cancelled_requests_total", cancelledCount.get()),
                    Map.entry("mock_engine_cache_evictions_total", cache.evictions()),
                    Map.entry("mock_engine_cache_key_hits_total", cacheKeyHits.sum()),
                    Map.entry("mock_engine_cache_keys_requested_total", cacheKeysRequested.sum()),
                    Map.entry("rtp_llm_running_stream_size", executingPrefillRequests.get() + activeDecodeRequests.get()),
                    Map.entry("rtp_llm_wait_stream_size", schedulerWaitingStreamSize()),
                    // Real scheduler metrics exclude the pre-GENERATE D lease. The
                    // mock LOAD exchange is immediate, not a scheduler cache-load queue.
                    Map.entry("rtp_llm_remote_running_stream_size", 0),
                    Map.entry("rtp_llm_loading_cache_stream_size", schedulerLoadingStreamSize()),
                    Map.entry("rtp_llm_context_batch_size", activePrefillRequests.get()),
                    Map.entry("rtp_llm_generate_batch_size", activeDecodeRequests.get()),
                    Map.entry("rtp_llm_kv_cache_item_num", cache.lruKeyBlocks()),
                    Map.entry("rtp_llm_kv_cache_left_seq", (long) cache.availableBlocks() * seqSizePerBlock),
                    Map.entry("rtp_llm_kv_cache_free_blocks", cache.freeBlocks()),
                    Map.entry("rtp_llm_kv_cache_available_blocks", cache.availableBlocks()),
                    Map.entry("rtp_llm_kv_cache_total_blocks", cache.totalBlocks()),
                    Map.entry("rtp_llm_kv_cache_used_ratio", cache.totalBlocks() == 0 ? 0.0
                            : 100.0 * (cache.totalBlocks() - cache.freeBlocks()) / cache.totalBlocks())));
            if (memoryCache != null) {
                metrics.put("mock_memory_cache_total_blocks", memoryCache.capacity());
                metrics.put("mock_memory_cache_allocated_blocks", memoryCache.size() + memoryCache.pendingBlocks());
                // Cached unpinned entries are reclaimable; in-flight copies are not.
                metrics.put("mock_memory_cache_available_blocks", memoryCache.availableBlocks());
                metrics.put("mock_memory_cache_unavailable_ratio", 100.0 * (memoryCache.capacity() - memoryCache.availableBlocks()) / memoryCache.capacity());
                metrics.put("mock_memory_cache_occupancy_ratio", (double) memoryCache.size() / memoryCache.capacity());
                metrics.put("mock_memory_cache_total_tokens", (long) memoryCache.capacity() * seqSizePerBlock);
                metrics.put("mock_memory_cache_evicted_blocks_total", memoryCache.evictions());
                metrics.put("mock_memory_cache_read_blocks_total", memoryReadBlocks.sum());
                metrics.put("mock_memory_cache_pinned_blocks", memoryCache.pinnedBlocks());
                metrics.put("mock_memory_cache_pending_write_blocks", memoryCache.pendingBlocks());
                metrics.put("mock_memory_cache_write_rejected_total", memoryCache.writeRejected());
            }
            return metrics;
        }
        int getGrpcPort() { return grpcPort; }
        int getDownstreamOwnershipCount() { return downstreamDecodeOwners.size(); }
        int getUpstreamOwnershipCount() { return upstreamPrefillOwners.size(); }
        boolean hasDownstreamOwnership(long requestId) { return downstreamDecodeOwners.containsKey(requestId); }
        boolean hasUpstreamOwnership(long requestId) { return upstreamPrefillOwners.containsKey(requestId); }
        String getRoleName() { return roleName; }
        String getEngineName() { return engineName; }
        String getHost() { return host; }
        long getTotalKvTokens() { return totalKvTokens; }

        MockPerformanceModel getPerformance() { return performance; }
        int getRunningCount() { return runningTasks.size(); }
        int getWaitingCount() { return waitingPrefillRequests.get(); }
        int getActivePrefillRequestCount() { return activePrefillRequests.get(); }
        long getAcceptedCount() { return acceptedCount.get(); }
        long getCompletedCount() { return completedCount.get(); }
        long getCancelledCount() { return cancelledCount.get(); }
        /** Master-facing used tokens (occupied + pressure, clamped to total) —
         * the pool-derived caliber behind "active" everywhere. */
        long getActiveKvTokens() { return usedKvTokens(); }
        /** Pool availability (free + pure-LRU) clamped to total, minus pressure. */
        long getAvailableKvTokens() { return availableKvTokens(); }
        /** Total pool blocks (ceil(totalKvTokens/spb) or explicit override). */
        Map<String, Object> effectiveConfiguration() {
            Map<String, Object> state = new LinkedHashMap<>();
            state.put("engine", engineName);
            state.put("config", effectiveConfig == null ? Map.of() : effectiveConfig);
            state.put("calibration", performance.calibrationState());
            state.put("runtime", Map.of("auto_fetch", autoFetch, "native_token_keys", performance.nativeTokenCacheKeys));
            state.put("limits", List.of("single KV pool", "metadata-only KV transfer; host load latency is calibrated"));
            return state;
        }

        int getCacheBlocks() { return cache.totalBlocks(); }
        void setCacheRetentionBlocks(int limit) {
            cache.setRetentionBlocks(limit);
        }
        /** spb — the pool's token<->block conversion factor (reported as block_size). */
        int getSeqSizePerBlock() { return seqSizePerBlock; }
        boolean isLeakDetected() { return leakDetected.get(); }
        boolean isShuttingDown() { return shuttingDown; }
        int getActiveDecodeCount() { return activeDecodeRequests.get(); }
        int getActivePrefillBatchCount() { return activePrefillBatches.get(); }
        int getDecodePendingQueueDepth() { return decodePendingQueueSize(); }
        int getPrefillPendingQueueDepth() { return prefillPendingQueueSize(); }

        /**
         * Whether this engine still has ANY work that must finish before a
         * GRACEFUL removal may tear the gRPC server down: running tasks, both
         * prefill wait queues (master-composed batches + direct-path parks),
         * the decode pending queue, running batch/decode counters, and
         * cross-engine P→D ownership in BOTH directions (a prefill victim must
         * outlive requests it already handed to a decode engine; a decode
         * victim must outlive streams an upstream prefill parked on it).
         *
         * <p>Deliberately NOT included: the un-acked completion backlog
         * ({@code completions}) — its consumer is the master's status poll, and
         * once the master has dropped the endpoint's route it will never come;
         * waiting on it would only stretch every graceful drain to its deadline.
         * Post-completion visibility is the master's scale-in contract, not the
         * engine's.
         */
        Map<String, Integer> inflightWorkSnapshot() {
            Map<String, Integer> work = new LinkedHashMap<>();
            work.put("running_tasks", runningTasks.size());
            work.put("waiting_prefill", waitingPrefillRequests.get());
            work.put("active_prefill_batches", activePrefillBatches.get());
            work.put("active_prefill_requests", activePrefillRequests.get());
            work.put("active_decode_requests", activeDecodeRequests.get());
            work.put("prefill_pending", prefillPendingQueueSize());
            work.put("decode_pending", decodePendingQueueSize());
            work.put("direct_prefill", directPrefillQueueSize());
            work.put("downstream_decode_owners", downstreamDecodeOwners.size());
            work.put("remote_decode_owners", remoteDecodeLeaseOwners.size());
            work.put("remote_prefill_owners", remotePrefillOwners.size());
            work.put("upstream_prefill_owners", upstreamPrefillOwners.size());
            return work;
        }

        boolean hasInflightWork() {
            return !runningTasks.isEmpty()
                    || waitingPrefillRequests.get() != 0
                    || activePrefillBatches.get() != 0
                    || activePrefillRequests.get() != 0
                    || activeDecodeRequests.get() != 0
                    || prefillPendingQueueSize() != 0
                    || decodePendingQueueSize() != 0
                    || directPrefillQueueSize() != 0
                    || !downstreamDecodeOwners.isEmpty()
                    || !remoteDecodeLeaseOwners.isEmpty()
                    || !remotePrefillOwners.isEmpty()
                    || !upstreamPrefillOwners.isEmpty();
        }

        /** Snapshot size of the direct (NON_BATCH) prefill park queue. */
        private int directPrefillQueueSize() {
            synchronized (prefillQueueLock) {
                return fifoPrefillQueue.ordinarySize();
            }
        }
        Map<Long, String> getRequestStates() { return requests.states(); }

        /**
         * Python /set_kv_pressure uses ABSOLUTE active_kv_tokens semantics
         * (Python semantics: state._active_kv_tokens = value). The Java engine models
         * pressure as an additive fault-config term on top of live decode tokens, so
         * convert the requested absolute value into the equivalent additive pressure.
         */
        void setAbsoluteActiveKvTokens(long absoluteTokens) {
            // occupied (not used) so the injected pressure lands on TOP of the
            // live pool occupancy — repeated absolute injections stay additive
            // relative to what is really running.
            long pressure = Math.max(0, absoluteTokens - occupiedKvTokens());
            faultConfig = faultConfig.toBuilder().kvPressureTokens(pressure).build();
        }

        private AtomicLong pickPrefillLane(int dpRank) {
            return nextPrefillAvailableNanosByDp.computeIfAbsent(dpRank, ignored -> new AtomicLong());
        }

        private void addCancelledRid(long requestId) {
            synchronized (cancelledRidHistory) {
                if (cancelledRidHistory.add(requestId) && cancelledRidHistory.size() > CANCELLED_RID_CAP) {
                    cancelledRidHistory.remove(cancelledRidHistory.iterator().next());
                }
            }
        }

        private void addPriorityCancelledRequestId(long requestId) {
            priorityCancelledRequestIds.add(requestId);
        }

        private boolean hasPriorityCancelledRequestId(long requestId) {
            return priorityCancelledRequestIds.contains(requestId);
        }

        private void addFencedRequestId(long requestId) {
            fencedRequestIds.add(requestId);
        }

        private boolean hasFencedRequestId(long requestId) {
            return fencedRequestIds.contains(requestId);
        }

        /** Typed internal row; only the diagnostic boundary emits the legacy JSON keys.
         * All access is guarded by requestLifecycles. */
        private static final class RequestLifecycle {
            final long requestId;
            final long batchId;
            final String method;
            final long arrivedMs = System.currentTimeMillis();
            long runningMs = arrivedMs;
            long endMs;
            String endState = "running";

            RequestLifecycle(long requestId, long batchId, String method) {
                this.requestId = requestId;
                this.batchId = batchId;
                this.method = method;
            }

            boolean running() { return "running".equals(endState); }

            Map<String, Object> snapshot() {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("rid", requestId);
                row.put("method", method);
                row.put("batch_id", batchId);
                row.put("arrived_ms", arrivedMs);
                row.put("running_ms", runningMs);
                row.put("end_ms", endMs);
                row.put("end_state", endState);
                return row;
            }
        }

        private void recordLifecycleStart(long requestId, long batchId, String method) {
            RequestLifecycle lifecycle = new RequestLifecycle(requestId, batchId, method);
            synchronized (requestLifecycles) {
                requestLifecycles.compute(requestId, (id, previous) ->
                        previous == null || !previous.running()
                                ? lifecycle : previous);
                // Bounded like Python _prune_lifecycle: evict the oldest entries once over cap.
                while (requestLifecycles.size() > LIFECYCLE_CAP) {
                    requestLifecycles.remove(requestLifecycles.keySet().iterator().next());
                }
            }
        }

        private void recordLifecycleEnd(long requestId, boolean cancelled) {
            recordLifecycleEnd(requestId, cancelled ? "cancelled" : "completed");
        }

        /**
         * String-state variant (20260903): decode KV terminal failures record
         * end_state "failed" — distinct from both "completed" and "cancelled"
         * in the /requests view; the existing boolean callers are unchanged.
         */
        private void recordLifecycleEnd(long requestId, String endState) {
            recentlySeenRequests.add(requestId);
            synchronized (requestLifecycles) {
                RequestLifecycle lifecycle = requestLifecycles.get(requestId);
                if (lifecycle != null && lifecycle.running()) {
                    lifecycle.endMs = System.currentTimeMillis();
                    lifecycle.endState = endState;
                }
            }
        }

        /**
         * Mark the epoch-ms moment a decode request actually entered RUNNING
         * (promoted out of the waiting queue into a running slot). The lifecycle
         * row is created at enqueue with running_ms == arrived_ms; this rewrites
         * running_ms to the true admission time so waiting dwell is visible in
         * the final_snapshot. Lock order decodeQueueLock -> requestLifecycles
         * matches recordLifecycleStart (also called under decodeQueueLock on
         * the queued path).
         */
        private void recordLifecycleRunning(long requestId) {
            synchronized (requestLifecycles) {
                RequestLifecycle lifecycle = requestLifecycles.get(requestId);
                if (lifecycle != null && lifecycle.running()) lifecycle.runningMs = System.currentTimeMillis();
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

        /** Stamp the engine-side arrival epoch-ms for engine_events.jsonl (first arrival wins). */
        private void recordEventArrival(long requestId) {
            if (engineEventLog == null && eventMetricReporter == null) {
                return;
            }
            long now = System.currentTimeMillis();
            eventTimings.compute(requestId, (id, timing) -> timing == null ? new EventTiming(now, 0L)
                    : timing.arrivalMs == 0L ? new EventTiming(now, timing.startMs) : timing);
        }

        /** Stamp the execution-start epoch-ms (prefill batch start / decode slot admission). */
        private void recordEventStart(long requestId) {
            if (engineEventLog == null) {
                return;
            }
            long now = System.currentTimeMillis();
            eventTimings.compute(requestId, (id, timing) -> timing == null ? new EventTiming(0L, now)
                    : timing.startMs == 0L ? new EventTiming(timing.arrivalMs, now) : timing);
        }

        private EventTiming takeEventTiming(long requestId) {
            EventTiming timing = eventTimings.remove(requestId);
            return timing == null ? EventTiming.EMPTY : timing;
        }

        private long eventArrivalMs(long requestId) {
            return eventTimings.getOrDefault(requestId, EventTiming.EMPTY).arrivalMs;
        }

        // Production trans_input uses epoch microseconds; the Java replay
        // client and older case fixtures use epoch milliseconds. A missing or
        // implausible stamp must not become a fabricated end-to-end latency.
        static long requestStartEpochMs(EngineRpcService.GenerateInputPB input, long doneMs) {
            long raw = input.getStartTime();
            long startMs = raw >= 100_000_000_000_000L ? raw / 1000L : raw;
            return startMs >= 1_000_000_000_000L && startMs <= doneMs
                    && doneMs - startMs <= TimeUnit.HOURS.toMillis(1) ? startMs : 0L;
        }

        /**
         * engine_events.jsonl terminal row for one prefill batch member
         * (normal or cancelled) — the structured replacement of the former
         * mock_prefill_done stdout line. exec_ms is the BATCH execution
         * duration (prefill runs whole batches, every member logs the same
         * value); ttft_ms is the engine-resident time arrival → prefill
         * terminal (includes engine-side queueing — the engine-internal TTFT
         * caliber); kv_used_tokens reads the still-live block lease (blocks ×
         * spb). Missing stamps (row written for a request whose arrival was
         * never stamped — e.g. log injected mid-run) serialize as 0.
         */
        private void writePrefillDoneEvent(MockPerformanceModel.RequestShape shape,
                                           long requestId, long batchId, long doneTsMs,
                                           long executionMs, int batchSize, boolean cancelled) {
            EngineEventLog log = engineEventLog;
            EventTiming timing = takeEventTiming(requestId);
            long arrivalMs = timing.arrivalMs;
            long startMs = timing.startMs;
            if (log == null) {
                return;
            }
            ObjectNode row = OBJECT_MAPPER.createObjectNode();
            row.put("event", "prefill_done");
            row.put("rid", requestId);
            row.put("engine_name", engineName);
            row.put("engine_incarnation", engineIncarnation);
            row.put("engine_address", host + ":" + grpcPort);
            row.put("batch_id", batchId);
            row.put("engine_arrival_ms", arrivalMs);
            row.put("prefill_start_ms", startMs);
            row.put("prefill_done_ms", doneTsMs);
            row.put("ttft_ms", arrivalMs > 0 ? Math.max(0L, doneTsMs - arrivalMs) : 0L);
            row.put("exec_ms", executionMs);
            row.put("batch_size", batchSize);
            row.put("input_len", shape.inputLen());
            row.put("cache_hit_tokens", shape.hitTokens());
            long memoryHitTokens = memoryCache == null ? 0L
                    : Math.min(shape.hitTokens(), (long) shape.memoryHitBlocks() * seqSizePerBlock);
            row.put("cache_memory_hit_tokens", memoryHitTokens);
            row.put("cache_device_hit_tokens", shape.hitTokens() - memoryHitTokens);
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            row.put("kv_used_tokens",
                    lease != null ? (long) lease.totalBlocks() * seqSizePerBlock : 0L);
            row.put("cancelled", cancelled);
            log.write(row);
        }

        /**
         * engine_events.jsonl terminal row for one decode stream — the
         * structured replacement of the former mock_decode_done stdout line.
         * exec_ms is the decode_done_ms − decode_start_ms WALL-SPAN (20260903
         * caliber change, symmetric with the prefill batch's done−start
         * semantics — the former summed-booked-step-durations stream caliber
         * retired; under the step-boundary admission model wall-span ≈ Σstep
         * durations, the only error being the ≤1-step admission quantization
         * plus scheduler jitter). batch_size is the terminal-step running batch
         * (claimed under decodeQueueLock, includes this stream); kv_used_tokens
         * reads the still-live block lease (the lease hands over to the LRU
         * only AFTER this row, on the normal path).
         */
        private void writeDecodeDoneEvent(DecodeStream stream, boolean cancelled) {
            writeDecodeDoneEventCore(stream.shape, stream.batchId,
                    stream.terminalBatchSize, cancelled, 0L);
        }

        /**
         * Core decode_done row writer. {@code errorCode} != 0 marks a KV
         * terminal-failure row (production-aligned decode LACK_MEM, 20260903):
         * the row carries an extra error_code key so downstream joins
         * (aggregate_canvas_run.py) can exclude failed requests from latency
         * calibers exactly like cancelled ones; normal completion rows keep
         * the 12-key schema unchanged.
         */
        private void writeDecodeDoneEventCore(
                MockPerformanceModel.RequestShape shape, long batchId,
                int terminalBatchSize, boolean cancelled, long errorCode) {
            long requestId = shape.input().getRequestId();
            EngineEventLog log = engineEventLog;
            EventTiming timing = takeEventTiming(requestId);
            long arrivalMs = timing.arrivalMs;
            long startMs = timing.startMs;
            if (log == null) {
                return;
            }
            long doneMs = System.currentTimeMillis();
            ObjectNode row = OBJECT_MAPPER.createObjectNode();
            row.put("event", "decode_done");
            row.put("rid", requestId);
            row.put("engine_name", engineName);
            row.put("engine_incarnation", engineIncarnation);
            row.put("engine_address", host + ":" + grpcPort);
            row.put("batch_id", batchId);
            row.put("engine_arrival_ms", arrivalMs);
            row.put("decode_start_ms", startMs);
            row.put("decode_done_ms", doneMs);
            // Missing start stamps (log injected mid-run) serialize as 0,
            // mirroring the prefill ttft_ms guard — never a fabricated span.
            row.put("exec_ms", startMs > 0 ? Math.max(0L, doneMs - startMs) : 0L);
            row.put("batch_size", terminalBatchSize);
            row.put("output_len", shape.outputLen());
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            row.put("kv_used_tokens",
                    lease != null ? (long) lease.totalBlocks() * seqSizePerBlock : 0L);
            row.put("cancelled", cancelled);
            if (errorCode != 0L) {
                row.put("error_code", errorCode);
            }
            log.write(row);
        }

        /** Snapshot of the bounded request lifecycle map with string keys (Python /requests). */
        Map<String, Map<String, Object>> getRequestLifecycleSnapshot() {
            Map<String, Map<String, Object>> result = new LinkedHashMap<>();
            synchronized (requestLifecycles) {
                requestLifecycles.forEach((id, lifecycle) -> result.put(String.valueOf(id), lifecycle.snapshot()));
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

        /** Settle the HTTP reader once per scrape; snapshots themselves are read-only. */
        void drainTpsWindows() {
            prometheusCacheStatus.sample(cacheStatusRates.snapshot(), System.nanoTime());
            prometheusWorkerStatus.sample(workerStatusRates.snapshot(), System.nanoTime());
            prometheusDecodeTps.sample(decodeTps.snapshot(), System.nanoTime());
            prometheusPrefillTps.sample(prefillTps.snapshot(), System.nanoTime());
        }

        CounterRateMetrics.Snapshot decodeTpsSnapshot() { return decodeTps.snapshot(); }

        PrefillTpsMetrics.Snapshot prefillTpsSnapshot() { return prefillTps.snapshot(); }

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
        Map<String, Object> getTopologySnapshot() {
            return Map.of("name", engineName, "role", roleName.toLowerCase(),
                    "grpc_addr", host + ":" + grpcPort,
                    "http_addr", host + ":" + (grpcPort - 1),
                    "engine_incarnation", engineIncarnation);
        }

        Map<String, Object> getSnapshot() {
            return getSnapshot(true);
        }

        Map<String, Object> getMetricsSnapshot() {
            return getSnapshot(false);
        }

        private Map<String, Object> getSnapshot(boolean includeDetails) {
            Map<String, Object> snap = new LinkedHashMap<>();
            // Capacity model v2: one pool-derived caliber everywhere —
            // active = occupied + pressure (clamped), available = pool
            // availability (free + pure-LRU, LRU counts as available) clamped
            // to total minus pressure. cache_keys growth shows up as occupancy
            // only while requests are RUNNING; parked LRU keys restore
            // availability, matching the production master's view.
            long effectiveActiveKv = usedKvTokens();
            snap.put("name", engineName);
            snap.put("engine_incarnation", engineIncarnation);
            snap.put("role", roleName.toLowerCase());
            snap.put("grpc_addr", host + ":" + grpcPort);
            snap.put("http_addr", host + ":" + (grpcPort - 1));
            snap.put("running", runningTasks.size());
            // Lifecycle inventory above includes queued/allocated requests. The real
            // scheduler gauge counts only executing streams, as in whaleMetrics().
            snap.put("scheduler_running", executingPrefillRequests.get() + activeDecodeRequests.get());
            // Python: max(_injected_queue_depth, _prefill_waiting). Java has no fake injected
            // depth (see /set_queue_depth note), so this is the real waiting count.
            // For decode engines, report the decode pending queue depth (consistent
            // with scheduler metrics) instead of waitingPrefillRequests
            // which is always 0 for decode engines.
            snap.put("waiting", schedulerWaitingStreamSize());
            snap.put("loading_cache", schedulerLoadingStreamSize());
            if (hostTransfer != null) snap.put("host_queues", hostTransfer.snapshot());
            snap.put("prepare_running", prepareExecutor.getActiveCount());
            snap.put("prepare_waiting", prepareExecutor.getQueue().size());
            // Group queue depth; "waiting" above counts individual requests.
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                snap.put("prefill_waiting_batches", prefillPendingQueueSize());
            }
            snap.put("accepted", acceptedCount.get());
            snap.put("auto_fetch", autoFetch);
            snap.put("active_decode_requests", activeDecodeRequests.get());
            snap.put("response_buffers", requests.outputCount());
            snap.put("prefill_contexts", prefillSessions.size());
            snap.put("decode_waiting_for_kv", decodeWaitingForKv.size());
            snap.put("fetch_attach_expirations", fetchAttachExpirations.get());
            snap.put("completed", completedCount.get());
            snap.put("cache_keys", cache.lruKeyBlocks());
            snap.put("cache_evictions", cache.evictions());
            // Full per-engine key list (flexlb_test_framework KV cases' per-engine key-set
            // exposure). Debug endpoint — the whole list is exposed by design
            // (default 6000-block caches included); sorted for determinism.
            if (includeDetails) {
                List<Long> cacheKeySet = new ArrayList<>(cache.snapshotKeys());
                cacheKeySet.sort(Long::compareTo);
                snap.put("cache_key_set", cacheKeySet);
            }
            snap.put("active_kv_tokens", effectiveActiveKv);
            snap.put("available_kv_tokens", availableKvTokens());
            // Pool observability (per-engine series, /metrics passthrough):
            // spb and the block-level split of the same capacity model.
            snap.put("total_kv_tokens", totalKvTokens);
            snap.put("block_size", seqSizePerBlock);
            snap.put("cache_blocks", cache.totalBlocks());
            snap.put("gpu_prefix_tree_enabled", cache.prefixTreeEnabled());
            snap.put("cache_retention_blocks", cache.retentionBlocks());
            snap.put("cache_retention_evictions", cache.retentionEvictions());
            snap.put("available_blocks", cache.availableBlocks());
            snap.put("held_blocks", cache.heldBlocks());
            snap.put("referenced_blocks", cache.referencedKeyBlocks());
            snap.put("kv_admission_fails", kvAdmissionFails.sum());
            snap.put("lack_mem_rejects", prefillLackMemRejects.sum());
            snap.put("decode_reuse_blocks", decodeReuseBlocks.sum());
            snap.put("cache_key_hits", cacheKeyHits.sum());
            snap.put("cache_keys_requested", cacheKeysRequested.sum());
            snap.put("prefill_async_partial_fails", prefillAsyncPartialFails.sum());
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
            if (includeDetails) {
                List<Long> cancelledRids;
                synchronized (cancelledRidHistory) {
                    cancelledRids = new ArrayList<>(cancelledRidHistory);
                }
                cancelledRids.sort(Long::compareTo);
                snap.put("cancelled_rids", cancelledRids);
                snap.put("request_lifecycle", getRequestLifecycleSnapshot());
            }
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
            snap.putAll(admissionSnapshot());
            // Java-only fields retained (do not rename Python fields above).
            snap.put("port", grpcPort);
            // Cumulative per-engine busy time (ms): prefill batches (resp. decode
            // requests) executed by this engine — see busyMs field comment.
            snap.put("busy_ms", busyMs.get());
            // Per-engine prefill batch composition (20260903 #8): the
            // REGROUPED execution batches — prefill_batches counts batches
            // executed, prefill_batch_requests the requests they carried
            // (avg size = requests/batches), max_prefill_batch_size the
            // largest. Batch-caliber consumers read these instead of
            // inferring composition from per-request counters.
            snap.put("prefill_batches", prefillBatchesExecuted.sum());
            snap.put("prefill_batch_requests", prefillBatchRequestsExecuted.sum());
            snap.put("max_prefill_batch_size", maxPrefillBatchSizeExecuted.get());
            List<Long> batchSizeBuckets = new ArrayList<>(prefillBatchSizeCounts.length());
            long cumulativeBatches = 0;
            for (int i = 0; i < prefillBatchSizeCounts.length(); i++) {
                cumulativeBatches += prefillBatchSizeCounts.get(i);
                batchSizeBuckets.add(cumulativeBatches);
            }
            snap.put("prefill_batch_size_buckets", batchSizeBuckets);
            // Business completion totals stay cumulative. TPS is an execution
            // window with its own atomic numerator/time ledger (see METRICS.md).
            snap.put("context_compute_tokens_total", lifetimeContextComputeTokens.sum());
            snap.put("context_tokens_total", lifetimeContextTokens.sum());
            snap.put("context_requests_total", lifetimeContextRequests.sum());
            snap.put("generate_tokens_total", lifetimeGenerateTokens.sum());
            snap.put("decode_step_tokens_total", decodeTps.snapshot().tokens());
            snap.putAll(prometheusPrefillTps.last());
            snap.put("generate_tps", prometheusDecodeTps.lastTokens());
            snap.put("cache_status_qps", prometheusCacheStatus.last());
            snap.put("cache_status_rt_us", cacheStatusRtUs.get());
            snap.put("cache_status_rpcs_total", cacheStatusRates.snapshot().tokens());
            snap.put("worker_status_qps", prometheusWorkerStatus.last());
            snap.put("worker_status_rt_us", workerStatusRtUs.get());
            snap.put("worker_status_rpcs_total", workerStatusRates.snapshot().tokens());
            if (rpcEngine != null) snap.put("rpc_runtime", rpcEngine.snapshot());
            snap.put("decode_wall_tps", prometheusDecodeTps.last());
            snap.put("hit_tokens_total", hitTokensTotal.get());
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

    }

    /**
     * Result of {@link FastRpcService#cancelRequest(long)}, mapped by every
     * caller to the production C++ Prefill cancel contract: {@code found}
     * (live, or an idempotent ACTIVE_CANCEL terminal record retry) → ACCEPTED;
     * {@code alreadyFinished} (seen-but-terminal: completed or previously
     * cancelled) → NOT_FOUND (the completion record stays deliverable from
     * the retain window); unknown (never-seen rid, ABSENT_FENCE record
     * installed by cancelRequest) → REQUEST_FENCED, and later Enqueues of that
     * rid are rejected with the typed 8429.
     */
    record CancelResult(boolean found, EngineRpcService.TaskPhase phase, boolean alreadyFinished) {
    }

    /**
     * Armed cancel-RPC fault injection kind returned by
     * {@link FastRpcService#arriveCancelRpc} — the RPC-layer failure family
     * ({@code cancel_no_respond} / {@code cancel_error} /
     * {@code cancel_unexpected_status}) simulated before the engine cancel
     * state machine is touched.
     */
    enum CancelFaultKind {
        /** cancel_no_respond: the RPC hangs — no response ever (mirrors statusNoRespond). */
        NO_RESPOND,
        /** cancel_error: transport-layer failure (gRPC INTERNAL / HTTP 500). */
        ERROR,
        /** cancel_unexpected_status: ack status outside the contract (UNSPECIFIED). */
        UNEXPECTED_STATUS
    }

    static final class ClusterStats {
        private final LongAdder enqueueRpcs = new LongAdder();
        private final LongAdder enqueuedRequests = new LongAdder();
        private final LongAdder statusRpcs = new LongAdder();
        private final LongAdder cacheRpcs = new LongAdder();
        private final LongAdder generateStreamRpcs = new LongAdder();
        private final LongAdder fetchResponseRpcs = new LongAdder();
        private final LongAdder cancelRpcs = new LongAdder();
        // Leak-attribution census (Jack A1/A2): how incoming cancel RPCs
        // distribute across tracked / already-finished / unknown / terminal record.
        // B1 on the master side should stay close to zero once the cancel
        // setBatchId fix lands; a sustained B1 ~= tracked count means typed
        // CANCELLED terminals are still being dropped in reconcile.
        final LongAdder cancelCensusTracked = new LongAdder();
        final LongAdder cancelCensusAlreadyFinished = new LongAdder();
        final LongAdder cancelCensusUnknown = new LongAdder();
        final LongAdder cancelCensusAlreadyCancelled = new LongAdder();
        // Cancel-RPC fault-injection census: arrivals short-circuited by an
        // armed cancel_no_respond / cancel_error / cancel_unexpected_status
        // injection (arriveCancelRpc). Separate from the state-machine
        // branches so the census family stays an exhaustive split of cancel
        // RPC arrivals — an injected cancel is neither tracked, finished,
        // unknown nor a terminal record: the engine state never saw it.
        final LongAdder cancelCensusInjected = new LongAdder();
        // Autonomous client-gone cancellations (broken GenerateStream /
        // FetchResponse stream): how many in-flight requests the engine
        // cleaned up by itself because the client stream died, split from the
        // explicit-Cancel census above so leak attribution stays clean.
        final LongAdder cancelCensusClientGone = new LongAdder();
        // Epoch-aligned event counter: cumulative decode RUNNING admissions
        // (immediate admission + waiting-queue top-up). Differencing against
        // the stats line's ts_epoch_ms gives per-second "entered running" QPS;
        // paired with decode_done differencing it separates admission rate
        // from completion rate on the same epoch axis.
        final LongAdder decodeAdmitted = new LongAdder();
        // Package-private: DirectPrefillCoalescingTest asserts the coalesced
        // batch structure through these counters (java_mock_stats same source).
        final LongAdder prefillBatches = new LongAdder();
        final LongAdder prefillBatchRequests = new LongAdder();
        private final LongAdder prefillBatchExecutionMs = new LongAdder();
        final AtomicInteger maxPrefillBatchSize = new AtomicInteger();
        private final AtomicLong maxPrefillBatchExecutionMs = new AtomicLong();

        void recordPrefillBatch(int batchSize, long executionMs) {
            prefillBatches.increment();
            prefillBatchRequests.add(batchSize);
            prefillBatchExecutionMs.add(executionMs);
            maxPrefillBatchSize.accumulateAndGet(batchSize, Math::max);
            maxPrefillBatchExecutionMs.accumulateAndGet(executionMs, Math::max);
            synchronized (prefillWindowLock) {
                prefillWindowCount++;
                prefillWindowMaxMs = Math.max(prefillWindowMaxMs, executionMs);
                if (prefillWindowSize < PREFILL_WINDOW_SAMPLE_CAP) {
                    prefillWindowSamples[prefillWindowSize++] = executionMs;
                } else {
                    long slot = ThreadLocalRandom.current().nextLong(prefillWindowCount);
                    if (slot < PREFILL_WINDOW_SAMPLE_CAP) {
                        prefillWindowSamples[(int) slot] = executionMs;
                    }
                }
            }
        }

        // ---- Prefill completion window (drained on every java_mock_stats tick) ----
        // Same bounded-reservoir scheme as the decode window: per-tick prefill
        // batch execution-time summary feeding java_mock_stats prefill_exec_p50/p95
        // (the legacy avg_batch_ms is a since-start cumulative mean and cannot
        // show per-tick execution-time drift).
        private static final int PREFILL_WINDOW_SAMPLE_CAP = 8192;
        private final Object prefillWindowLock = new Object();
        private final long[] prefillWindowSamples = new long[PREFILL_WINDOW_SAMPLE_CAP];
        private long prefillWindowCount;
        private long prefillWindowMaxMs;
        private int prefillWindowSize;

        // ---- Decode completion window (drained on every java_mock_stats tick) ----
        // A bounded reservoir keeps p50/p95 approximation cheap under load: exact
        // count/max are always tracked, samples beyond the cap replace a random
        // slot (unbiased reservoir sampling). Guarded by decodeWindowLock only —
        // one short critical section per decode completion and per stats tick.
        private static final int DECODE_WINDOW_SAMPLE_CAP = 8192;
        private final Object decodeWindowLock = new Object();
        private final long[] decodeWindowSamples = new long[DECODE_WINDOW_SAMPLE_CAP];
        private long decodeWindowCount;
        private long decodeWindowMaxMs;
        private int decodeWindowSize;

        void recordDecodeDone(long executionMs) {
            synchronized (decodeWindowLock) {
                decodeWindowCount++;
                decodeWindowMaxMs = Math.max(decodeWindowMaxMs, executionMs);
                if (decodeWindowSize < DECODE_WINDOW_SAMPLE_CAP) {
                    decodeWindowSamples[decodeWindowSize++] = executionMs;
                } else {
                    long slot = ThreadLocalRandom.current().nextLong(decodeWindowCount);
                    if (slot < DECODE_WINDOW_SAMPLE_CAP) {
                        decodeWindowSamples[(int) slot] = executionMs;
                    }
                }
            }
        }

        DecodeWindow drainDecodeWindow() {
            long count;
            long maxMs;
            long[] samples;
            synchronized (decodeWindowLock) {
                count = decodeWindowCount;
                maxMs = decodeWindowMaxMs;
                samples = Arrays.copyOf(decodeWindowSamples, decodeWindowSize);
                decodeWindowCount = 0;
                decodeWindowMaxMs = 0;
                decodeWindowSize = 0;
            }
            if (samples.length == 0) {
                return new DecodeWindow(count, 0, 0, maxMs);
            }
            Arrays.sort(samples);
            return new DecodeWindow(count,
                    samples[percentileIndex(samples.length, 0.50)],
                    samples[percentileIndex(samples.length, 0.95)],
                    maxMs);
        }

        private static int percentileIndex(int size, double quantile) {
            return Math.max(0, Math.min(size - 1, (int) Math.ceil(quantile * size) - 1));
        }

        /** Decode completions since the previous stats sample, with execution-time summary. */
        record DecodeWindow(long count, long p50Ms, long p95Ms, long maxMs) {
        }

        PrefillWindow drainPrefillWindow() {
            long count;
            long maxMs;
            long[] samples;
            synchronized (prefillWindowLock) {
                count = prefillWindowCount;
                maxMs = prefillWindowMaxMs;
                samples = Arrays.copyOf(prefillWindowSamples, prefillWindowSize);
                prefillWindowCount = 0;
                prefillWindowMaxMs = 0;
                prefillWindowSize = 0;
            }
            if (samples.length == 0) {
                return new PrefillWindow(count, 0, 0, maxMs);
            }
            Arrays.sort(samples);
            return new PrefillWindow(count,
                    samples[percentileIndex(samples.length, 0.50)],
                    samples[percentileIndex(samples.length, 0.95)],
                    maxMs);
        }

        /** Prefill batches since the previous stats sample, with execution-time summary. */
        record PrefillWindow(long count, long p50Ms, long p95Ms, long maxMs) {
        }
    }

    /**
     * Cluster-shared append-only JSONL event log (engine_events.jsonl): one
     * structured row per request terminal per engine (prefill_done on the
     * prefill engine, decode_done on the decode engine), replacing the former
     * mock_prefill_done / mock_decode_done stdout trace lines. The offline
     * aggregator (aggregate_canvas_run.py) rid-joins these rows against the
     * load client's client_events.jsonl to rebuild each request's full
     * lifecycle; components keep ZERO summarization duty (all derived stats
     * live in the aggregation layer).
     *
     * <p>Writes are synchronized (every engine in the cluster shares one file)
     * and autoflush per line — the same durability the per-line stdout printf
     * had: a killed JVM keeps every row a completion callback already wrote.
     * Opened (appending) from {@code --events-file}; a null/blank path
     * disables the stream entirely.
     */
    static final class EngineEventLog implements AutoCloseable {
        private final PrintWriter writer;

        private EngineEventLog(PrintWriter writer) {
            this.writer = writer;
        }

        /** Opens (appending) the events file; null/blank path → null (disabled). */
        static EngineEventLog open(String path) {
            if (path == null || path.isBlank()) {
                return null;
            }
            try {
                Path file = Path.of(path);
                if (file.getParent() != null) {
                    Files.createDirectories(file.getParent());
                }
                PrintWriter printWriter = new PrintWriter(Files.newBufferedWriter(file,
                        StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                        StandardOpenOption.APPEND), true);
                return new EngineEventLog(printWriter);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot open engine events file: " + path, e);
            }
        }

        /** Appends one serialized row (synchronized: all engines share one file). */
        synchronized void write(ObjectNode row) {
            writer.println(row.toString());
        }

        @Override
        public synchronized void close() {
            writer.flush();
            writer.close();
        }
    }

    static final class Config {
        private MockRpcExecutors rpcExecutors;
        private EventLoopGroup rpcTransport;
        synchronized MockRpcExecutors rpcExecutors(EventLoopGroup transport) {
            if (rpcExecutors == null || rpcTransport.isTerminated()) {
                rpcExecutors = new MockRpcExecutors(rpcThreads, rpcMaxCalls);
                rpcTransport = transport;
            }
            if (rpcTransport != transport) throw new IllegalStateException("RPC runtime belongs to another transport");
            return rpcExecutors;
        }
        java.util.function.Consumer<FastRpcService> engineInitializer;
        boolean whale = false;
        boolean whaleBundle = false;
        boolean kmonitor = false;
        // Package-private for direct assertions in ClusterConfigParamTest.
        int nPrefill = 2;
        int nDecode = 4;
        int baseGrpcPort = 61_000;
        int eventLoopThreads = 32;
        boolean cacheStatusCompactEncoding = false;
        int rpcThreads = 2;
        int rpcMaxCalls = 32768;
        int completionThreads = 8;
        // Match RtpLLMOp.cc's default receive limit; this does not reserve memory.
        int grpcMaxInboundMessageBytes = 1024 * 1024 * 1024;
        String host = "127.0.0.1";
        String bindHost = "0.0.0.0";
        String prefillDomain = "mock.prefill.hosts.address";
        String decodeDomain = "mock.decode.hosts.address";
        String endpointFile;
        String envFile;
        String discoveryFile;
        String calibrationFile;
        MockEngineConfig engineConfig;
        int statsIntervalMs = 5000;
        /**
         * Per-request JSONL event stream target (engine_events.jsonl). Null =
         * disabled (no event rows; the former mock_prefill_done /
         * mock_decode_done stdout lines are gone either way — the jsonl file is
         * the ONLY per-request engine-side output now).
         */
        String eventsFile;
        /**
         * Emit the java_mock_stats line on stdout every statsIntervalMs.
         * Default OFF: the stats timeline is a debug surface, and a quiet
         * stdout keeps mock_engine.log small. run_online_eval.sh passes
         * --stats-stdout explicitly because consolidate_run_outputs.py still
         * builds mock.json's stats timeline by parsing those lines from
         * mock_engine.log.
         */
        boolean statsStdout = false;
        boolean autoFetch = false;
        long fetchAttachTimeoutMs = 600_000L;
        /**
         * Unique per-engine loopback advertisement IPs (127.x.y.z), default on:
         * keeps the master-side engineIp Prometheus label distinct per engine.
         * Disable with --unique-engine-ips=false for the legacy single-host
         * behavior (every engine declares Config.host).
         */
        boolean uniqueEngineIps = true;

        static Config parse(String[] args) {
            Config config = new Config();
            for (int i = 0; i < args.length; i++) {
                String key = args[i];
                // Boolean flag glued form (--unique-engine-ips=false): the
                // value travels with the key, so no separate value is consumed.
                if (key.startsWith(UNIQUE_ENGINE_IPS_FLAG + "=")) {
                    config.uniqueEngineIps = parseBooleanFlag(
                            key.substring(UNIQUE_ENGINE_IPS_FLAG.length() + 1), UNIQUE_ENGINE_IPS_FLAG);
                    continue;
                }
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("Missing value for " + key);
                }
                String value = args[++i];
                switch (key) {
                    case "--n-prefill" -> config.nPrefill = Integer.parseInt(value);
                    case "--n-decode" -> config.nDecode = Integer.parseInt(value);
                    case "--base-grpc-port" -> config.baseGrpcPort = Integer.parseInt(value);
                    case "--event-loop-threads" -> config.eventLoopThreads = Integer.parseInt(value);
                    case "--cache-status-compact-encoding" -> config.cacheStatusCompactEncoding = parseBooleanFlag(value, key);
                    case "--rpc-threads" -> config.rpcThreads = Integer.parseInt(value);
                    case "--rpc-max-calls" -> config.rpcMaxCalls = Integer.parseInt(value);
                    case "--completion-threads" -> config.completionThreads = Integer.parseInt(value);
                    case "--grpc-max-inbound-message-bytes" -> config.grpcMaxInboundMessageBytes = Integer.parseInt(value);
                    case "--auto-fetch" -> config.autoFetch = parseBooleanFlag(value, key);
                    case "--fetch-attach-timeout-ms" -> config.fetchAttachTimeoutMs = Long.parseLong(value);
                    case "--host" -> config.host = value;
                    case "--bind-host" -> config.bindHost = value;
                    case "--whale" -> config.whale = parseBooleanFlag(value, key);
                    case "--whale-bundle" -> config.whaleBundle = parseBooleanFlag(value, key);
                    case "--kmonitor" -> config.kmonitor = parseBooleanFlag(value, key);
                    case "--prefill-domain" -> config.prefillDomain = value;
                    case "--decode-domain" -> config.decodeDomain = value;
                    case "--endpoint-file" -> config.endpointFile = value;
                    case "--env-file" -> config.envFile = value;
                    case "--discovery-file" -> config.discoveryFile = value;
                    case "--calibration" -> {
                        if (config.calibrationFile != null) throw new IllegalArgumentException("duplicate calibration input");
                        config.calibrationFile = value;
                    }
                    case "--engine-config" -> {
                        if (config.engineConfig != null) throw new IllegalArgumentException("duplicate engine config input");
                        try { config.engineConfig = MockEngineConfig.load(value); }
                        catch (IOException error) { throw new IllegalArgumentException("cannot read engine config", error); }
                    }
                    case "--stats-interval-ms" -> config.statsIntervalMs = Integer.parseInt(value);
                    case "--events-file" -> config.eventsFile = value;
                    case "--stats-stdout" -> config.statsStdout = parseBooleanFlag(value, "--stats-stdout");
                    case UNIQUE_ENGINE_IPS_FLAG -> config.uniqueEngineIps =
                            parseBooleanFlag(value, UNIQUE_ENGINE_IPS_FLAG);
                    default -> throw new IllegalArgumentException("Unknown argument: " + key);
                }
            }
            if (config.engineConfig == null || config.endpointFile == null || config.calibrationFile == null) {
                throw new IllegalArgumentException(
                        "--engine-config, --endpoint-file, and --calibration are required");
            }
            if (config.whaleBundle && !config.whale) {
                throw new IllegalArgumentException("--whale-bundle requires --whale true");
            }
            if (config.whale) {
                if ((!config.whaleBundle && config.nPrefill + config.nDecode != 1)
                        || config.host.equals("127.0.0.1") || config.host.equals("0.0.0.0")
                        || config.host.isBlank()) {
                    throw new IllegalArgumentException("Whale requires exactly one engine and an advertised Pod IP");
                }
                if (config.discoveryFile != null && !config.whaleBundle) {
                    throw new IllegalArgumentException("Whale uses platform discovery, not --discovery-file");
                }
            }
            if (config.kmonitor && !config.whale) {
                throw new IllegalArgumentException("--kmonitor requires --whale true");
            }
            if (config.discoveryFile == null && (!config.whale || config.whaleBundle)) {
                config.discoveryFile = Path.of(config.endpointFile).toAbsolutePath()
                        .resolveSibling("discovery.json").toString();
            }
            // Single-role clusters are allowed (e.g. engine_kill_restart_test victim JVMs
            // hosting only prefill or only decode engines), but at least one engine is required.
            if (config.nPrefill < 0 || config.nDecode < 0
                    || config.nPrefill + config.nDecode < 1) {
                throw new IllegalArgumentException(
                        "n-prefill/n-decode must be >= 0 with at least one engine in total");
            }
            if (config.eventLoopThreads < 1 || config.completionThreads < 1 || config.rpcThreads < 1 || config.rpcMaxCalls < 1) {
                throw new IllegalArgumentException("thread counts must be positive");
            }
            if (config.grpcMaxInboundMessageBytes < 1) {
                throw new IllegalArgumentException("--grpc-max-inbound-message-bytes must be >= 1");
            }
            if (config.fetchAttachTimeoutMs < 1) {
                throw new IllegalArgumentException("--fetch-attach-timeout-ms must be >= 1");
            }
            if (config.statsIntervalMs < 1) {
                throw new IllegalArgumentException("--stats-interval-ms must be >= 1");
            }
            if (config.nPrefill > 0) config.engineConfig.role("prefill");
            if (config.nDecode > 0) config.engineConfig.role("decode");
            return config;
        }
    }
}
