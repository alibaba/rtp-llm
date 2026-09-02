package org.flexlb.mockengine;

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
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.dao.route.RoleType;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
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
    /** Default decode available_concurrency reported to the master (CONCURRENCY_LIMIT-aligned, previously 132). */
    static final int DEFAULT_DECODE_MAX_CONCURRENCY = 128;
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
        // Per-request JSONL event stream (engine_events.jsonl): opened BEFORE
        // any engine starts so every FastRpcService (initial and dynamically
        // added) receives the same log instance via setEngineEventLog.
        EngineEventLog engineEventLog = EngineEventLog.open(config.eventsFile);
        try {
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    0, config.nPrefill, "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL);
            startRole(config, performance, serversByPort, bossGroup, workerGroup, services, scheduler, stats,
                    config.nPrefill, config.nDecode, "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE);
            for (FastRpcService service : services.values()) {
                service.setEngineEventLog(engineEventLog);
            }
            writeDiscoveryFiles(config);
            // File-based discovery mode (--discovery-file): maintain the dynamic
            // domain→hosts mapping consumed by FileServiceDiscovery on the master,
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
            shutdown(serversByPort, bossGroup, workerGroup);
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
            // Drain BEFORE killing the scheduler: cancel every in-flight request
            // through the existing cancel() bookkeeping so counters net to zero
            // and checkLeakDrain stops evaluating. Without this, requests whose
            // simulated completion (up to ~90s for long decodes) has not fired
            // yet were reported as LEAK DETECTED during test teardown.
            for (FastRpcService service : services.values()) {
                service.drainAndShutdown();
            }
            scheduler.shutdownNow();
            for (FastRpcService service : services.values()) {
                service.shutdown();
            }
            shutdown(serversByPort, bossGroup, workerGroup);
            if (engineEventLog != null) {
                engineEventLog.close();
            }
        }, "java-mock-engine-shutdown"));

        System.out.printf("Java mock engine ready: prefill=%d decode=%d ports=%d-%d eventLoops=%d performance=%s completionThreads=%d statsIntervalMs=%d%n",
                config.nPrefill, config.nDecode, config.baseGrpcPort,
                config.baseGrpcPort + config.nPrefill + config.nDecode - 1,
                config.eventLoopThreads, config.performanceFile, config.completionThreads,
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
                                  String roleName,
                                  EngineRpcService.RoleTypePB roleType) throws IOException {
        for (int i = 0; i < count; i++) {
            int grpcPort = config.baseGrpcPort + portOffset + i;
            startEngine(config, performance, serversByPort, bossGroup, workerGroup,
                    services, scheduler, stats, roleName, roleName + "-" + i, grpcPort,
                    portOffset + i);
        }
    }

    /**
     * Create, register, and start ONE engine gRPC server. Extracted from
     * {@link #startRole} so dynamic scale-out ({@link DynamicEngineManager}
     * /add_engine) can create engines at runtime with the same construction
     * rules: shared scheduler/stats/eventLoopGroups, per-role cache capacity,
     * SO_REUSEADDR, direct executor.
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
        // Per-role KV pool sizing (capacity model v2): totalKvTokens decides the
        // reported token capacity, the pool itself is sized in blocks
        // (ceil(total/spb)); --prefill-cache-blocks/--decode-cache-blocks override
        // the block count directly (legacy flags repurposed from key-count caps
        // to pool-size overrides so the load scripts keep working unchanged).
        long roleTotalKvTokens = roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                ? config.prefillTotalKvTokens : config.decodeTotalKvTokens;
        int blocksOverride = roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL
                ? config.prefillCacheBlocks : config.decodeCacheBlocks;
        int spb = performance.blockSize();
        int totalBlocks = blocksOverride > 0
                ? blocksOverride
                : (int) ((roleTotalKvTokens + spb - 1) / spb);
        FastRpcService service = new FastRpcService(
                engineName, declaredHost(config, engineIndex), roleName, roleType, grpcPort,
                services, scheduler, performance, totalBlocks, stats,
                roleTotalKvTokens, config.decodeMaxConcurrency);
        service.setResponsePollTimeoutMs(DEFAULT_RESPONSE_POLL_TIMEOUT_MS);
        services.put(grpcPort, service);
        try {
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
                                 EventLoopGroup workerGroup) {
        for (Server server : serversByPort.values()) {
            server.shutdownNow();
        }
        bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
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
                        + "cancel_census_tracked=%d cancel_census_finished=%d cancel_census_unknown=%d cancel_census_tombstone=%d "
                        + "cancel_census_client_gone=%d%n",
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
                stats.cancelCensusUnknown.sum(), stats.cancelCensusTombstone.sum(),
                stats.cancelCensusClientGone.sum());
    }

    static void writeDiscoveryFiles(Config config) throws IOException {
        String prefillAddresses = addressList(config, 0, config.baseGrpcPort, config.nPrefill);
        String decodeAddresses = addressList(
                config, config.nPrefill, config.baseGrpcPort + config.nPrefill, config.nDecode);

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

    private static String addressList(Config config, int firstEngineIndex, int firstGrpcPort, int count) {
        StringBuilder addresses = new StringBuilder(count * 20);
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                addresses.append(',');
            }
            addresses.append(declaredHost(config, firstEngineIndex + i))
                    .append(':').append(firstGrpcPort + i - 1);
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

        private final String engineName;
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
        /**
         * Per-request engine-side arrival stamps (epoch ms) for
         * engine_events.jsonl: recorded at enqueue/admission, consumed and
         * removed by the terminal callback that writes the event row. Bounded
         * indirectly by periodicCleanup (rids no longer tracked anywhere).
         */
        private final ConcurrentHashMap<Long, Long> eventArrivalMs = new ConcurrentHashMap<>();
        /**
         * Per-request execution-start stamps (epoch ms): prefill batch start /
         * decode running-slot admission, consumed with {@link #eventArrivalMs}
         * by the terminal callback.
         */
        private final ConcurrentHashMap<Long, Long> eventStartMs = new ConcurrentHashMap<>();
        private final Map<Integer, FastRpcService> services;
        private final ScheduledExecutorService scheduler;
        private final MockPerformanceModel performance;
        private final MockLruBlockCache cache;
        private final ClusterStats stats;
        /** Reported token capacity (per-role; pool is ceil(total/spb) blocks). */
        private final long totalKvTokens;
        /** Blocks per cache block (spb) — the pool's token<->block conversion factor. */
        private final int seqSizePerBlock;
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
        // Priority-Cancel tombstones mirror the C++ Prefill contract: once a
        // request accepted priority cancellation, retries stay ACCEPTED even
        // after the live ownership entry has been removed.  Keep this separate
        // from client-cancel history so a normal terminal remains NOT_FOUND.
        private final LinkedHashSet<Long> priorityCancelTombstones = new LinkedHashSet<>();
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
        // ── KV capacity model v2: single block-pool truth ──
        // In-flight requests pin blocks via per-request leases (acquired at
        // prefill enqueue / decode run-start, grown per decode step, handed to
        // the LRU on completion, returned to free on cancel). Occupied tokens
        // are derived from the pool — never tracked as an independent counter.
        private final Map<Long, MockLruBlockCache.BlockLease> activeBlockLeases = new ConcurrentHashMap<>();
        /** Decode requests that ran un-pooled because admission/growth failed (overflow observability). */
        private final LongAdder kvAdmissionFails = new LongAdder();
        /** Prefill requests synchronously rejected with LACK_MEM 602 (the
         *  enqueue-batch Phase-1.5 gate + the direct generate_stream gate —
         *  distinct from kvAdmissionFails, which counts DECODE degradations:
         *  prefill rejects, decode degrades). /metrics carries it as
         *  mock_engine_lack_mem_rejects_total. */
        private final LongAdder prefillLackMemRejects = new LongAdder();
        /** Decode prefix-reuse blocks accumulated (KV v2 fix #5): every
         *  acquireWithReuse hit key — the net-demand deduction the decode
         *  engine makes against its OWN LRU. Cumulative counter (never
         *  drained): /metrics carries it as mock_engine_decode_reuse_blocks_total,
         *  /snapshot as decode_reuse_blocks. */
        private final LongAdder decodeReuseBlocks = new LongAdder();
        /** Key-level cache-hit observability (production recent_cache_key_hit_count /
         *  total_count caliber): cumulative counters recorded at the prefill
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
        // Per-engine busy time. Prefill engines accumulate batch execution ms
        // (maxPrefillConcurrency=1 -> busy == wall-clock occupancy; utilization =
        // busy/elapsed). Decode engines accumulate per-request execution ms under
        // soft concurrency, so busy/elapsed reads as the average concurrent
        // request count, not a <=100% utilization. Exported via /snapshot busy_ms
        // (never per-engine keys on the per-second stats line: 1250 engines would
        // bloat every tick line).
        private final AtomicLong busyMs = new AtomicLong();
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
            /** Total step budget at admission — with remainingSteps it yields
             * tokens generated so far (tokensPerStep x (total - remaining)),
             * which drives the per-step KV block growth (production incrMalloc). */
            final int totalSteps;
            /** Σ actual step durations (the per-step exec caliber). */
            double accumulatedExecMs;
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
            /** True while a step is already executing (tick pending) and this
             * stream joined mid-step: the stream joins the running batch at the
             * NEXT step boundary and produces its first tokens there, mirroring
             * production where a request arriving during step k first participates
             * in step k+1. Flipped to false by the boundary tick. */
            boolean awaitsFirstStep;

            DecodeStream(MockPerformanceModel.RequestShape shape, long batchId,
                         LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue,
                         int totalSteps,
                         boolean awaitsFirstStep) {
                this.shape = shape;
                this.batchId = batchId;
                this.responseQueue = responseQueue;
                this.totalSteps = totalSteps;
                this.remainingSteps = totalSteps;
                this.awaitsFirstStep = awaitsFirstStep;
            }
        }
        // ── Prefill batch-level wait queue (change 2) ──
        // Pending prefill batches waiting for a maxPrefillConcurrency slot. Drained
        // by the prefill completion callback. waitingPrefillRequests now reports the
        // real queued depth (queued requests) instead of lane-delayed batches.
        private final ArrayDeque<PrefillPendingBatch> prefillPendingQueue = new ArrayDeque<>();
        // Direct (generate_stream / NON_BATCH) requests parked while every
        // maxPrefillConcurrency slot is busy. Unlike prefillPendingQueue (whose
        // elements are master-composed batches), entries here are individual
        // requests: the drain coalesces up to performance.directBatchSizeMax() of
        // them into ONE batch, mirroring production engines' prefill continuous
        // batching. waitingPrefillRequests counts members of BOTH queues.
        private final ArrayDeque<MockPerformanceModel.RequestShape> directPrefillQueue = new ArrayDeque<>();
        private final Object prefillQueueLock = new Object();
        private final ConcurrentLinkedQueue<VersionedTask> completions = new ConcurrentLinkedQueue<>();
        /**
         * Publishes completion records and their cursor as one ordered operation.
         * A status reader must never observe a latest version whose record has
         * not yet been inserted, otherwise advancing its cursor loses that
         * completion permanently.
         */
        private final Object completionLock = new Object();
        private final Map<Long, EngineRpcService.TaskInfoPB> runningTasks = new ConcurrentHashMap<>();
        private final Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>> responseQueues = new ConcurrentHashMap<>();
        private final Map<Long, String> requestStates = new ConcurrentHashMap<>();
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

        private volatile FaultInjectionConfig faultConfig = FaultInjectionConfig.builder().build();
        private final AtomicInteger enqueueCount = new AtomicInteger();
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
        // Set once by drainAndShutdown(): rejects new admissions and disables
        // checkLeakDrain so shutdown-time in-flight requests are not misreported
        // as leaks. Never reset (the process is exiting).
        private volatile boolean shuttingDown = false;
        private final AtomicBoolean leakDetected = new AtomicBoolean(false);
        private final AtomicLong lastEnqueueTime = new AtomicLong(System.nanoTime());
        private final AtomicLong acceptedCount = new AtomicLong();
        private final AtomicLong completedCount = new AtomicLong();
        private final AtomicLong cancelledCount = new AtomicLong();
        // ── Production-caliber TPS observation (rtp_llm_* /metrics series) ──
        // Pure accounting on completion events: token sums accumulate into
        // the *Tokens counters and every /metrics scrape drains them into the
        // lastWindow* values (window = scrape interval, 1s for the G1 poller
        // — the value IS tokens-per-second because the window is 1s). Caliber
        // note: the mock's execution time is itself a formula product, so
        // unlike production there is no execute/wall dual denominator — the
        // fixed 1s window is the whole denominator. Only NON-cancelled
        // completions count (production semantics: tokens actually accepted
        // and generated). hit_tokens_total is cumulative and never drained
        // (the cache_saved_tokens source via final_snapshot).
        private final AtomicLong contextComputeTokens = new AtomicLong();
        private final AtomicLong contextWithCacheTokens = new AtomicLong();
        private final AtomicLong generateTokens = new AtomicLong();
        private final AtomicLong hitTokensTotal = new AtomicLong();
        private final AtomicLong lastWindowContextCompute = new AtomicLong();
        private final AtomicLong lastWindowContextCache = new AtomicLong();
        private final AtomicLong lastWindowGenerate = new AtomicLong();
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
            this.performance = performance;
            this.cache = new MockLruBlockCache(totalBlocks);
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

            Runnable process = () -> {
                for (EngineRpcService.EnqueueBatchDpSlotPB slot : request.getDpSlotsList()) {
                    List<MockPerformanceModel.RequestShape> shapes = new ArrayList<>(slot.getRequestsCount());
                    // Phase 1: register per-request state the completion callback
                    // depends on (responseQueues/requestStates) BEFORE admission so
                    // an immediately-admitted batch can never complete against a
                    // missing response queue.
                    for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                        long requestId = input.getInput().getRequestId();
                        MockPerformanceModel.RequestShape shape = performance.shape(input.getInput(), cache);
                        // Key-level cache-hit accounting at the admission hit
                        // computation point (recorded whether or not the request
                        // later admits — a rejected request still observed the
                        // engine's index state for its keys).
                        cacheKeyHits.add(shape.hitBlocks());
                        cacheKeysRequested.add(shape.blockKeys().size());
                        // Phase 1.5 (KV capacity model v2): block-pool admission.
                        // A request whose blocks cannot be provisioned (free + LRU
                        // below need, or the reserve watermark would be breached) is
                        // rejected SYNCHRONOUSLY in this ack with MALLOC_FAILED —
                        // the engine-side KV gate the master turns into
                        // EngineRejectedException on its dispatch path. Rejected
                        // requests leave no residue (state rolled back below).
                        MockLruBlockCache.BlockLease lease =
                                acquireBlockLease(requestId, shape);
                        if (lease == null) {
                            prefillLackMemRejects.increment();
                            String message = String.format(
                                    "LACK_MEM: insufficient KV cache blocks (need=%d, avail=%d, spb=%d)",
                                    needBlocks(shape), cache.availableBlocks(), seqSizePerBlock);
                            response.addErrorsBuilder()
                                    .setRequestId(requestId)
                                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                            .setErrorCode(LACK_MEM_ERROR_CODE)
                                            .setErrorMessage(message)
                                            .build());
                            requestStates.put(requestId, "rejected");
                            continue;
                        }
                        shapes.add(shape);
                        responseQueues.computeIfAbsent(requestId, k -> new LinkedBlockingQueue<>());
                        requestStates.put(requestId, "running");
                    }
                    // Phase 2: admission. false = prefill waiting-queue cap hit
                    // (batch-level backpressure, independent of the request-level
                    // queue_depth_limit fault-injection gate checked at the RPC
                    // entry above). Roll back phase-1 state so rejected requests
                    // leave no residue (no pendingRequests/waitingPrefillRequests/
                    // runningTasks were claimed — the cap check rejects before any
                    // counter is touched).
                    if (!schedulePrefillCompletion(shapes, request.getBatchId(), slot.getDpRank())) {
                        String message = String.format(
                                "prefill waiting queue full (backpressure): waiting=%d cap=%d",
                                prefillPendingQueueSize(), performance.maxWaitingPrefillBatches());
                        for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                            long requestId = input.getInput().getRequestId();
                            responseQueues.remove(requestId);
                            requestStates.put(requestId, "rejected");
                            response.addErrorsBuilder()
                                    .setRequestId(requestId)
                                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                            .setErrorMessage(message)
                                            .build());
                        }
                        continue;
                    }
                    // Phase 3: success bookkeeping (only admitted requests count).
                    for (EngineRpcService.EnqueueBatchExternalInputPB input : slot.getRequestsList()) {
                        stats.enqueuedRequests.increment();
                        acceptedCount.incrementAndGet();
                        long requestId = input.getInput().getRequestId();
                        response.addSuccessesBuilder().setRequestId(requestId);
                        recordLifecycleStart(requestId, request.getBatchId(), "enqueue_batch");
                        recordEventArrival(requestId);
                    }
                }
                // ── EnqueueBatch ack fault injections: all phases above ran
                // exactly as usual (the engine really admitted and will
                // execute every member); only the ACK content is corrupted,
                // so the master must tolerate an ack that lies. ──
                if (faultConfig.isEnqueueAckDrop()) {
                    // enqueue_ack_drop: empty ack — no successes, no errors,
                    // stopped stays false (unlike crash_after) so the engine
                    // keeps serving subsequent RPCs normally.
                    observer.onNext(EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                            .setBatchId(request.getBatchId())
                            .build());
                    observer.onCompleted();
                    return;
                }
                applyEnqueueAckFaults(response);
                observer.onNext(response.build());
                observer.onCompleted();
            };

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

        /**
         * EnqueueBatch ack fault application (enqueue_ack_partial_fail +
         * enqueue_ack_error_code): move the first k admitted members from
         * successes to errors in the ACK only — the engine still executes all
         * of them, so their completions surface later via getWorkerStatus.
         * The error code defaults to 13 unless enqueue_ack_error_code
         * overrides it (per-request: each moved member's error_info entry
         * carries the code).
         */
        private void applyEnqueueAckFaults(EngineRpcService.EnqueueBatchResponsePB.Builder response) {
            int k = faultConfig.getEnqueueAckPartialFail();
            if (k <= 0 || response.getSuccessesCount() == 0) {
                return;
            }
            long errorCode = faultConfig.getEnqueueAckErrorCode() != 0
                    ? faultConfig.getEnqueueAckErrorCode() : 13L;
            int moves = Math.min(k, response.getSuccessesCount());
            List<Long> moved = new ArrayList<>(moves);
            for (int i = 0; i < moves; i++) {
                moved.add(response.getSuccesses(i).getRequestId());
            }
            for (int i = 0; i < moves; i++) {
                response.removeSuccesses(0);
            }
            for (long requestId : moved) {
                response.addErrorsBuilder()
                        .setRequestId(requestId)
                        .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                .setErrorCode(errorCode)
                                .setErrorMessage("injected enqueue_ack_partial_fail")
                                .build());
            }
        }

        @Override
        public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                    StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
            stats.statusRpcs.increment();
            // status_no_respond: hang the RPC — no onNext/onCompleted ever,
            // mirroring generateStreamCall's noRespond handling.
            if (faultConfig.isStatusNoRespond()) {
                return;
            }
            long requestedVersion = request.getLatestFinishedVersion();
            long latestVersion;
            List<VersionedTask> visibleCompletions = new ArrayList<>();
            synchronized (completionLock) {
                VersionedTask head;
                while ((head = completions.peek()) != null
                        && head.version <= requestedVersion) {
                    completions.poll();
                }
                latestVersion = completionVersion.get();
                for (VersionedTask completion : completions) {
                    if (completion.version > requestedVersion
                            && completion.version <= latestVersion) {
                        visibleCompletions.add(completion);
                    }
                }
            }
            long runningCount = runningTasks.values().stream()
                    .filter(task -> task.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .count();
            // Capacity model v2: used/available both derive from the block pool
            // (occupied = held + referenced-key blocks) plus injected pressure —
            // the same caliber getCacheStatus and /snapshot report, so the master
            // sees one consistent number on every surface.
            EngineRpcService.WorkerStatusPB.Builder status = EngineRpcService.WorkerStatusPB.newBuilder()
                    .setAlive(!stopped)
                    .setRole("RoleType." + roleName)
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
                    .setAvailableKvCache(availableKvTokens())
                    .setTotalKvCache(totalKvTokens)
                    .setStatusVersion(statusVersion.incrementAndGet())
                    .setLatestFinishedVersion(latestVersion)
                    .setDpSize(1)
                    .setTpSize(1)
                    .setDpRank(0)
                    // Static engine limits. Production engines publish max_seq_len /
                    // max_batch_tokens_size so the master can clamp decision-group
                    // token capacity (BatcherContext); reporting them keeps the mock's
                    // admission semantics aligned with production instead of the
                    // implicit unlimited fallback.
                    .setMaxSeqLen(1048576L)
                    .setMaxBatchTokensSize(1048576L);
            status.addAllRunningTaskInfo(runningTasks.values().stream()
                    .map(FastRpcService::withLegacyTaskState)
                    .toList());
            for (VersionedTask completion : visibleCompletions) {
                status.addFinishedTaskList(withLegacyTaskState(completion.task));
            }
            // ── Status-report fault injections: pure output-layer filters on
            // the assembled status. The completion queue, its head-trim and
            // the version bookkeeping above are untouched — a completion
            // suppressed here is permanently lost once the (real) cursor
            // advances past it, which is exactly the fault under test. ──
            applyStatusReportFaults(status);
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
            if (faultConfig.isStatusVersionRegress()) {
                status.setStatusVersion(statusVersion.addAndGet(-2L));
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

        @Override
        public void generateStreamCall(EngineRpcService.GenerateInputPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
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
            // Key-level cache-hit accounting (direct path, same admission hit
            // computation point as the enqueue-batch path above).
            cacheKeyHits.add(shape.hitBlocks());
            cacheKeysRequested.add(shape.blockKeys().size());
            acceptedCount.incrementAndGet();
            lastEnqueueTime.set(System.nanoTime());
            requestStates.put(requestId, "running");
            recordLifecycleStart(requestId, -1, "generate_stream");
            recordEventArrival(requestId);

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
                // KV capacity model v2: direct-prefill admission mirrors the
                // EnqueueBatch Phase-1.5 gate — provision blocks BEFORE claiming
                // any queue state so a LACK_MEM rejection leaves no residue.
                // Same master-visible surface as the enqueue path (synchronous
                // error on the dispatch RPC, code MALLOC_FAILED=602 in the
                // EnqueueBatch flavor; generate_stream carries it in the
                // RuntimeException message).
                if (acquireBlockLease(requestId, shape) == null) {
                    prefillLackMemRejects.increment();
                    responseQueues.remove(requestId);
                    requestStates.put(requestId, "rejected");
                    observer.onError(new RuntimeException(String.format(
                            "LACK_MEM: insufficient KV cache blocks (need=%d, avail=%d, spb=%d)",
                            needBlocks(shape), cache.availableBlocks(), seqSizePerBlock)));
                    return;
                }
                if (!admitDirectPrefill(shape)) {
                    // Backpressure: direct waiting-queue cap hit — reject so the
                    // caller (client/master) perceives prefill overload. Clean up
                    // the per-request state set up above; the cap check rejects
                    // before claiming any counter. The lease provisioned above
                    // returns to the pool too.
                    releaseBlockLease(requestId);
                    responseQueues.remove(requestId);
                    requestStates.put(requestId, "rejected");
                    observer.onError(new RuntimeException(String.format(
                            "prefill waiting queue full (backpressure): waiting=%d cap=%d",
                            waitingPrefillRequests.get(), directWaitingRequestCap())));
                    return;
                }
            }

            // Admitted: arm the autonomous client-gone detector for this
            // stream (production C++ engine IsCancelled semantics).
            registerClientGoneListener(requestId, rpcContext);

            // Use a separate executor for blocking poll to avoid starving the
            // completion scheduler which is responsible for producing responses.
            // Loop poll until a finished=true frame or timeout, so the mock can
            // emit a first-token frame (finished=false) at prefill completion
            // and a terminal frame (finished=true) at decode completion. Client
            // measures ttft from the first frame and total from the finished
            // frame; without the loop both metrics collapse to a single
            // timestamp.
            responseExecutor.execute(() -> {
                try {
                    boolean anyDelivered = false;
                    while (true) {
                        if (rpcContext.isCancelled()) {
                            // Client stream broke mid-flight: the listener has
                            // already driven the autonomous cancellation; stop
                            // delivering frames to the dead observer and exit
                            // WITHOUT onCompleted (the call is gone).
                            return;
                        }
                        EngineRpcService.GenerateOutputsPB output =
                                queue.poll(responsePollTimeoutMs, TimeUnit.MILLISECONDS);
                        if (output == null) {
                            // Timeout with no frame — mirror the pre-change
                            // behavior of completing the stream. If nothing was
                            // ever delivered the client sees zero outputs and
                            // reports empty_response.
                            break;
                        }
                        observer.onNext(output);
                        anyDelivered = true;
                        boolean terminal = false;
                        if (output.hasErrorInfo()) {
                            // Error-only frames (cancel, P->D downstream
                            // cancel, preemption tombstones) terminate the
                            // stream: gRPC semantics deliver the failure then
                            // close. Without this the pump keeps polling until
                            // the frame-gap timeout (600s default), leaving
                            // cancelled requests' client streams open far past
                            // the cancel (functional cancel cases wait 5s).
                            terminal = true;
                        } else {
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
                    // anyDelivered is intentionally unread beyond diagnostics —
                    // the client already treats a stream with zero onNext as
                    // empty_response via its own null firstFrameNanos check.
                    if (!anyDelivered) {
                        // no-op branch, kept explicit for reader clarity
                    }
                } catch (InterruptedException e) {
                    observer.onError(e);
                } catch (RuntimeException e) {
                    if (!rpcContext.isCancelled()) {
                        throw e;
                    }
                    // Delivering a frame (e.g. the CANCELLED error frame the
                    // autonomous cancel just queued) to an already-cancelled
                    // call is expected here; swallow so the pump thread exits
                    // cleanly instead of dying in the executor.
                }
            });
        }

        @Override
        public void fetchResponse(EngineRpcService.FetchRequestPB request,
                StreamObserver<EngineRpcService.GenerateOutputsPB> observer) {
            stats.fetchResponseRpcs.increment();
            rpcFetchResponse.incrementAndGet();
            // Same client-gone capture as generateStreamCall: under the BATCH
            // dispatcher the client's FetchResponse stream is glued to the
            // ORIGINAL PREFILL engine, so a broken fetch must drive that
            // prefill's autonomous cancel (and the P->D propagation when the
            // hand-off already happened).
            Context rpcContext = Context.current();

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

            // Arm the autonomous client-gone detector for this fetch stream.
            registerClientGoneListener(requestId, rpcContext);

            // Loop poll until a finished=true frame or timeout — same pump as
            // generateStreamCall. Under the BATCH dispatcher the client polls
            // FetchResponse against the ORIGINAL PREFILL engine (startDecode
            // handed the prefill-side queue to the Decode engine), so the stream
            // carries TWO frames per request: a first-token frame (finished=false)
            // offered at prefill completion and a terminal frame (finished=true)
            // offered at decode completion. Polling exactly ONE frame collapsed
            // the stream after the first-token frame, making the client's ttft
            // and e2e identical by construction (verified: run 20260827_112212
            // per-second ttft_p50==e2e_p50 everywhere) and truncating e2e for
            // every request that actually spent decode time. The loop keeps the
            // empty_response semantics: zero frames delivered before the timeout
            // still completes the stream without error.
            responseExecutor.execute(() -> {
                try {
                    while (true) {
                        if (rpcContext.isCancelled()) {
                            // Client fetch stream broke mid-flight: the
                            // listener already drove the autonomous cancel;
                            // exit without touching the dead observer.
                            return;
                        }
                        EngineRpcService.GenerateOutputsPB output =
                                queue.poll(responsePollTimeoutMs, TimeUnit.MILLISECONDS);
                        if (output == null) {
                            break;
                        }
                        observer.onNext(output);
                        boolean terminal = false;
                        if (output.hasErrorInfo()) {
                            // Error-only frames (cancel, P->D downstream
                            // cancel, preemption tombstones) terminate the
                            // stream — mirrors the generateStreamCall pump.
                            // Without this a cancelled request's FetchResponse
                            // hangs until the 600s frame-gap timeout.
                            terminal = true;
                        } else {
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
            if (countRpc) {
                stats.cancelRpcs.increment();
                rpcCancel.incrementAndGet();
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
            addCancelledRid(requestId);
            if (priorityPreemption) {
                addPriorityCancelTombstone(requestId);
            }
            recordLifecycleEnd(requestId, true);
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
                    // If the request is still parked in the decode pending queue
                    // (not yet running), it has NOT been counted in
                    // activeDecodeRequests (and, in default mode, holds no block
                    // lease either), so those must not be released.
                    // The opt-in accepted-layer mode DID claim its lease at
                    // enqueue — released in the else-if below. removeIf under the
                    // admission lock atomically
                    // determines queued-vs-running: if the task is still in the
                    // queue it is removed here (wasQueued=true); if it was already
                    // drained into a running slot, removeIf finds nothing and the
                    // request is treated as running (release the slot below).
                    boolean wasQueuedDecode = decodePendingQueue.removeIf(
                            t -> t.shape().input().getRequestId() == requestId);
                    EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
                    if (removed != null) {
                        cancelledPhase = removed.getPhase();
                        pendingRequests.decrementAndGet();
                        if (!wasQueuedDecode) {
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
                        } else if (performance.reportQueuedAsKvAllocated()) {
                            // Opt-in KV fidelity (P2-5): the queued request's block
                            // lease was claimed at enqueue — release it here.
                            // Default OFF queued entries never claimed a lease;
                            // nothing to release.
                            releaseBlockLease(requestId);
                        }
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
                EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
                if (removed != null) {
                    cancelledPhase = removed.getPhase();
                    pendingRequests.decrementAndGet();
                    releaseBlockLease(requestId);
                }
            }
            requestStates.put(requestId, "cancelled");
            cancelledCount.incrementAndGet();
            EngineRpcService.TaskInfoPB.Builder taskBuilder = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(requestId)
                    // Pass the ACTUAL phase the request was cancelled in through
                    // to the finished entry (P2-1): a queued opt-in decode
                    // request surfaces KV_ALLOCATED, a queued prefill RECEIVED.
                    // RUNNING remains the fallback when no entry was found.
                    .setPhase(cancelledPhase != null
                            ? cancelledPhase : EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(priorityPreemption
                                    ? PRIORITY_PREEMPTED_ERROR_CODE
                                    : EngineRpcService.ErrorCodePB.CANCELLED.getNumber())
                            .setErrorMessage(priorityPreemption
                                    ? "preempted by higher-priority request"
                                    : "cancelled by client")
                            .build())
                    .setEndTimeMs(System.currentTimeMillis())
                    .setDpRank(0);
            // Same upstream fix as recordPriorityPreemptionCanceled: the typed
            // CANCELLED terminal must carry the exact EnqueueBatch identity so
            // the master's reconcile can correlate the cancel with the batch it
            // displaced (a terminal without batch_id falls into the batch-
            // mismatch dead branch and leaks the inflight slot).
            long cancelLifecycleBatchId = positiveLifecycleBatchId(requestId);
            if (cancelLifecycleBatchId > 0L) {
                taskBuilder.setBatchId(cancelLifecycleBatchId);
            }
            if (priorityPreemption) {
                taskBuilder.setPriorityPreemptionProgress(
                        EngineRpcService.PriorityPreemptionProgressPB
                                .PRIORITY_PREEMPTION_CANCELED);
            }
            EngineRpcService.TaskInfoPB task = taskBuilder.build();
            publishCompletion(task);
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
            // (A cancelled running slot's top-up ran under decodeQueueLock inside
            // the decode branch above — nothing to schedule outside the lock.)
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE) {
                clearUpstreamOwnership(requestId);
            }
            return cancelledPhase;
        }

        /**
         * Three-branch cancel used by {@link MockEngineCancelChannel}: reports whether
         * the request was actively tracked (and at which phase), already finished
         * (completed or previously cancelled), or entirely unknown to this engine.
         * The mock behaviour on the found branch is identical to {@link #cancel(long)}:
         * the request is removed and a CANCELLED completion is surfaced in the next
         * WorkerStatus finished list.
         */
        CancelResult cancelRequest(long requestId) {
            stats.cancelRpcs.increment();
            rpcCancel.incrementAndGet();
            if (roleType != EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                throw new UnsupportedOperationException(
                        "priority Cancel is only implemented by the original Prefill");
            }
            if (hasPriorityCancelTombstone(requestId)) {
                stats.cancelCensusTombstone.increment();
                return new CancelResult(true, null, true);
            }
            EngineRpcService.TaskInfoPB tracked = runningTasks.get(requestId);
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
            boolean alreadyFinished;
            synchronized (cancelledRidHistory) {
                alreadyFinished = cancelledRidHistory.contains(requestId);
            }
            if (!alreadyFinished) {
                synchronized (requestLifecycles) {
                    Map<String, Object> lifecycle = requestLifecycles.get(requestId);
                    alreadyFinished = lifecycle != null
                            && !"running".equals(lifecycle.get("end_state"));
                }
            }
            if (alreadyFinished) {
                // A2 census: cancel raced a terminal that already finished — the
                // master's reconcile should see the original terminal instead.
                stats.cancelCensusAlreadyFinished.increment();
                return new CancelResult(false, null, true);
            }
            // A2 census: cancel addressed a request this engine never knew —
            // stale master bookkeeping or a cancelled generation.
            stats.cancelCensusUnknown.increment();
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
                // This is downstream stream cancellation, not a Decode Cancel
                // RPC.  Preserve ordinary Decode accounting/terminal behavior
                // without incrementing the Decode RPC counter.
                EngineRpcService.TaskPhase phase = cancel(requestId, false, false);
                EngineRpcService.TaskPhase observedPhase = phase != null
                        ? phase : EngineRpcService.TaskPhase.TASK_PHASE_RUNNING;
                // Terminate the client-facing stream: after the P->D hand-off the
                // response queue the client's FetchResponse / GenerateStreamCall
                // poller hangs on lives on the ORIGINAL PREFILL (startDecode passed
                // it to this Decode). cancel() above only offers into this
                // Decode's own responseQueues, which never contained the request,
                // so without this delivery the cancelled request's stream hangs
                // until the 60s poll timeout (Python mock terminates the stream
                // from its async cancel finalizer instead).
                LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> prefillQueue =
                        expectedPrefill.responseQueues.get(requestId);
                if (prefillQueue != null) {
                    prefillQueue.offer(EngineRpcService.GenerateOutputsPB.newBuilder()
                            .setRequestId(requestId)
                            .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                                    .setErrorCode(EngineRpcService.ErrorCodePB.CANCELLED)
                                    .setErrorMessage("cancelled by client")
                                    .build())
                            .build());
                    expectedPrefill.responseQueues.remove(requestId);
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
        private void handleClientGone(long requestId) {
            if (runningTasks.containsKey(requestId)) {
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
            EngineRpcService.TaskInfoPB.Builder task = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(requestId)
                    .setPhase(phase)
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(EngineRpcService.ErrorCodePB.CANCELLED.getNumber())
                            .setErrorMessage("cancelled by client (stream gone)")
                            .build())
                    .setEndTimeMs(System.currentTimeMillis())
                    .setDpRank(0);
            long batchId = positiveLifecycleBatchId(requestId);
            if (batchId > 0L) {
                task.setBatchId(batchId);
            }
            publishCompletion(task.build());
            statusVersion.incrementAndGet();
        }

        private void recordPriorityPreemptionCanceled(long requestId,
                                                      EngineRpcService.TaskPhase phase) {
            addPriorityCancelTombstone(requestId);
            EngineRpcService.TaskInfoPB.Builder task = EngineRpcService.TaskInfoPB.newBuilder()
                    .setRequestId(requestId)
                    .setPhase(phase)
                    .setPriorityPreemptionProgress(EngineRpcService.PriorityPreemptionProgressPB
                            .PRIORITY_PREEMPTION_CANCELED)
                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                            .setErrorCode(PRIORITY_PREEMPTED_ERROR_CODE)
                            .setErrorMessage("preempted by higher-priority request")
                            .build())
                    .setEndTimeMs(System.currentTimeMillis())
                    .setDpRank(0);
            // Upstream fix preserved through the intake2 merge: typed
            // preemption terminals must carry the exact EnqueueBatch identity
            // so the master's ownership bookkeeping can correlate the cancel
            // with the batch it displaced.
            long batchId = positiveLifecycleBatchId(requestId);
            if (batchId > 0L) {
                task.setBatchId(batchId);
            }
            publishCompletion(task.build());
            statusVersion.incrementAndGet();
        }

        /** Preserve the exact EnqueueBatch identity on typed Prefill terminals. */
        private long positiveLifecycleBatchId(long requestId) {
            synchronized (requestLifecycles) {
                Map<String, Object> lifecycle = requestLifecycles.get(requestId);
                Object rawBatchId = lifecycle == null
                        ? null : lifecycle.get("batch_id");
                if (rawBatchId instanceof Number number
                        && number.longValue() > 0L) {
                    return number.longValue();
                }
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
                }
            }
        }

        /**
         * Admission point for a prefill batch. Returns true when the batch was
         * admitted (running slot reserved or parked in the pending queue); false
         * when the waiting-queue cap is reached (backpressure) — in that case
         * nothing was claimed (no pendingRequests/waitingPrefillRequests/
         * runningTasks residue) and the caller must reject the batch's requests.
         *
         * <p>The cap (performance JSON {@code prefill.max_waiting_batches},
         * default 0 = unbounded) bounds QUEUED batches only — running
         * batches never count toward it. Derivation: batches run FIFO, so the
         * k-th queued batch starts after k × batch_ms; with a 1000 ms target
         * latency and ~150 ms prefill execution, the wait allowance is ~850 ms,
         * and n = 4 keeps
         * the deepest wait at 600 ms (750 ms total, ~25% headroom). Rule of
         * thumb: n ≈ target_latency_ms / batch_ms − 1.
         *
         * <p>Independent of the fault-injection {@code queue_depth_limit} gate in
         * enqueueBatch: that one is request-level (pendingRequests, waiting +
         * running) at the RPC entry, this one is batch-level on the pure waiting
         * queue; both stack.
         */
        private boolean schedulePrefillCompletion(List<MockPerformanceModel.RequestShape> shapes,
                                                  long batchId,
                                                  int dpRank) {
            if (shapes.isEmpty()) {
                return true;
            }
            // Shutdown drain in progress — reject before claiming any counter so
            // a racing enqueue (entry stopped-check passed pre-drain) leaves no
            // residue behind the drain's cancel sweep.
            if (shuttingDown) {
                return false;
            }
            // Hard gate on maxPrefillConcurrency (change 2): a batch is either admitted
            // immediately (occupies a concurrency slot) or parked in the pending
            // queue until a running batch finishes. activePrefillBatches is reserved
            // under prefillQueueLock at admission/drain time so admission and the
            // completion-callback drain cannot over-admit into the same slot.
            synchronized (prefillQueueLock) {
                if (activePrefillBatches.get() < maxPrefillConcurrency) {
                    activePrefillBatches.incrementAndGet();
                    activePrefillRequests.addAndGet(shapes.size());
                    pendingRequests.addAndGet(shapes.size());
                    for (MockPerformanceModel.RequestShape shape : shapes) {
                        runningTasks.put(shape.input().getRequestId(),
                                task(shape, batchId, dpRank,
                                        EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    }
                } else {
                    int cap = performance.maxWaitingPrefillBatches();
                    if (cap > 0 && prefillPendingQueue.size() >= cap) {
                        // Waiting-queue cap hit — reject before claiming anything.
                        return false;
                    }
                    prefillPendingQueue.addLast(new PrefillPendingBatch(shapes, batchId, dpRank));
                    waitingPrefillRequests.addAndGet(shapes.size());
                    pendingRequests.addAndGet(shapes.size());
                    for (MockPerformanceModel.RequestShape shape : shapes) {
                        runningTasks.put(shape.input().getRequestId(),
                                task(shape, batchId, dpRank,
                                        EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    }
                    return true;
                }
            }
            runPrefillBatch(shapes, batchId, dpRank);
            return true;
        }

        /**
         * Effective direct-path waiting cap in REQUESTS. The performance JSON
         * cap ({@code prefill.max_waiting_batches}) counts queued batches; a
         * coalesced direct batch holds up to {@code directBatchSizeMax()}
         * requests, so the equivalent request-level cap is the product. 0
         * (unbounded) stays 0.
         */
        private int directWaitingRequestCap() {
            int batchCap = performance.maxWaitingPrefillBatches();
            return batchCap > 0 ? batchCap * performance.directBatchSizeMax() : 0;
        }

        /**
         * Admission point for a direct (generate_stream / NON_BATCH) prefill
         * request. Mirrors {@link #schedulePrefillCompletion} but parks
         * individual REQUESTS (not master-composed batches) and, whenever a
         * concurrency slot is free, coalesces the queued requests with the
         * newcomer into a single batch of up to
         * {@code performance.directBatchSizeMax()} members — production engines
         * run prefill continuous batching, so engine-side drain scales with
         * batch size instead of one request per {@code prefillMs}.
         *
         * <p>Counting contract identical to schedulePrefillCompletion: true →
         * pendingRequests/runningTasks claimed for every member (RECEIVED while
         * queued, RUNNING via {@link #startPrefillBatch}); waitingPrefillRequests
         * counts queued members of both waiting queues and is decremented when
         * a request leaves the queue (drain), not at cancel time.
         *
         * <p>Direct batches carry batchId -1 and dpRank 0 (single-dp mock).
         *
         * @return false when the direct waiting-queue cap is reached — nothing
         *         was claimed and the caller must reject the request.
         */
        private boolean admitDirectPrefill(MockPerformanceModel.RequestShape shape) {
            int maxBatch = performance.directBatchSizeMax();
            List<MockPerformanceModel.RequestShape> merged;
            synchronized (prefillQueueLock) {
                // Shutdown drain in progress — reject before claiming any counter.
                if (shuttingDown) {
                    return false;
                }
                if (activePrefillBatches.get() >= maxPrefillConcurrency) {
                    int cap = directWaitingRequestCap();
                    if (cap > 0 && directPrefillQueue.size() >= cap) {
                        // Waiting-queue cap hit — reject before claiming anything.
                        return false;
                    }
                    directPrefillQueue.addLast(shape);
                    waitingPrefillRequests.incrementAndGet();
                    pendingRequests.incrementAndGet();
                    runningTasks.put(shape.input().getRequestId(),
                            task(shape, -1L, 0, EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                    return true;
                }
                merged = new ArrayList<>(Math.min(maxBatch, directPrefillQueue.size() + 1));
                // Coalesce queued requests with the newcomer: every polled entry
                // leaves the waiting queue (cancelled members have no
                // runningTasks entry and are skipped, matching the BATCH-queue
                // drain's anyAlive semantics).
                while (!directPrefillQueue.isEmpty() && merged.size() < maxBatch - 1) {
                    MockPerformanceModel.RequestShape candidate = directPrefillQueue.pollFirst();
                    waitingPrefillRequests.decrementAndGet();
                    if (runningTasks.containsKey(candidate.input().getRequestId())) {
                        merged.add(candidate);
                    }
                }
                merged.add(shape);
                activePrefillBatches.incrementAndGet();
                activePrefillRequests.addAndGet(merged.size());
                pendingRequests.addAndGet(merged.size());
                for (MockPerformanceModel.RequestShape member : merged) {
                    runningTasks.put(member.input().getRequestId(),
                            task(member, -1L, 0, EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED));
                }
            }
            runPrefillBatch(merged, -1L, 0);
            return true;
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
            // Per-engine prefill busy: one executionMs per scheduled batch (the
            // execution duration is known at schedule time in the mock model).
            busyMs.addAndGet(executionMs);
            // Crash fence: callbacks queued before a crash_after must never
            // touch the wiped state — capture the epoch at schedule time and
            // drop out on mismatch (the late callback cannot be unscheduled).
            final long epoch = crashEpoch.get();
            long startDelayNanos = Math.max(0, startNanos - now);
            if (startDelayNanos == 0) {
                startPrefillBatch(shapes, batchId, dpRank);
            } else {
                scheduler.schedule(() -> {
                    if (crashEpoch.get() == epoch) {
                        startPrefillBatch(shapes, batchId, dpRank);
                    }
                }, startDelayNanos, TimeUnit.NANOSECONDS);
            }

            long delayNanos = Math.max(0, finishNanos - now);
            scheduler.schedule(() -> {
                if (crashEpoch.get() != epoch) {
                    return; // crashed mid-flight: this batch died with the process
                }
                int activeCount = 0;
                // One wall-clock stamp for the whole batch: every member shares
                // the same completion instant (the batch is the execution unit).
                long doneTsMs = System.currentTimeMillis();
                for (MockPerformanceModel.RequestShape shape : shapes) {
                    long requestId = shape.input().getRequestId();
                    boolean alreadyCancelled = cancelledRequests.containsKey(requestId);
                    EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
                    // status_zombie_running: re-insert the entry right after the
                    // removal so this request keeps being reported RUNNING
                    // forever (its completion record is dropped inside
                    // publishCompletion). Every counter below still releases
                    // normally — the zombie poisons only the status report,
                    // not engine capacity.
                    if (faultConfig.isStatusZombieRunning() && removed != null) {
                        runningTasks.put(requestId, removed);
                    }
                    // Only count non-cancelled requests toward pendingRequests
                    // decrement. A cancelled member was re-put to RUNNING by
                    // startPrefillBatch (which loops all shapes), so removed!=null
                    // alone would double-decrement pendingRequests (cancel already
                    // decremented it). The !alreadyCancelled guard fixes this.
                    if (removed != null && !alreadyCancelled) {
                        activeCount++;
                    }
                    recordCompletion(shape, batchId, executionMs, dpRank);
                    // Per-rid prefill terminal event row (engine_events.jsonl,
                    // replaces the former mock_prefill_done stdout trace line —
                    // per-request data now streams as structured JSONL, not
                    // stdout grep). aggregate_canvas_run.py joins prefill_done_ms
                    // / exec_ms against the load client's send_start_epoch_ms by
                    // request_id to bucket prefill exec percentiles on the
                    // request-BIRTH axis (same axis as e2e/full_e2e); exec_ms is
                    // the BATCH execution duration — prefill runs whole batches,
                    // so every member of one batch logs the same value.
                    writePrefillDoneEvent(shape, requestId, batchId, doneTsMs,
                            executionMs, shapes.size(), alreadyCancelled);
                    if (!alreadyCancelled) {
                        // rtp_llm_context_tps accounting (production caliber):
                        // compute = il - hit (actually-computed context tokens,
                        // the rtp_llm_context_tps numerator — cache reuse is
                        // excluded), with_cache = il (the
                        // rtp_llm_context_tps_with_cache numerator, the
                        // DeepSeek-style "input tokens/s incl. cache hits").
                        long inputLen = shape.inputLen();
                        long hitTokens = shape.hitTokens();
                        contextComputeTokens.addAndGet(Math.max(0L, inputLen - hitTokens));
                        contextWithCacheTokens.addAndGet(inputLen);
                        hitTokensTotal.addAndGet(hitTokens);
                    }
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
                    if (decodeStarted) {
                        // Emit a first-token frame (finished=false) so the
                        // client stream loop records firstFrameNanos at prefill
                        // completion. The terminal frame (finished=true) is
                        // emitted later by scheduleDecodeCompletionInternal
                        // after decodeMs elapses. Without this frame ttft and
                        // total collapse to the same nanos value (single-frame
                        // stream), even though the engine really spent decodeMs
                        // producing outputs.
                        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                                responseQueues.get(requestId);
                        if (queue != null) {
                            queue.offer(buildOutput(shape, false));
                        }
                    } else {
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
                    if (alreadyCancelled) {
                        // Cancelled member: blocks return to the pool directly
                        // (no LRU handover — a cancelled request leaves no cache).
                        releaseBlockLease(requestId);
                    } else if (admitBlockLease(requestId, shape)) {
                        cacheVersion.incrementAndGet();
                    }
                }
                activePrefillBatches.decrementAndGet();
                // Mirror the addAndGet(shapes.size()) made when this batch reserved
                // its running slot (admission or drain). Cancelled members stay
                // counted until the batch finishes — the batch keeps executing.
                activePrefillRequests.addAndGet(-shapes.size());
                pendingRequests.addAndGet(-activeCount);
                // Drain one queued batch under the same lock that guards admission,
                // handing this completion's freed slot to a queued batch atomically.
                // Direct-path (generate_stream) requests drain first, coalesced into
                // ONE batch of up to directBatchSizeMax() alive members (production
                // prefill continuous batching); the legacy BATCH queue drains only
                // when no direct batch claimed the slot.
                List<MockPerformanceModel.RequestShape> directBatch = null;
                PrefillPendingBatch nextBatch = null;
                synchronized (prefillQueueLock) {
                    if (activePrefillBatches.get() < maxPrefillConcurrency
                            && !directPrefillQueue.isEmpty()) {
                        int maxBatch = performance.directBatchSizeMax();
                        directBatch = new ArrayList<>(
                                Math.min(maxBatch, directPrefillQueue.size()));
                        while (!directPrefillQueue.isEmpty()
                                && directBatch.size() < maxBatch) {
                            MockPerformanceModel.RequestShape candidate =
                                    directPrefillQueue.pollFirst();
                            waitingPrefillRequests.decrementAndGet();
                            if (runningTasks.containsKey(candidate.input().getRequestId())) {
                                directBatch.add(candidate);
                            }
                        }
                        if (directBatch.isEmpty()) {
                            // Every member was cancelled while queued — the slot
                            // stays free for the BATCH queue below.
                            directBatch = null;
                        } else {
                            activePrefillBatches.incrementAndGet();
                            activePrefillRequests.addAndGet(directBatch.size());
                        }
                    }
                    if (directBatch == null) {
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
                            activePrefillRequests.addAndGet(candidate.shapes().size());
                            waitingPrefillRequests.addAndGet(-candidate.shapes().size());
                            nextBatch = prefillPendingQueue.pollFirst();
                            break;
                        }
                    }
                }
                if (directBatch != null) {
                    runPrefillBatch(directBatch, -1L, 0);
                } else if (nextBatch != null) {
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
                // engine_events.jsonl: execution-start stamp (lane serialization
                // actually begins for this batch member).
                recordEventStart(shape.input().getRequestId());
            }
        }

        private boolean startDecode(MockPerformanceModel.RequestShape shape, long batchId) {
            EngineRpcService.GenerateInputPB input = shape.input();
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue =
                    responseQueues.get(input.getRequestId());
            for (EngineRpcService.RoleAddrPB addr : input.getGenerateConfig().getRoleAddrsList()) {
                if (RoleTypeProtoConverter.fromRoleAddr(addr)
                        != RoleType.DECODE) {
                    continue;
                }
                FastRpcService decode = services.get(addr.getGrpcPort());
                if (decode != null && decode.grpcPort != grpcPort) {
                    // Propagate the decode admission result: true = admitted/queued,
                    // false = backpressure (decode pending queue full). On false the
                    // caller treats decodeStarted=false and delivers the finished
                    // output from the prefill side (degraded but no request lost).
                    registerDecodeOwnership(input.getRequestId(), decode);
                    boolean accepted = decode.scheduleDecodeCompletion(shape, batchId, queue);
                    if (!accepted) {
                        clearDecodeOwnership(input.getRequestId(), decode);
                    }
                    return accepted;
                }
                return false;
            }
            return false;
        }

        /**
         * Admission point for a decode request. Returns true if the request was
         * accepted (scheduled immediately, queued behind the concurrency gate,
         * already scheduled previously, or already cancelled); false only while
         * a shutdown drain is in progress so the caller can degrade.
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
                // Cancel raced ahead of scheduling: bail out before claiming
                // anything. The cancel path has already surfaced the CANCELLED
                // completion/response, so treat the request as accepted-and-
                // cancelled rather than as backpressure.
                if (cancelledRequests.containsKey(requestId)) {
                    return true;
                }
                // Guard: never schedule the same requestId twice on this engine.
                EngineRpcService.TaskInfoPB existing = runningTasks.putIfAbsent(
                        requestId,
                        task(shape, batchId, 0, EngineRpcService.TaskPhase.TASK_PHASE_RUNNING));
                if (existing != null) {
                    return true; // already accepted/scheduled on this engine
                }
                if (activeDecodeRequests.get() < decodeMaxConcurrency) {
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
                    // KV capacity model v2: run start provisions the request's
                    // blocks from the pool (default mode counts KV at run start,
                    // the opt-in accepted-layer mode counts it at admission — both
                    // funnel through the same lease). Admission failure degrades
                    // to un-pooled execution (the request is NOT rejected here:
                    // the decode engine parks/retries, never rejects) and bumps
                    // kvAdmissionFails for overflow observability.
                    // Fix #5: the decode engine re-matches the request's keys
                    // against its OWN LRU here — net demand
                    // ceil(il/spb) − hitBlocks, reused blocks referenced not
                    // re-allocated (production reuse_block_size semantics).
                    if (acquireDecodeBlockLease(requestId, shape) == null) {
                        kvAdmissionFails.increment();
                    }
                    // awaitsFirstStep: if a step tick is already pending (another
                    // stream is running), this stream joins MID-step and first
                    // produces a token at the next boundary — production: a
                    // request arriving during step k participates from step k+1.
                    // When no tick is pending, the arm below prices the step with
                    // this stream already in the count.
                    decodeRunning.put(requestId,
                            new DecodeStream(shape, batchId, responseQueue,
                                    performance.decodeSteps(shape.outputLen()), decodeStepScheduled));
                    stats.decodeAdmitted.increment();
                    recordEventStart(requestId);
                    scheduleDecodeStepLocked();
                } else {
                    // Concurrency gate hit — park the request in the unbounded
                    // pending queue (production waiting_streams_ semantics:
                    // engine waits, never rejects). It will be drained when a
                    // running request completes.
                    // Opt-in accepted-layer window (C1): when
                    // decode.report_queued_as_kv_allocated is enabled, the parked
                    // request is reported as KV_ALLOCATED (KV reserved, not
                    // running yet) so the scheduler's DecodeEndpoint.trackConfirmed
                    // maps it into the accepted layer. Overwrite is safe: we hold
                    // decodeQueueLock and putIfAbsent above claimed the entry.
                    if (performance.reportQueuedAsKvAllocated()) {
                        runningTasks.put(requestId, task(shape, batchId, 0,
                                EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED));
                        // Opt-in KV fidelity (P2-5): a queued request holds its
                        // KV reservation from ENQUEUE — modeled as a real block
                        // lease claimed at park time (exactly once; the drain into
                        // a running slot must not claim it again). Failure degrades
                        // to un-pooled (kvAdmissionFails). Default OFF queued
                        // entries stay uncounted until run start — zero behavior
                        // change. Fix #5: decode-side reuse deduction applies here
                        // too (own-LRU re-match, net demand, referenced reuse).
                        if (acquireDecodeBlockLease(requestId, shape) == null) {
                            kvAdmissionFails.increment();
                        }
                    }
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
         * (completion, top-up, cancel) re-prices the NEXT step. Streams that
         * joined mid-step (awaitsFirstStep) first participate at the next
         * boundary. Streams that exhaust their step budget complete here; the
         * waiting batch is topped up immediately (production top-up semantics),
         * then the next step is chained. Terminal bookkeeping is claimed under
         * decodeQueueLock (racing cancel); the completion frame, stats and
         * cleanup publish outside the lock, mirroring the former one-shot
         * completion callback's split.
         */
        private void runDecodeStep() {
            List<DecodeStream> finished = new ArrayList<>();
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
                for (Iterator<DecodeStream> it = decodeRunning.values().iterator(); it.hasNext(); ) {
                    DecodeStream stream = it.next();
                    if (stream.awaitsFirstStep) {
                        // Joined while this step was already in flight (admission
                        // or cancel-path top-up saw decodeStepScheduled=true): it
                        // did not participate in THIS step — no step advance, no
                        // exec time. It joins the batch at the next boundary,
                        // which scheduleDecodeStepLocked prices with the new count.
                        stream.awaitsFirstStep = false;
                        continue;
                    }
                    // One step = tokensPerStep tokens (MTP fold): the step budget
                    // was pre-computed as ceil(outputLen / tokensPerStep) at
                    // admission, so the tick only decrements whole steps.
                    stream.remainingSteps--;
                    stream.accumulatedExecMs += stepDelayMs;
                    // Per-step KV growth (production incrMalloc): extend the
                    // lease toward ceil((inputLen + generated)/spb) blocks — a
                    // no-op except at spb/tokensPerStep step boundaries. Runs
                    // under decodeQueueLock (lock order: queue -> cache).
                    growDecodeLeaseLocked(stream.shape.input().getRequestId(),
                            stream.shape.inputLen() + (int) Math.ceil(
                                    performance.tokensPerStep() * (stream.totalSteps - stream.remainingSteps)));
                    if (stream.remainingSteps <= 0) {
                        it.remove();
                        finished.add(stream);
                    }
                }
                for (DecodeStream stream : finished) {
                    claimDecodeTerminalLocked(stream);
                }
                topUpDecodeRunningLocked();
                scheduleDecodeStepLocked();
            }
            for (DecodeStream stream : finished) {
                if (stream.owned) {
                    publishDecodeCompletion(stream);
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
            if (decodeStepScheduled || decodeRunning.isEmpty() || shuttingDown) {
                return;
            }
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
            while (!decodePendingQueue.isEmpty()
                    && activeDecodeRequests.get() < decodeMaxConcurrency) {
                DecodePendingTask candidate = decodePendingQueue.peekFirst();
                long candidateId = candidate.shape().input().getRequestId();
                if (!runningTasks.containsKey(candidateId)) {
                    decodePendingQueue.pollFirst();
                    continue;
                }
                decodePendingQueue.pollFirst();
                activeDecodeRequests.incrementAndGet();
                // Run-start bookkeeping, mirroring the immediate-admission path:
                // default mode provisions blocks at run start; the opt-in
                // accepted-layer mode claimed them at enqueue and only needs the
                // phase flip. A queued request whose opt-in claim degraded takes
                // its second chance here. Fix #5: decode-side reuse deduction
                // (own-LRU re-match at run start, net demand, referenced reuse).
                if (activeBlockLeases.get(candidateId) == null) {
                    if (acquireDecodeBlockLease(candidateId, candidate.shape()) == null) {
                        kvAdmissionFails.increment();
                    }
                }
                if (performance.reportQueuedAsKvAllocated()) {
                    runningTasks.computeIfPresent(candidateId, (id, tracked) ->
                            tracked.getPhase() == EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED
                                    ? tracked.toBuilder()
                                            .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING)
                                            .build()
                                    : tracked);
                }
                // awaitsFirstStep = decodeStepScheduled: from inside a tick the
                // flag is already false (the tick resets it first), so a boundary
                // top-up joins the next armed step immediately; from cancel-path
                // top-up a tick may be pending, and the promoted request waits
                // for that boundary like a mid-step admission.
                decodeRunning.put(candidateId,
                        new DecodeStream(candidate.shape(), candidate.batchId(),
                                candidate.responseQueue(),
                                performance.decodeSteps(candidate.shape().outputLen()),
                                decodeStepScheduled));
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
        private void claimDecodeTerminalLocked(DecodeStream stream) {
            long requestId = stream.shape.input().getRequestId();
            clearUpstreamOwnership(requestId);
            EngineRpcService.TaskInfoPB removed = runningTasks.remove(requestId);
            if (removed == null) {
                return; // cancel won the terminal race; it released everything
            }
            // status_zombie_running: re-insert the entry after the removal so
            // this request keeps being reported RUNNING forever (its completion
            // record is dropped inside publishCompletion); the slot/KV/pending
            // counters below still release normally so the engine keeps
            // admitting — the zombie poisons only the status report.
            if (faultConfig.isStatusZombieRunning()) {
                runningTasks.put(requestId, removed);
            }
            stream.owned = true;
            stream.terminalBatchSize = decodeRunning.size() + 1;
            activeDecodeRequests.decrementAndGet();
            // Capacity model v2: the lease is handed to the LRU OUTSIDE this
            // lock by publishDecodeCompletion (admitBlockLease) — release !=
            // delete — so cancelling cannot double-release it.
            pendingRequests.decrementAndGet();
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
            recordCompletion(shape, stream.batchId, executionMs, 0);
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
            writeDecodeDoneEvent(stream, executionMs,
                    cancelledRequests.containsKey(requestId));
            // Per-engine decode busy: one executionMs per completed request.
            busyMs.addAndGet(executionMs);
            boolean alreadyCancelled = cancelledRequests.containsKey(requestId);
            recordLifecycleEnd(requestId, alreadyCancelled);
            if (!alreadyCancelled) {
                completedCount.incrementAndGet();
                requestStates.put(requestId, "completed");
                // rtp_llm_generate_tps accounting (production caliber): the
                // numerator is the stream's accepted output token count
                // (the MTP fold), not the decode batch size.
                generateTokens.addAndGet(shape.outputLen());
            }
            // Python compat (_run_decode): no_respond on the decode engine only
            // suppresses the intermediate first-step output; the finished output
            // is still delivered, so keep this unconditional.
            if (stream.responseQueue != null && !alreadyCancelled) {
                stream.responseQueue.offer(buildOutput(shape, true));
            }
            responseQueues.remove(requestId);
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
            } else if (admitBlockLease(requestId, shape)) {
                cacheVersion.incrementAndGet();
            }
        }

        /** Snapshot size of the decode pending queue (for waitingQueryLen reporting). */
        private int decodePendingQueueSize() {
            synchronized (decodeQueueLock) {
                return decodePendingQueue.size();
            }
        }

        /** Snapshot size of the prefill pending queue in BATCHES (cap accounting unit). */
        private int prefillPendingQueueSize() {
            synchronized (prefillQueueLock) {
                return prefillPendingQueue.size();
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

        private void publishCompletion(EngineRpcService.TaskInfoPB task) {
            synchronized (completionLock) {
                // status_zombie_running: drop the completion record entirely —
                // the request finished internally but is never reported
                // finished (paired with the runningTasks re-insert at the
                // completion points).
                if (faultConfig.isStatusZombieRunning()) {
                    return;
                }
                long version = completionVersion.incrementAndGet();
                completions.add(new VersionedTask(version, task));
                // status_duplicate_finished: enqueue the SAME completion twice
                // under the SAME version — one poll reports the rid twice, and
                // advancing the cursor past that version consumes both copies.
                if (faultConfig.isStatusDuplicateFinished()) {
                    completions.add(new VersionedTask(version, task));
                }
            }
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
            // Capacity model v2 + pressure consistency: available derives from
            // the block pool (free + pure-LRU blocks count) minus injected
            // pressure, clamped to the reported total — the SAME caliber as
            // WorkerStatus.availableKvCache and /snapshot available_kv_tokens
            // (previously getCacheStatus ignored kv_pressure while WorkerStatus
            // subtracted it).
            EngineRpcService.CacheStatusPB.Builder status = EngineRpcService.CacheStatusPB.newBuilder()
                    .setAvailableKvCache(availableKvTokens())
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

        /**
         * Force-evict block keys from this engine's LRU (control-plane POST
         * /cache_evict — the flexlb_ft KV family's forced-eviction hook).
         * Idempotent: evicting keys that are not present is a no-op. When
         * the key set changes the engine's cacheVersion is bumped, so the
         * master's next cache-status poll re-pulls the key set and its
         * global key→holder index converges on the eviction (the test
         * family's sync premise).
         *
         * @return true when at least one key was actually removed
         */
        boolean evictCacheKeys(List<Long> keys) {
            boolean changed = cache.evict(keys);
            if (changed) {
                cacheVersion.incrementAndGet();
            }
            return changed;
        }

        // ──────────── KV capacity model v2: block-pool helpers ────────────

        /**
         * A request's total block demand: the hash-channel block count when the
         * request carries block_cache_keys, else ceil(inputLen/spb) computed live
         * (the ~22% empty-bh share of the production trace).
         */
        private int needBlocks(MockPerformanceModel.RequestShape shape) {
            List<Long> keys = shape.blockKeys();
            if (!keys.isEmpty()) {
                return keys.size();
            }
            return (shape.inputLen() + seqSizePerBlock - 1) / seqSizePerBlock;
        }

        /**
         * Try to provision {@code shape}'s blocks from the pool (TOTAL_AND_AVAILABLE
         * gate + LRU-tail eviction coupling). On success the lease is registered in
         * {@code activeBlockLeases} so completion/cancel can hand it back.
         *
         * @return the lease, or null = LACK_MEM (nothing claimed — callers either
         *         reject the request or degrade with a kvAdmissionFails bump)
         */
        private MockLruBlockCache.BlockLease acquireBlockLease(long requestId,
                                                                MockPerformanceModel.RequestShape shape) {
            MockLruBlockCache.BlockLease lease = cache.acquire(needBlocks(shape), shape.blockKeys());
            if (lease == null) {
                return null;
            }
            activeBlockLeases.put(requestId, lease);
            return lease;
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
         * hitBlocks + netNew = ceil(inputLen/spb), so the per-step growth toward
         * ceil((inputLen+generated)/spb) only ever allocates the GENERATED
         * increment — the reuse deduction persists for the stream's whole
         * life (the old keys.size-caliber lease started below the token
         * demand and back-filled the uncovered suffix at the first grow
         * boundary, diluting the deduction whenever the pool ran dry).
         *
         * @return the lease (registered in {@code activeBlockLeases}), or null
         *         = LACK_MEM (caller degrades to un-pooled + kvAdmissionFails)
         */
        private MockLruBlockCache.BlockLease acquireDecodeBlockLease(
                long requestId, MockPerformanceModel.RequestShape shape) {
            int totalDemand = (shape.inputLen() + seqSizePerBlock - 1) / seqSizePerBlock;
            MockLruBlockCache.BlockLease lease =
                    cache.acquireWithReuse(totalDemand, shape.blockKeys());
            if (lease == null) {
                return null;
            }
            // Reuse observability (KV v2 fix #5): the hit keys are exactly the
            // blocks this decode admission did NOT re-allocate — the
            // net-demand deduction, accumulated for the /metrics counter.
            decodeReuseBlocks.add(lease.hitKeys.size());
            activeBlockLeases.put(requestId, lease);
            return lease;
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
            return cache.admit(lease, shape.blockKeys());
        }

        /** Cancel path: return the lease's blocks to the pool without LRU handover. */
        private void releaseBlockLease(long requestId) {
            MockLruBlockCache.BlockLease lease = activeBlockLeases.remove(requestId);
            if (lease != null) {
                cache.release(lease);
            }
        }

        /**
         * Per-step decode growth (production incrMalloc): extend the running
         * request's allocation toward ceil((inputLen+grown)/spb) blocks. Free
         * blocks first, LRU-tail eviction second; on exhaustion the growth
         * stalls (counted in kvAdmissionFails) — the request keeps running with
         * its current allocation rather than being aborted.
         */
        private void growDecodeLeaseLocked(long requestId, int totalTokensSoFar) {
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            if (lease == null) {
                return; // un-pooled degraded request (admission failed earlier)
            }
            int targetBlocks = (totalTokensSoFar + seqSizePerBlock - 1) / seqSizePerBlock;
            while (lease.totalBlocks() < targetBlocks) {
                if (!cache.grow(lease)) {
                    kvAdmissionFails.increment();
                    return;
                }
            }
        }

        /** Tokens pinned by in-flight requests: (held + referenced key blocks) x spb. */
        private long occupiedKvTokens() {
            return (long) (cache.heldBlocks() + cache.referencedKeyBlocks()) * seqSizePerBlock;
        }

        /** Tokens available per the pool (free + pure-LRU blocks count) — LRU included. */
        private long poolAvailableKvTokens() {
            return (long) cache.availableBlocks() * seqSizePerBlock;
        }

        /**
         * Master-facing available tokens: pool availability clamped to the reported
         * total, minus injected KV pressure. The clamp keeps total/spb non-divisible
         * configs reporting at most totalKvTokens when idle (legacy compatibility).
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
            return cacheVersion.get();
        }

        @Override
        public void checkHealth(EngineRpcService.EmptyPB request,
                                StreamObserver<EngineRpcService.CheckHealthResponsePB> observer) {
            observer.onNext(EngineRpcService.CheckHealthResponsePB.newBuilder().setHealth("OK").build());
            observer.onCompleted();
        }

        /**
         * gRPC Cancel (proto {@code RpcService/Cancel}, the priority-preemption
         * engine contract). Mirrors the in-process MockEngineCancelChannel and
         * the HTTP control-plane {@code /cancel_request}: a live request and its
         * accepted-cancel tombstone both return ACCEPTED (idempotent retry,
         * matching the Python mock's {@code _cancelled} fast path), a request
         * unknown to — or already finished on — this specifically addressed
         * Prefill returns NOT_FOUND, and Decode rejects the RPC with
         * UNIMPLEMENTED (production role contract). TOMBSTONED stays reserved
         * for the production engine; every mock cancel channel (in-process,
         * HTTP, and this gRPC handler) maps the three-branch CancelResult with
         * found -> accepted so the Master-side settlement semantics stay
         * identical across transports.
         */
        @Override
        public void cancel(EngineRpcService.CancelRequestPB request,
                           StreamObserver<EngineRpcService.CancelResponsePB> observer) {
            try {
                CancelResult result = cancelRequest(request.getRequestId());
                observer.onNext(EngineRpcService.CancelResponsePB.newBuilder()
                        .setStatus(result.found()
                                ? EngineRpcService.CancelStatusPB.CANCEL_STATUS_ACCEPTED
                                : EngineRpcService.CancelStatusPB.CANCEL_STATUS_NOT_FOUND)
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
            // engine_events.jsonl stamp safety net: a cancelled-in-queue request
            // that never reached a terminal callback leaves its arrival/start
            // stamps behind. A rid is truly gone once it is neither a running
            // task nor has a live response queue (queued prefill/decode members
            // always hold a runningTasks entry).
            eventArrivalMs.keySet().removeIf(id ->
                    !runningTasks.containsKey(id) && !responseQueues.containsKey(id));
            eventStartMs.keySet().removeIf(id ->
                    !runningTasks.containsKey(id) && !responseQueues.containsKey(id));
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
            stopped = true; // same rejection semantics as the control-plane /stop_engine
            // Cancel sweep. A pass can promote queued decode tasks into running
            // slots (cancel's slot hand-off) and a racing cross-engine hand-off
            // may slip in before the shuttingDown guard was observed, so retry
            // a bounded number of passes until runningTasks is empty.
            for (int pass = 0; pass < 3 && !runningTasks.isEmpty(); pass++) {
                for (Long requestId : List.copyOf(runningTasks.keySet())) {
                    cancel(requestId);
                }
            }
            // Direct-path parked requests were cancelled above (their
            // runningTasks entries are gone); drop the shapes themselves so a
            // post-shutdown snapshot sees an empty queue.
            synchronized (prefillQueueLock) {
                directPrefillQueue.clear();
            }
            // Queued prefill batches: cancel() removed every member's
            // runningTasks entry (and decremented pendingRequests) but by design
            // leaves the batch parked for the completion drain, which will never
            // fire once the scheduler stops. Drop dead batches with the same
            // anyAlive bookkeeping as the completion drain; any member still
            // alive (raced past the sweep) goes through cancel() first so no
            // counter is decremented twice.
            synchronized (prefillQueueLock) {
                while (!prefillPendingQueue.isEmpty()) {
                    PrefillPendingBatch batch = prefillPendingQueue.pollFirst();
                    for (MockPerformanceModel.RequestShape shape : batch.shapes()) {
                        if (runningTasks.containsKey(shape.input().getRequestId())) {
                            cancel(shape.input().getRequestId());
                        }
                    }
                    waitingPrefillRequests.addAndGet(-batch.shapes().size());
                }
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
         * closes the port but deliberately KEEPS every pool and queue, so the
         * engine resumes in place once restarted — a network-level outage, not
         * a process death. {@link #drainAndShutdown()} is the graceful process
         * EXIT path and cancels everything through the normal terminal
         * machinery instead of discarding it.
         */
        void crashNow() {
            stopped = true;
            crashEpoch.incrementAndGet();
            runningTasks.clear();
            responseQueues.clear();
            requestStates.clear();
            activeBlockLeases.clear();
            cancelledRequests.clear();
            // Queued work dies with the process: pending prefill batches,
            // direct-stream parks, decode wait queue and running streams.
            synchronized (prefillQueueLock) {
                prefillPendingQueue.clear();
                directPrefillQueue.clear();
            }
            synchronized (decodeQueueLock) {
                decodePendingQueue.clear();
                decodeRunning.clear();
                decodeStepScheduled = false;
                pendingStepDelayMs = 0;
            }
            // Un-acked completion backlog dies too: finished-but-unreported
            // work is lost, the master's poller will never see it again.
            synchronized (completionLock) {
                completions.clear();
            }
            // KV memory: every held block and LRU entry is gone with the process.
            cache.clear();
            cacheVersion.incrementAndGet();
            statusVersion.incrementAndGet();
            // Observability histories are process memory as well.
            synchronized (requestLifecycles) {
                requestLifecycles.clear();
            }
            synchronized (cancelledRidHistory) {
                cancelledRidHistory.clear();
            }
            synchronized (priorityCancelTombstones) {
                priorityCancelTombstones.clear();
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
            // Admission gauges: nothing is queued or running on a dead process.
            pendingRequests.set(0);
            waitingPrefillRequests.set(0);
            activePrefillBatches.set(0);
            activePrefillRequests.set(0);
            activeDecodeRequests.set(0);
            // Fresh process: lane time axes start from "available now" and the
            // observability counters restart from zero.
            nextPrefillAvailableNanosByDp.clear();
            AtomicLong[] lanes = prefillLanes;
            if (lanes != null) {
                for (AtomicLong lane : lanes) {
                    lane.set(0L);
                }
            }
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
            contextComputeTokens.set(0);
            contextWithCacheTokens.set(0);
            generateTokens.set(0);
            hitTokensTotal.set(0);
            lastWindowContextCompute.set(0);
            lastWindowContextCache.set(0);
            lastWindowGenerate.set(0);
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
            responseExecutor.shutdownNow();
        }

        // ──────────── Getters and setters for MockControlServer ────────────

        FaultInjectionConfig getFaultConfig() { return faultConfig; }
        void setFaultConfig(FaultInjectionConfig config) { this.faultConfig = config; }
        void clearFaultConfig() { this.faultConfig = FaultInjectionConfig.builder().build(); }
        void resetEnqueueCount() { this.enqueueCount.set(0); }
        void setStopped(boolean s) { this.stopped = s; }
        void setGrpcServer(Server server) { this.grpcServer = server; }
        long getCrashEpoch() { return crashEpoch.get(); }
        boolean isStopped() { return stopped; }
        int getGrpcPort() { return grpcPort; }
        int getDownstreamOwnershipCount() { return downstreamDecodeOwners.size(); }
        int getUpstreamOwnershipCount() { return upstreamPrefillOwners.size(); }
        boolean hasDownstreamOwnership(long requestId) { return downstreamDecodeOwners.containsKey(requestId); }
        boolean hasUpstreamOwnership(long requestId) { return upstreamPrefillOwners.containsKey(requestId); }
        String getRoleName() { return roleName; }
        String getEngineName() { return engineName; }
        String getHost() { return host; }
        long getTotalKvTokens() { return totalKvTokens; }
        int getMaxPrefillConcurrency() { return maxPrefillConcurrency; }
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
        /** Blocks currently pinned by in-flight leases (held + referenced). */
        long getOccupiedKvTokens() { return occupiedKvTokens(); }
        /** Pool availability (free + pure-LRU) clamped to total, minus pressure. */
        long getAvailableKvTokens() { return availableKvTokens(); }
        /** Total pool blocks (ceil(totalKvTokens/spb) or explicit override). */
        int getCacheBlocks() { return cache.totalBlocks(); }
        /** spb — the pool's token<->block conversion factor (reported as block_size). */
        int getSeqSizePerBlock() { return seqSizePerBlock; }
        /** Count of KV admission/growth failures (LACK_MEM degradations + stalls). */
        long getKvAdmissionFails() { return kvAdmissionFails.sum(); }
        boolean isLeakDetected() { return leakDetected.get(); }
        boolean isShuttingDown() { return shuttingDown; }
        int getActiveDecodeCount() { return activeDecodeRequests.get(); }
        int getActivePrefillBatchCount() { return activePrefillBatches.get(); }
        int getDecodePendingQueueDepth() { return decodePendingQueueSize(); }
        int getPrefillPendingQueueDepth() { return prefillPendingQueueSize(); }
        Map<Long, String> getRequestStates() { return requestStates; }

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

        private void addPriorityCancelTombstone(long requestId) {
            synchronized (priorityCancelTombstones) {
                priorityCancelTombstones.add(requestId);
                while (priorityCancelTombstones.size() > CANCELLED_RID_CAP) {
                    var iterator = priorityCancelTombstones.iterator();
                    if (!iterator.hasNext()) {
                        break;
                    }
                    iterator.next();
                    iterator.remove();
                }
            }
        }

        private boolean hasPriorityCancelTombstone(long requestId) {
            synchronized (priorityCancelTombstones) {
                return priorityCancelTombstones.contains(requestId);
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
                Map<String, Object> lifecycle = requestLifecycles.get(requestId);
                if (lifecycle != null && "running".equals(lifecycle.get("end_state"))) {
                    lifecycle.put("running_ms", System.currentTimeMillis());
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

        /** Stamp the engine-side arrival epoch-ms for engine_events.jsonl (first arrival wins). */
        private void recordEventArrival(long requestId) {
            if (engineEventLog == null) {
                return;
            }
            eventArrivalMs.putIfAbsent(requestId, System.currentTimeMillis());
        }

        /** Stamp the execution-start epoch-ms (prefill batch start / decode slot admission). */
        private void recordEventStart(long requestId) {
            if (engineEventLog == null) {
                return;
            }
            eventStartMs.putIfAbsent(requestId, System.currentTimeMillis());
        }

        private static long orZero(Long value) {
            return value != null ? value : 0L;
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
            long arrivalMs = orZero(eventArrivalMs.remove(requestId));
            long startMs = orZero(eventStartMs.remove(requestId));
            if (log == null) {
                return;
            }
            ObjectNode row = OBJECT_MAPPER.createObjectNode();
            row.put("event", "prefill_done");
            row.put("rid", requestId);
            row.put("engine_name", engineName);
            row.put("batch_id", batchId);
            row.put("engine_arrival_ms", arrivalMs);
            row.put("prefill_start_ms", startMs);
            row.put("prefill_done_ms", doneTsMs);
            row.put("ttft_ms", arrivalMs > 0 ? Math.max(0L, doneTsMs - arrivalMs) : 0L);
            row.put("exec_ms", executionMs);
            row.put("batch_size", batchSize);
            row.put("input_len", shape.inputLen());
            row.put("cache_hit_tokens", shape.hitTokens());
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            row.put("kv_used_tokens",
                    lease != null ? (long) lease.totalBlocks() * seqSizePerBlock : 0L);
            row.put("cancelled", cancelled);
            log.write(row);
        }

        /**
         * engine_events.jsonl terminal row for one decode stream — the
         * structured replacement of the former mock_decode_done stdout line.
         * exec_ms is the summed booked step durations (per-step continuous
         * batching caliber); batch_size is the terminal-step running batch
         * (claimed under decodeQueueLock, includes this stream); kv_used_tokens
         * reads the still-live block lease (the lease hands over to the LRU
         * only AFTER this row, on the normal path).
         */
        private void writeDecodeDoneEvent(DecodeStream stream, long executionMs, boolean cancelled) {
            MockPerformanceModel.RequestShape shape = stream.shape;
            long requestId = shape.input().getRequestId();
            EngineEventLog log = engineEventLog;
            long arrivalMs = orZero(eventArrivalMs.remove(requestId));
            long startMs = orZero(eventStartMs.remove(requestId));
            if (log == null) {
                return;
            }
            ObjectNode row = OBJECT_MAPPER.createObjectNode();
            row.put("event", "decode_done");
            row.put("rid", requestId);
            row.put("engine_name", engineName);
            row.put("batch_id", stream.batchId);
            row.put("engine_arrival_ms", arrivalMs);
            row.put("decode_start_ms", startMs);
            row.put("decode_done_ms", System.currentTimeMillis());
            row.put("exec_ms", executionMs);
            row.put("batch_size", stream.terminalBatchSize);
            row.put("output_len", shape.outputLen());
            MockLruBlockCache.BlockLease lease = activeBlockLeases.get(requestId);
            row.put("kv_used_tokens",
                    lease != null ? (long) lease.totalBlocks() * seqSizePerBlock : 0L);
            row.put("cancelled", cancelled);
            log.write(row);
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

        /**
         * Settle the per-scrape TPS windows (rtp_llm_* series): the /metrics
         * handler calls this on EVERY scrape before reading snapshots, so a
         * window = one scrape interval (1s for the G1 poller — the drained
         * value is tokens-per-second by construction). Events landing
         * between this drain and the snapshot read roll into the next
         * window via the pending counters added back in getSnapshot().
         */
        void drainTpsWindows() {
            lastWindowContextCompute.set(contextComputeTokens.getAndSet(0));
            lastWindowContextCache.set(contextWithCacheTokens.getAndSet(0));
            lastWindowGenerate.set(generateTokens.getAndSet(0));
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
            // Capacity model v2: one pool-derived caliber everywhere —
            // active = occupied + pressure (clamped), available = pool
            // availability (free + pure-LRU, LRU counts as available) clamped
            // to total minus pressure. cache_keys growth shows up as occupancy
            // only while requests are RUNNING; parked LRU keys restore
            // availability, matching the production master's view.
            long effectiveActiveKv = usedKvTokens();
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
            // Queued prefill batches (same unit as prefill.max_waiting_batches, for
            // cap observation). Requests-vs-batches: "waiting" above counts requests.
            if (roleType == EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL) {
                snap.put("prefill_waiting_batches", prefillPendingQueueSize());
            }
            snap.put("accepted", acceptedCount.get());
            snap.put("completed", completedCount.get());
            snap.put("cache_keys", cache.snapshotKeys().size());
            snap.put("cache_evictions", cache.evictions());
            // Full per-engine key list (flexlb_ft KV cases' per-engine key-set
            // exposure). Debug endpoint — the whole list is exposed by design
            // (default 6000-block caches included); sorted for determinism.
            List<Long> cacheKeySet = new ArrayList<>(cache.snapshotKeys());
            cacheKeySet.sort(Long::compareTo);
            snap.put("cache_key_set", cacheKeySet);
            snap.put("active_kv_tokens", effectiveActiveKv);
            snap.put("available_kv_tokens", availableKvTokens());
            // Pool observability (per-engine series, /metrics passthrough):
            // spb and the block-level split of the same capacity model.
            snap.put("total_kv_tokens", totalKvTokens);
            snap.put("block_size", seqSizePerBlock);
            snap.put("cache_blocks", cache.totalBlocks());
            snap.put("available_blocks", cache.availableBlocks());
            snap.put("held_blocks", cache.heldBlocks());
            snap.put("referenced_blocks", cache.referencedKeyBlocks());
            snap.put("kv_admission_fails", kvAdmissionFails.sum());
            snap.put("lack_mem_rejects", prefillLackMemRejects.sum());
            snap.put("decode_reuse_blocks", decodeReuseBlocks.sum());
            snap.put("cache_key_hits", cacheKeyHits.sum());
            snap.put("cache_keys_requested", cacheKeysRequested.sum());
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
            // Cumulative per-engine busy time (ms): prefill batches (resp. decode
            // requests) executed by this engine — see busyMs field comment.
            snap.put("busy_ms", busyMs.get());
            // Production-caliber TPS observation: the rtp_llm_* /metrics
            // series read these. Window value = last settled scrape window +
            // events since (the /metrics handler drains first, so a scrape
            // reads exactly its own window; /snapshot sees the in-progress
            // window too). hit_tokens_total is cumulative cache-reuse
            // accounting (the cache_saved_tokens source).
            snap.put("context_tps",
                    lastWindowContextCompute.get() + contextComputeTokens.get());
            snap.put("context_tps_with_cache",
                    lastWindowContextCache.get() + contextWithCacheTokens.get());
            snap.put("generate_tps",
                    lastWindowGenerate.get() + generateTokens.get());
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
        private record PrefillPendingBatch(
                List<MockPerformanceModel.RequestShape> shapes,
                long batchId,
                int dpRank) {
        }
    }

    /**
     * Result of {@link FastRpcService#cancelRequest(long)}: mirrors the three
     * branches of the v3 EngineCancelChannel contract — accepted (live or an
     * idempotent priority-cancel tombstone), already finished before the first
     * cancel arrived, or unknown to this engine.
     */
    record CancelResult(boolean found, EngineRpcService.TaskPhase phase, boolean alreadyFinished) {
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
        // distribute across tracked / already-finished / unknown / tombstone.
        // B1 on the master side should stay close to zero once the cancel
        // setBatchId fix lands; a sustained B1 ~= tracked count means typed
        // CANCELLED terminals are still being dropped in reconcile.
        final LongAdder cancelCensusTracked = new LongAdder();
        final LongAdder cancelCensusAlreadyFinished = new LongAdder();
        final LongAdder cancelCensusUnknown = new LongAdder();
        final LongAdder cancelCensusTombstone = new LongAdder();
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
        // Package-private for direct assertions in ClusterConfigParamTest.
        int nPrefill = 2;
        int nDecode = 4;
        int baseGrpcPort = 61_000;
        int eventLoopThreads = 32;
        int completionThreads = 8;
        /**
         * Block-count pool overrides (capacity model v2): 0 = derive the pool
         * from the per-role token capacity (ceil(totalKvTokens/spb)). The legacy
         * flag NAMES are kept (run_online_eval.sh L861-862 / lib_load_client.sh /
         * harness.py still pass them) but the MEANING changed from "max cache
         * keys" to "total pool blocks" — a non-zero value overrides derivation.
         */
        int prefillCacheBlocks = 0;
        int decodeCacheBlocks = 0;
        String host = "127.0.0.1";
        String prefillDomain = "mock.prefill.hosts.address";
        String decodeDomain = "mock.decode.hosts.address";
        String endpointFile;
        String envFile;
        String discoveryFile;
        String performanceFile;
        String masterConfigFile;
        /** Legacy uniform token-capacity knob — applies to BOTH per-role pools when set. */
        long totalKvTokens = DEFAULT_TOTAL_KV_TOKENS;
        /**
         * Per-role token capacities (capacity model v2, heterogeneous defaults):
         * prefill 6,291,456 (6144 blocks x 1024 spb) vs decode 4,194,304 (2/3)
         * — decode engines hold each request's KV for its whole life, so the
         * smaller pool lets the master's cross-engine comparisons (min
         * kvCacheUsed / KV% gates) exercise real per-role divergence.
         */
        long prefillTotalKvTokens = DEFAULT_TOTAL_KV_TOKENS;
        long decodeTotalKvTokens = DEFAULT_DECODE_TOTAL_KV_TOKENS;
        int blockSize = 0;
        int decodeMaxConcurrency = DEFAULT_DECODE_MAX_CONCURRENCY;
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
                    case "--completion-threads" -> config.completionThreads = Integer.parseInt(value);
                    case "--prefill-cache-blocks" -> config.prefillCacheBlocks = Integer.parseInt(value);
                    case "--decode-cache-blocks" -> config.decodeCacheBlocks = Integer.parseInt(value);
                    case "--host" -> config.host = value;
                    case "--prefill-domain" -> config.prefillDomain = value;
                    case "--decode-domain" -> config.decodeDomain = value;
                    case "--endpoint-file" -> config.endpointFile = value;
                    case "--env-file" -> config.envFile = value;
                    case "--discovery-file" -> config.discoveryFile = value;
                    case "--performance" -> config.performanceFile = value;
                    case "--master-config" -> config.masterConfigFile = value;
                    // --total-kv-tokens stays the uniform knob: sets BOTH
                    // per-role pools (Python compat: one number, both roles).
                    case "--total-kv-tokens" -> {
                        config.totalKvTokens = Long.parseLong(value);
                        config.prefillTotalKvTokens = config.totalKvTokens;
                        config.decodeTotalKvTokens = config.totalKvTokens;
                    }
                    case "--prefill-total-kv-tokens" -> config.prefillTotalKvTokens = Long.parseLong(value);
                    case "--decode-total-kv-tokens" -> config.decodeTotalKvTokens = Long.parseLong(value);
                    case "--block-size" -> config.blockSize = Integer.parseInt(value);
                    case "--decode-max-concurrency" -> config.decodeMaxConcurrency = Integer.parseInt(value);
                    case "--stats-interval-ms" -> config.statsIntervalMs = Integer.parseInt(value);
                    case "--events-file" -> config.eventsFile = value;
                    case "--stats-stdout" -> config.statsStdout = parseBooleanFlag(value, "--stats-stdout");
                    case UNIQUE_ENGINE_IPS_FLAG -> config.uniqueEngineIps =
                            parseBooleanFlag(value, UNIQUE_ENGINE_IPS_FLAG);
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
            if (config.statsIntervalMs < 1) {
                throw new IllegalArgumentException("--stats-interval-ms must be >= 1");
            }
            return config;
        }
    }
}
