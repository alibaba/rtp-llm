package org.flexlb.mock.grpc;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.scheduler.QueueRoutingResult;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.FileServiceDiscovery;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockPrefillWorker;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * End-to-end validation of dynamic service discovery through the REAL master
 * pipeline: FileServiceDiscovery (file re-read per poll) → WorkerAddressService
 * (domain → hosts, http→grpc conversion) → EngineSyncRunner (20 ms re-pull,
 * getOrCreateWorkerStatus, eviction of vanished entries) → GrpcWorkerStatusRunner
 * (gRPC status check) → EndpointRegistry.ensureEndpoint (routing candidate set)
 * → round-robin Router over the live endpoint set.
 *
 * <p>Flow (mirrors the required acceptance scenario):
 * <ol>
 *   <li>Start with workers A + B listed in the discovery file; assert the sync
 *       loop polls both and requests are distributed across them.</li>
 *   <li>Start worker C — deliberately NOT registered anywhere — then atomically
 *       rewrite the discovery file (tmp + ATOMIC_MOVE, exactly like
 *       {@code DiscoveryFileStore}) to include C. Keep sending requests; assert
 *       C starts receiving them (poll until convergence, hard cap 10 s).</li>
 *   <li>Atomically rewrite the file WITHOUT worker A. Assert A is evicted from
 *       the EndpointRegistry and receives no further requests, while B + C
 *       keep serving.</li>
 * </ol>
 *
 * <p>The only mocked master-side pieces are the metrics reporter
 * ({@link EngineHealthReporter}) and the cache-aware manager — neither
 * participates in the discovery/registration decision path.
 */
class FileDiscoveryDynamicScaleEndToEndTest extends FlexLBMockTestBase {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String PREFILL_DOMAIN = "mock.prefill.hosts.address";
    private static final String DECODE_DOMAIN = "mock.decode.hosts.address";
    private static final String MODEL_NAME = "mock-model";
    /** Master SYNC_STATUS_INTERVAL default — the production re-pull cadence. */
    private static final long SYNC_INTERVAL_MS = 20L;
    private static final long CONVERGENCE_CAP_MS = 10_000L;
    /**
     * Upstream's schedule refactor made disappearance eviction
     * staleness-gated (retireMissingGenerationIfExpired): a worker dropped
     * from discovery is retired only once its last successful status poll
     * is older than this threshold — the 15-arg constructor defaults to
     * 10 s, which would land exactly on Phase 3's 10 s convergence cap.
     * The 16-arg constructor exists to inject a short staleness so this
     * test verifies the eviction mechanism without waiting out the
     * production TTL.
     */
    private static final long STATUS_STALE_AFTER_US = 500_000L;

    @TempDir
    Path tempDir;

    private MockPrefillWorker workerB;
    private MockPrefillWorker workerC;
    private Path discoveryFile;
    private FileServiceDiscovery fileServiceDiscovery;
    private WorkerAddressService workerAddressService;
    private EngineGrpcService engineGrpcService;
    private EngineHealthReporter healthReporter;
    private ScheduledExecutorService syncScheduler;
    private ExecutorService statusCheckExecutor;
    private final AtomicInteger routerCounter = new AtomicInteger();
    private final AtomicLong requestIdCounter = new AtomicLong(810_000L);

    @BeforeEach
    void setUpFileDiscoveryPipeline() throws Exception {
        // Second initial prefill worker (B), started by the base helper.
        workerB = addPrefillWorker(MockWorkerBehavior.builder().build());

        // Initial discovery file: A + B on the prefill domain, base decode worker.
        discoveryFile = tempDir.resolve("discovery-" + System.nanoTime() + ".json");
        writeDiscoveryFileAtomic(List.of(prefillIpPort, workerIpPort(workerB)));

        // Real file-backed ServiceDiscovery — re-reads the file on every poll.
        fileServiceDiscovery = new FileServiceDiscovery(discoveryFile.toString());

        // Model topology: upstream builds ModelMetaConfig from the
        // MODEL_SERVICE_CONFIG env; mocking it keeps this test hermetic while
        // still resolving the domain → address mapping the sync loop consumes.
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress(PREFILL_DOMAIN);
        prefillEndpoint.setProtocol("http");
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress(DECODE_DOMAIN);
        decodeEndpoint.setProtocol("http");
        ModelMetaConfig modelMetaConfig = mock(ModelMetaConfig.class);
        when(modelMetaConfig.endpointsWithGroup(MODEL_NAME, RoleType.PREFILL))
                .thenReturn(List.of(Pair.of("mock", prefillEndpoint)));
        when(modelMetaConfig.endpointsWithGroup(MODEL_NAME, RoleType.DECODE))
                .thenReturn(List.of(Pair.of("mock", decodeEndpoint)));

        // Real master discovery wiring (metrics reporter mocked — not on the decision path).
        healthReporter = mock(EngineHealthReporter.class);
        workerAddressService = new WorkerAddressService(
                healthReporter, modelMetaConfig, fileServiceDiscovery, configService);
        engineGrpcService = new EngineGrpcService(grpcClient);
        // Upstream's schedule refactor removed the cache-generation
        // activate/retire gate from the discovery publish path; the cache
        // poll now runs as an independent GrpcCacheStatusCheckRunner whose
        // updateEngineBlockCache result never gates worker publication.
        // A bare mock (null result -> debug-report path) mirrors upstream
        // EngineSyncRunnerTest's stub contract.
        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
        DynamicCacheIntervalService cacheIntervalService =
                mock(DynamicCacheIntervalService.class);
        statusCheckExecutor = Executors.newFixedThreadPool(4, r -> {
            Thread thread = new Thread(r, "e2e-status-check");
            thread.setDaemon(true);
            return thread;
        });

        EngineSyncRunner prefillSyncRunner = new EngineSyncRunner(
                MODEL_NAME,
                // Upstream's directory refactor removed the shared static
                // ledger (EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS), so the
                // sync runner now takes the status map via constructor. Reuse
                // the base fixture's WorkerDirectory map — it holds the
                // preinitialized placeholder generations paired with the
                // registry's preinitialized endpoints, so the FIRST sync pass
                // retires them through the group-change path (detaching
                // those endpoints) and republishes A/B from discovery. A bare
                // new ConcurrentHashMap<>() would orphan the placeholder
                // endpoints: every newly created generation would then hit
                // "existing endpoint generation must be withdrawn before
                // publication" and be retired, oscillating forever.
                engineWorkerStatus.statusMap(RoleType.PREFILL),
                workerAddressService,
                statusCheckExecutor,
                healthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                cacheAwareService,
                cacheIntervalService,
                5_000L,
                new LongAdder(),
                1L,
                false,
                // Upstream's EndpointEventSink now declares three retirement
                // callbacks (no longer a functional interface), so a Mockito
                // no-op mock replaces the former single-method lambda —
                // same contract as upstream EngineSyncRunnerTest's
                // RunnerTestSupport.eventSink().
                mock(EndpointEventSink.class),
                endpointRegistry,
                STATUS_STALE_AFTER_US);

        syncScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread thread = new Thread(r, "e2e-engine-sync");
            thread.setDaemon(true);
            return thread;
        });
        // Production cadence: the master re-pulls the discovery list every SYNC_STATUS_INTERVAL.
        syncScheduler.scheduleAtFixedRate(prefillSyncRunner, 0, SYNC_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    @AfterEach
    void tearDownFileDiscoveryPipeline() {
        if (syncScheduler != null) {
            syncScheduler.shutdownNow();
            syncScheduler = null;
        }
        if (statusCheckExecutor != null) {
            statusCheckExecutor.shutdownNow();
            statusCheckExecutor = null;
        }
        if (workerC != null) {
            workerC.stop();
            workerC = null;
        }
        if (workerAddressService != null) {
            workerAddressService.destroy();
        }
    }

    @Test
    @Timeout(60)
    void fileDiscoveryDynamicallyAddsAndRemovesWorkers() throws Exception {
        // ─── Phase 1: initial convergence — the sync loop polls A and B ───
        // The base fixture preinitializes A/B endpoint generations with a
        // placeholder group ("test-group"), while discovery reports group
        // "mock", so the FIRST sync pass retires those generations
        // (EngineSyncRunner.getOrCreateWorkerStatus group-change path) and
        // detaches both addresses from the EndpointRegistry — the routable
        // prefill set is briefly EMPTY until the next pass re-publishes A/B
        // from discovery. "Both polled" alone can pass inside that window:
        // the poll counter increments on the mock worker when the status RPC
        // ARRIVES, but the endpoint is re-published only when the response
        // callback runs (GrpcWorkerStatusRunner ->
        // initializeAndPublishNewStatusEndpoint). A drain in that gap reads
        // an empty address snapshot and the router rejects every request
        // with a bare success=false/code=200 response before admission.
        // Gate on both workers polled AND both addresses live in the exact
        // prefill snapshot the router consumes before asserting any request.
        long startMs = System.nanoTime();
        awaitUntil(3_000, () -> mockPrefillWorker.getWorkerStatusCallCount() > 0
                        && workerB.getWorkerStatusCallCount() > 0
                        && prefillAddressesRoutable(prefillIpPort, workerIpPort(workerB)),
                "discovery loop should poll both initial workers and publish "
                        + "routable prefill endpoints for both");
        long initialConvergenceMs = elapsedMs(startMs);

        drainRequests(6);
        awaitUntil(3_000, () -> mockPrefillWorker.getEnqueueCount() >= 1
                        && workerB.getEnqueueCount() >= 1,
                "requests should be distributed across the two initial workers");
        int countABeforeRemoval = mockPrefillWorker.getEnqueueCount();
        int countBBeforeAdd = workerB.getEnqueueCount();
        assertTrue(countABeforeRemoval >= 1 && countBBeforeAdd >= 1,
                "both initial workers must have received requests");

        // ─── Phase 2: start C (NOT registered) → add to file → it must converge ───
        workerC = new MockPrefillWorker(MockWorkerBehavior.builder().build());
        workerC.start(0);
        String cIpPort = workerIpPort(workerC);
        writeDiscoveryFileAtomic(List.of(prefillIpPort, workerIpPort(workerB), cIpPort));

        long addStartMs = System.nanoTime();
        AtomicBoolean cReceived = new AtomicBoolean(false);
        long addConvergenceMs = -1;
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(CONVERGENCE_CAP_MS);
        while (System.nanoTime() < deadline) {
            drainRequests(3);
            if (workerC.getEnqueueCount() > 0) {
                cReceived.set(true);
                addConvergenceMs = elapsedMs(addStartMs);
                break;
            }
        }
        assertTrue(cReceived.get(),
                "worker C should start receiving requests within " + CONVERGENCE_CAP_MS + " ms "
                        + "of appearing in the discovery file");
        assertNotNull(endpointRegistry.get(RoleType.PREFILL, cIpPort),
                "worker C must be registered in the EndpointRegistry after discovery");

        // ─── Phase 3: remove A from the file → A must stop receiving new requests ───
        writeDiscoveryFileAtomic(List.of(workerIpPort(workerB), cIpPort));

        long removeStartMs = System.nanoTime();
        awaitUntil(CONVERGENCE_CAP_MS, () ->
                        endpointRegistry.get(RoleType.PREFILL, prefillIpPort) == null,
                "worker A should be evicted from the EndpointRegistry after removal from the file");
        long removeConvergenceMs = elapsedMs(removeStartMs);

        // EndpointRegistry no longer offers A, so the round-robin router cannot pick it.
        int countAAtRemoval = mockPrefillWorker.getEnqueueCount();
        int countBAtRemoval = workerB.getEnqueueCount();
        int countCAtRemoval = workerC.getEnqueueCount();
        drainRequests(9);
        assertEquals(countAAtRemoval, mockPrefillWorker.getEnqueueCount(),
                "removed worker A must not receive any new requests");
        assertTrue(workerB.getEnqueueCount() > countBAtRemoval,
                "worker B must keep serving after A's removal");
        assertTrue(workerC.getEnqueueCount() > countCAtRemoval,
                "worker C must keep serving after A's removal");

        System.out.printf(
                "FileDiscoveryDynamicScaleEndToEndTest convergence: initial=%dms, add=%dms, remove=%dms%n",
                initialConvergenceMs, addConvergenceMs, removeConvergenceMs);
    }

    // ════════════════════════════════════════════════════════════════
    //  Discovery file writer — same atomic protocol as DiscoveryFileStore
    // ════════════════════════════════════════════════════════════════

    private void writeDiscoveryFileAtomic(List<String> prefillHosts) throws IOException {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put(PREFILL_DOMAIN, prefillHosts);
        payload.put(DECODE_DOMAIN, List.of(decodeIpPort));
        Path tmp = discoveryFile.resolveSibling(discoveryFile.getFileName() + ".tmp");
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(tmp.toFile(), payload);
        try {
            Files.move(tmp, discoveryFile,
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(tmp, discoveryFile, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Round-robin router over the LIVE endpoint set
    // ════════════════════════════════════════════════════════════════

    /**
     * Route each request to the next live prefill endpoint from the
     * {@code EndpointRegistry} — the set the discovery pipeline maintains —
     * so additions and evictions are reflected in routing immediately.
     */
    @Override
    protected Router createRouter() {
        Router roundRobin = mock(Router.class);
        when(roundRobin.routeForQueue(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            List<String> candidates = new ArrayList<>(
                    endpointRegistry.endpointAddressSnapshot(RoleType.PREFILL));
            if (candidates.isEmpty()) {
                Response empty = new Response();
                empty.setSuccess(false);
                return QueueRoutingResult.rejected(empty);
            }
            Collections.sort(candidates);
            String chosen = candidates.get(routerCounter.getAndIncrement() % candidates.size());
            String[] parts = chosen.split(":");
            String ip = parts[0];
            int httpPort = Integer.parseInt(parts[1]);
            // admittedRoute() converts this response into the exact pinned
            // queue admission the scheduler consumes — including the Decode KV
            // reservation (QueueRouteAdmission.reserveQueuedPinned) the batcher
            // later marks queued; without it admission would hit NOT_QUEUED ->
            // OwnershipLost and the request would never complete.
            return admittedRoute(ctx,
                    routeResponse(ctx.getRequestId(), ip, httpPort, httpPort + 1));
        });
        return roundRobin;
    }

    private Response routeResponse(long requestId, String prefillIpAddr, int prefillHttpPort,
                                    int prefillGrpcPort) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                serverStatus(RoleType.PREFILL, prefillIpAddr, prefillHttpPort, prefillGrpcPort, requestId),
                serverStatus(RoleType.DECODE, decodeIp, decodeHttpPort, decodeGrpcPort, requestId)));
        return response;
    }

    private static ServerStatus serverStatus(RoleType role, String ip, int httpPort, int grpcPort,
                                              long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("test-group");
        status.setRequestId(requestId);
        return status;
    }

    // ════════════════════════════════════════════════════════════════
    //  Helpers
    // ════════════════════════════════════════════════════════════════

    /** Submit {@code count} requests and wait for every ACK to complete successfully. */
    private void drainRequests(int count) throws Exception {
        List<CompletableFuture<Response>> futures = new ArrayList<>(count);
        long firstId = requestIdCounter.getAndAdd(count);
        for (int i = 0; i < count; i++) {
            futures.add(submitRequest(firstId + i));
        }
        for (int i = 0; i < count; i++) {
            Response response = futures.get(i).get(5, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(),
                    "request " + (firstId + i) + " should succeed, code=" + response.getCode());
        }
    }

    /**
     * Every given worker is addressable in the exact prefill snapshot the
     * router consumes — the placement path's true readiness signal, not
     * merely "the sync loop contacted the worker once". The poll-count
     * precondition in the caller guarantees the preinitialized placeholder
     * generations have already been retired, so a live address can only
     * come from a discovery-driven publication.
     */
    private boolean prefillAddressesRoutable(String... ipPorts) {
        List<String> routable =
                endpointRegistry.endpointAddressSnapshot(RoleType.PREFILL);
        for (String ipPort : ipPorts) {
            if (!routable.contains(ipPort)) {
                return false;
            }
        }
        return true;
    }

    private static void awaitUntil(long timeoutMs, BooleanSupplier condition, String message)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(20);
        }
        fail(message);
    }

    private static long elapsedMs(long startNano) {
        return TimeUnit.MILLISECONDS.convert(System.nanoTime() - startNano, TimeUnit.NANOSECONDS);
    }
}
