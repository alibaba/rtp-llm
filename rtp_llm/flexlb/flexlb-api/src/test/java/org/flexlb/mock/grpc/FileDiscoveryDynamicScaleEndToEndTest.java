package org.flexlb.mock.grpc;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.FileServiceDiscovery;
import org.flexlb.mock.FlexLBMockTestBase;
import org.flexlb.mock.MockPrefillWorker;
import org.flexlb.mock.MockWorkerBehavior;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.runner.EngineSyncRunner;
import org.flexlb.sync.status.EngineWorkerStatus;
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
    /** IdUtils.getServiceIdByModelName(MODEL_NAME) = CommonConstants.FUNCTION + "." + MODEL_NAME. */
    private static final String SERVICE_ID = "aigc.text-generation.generation." + MODEL_NAME;
    /** Master SYNC_STATUS_INTERVAL default — the production re-pull cadence. */
    private static final long SYNC_INTERVAL_MS = 20L;
    private static final long CONVERGENCE_CAP_MS = 10_000L;

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

        // Static ServiceRoute registry: domain → address mapping the sync loop resolves.
        Endpoint prefillEndpoint = new Endpoint();
        prefillEndpoint.setAddress(PREFILL_DOMAIN);
        prefillEndpoint.setProtocol("http");
        Endpoint decodeEndpoint = new Endpoint();
        decodeEndpoint.setAddress(DECODE_DOMAIN);
        decodeEndpoint.setProtocol("http");
        GroupRoleEndPoint groupEndpoint = new GroupRoleEndPoint();
        groupEndpoint.setGroup("mock");
        groupEndpoint.setPrefillEndpoint(prefillEndpoint);
        groupEndpoint.setDecodeEndpoint(decodeEndpoint);
        ServiceRoute serviceRoute = new ServiceRoute();
        serviceRoute.setRoleEndpoints(List.of(groupEndpoint));
        ModelMetaConfig.putServiceRoute(SERVICE_ID, serviceRoute);

        // Real master discovery wiring (metrics reporter mocked — not on the decision path).
        healthReporter = mock(EngineHealthReporter.class);
        workerAddressService = new WorkerAddressService(
                healthReporter, new ModelMetaConfig(), fileServiceDiscovery, configService);
        engineGrpcService = new EngineGrpcService(grpcClient);
        statusCheckExecutor = Executors.newFixedThreadPool(4, r -> {
            Thread thread = new Thread(r, "e2e-status-check");
            thread.setDaemon(true);
            return thread;
        });

        EngineSyncRunner prefillSyncRunner = new EngineSyncRunner(
                MODEL_NAME,
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap(),
                workerAddressService,
                statusCheckExecutor,
                healthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                mock(CacheAwareService.class),
                5_000L,
                new LongAdder(),
                1L,
                // Local intake: EngineSyncRunner carries an extra
                // cacheFullSnapshotDebugMode flag between the sync interval and
                // the scheduler.
                false,
                scheduler,
                endpointRegistry);

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
        if (WorkerAddressService.serviceDiscoveryExecutor != null) {
            WorkerAddressService.serviceDiscoveryExecutor.shutdown();
        }
    }

    @Test
    @Timeout(60)
    void fileDiscoveryDynamicallyAddsAndRemovesWorkers() throws Exception {
        // ─── Phase 1: initial convergence — the sync loop polls A and B ───
        long startMs = System.nanoTime();
        awaitUntil(3_000, () -> mockPrefillWorker.getWorkerStatusCallCount() > 0
                        && workerB.getWorkerStatusCallCount() > 0,
                "discovery loop should poll both initial workers");
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
        when(roundRobin.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            List<String> candidates = new ArrayList<>(
                    endpointRegistry.getEndpoints(RoleType.PREFILL).keySet());
            if (candidates.isEmpty()) {
                Response empty = new Response();
                empty.setSuccess(false);
                return empty;
            }
            Collections.sort(candidates);
            String chosen = candidates.get(routerCounter.getAndIncrement() % candidates.size());
            // Mirror the production routing side effect: reserve Decode KV
            // capacity for this request. Without the reservation, submit()'s
            // decodeEp.markQueuedPhase() has no reservation to mark, the batcher
            // admission later hits NOT_QUEUED -> OwnershipLost, and the request
            // leaves the queue without ever completing (drainRequests timeout).
            reserveDecode(ctx);
            String[] parts = chosen.split(":");
            String ip = parts[0];
            int httpPort = Integer.parseInt(parts[1]);
            return routeResponse(ctx.getRequestId(), ip, httpPort, httpPort + 1);
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
