package org.flexlb.sync.synchronizer;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Wires a real {@link MasterEngineSynchronizer} to pin the assembly the unit tests of its parts
 * cannot see: the single-flight gate actually guards {@code submitRound}, the bean-owned grace
 * clock actually reaches every per-round runner, boot validation actually gates the scheduler
 * start, and one role's failed submit cannot starve the other roles of the same tick.
 */
@ExtendWith(MockitoExtension.class)
class MasterEngineSynchronizerTest {

    private static final String SERVICE_ID = "aigc.text-generation.generation.msynth-test";
    private static final String MODEL_NAME = "msynth-test";

    @Mock
    private WorkerAddressService workerAddressService;
    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private EngineGrpcService engineGrpcService;
    @Mock
    private CacheAwareService cacheAwareService;
    @Mock
    private ConfigService configService;

    private final ModelMetaConfig modelMetaConfig = new ModelMetaConfig();

    @AfterEach
    void tearDown() {
        // The constructor registers the route into a process-wide static table before validating.
        ModelMetaConfig.removeServiceRoute(SERVICE_ID);
    }

    private static String modelConfig(String... endpointFields) {
        return "{\"service_id\":\"" + SERVICE_ID + "\",\"role_endpoints\":[{\"group\":\"g1\","
                + String.join(",", endpointFields) + "}]}";
    }

    private static String pdFusionOnlyConfig() {
        return modelConfig("\"pd_fusion_endpoint\":{\"protocol\":\"http\",\"path\":\"/\"}");
    }

    private MasterEngineSynchronizer newSynchronizer(FlexlbConfig config, String modelConfigJson) {
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new MasterEngineSynchronizer(workerAddressService, engineHealthReporter,
                new EngineWorkerStatus(modelMetaConfig), engineGrpcService, modelMetaConfig,
                cacheAwareService, configService, modelConfigJson, s -> {
                });
    }

    @Test
    void embeddingWithLoadAwareStrategyFailsConstructionBeforeTheSchedulerStarts() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.SHORTEST_TTFT);
        when(configService.loadBalanceConfig()).thenReturn(config);

        AtomicBoolean schedulerStarted = new AtomicBoolean();
        assertThrows(IllegalStateException.class, () -> new MasterEngineSynchronizer(
                        workerAddressService, engineHealthReporter, new EngineWorkerStatus(modelMetaConfig),
                        engineGrpcService, modelMetaConfig, cacheAwareService, configService,
                        pdFusionOnlyConfig(), s -> schedulerStarted.set(true)),
                "EMBEDDING workers are never probed, so a load-aware strategy must fail boot");
        assertFalse(schedulerStarted.get(),
                "a construction that failed validation must never have started the periodic sync");
    }

    @Test
    void aSecondSubmitRoundForTheSameKeyIsSkippedWhileTheFirstIsInFlight() throws Exception {
        CountDownLatch discoveryStarted = new CountDownLatch(1);
        CountDownLatch releaseDiscovery = new CountDownLatch(1);
        when(workerAddressService.getEngineWorkerList(any(), any())).thenAnswer(inv -> {
            discoveryStarted.countDown();
            releaseDiscovery.await(5, TimeUnit.SECONDS);
            return List.of();
        });

        MasterEngineSynchronizer synchronizer = newSynchronizer(new FlexlbConfig(), pdFusionOnlyConfig());
        Map<String, WorkerStatus> roleStatusMap = new ConcurrentHashMap<>();

        synchronizer.submitRound(MODEL_NAME, RoleType.PDFUSION, roleStatusMap);
        assertTrue(discoveryStarted.await(5, TimeUnit.SECONDS), "first round must start");
        // Ticks arriving while the round is still resolving discovery must be skipped, not queued.
        synchronizer.submitRound(MODEL_NAME, RoleType.PDFUSION, roleStatusMap);
        synchronizer.submitRound(MODEL_NAME, RoleType.PDFUSION, roleStatusMap);
        releaseDiscovery.countDown();

        verify(workerAddressService, after(500).times(1)).getEngineWorkerList(any(), any());
    }

    @Test
    void runnersOfSuccessiveRoundsShareTheGraceClockSoTheGapRideOutSurvivesRounds() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.ROUND_ROBIN);
        // First round succeeds and stamps the grace baseline; every later round sees the empty
        // list a swallowed discovery failure produces.
        when(workerAddressService.getEngineWorkerList(any(), any()))
                .thenReturn(List.of(new WorkerHost("10.0.0.9", 23950, "site-a")))
                .thenReturn(List.of());

        MasterEngineSynchronizer synchronizer = newSynchronizer(config, pdFusionOnlyConfig());
        Map<String, WorkerStatus> roleStatusMap = new ConcurrentHashMap<>();

        synchronizer.submitRound(MODEL_NAME, RoleType.PDFUSION, roleStatusMap);
        WorkerStatus worker = awaitWorker(roleStatusMap, "10.0.0.9:23950");

        // Make the empty-round refresh observable, then keep ticking until a gap round ran. The
        // refresh only happens when the runner of a *later* round still sees the baseline the
        // first round stamped — i.e. when all rounds share one clock map instance.
        worker.getStatusLastUpdateTime().set(1L);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (worker.getStatusLastUpdateTime().get() == 1L && System.nanoTime() < deadline) {
            synchronizer.submitRound(MODEL_NAME, RoleType.PDFUSION, roleStatusMap);
            Thread.sleep(5);
        }

        assertTrue(worker.getStatusLastUpdateTime().get() > 1L,
                "an empty discovery round must ride the grace window stamped by the previous round; "
                        + "that only works if every round's runner receives the same grace-clock map");
        assertTrue(worker.isAlive(), "the known fleet must survive the discovery gap");
    }

    @Test
    void aRejectedSubmitForOneRoleDoesNotStarveTheRemainingRolesOfTheTick() throws Exception {
        CountDownLatch survivingRoleRan = new CountDownLatch(1);
        when(workerAddressService.getEngineWorkerList(any(), any())).thenAnswer(inv -> {
            survivingRoleRan.countDown();
            return List.of();
        });

        MasterEngineSynchronizer synchronizer = newSynchronizer(new FlexlbConfig(), modelConfig(
                "\"prefill_endpoint\":{\"protocol\":\"http\",\"path\":\"/\"}",
                "\"decode_endpoint\":{\"protocol\":\"http\",\"path\":\"/\"}"));

        AtomicBoolean rejectedOnce = new AtomicBoolean();
        ExecutorService original = AbstractEngineStatusSynchronizer.engineSyncExecutor;
        ThreadPoolExecutor rejectFirst = new ThreadPoolExecutor(2, 2, 0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>()) {
            @Override
            public Future<?> submit(Runnable task) {
                if (rejectedOnce.compareAndSet(false, true)) {
                    throw new RejectedExecutionException("synthetic executor saturation");
                }
                return super.submit(task);
            }
        };
        try {
            AbstractEngineStatusSynchronizer.engineSyncExecutor = rejectFirst;
            synchronizer.syncEngineStatus();

            assertTrue(survivingRoleRan.await(5, TimeUnit.SECONDS),
                    "the role after the rejected one must still get its round this tick");
            verify(workerAddressService, timeout(5000).times(1)).getEngineWorkerList(any(), any());
        } finally {
            AbstractEngineStatusSynchronizer.engineSyncExecutor = original;
            rejectFirst.shutdownNow();
        }
    }

    private static WorkerStatus awaitWorker(Map<String, WorkerStatus> map, String ipPort) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            WorkerStatus status = map.get(ipPort);
            if (status != null && status.isAlive()) {
                return status;
            }
            Thread.sleep(5);
        }
        throw new AssertionError("worker " + ipPort + " never appeared — the first round did not run");
    }
}
