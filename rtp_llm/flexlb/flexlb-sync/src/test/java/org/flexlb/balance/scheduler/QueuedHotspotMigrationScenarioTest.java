package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.RoutingConfig.QueuedRequestPolicy;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.StatusFailureReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class QueuedHotspotMigrationScenarioTest {
    private static final String SOURCE = "127.0.0.1";
    private static final String RECEIVER = "127.0.0.2";

    @ParameterizedTest
    @EnumSource(QueuedRequestPolicy.class)
    void existingBacklogRebalancesWithoutArrivalsAndDeliversOnce(QueuedRequestPolicy policy) throws Exception {
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        SchedulingTestConfig.usePrefillCostOrder(config);
        var release = config.getRouter().getRoles().getPrefill().getHotspotRelease();
        release.setQueuedRequestPolicy(policy);
        release.setSourceTopK(0);
        release.setSetpointFloor(6);
        release.setComputeTokenFloor(Long.MAX_VALUE);
        release.setMaxExtraComputePermille(0);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var reporter = mock(BatchSchedulerReporter.class);
        var requestReporter = mock(RequestSchedulerReporter.class);
        var registry = new RequestRegistry(service, reporter, requestReporter);
        var availability = new PlacementAvailability();
        var endpoints = new EndpointRegistry(service, new EndpointEventProjector(registry, mock(StatusFailureReporter.class)),
                reporter, new RouteDeliveryStrategy(registry, new DeliveryMetrics(reporter)), availability);
        var directory = new WorkerDirectory(endpoints);
        var cache = mock(CacheAwareService.class);
        // Alternating cold and half-hit requests, with identical resident prefixes on both Prefills.
        when(cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                List.of(42L).equals(call.getArgument(0))
                        ? Map.of(SOURCE + ":8080", 1, RECEIVER + ":8080", 1) : Map.of());
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        var router = spy(new DefaultRouter(new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class)),
                new DecodeSelector(directory), new RandomStrategy(directory), service, model));
        var firstRound = new AtomicBoolean(true);
        var resume = new AtomicBoolean(false);
        var nextRound = new CountDownLatch(1);
        doAnswer(call -> {
            @SuppressWarnings("unchecked")
            var sources = (List<PrefillEndpoint>) call.callRealMethod();
            if (sources.isEmpty() || firstRound.getAndSet(false) || resume.get()) { return sources; }
            nextRound.countDown();
            return List.of();
        }).when(router).queuedMigrationSources(release);
        var scheduler = new RequestScheduler(service, router, endpoints, reporter, mock(EvictionManager.class), registry, availability);
        var runtime = new SchedulerRuntime(registry, endpoints, reporter, requestReporter, scheduler);
        try {
            publish(endpoints, RoleType.PREFILL, SOURCE);
            var source = (PrefillEndpoint) endpoints.get(RoleType.PREFILL, SOURCE + ":8080");
            var requests = new ArrayList<CompletableFuture<Response>>();
            for (long id = 1; id <= 8; id++) {
                var context = RequestLifecycleTestSupport.context(config, id);
                context.getRequest().setCacheKeyBlockSize(8L);
                context.getRequest().setBlockCacheKeys(List.of(id % 2 == 0 ? 42L : 17L));
                requests.add(scheduler.submit(context));
            }
            await(() -> source.queuedRequestCount() == 8);
            publish(endpoints, RoleType.PREFILL, RECEIVER);
            var receiver = (PrefillEndpoint) endpoints.get(RoleType.PREFILL, RECEIVER + ":8080");
            var expected = switch (policy) {
                case DISABLED -> List.<Long>of();
                case LOWEST_HIT_RATE_FIRST -> List.of(5L, 7L);
                case HIGHEST_HIT_RATE_FIRST -> List.of(6L, 8L);
            };
            if (policy != QueuedRequestPolicy.DISABLED) {
                assertTrue(nextRound.await(5, TimeUnit.SECONDS));
                assertEquals(1, receiver.queuedRequestCount(), "one transfer per round across the fleet");
                resume.set(true);
            }
            await(() -> receiver.queuedRequestCount() == expected.size()
                    && source.queuedRequestCount() == 8 - expected.size());
            assertEquals(expected, receiver.captureQueueSnapshot().items().stream()
                    .map(ScheduledRequest::requestId).sorted().toList());
            assertTrue(requests.stream().noneMatch(CompletableFuture::isDone));
            publish(endpoints, RoleType.DECODE, "127.0.0.3");
            for (int i = 0; i < requests.size(); i++) {
                Response response = requests.get(i).get(5, TimeUnit.SECONDS);
                assertTrue(response.isSuccess(), response.getErrorMessage());
                String selected = response.getServerStatus().stream().filter(status -> status.getRole() == RoleType.PREFILL)
                        .findFirst().orElseThrow().getServerIp();
                assertEquals(expected.contains((long) i + 1) ? RECEIVER : SOURCE, selected);
            }
            assertEquals(0, source.queuedRequestCount() + receiver.queuedRequestCount());
        } finally {
            runtime.shutdown();
        }
    }

    private static void publish(EndpointRegistry endpoints, RoleType role, String ip) {
        WorkerStatus worker = WorkerStatusTestFixtures.readyForScheduling(
                WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test"));
        var status = new WorkerStatusResponse();
        status.setRole(role);
        status.setAlive(true);
        status.setStatusVersion(1L);
        status.setLatestFinishedVersion(0L);
        status.setTotalKvCacheTokens(100_000L);
        status.setAvailableKvCacheTokens(100_000L);
        status.setMaxSeqLen(10_000L);
        status.setMaxBatchTokensSize(10_000L);
        worker.lock.lock();
        try {
            endpoints.publishPreparedEndpoint(worker.getIpPort(), worker,
                    WorkerStatusTestFixtures.prepare(worker, WorkerStatusTestFixtures.observation(worker, status)));
        } finally {
            worker.lock.unlock();
        }
    }

    private static void await(BooleanSupplier predicate) throws InterruptedException {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!predicate.getAsBoolean() && System.nanoTime() < end) { Thread.sleep(10); }
        assertTrue(predicate.getAsBoolean(), "expected queue depths were not observed");
    }
}
