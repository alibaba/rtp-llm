package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRouteSelector;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.lang.management.ManagementFactory;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Production selection only: no RPC, Mockito invocation or asynchronous work in the measured loop.
 */
@Tag("performance-regression")
class SelectionPerformanceTest {
    private static final List<RoleType> ROLES = List.of(RoleType.DECODE, RoleType.PREFILL);

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void selectionCostAt750PrefillAndDecodeWorkers(boolean batch) throws Exception {
        var service =
                new ConfigService(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "dispatcher":{"type":"%s"},
                 "router":{"roles":{"prefill":{"executionTimeEstimator":
                    {"expression":"sum(computeTokens)"}}}}}
                """
                                .formatted(batch ? "BATCH" : "NON_BATCH"));
        long singleWorkerBytes = measure(service, batch, 1);
        long fleetBytes = measure(service, batch, 750);
        assertTrue(
                fleetBytes <= singleWorkerBytes + 128,
                "steady selection allocation must not grow with the fleet");
    }

    private long measure(ConfigService service, boolean batch, int workers) {
        var registry = StrategyTestSupport.endpointRegistry(service);
        try {
            for (var role : List.of(RoleType.PREFILL, RoleType.DECODE)) {
                for (int index = 0; index < workers; index++) {
                    var worker =
                            StrategyTestSupport.workerStatus(
                                    role,
                                    "g",
                                    "worker-" + index,
                                    8080,
                                    8081,
                                    true,
                                    1000000L,
                                    1000000L);
                    StrategyTestSupport.publishEndpoint(
                            registry, role, worker.getLogicalIpPort(), worker);
                }
            }
            var selector = selector(service, registry);
            var context = new RequestContext(service.loadBalanceConfig());
            context.data().setRequest(Request.builder().requestId("91001").seqLen(1000L).build());
            context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
            var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
            assertTrue(bean.isThreadAllocatedMemorySupported());
            bean.setThreadAllocatedMemoryEnabled(true);
            bean.setThreadCpuTimeEnabled(true);
            int operations = Integer.getInteger("flexlb.perf.selection.operations", 10000);
            long threadId = Thread.currentThread().threadId();
            long maximumBytes = 0;
            for (int round = 0; round < 4; round++) {
                long allocated = bean.getThreadAllocatedBytes(threadId);
                long cpu = bean.getCurrentThreadCpuTime();
                long start = System.nanoTime();
                long checksum = 0L;
                for (int index = 0; index < operations; index++) {
                    var result = selector.select(context, null);
                    assertEquals(PlacementResult.Status.SUCCESS, result.status());
                    checksum +=
                            ((org.flexlb.balance.scheduler.RoleSelection.Prefill)
                                            result.value().role(RoleType.PREFILL))
                                    .workMs();
                    var route = result.value();
                    route.prefillEndpoint().releaseRequest(route);
                    if (route.decodeReservation() != null)
                        route.decodeEndpoint()
                                .release(
                                        route.decodeReservation(),
                                        org.flexlb.balance.endpoint.DecodeResources.ReleaseReason
                                                .LOCAL_ROLLBACK);
                }
                long nanos = System.nanoTime() - start;
                long cpuNanos = bean.getCurrentThreadCpuTime() - cpu;
                long bytes = bean.getThreadAllocatedBytes(threadId) - allocated;
                assertEquals(1000L * operations, checksum);
                if (round > 0) {
                    System.out.printf(
                            "Selection: roles=%s workers=%d batch=%s round=%d ns/op=%d cpu_ns/op=%d bytes/op=%d%n",
                            ROLES,
                            workers,
                            batch,
                            round,
                            nanos / operations,
                            cpuNanos / operations,
                            bytes / operations);
                    maximumBytes = Math.max(maximumBytes, bytes / operations);
                }
            }
            return maximumBytes;
        } finally {
            registry.close();
        }
    }

    private RequestRouteSelector selector(ConfigService service, EndpointRegistry registry) {
        var monitor = new NoOpFlexMonitor();
        var metrics = new CacheMetricsReporter();
        ReflectionTestUtils.setField(metrics, "monitor", monitor);
        var health = new EngineHealthReporter(monitor, registry);
        var cache =
                new CacheAwareService(null, null, null, null) {
                    public org.flexlb.cache.domain.CacheMatchResult findMatchingEngines(
                            org.flexlb.cache.domain.CacheMatchQuery query) {
                        return org.flexlb.cache.domain.CacheMatchResult.empty(
                                org.flexlb.cache.domain.CacheMatchSource.LOCAL_SYNC);
                    }

                    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus status) {
                        throw new UnsupportedOperationException();
                    }

                    public void removeEngineBlockCache(String address) {}
                };
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(ROLES);
        return new RequestRouteSelector(
                registry,
                new CostBasedPrefillStrategy(cache, service),
                new DecodeSelector(service, new AdmissionCapacity(1000, 100)),
                new VitWorkerSelector(),
                org.mockito.Mockito.mock(
                        org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                null,
                health,
                metrics,
                service,
                model);
    }
}
