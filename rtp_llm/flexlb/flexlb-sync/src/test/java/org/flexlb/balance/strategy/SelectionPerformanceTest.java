package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestWorkerSelector;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Production selection only: no RPC, Mockito invocation or asynchronous work in the measured loop. */
@Tag("performance-regression")
class SelectionPerformanceTest {
    private static final List<RoleType> ROLES = List.of(RoleType.DECODE, RoleType.PREFILL);

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void selectionCostAt750PrefillAndDecodeWorkers(boolean batch) throws Exception {
        var config = StrategyTestSupport.config();
        if (!batch) config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        var constructor = ConfigService.class.getDeclaredConstructor(String.class);
        constructor.setAccessible(true);
        var service =
                constructor.newInstance(
                        "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":60000}}}");
        ReflectionTestUtils.setField(service, "flexlbConfig", config);
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
                    StrategyTestSupport.publishEndpoint(registry, role, worker.getIpPort(), worker);
                }
            }
            var selector = selector(service, registry);
            var context = new RequestContext(service.loadBalanceConfig());
            context.data().setRequest(Request.builder().requestId(91001L).seqLen(1000L).build());
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
                    try (var route = result.value()) {
                        checksum += route.prefillWorkMs();
                    }
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

    private RequestWorkerSelector selector(ConfigService service, EndpointRegistry registry) {
        var monitor = new NoOpFlexMonitor();
        var metrics = new CacheMetricsReporter();
        ReflectionTestUtils.setField(metrics, "monitor", monitor);
        var health = new EngineHealthReporter(monitor, registry);
        var cache =
                new CacheAwareService() {
                    public Map<String, Integer> findMatchingEngines(
                            List<Long> keys, RoleType role, List<String> addresses) {
                        return Map.of();
                    }

                    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus status) {
                        throw new UnsupportedOperationException();
                    }

                    public void removeEngineBlockCache(String address) {}
                };
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(ROLES);
        return new RequestWorkerSelector(
                registry,
                new CostBasedPrefillStrategy(cache, service),
                new DecodeSelector(service, new AdmissionCapacity(1000, 100)),
                new VitWorkerSelector(),
                null,
                health,
                metrics,
                service,
                model);
    }
}
