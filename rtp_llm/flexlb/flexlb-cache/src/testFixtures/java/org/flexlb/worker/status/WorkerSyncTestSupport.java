package org.flexlb.worker.status;

import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.cache.DynamicCacheIntervalService;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;

/**
 * Runs one real synchronization cycle on test-owned executors, without the periodic timer.
 */
public final class WorkerSyncTestSupport {
    private WorkerSyncTestSupport() {}

    public static Runnable runner(
            String modelName,
            EndpointRegistry endpoints,
            WorkerAddressService discovery,
            ExecutorService statuses,
            ServiceMetricsReporter reporter,
            WorkerStatusRpcClient client,
            RoleType role,
            CacheAwareService cache,
            DynamicCacheIntervalService intervals,
            long timeout,
            long cycle,
            Long interval,
            boolean fullCache,
            ExecutorService cacheExecutor,
            ExecutorService application) {
        var sync = mock(WorkerStatusSynchronizer.class, CALLS_REAL_METHODS);
        var config = new FlexlbConfig();
        config.getWorkerRegistry().getCacheStatus().setFullSnapshotDebugMode(fullCache);
        ReflectionTestUtils.setField(sync, "modelName", modelName);
        ReflectionTestUtils.setField(sync, "endpointRegistry", endpoints);
        ReflectionTestUtils.setField(sync, "workerAddressService", discovery);
        ReflectionTestUtils.setField(sync, "statusCheckExecutor", executor(statuses));
        ReflectionTestUtils.setField(sync, "serviceMetricsReporter", reporter);
        ReflectionTestUtils.setField(sync, "workerStatusRpcClient", client);
        ReflectionTestUtils.setField(sync, "cacheAwareService", cache);
        ReflectionTestUtils.setField(sync, "cacheIntervalService", intervals);
        ReflectionTestUtils.setField(sync, "syncRequestTimeoutMs", timeout);
        ReflectionTestUtils.setField(sync, "syncEngineStatusInterval", interval);
        ReflectionTestUtils.setField(sync, "flexlbConfig", config);
        ReflectionTestUtils.setField(sync, "cacheStatusExecutor", executor(cacheExecutor));
        ReflectionTestUtils.setField(sync, "statusApplicationExecutor", executor(application));
        return () -> sync.syncRole(role, cycle);
    }

    public static WorkerStatusSynchronizer cleanups(
            CacheAwareService cache, EndpointRegistry endpoints) {
        ReflectionTestUtils.setField(endpoints, "cacheAwareService", cache);
        var sync = mock(WorkerStatusSynchronizer.class, CALLS_REAL_METHODS);
        ReflectionTestUtils.setField(sync, "cacheAwareService", cache);
        ReflectionTestUtils.setField(sync, "endpointRegistry", endpoints);
        org.mockito.Mockito.doNothing().when(sync).destroy();
        return sync;
    }

    private static ThreadPoolExecutor executor(ExecutorService delegate) {
        if (delegate instanceof ThreadPoolExecutor pool) return pool;
        var pool = mock(ThreadPoolExecutor.class);
        org.mockito.Mockito.lenient()
                .doAnswer(call -> delegate.submit(call.getArgument(0, Runnable.class)))
                .when(pool)
                .submit(org.mockito.ArgumentMatchers.any(Runnable.class));
        org.mockito.Mockito.lenient()
                .doAnswer(
                        call -> {
                            delegate.execute(call.getArgument(0, Runnable.class));
                            return null;
                        })
                .when(pool)
                .execute(org.mockito.ArgumentMatchers.any(Runnable.class));
        return pool;
    }
}
