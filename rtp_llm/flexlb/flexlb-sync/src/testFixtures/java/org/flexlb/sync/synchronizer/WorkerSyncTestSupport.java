package org.flexlb.sync.synchronizer;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.doAnswer;

/** Runs one real synchronization cycle on test-owned executors, without the periodic timer. */
public final class WorkerSyncTestSupport {
    private WorkerSyncTestSupport() {}
    public static Runnable runner(String modelName, EndpointRegistry endpoints,
            WorkerAddressService discovery, ExecutorService statuses, EngineHealthReporter reporter,
            WorkerStatusRpcClient client, RoleType role, CacheAwareService cache,
            DynamicCacheIntervalService intervals, long timeout, long cycle, Long interval,
            boolean fullCache, ExecutorService cacheExecutor, ExecutorService application) {
        var sync = mock(WorkerStatusSynchronizer.class, CALLS_REAL_METHODS);
        var config = new FlexlbConfig();
        config.getWorkerRegistry().getCacheStatus().setFullSnapshotDebugMode(fullCache);
        ReflectionTestUtils.setField(sync, "modelName", modelName);
        ReflectionTestUtils.setField(sync, "endpointRegistry", endpoints);
        ReflectionTestUtils.setField(sync, "workerAddressService", discovery);
        ReflectionTestUtils.setField(sync, "statusCheckExecutor", executor(statuses));
        ReflectionTestUtils.setField(sync, "engineHealthReporter", reporter);
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

    public static WorkerStatusSynchronizer retirements(CacheAwareService cache, EndpointRegistry endpoints) {
        var sync = mock(WorkerStatusSynchronizer.class, CALLS_REAL_METHODS);
        ReflectionTestUtils.setField(sync, "cacheAwareService", cache);
        ReflectionTestUtils.setField(sync, "endpointRegistry", endpoints);
        org.mockito.Mockito.doNothing().when(sync).destroy();
        return sync;
    }

    private static ThreadPoolExecutor executor(ExecutorService delegate) {
        if (delegate instanceof ThreadPoolExecutor pool) return pool;
        var pool = mock(ThreadPoolExecutor.class);
        org.mockito.Mockito.lenient().doAnswer(call -> delegate.submit(call.getArgument(0, Runnable.class)))
                .when(pool).submit(org.mockito.ArgumentMatchers.any(Runnable.class));
        org.mockito.Mockito.lenient().doAnswer(call -> { delegate.execute(call.getArgument(0, Runnable.class)); return null; })
                .when(pool).execute(org.mockito.ArgumentMatchers.any(Runnable.class));
        return pool;
    }
}
