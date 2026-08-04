package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.PreDestroy;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Engine status synchronizer
 */
public abstract class AbstractEngineStatusSynchronizer {

    protected static final Logger logger = LoggerFactory.getLogger("syncLogger");

    protected final WorkerAddressService workerAddressService;

    protected final EngineWorkerStatus engineWorkerStatus;

    protected final EngineHealthReporter engineHealthReporter;

    protected ScheduledThreadPoolExecutor scheduler;

    /**
     * Engine worker status request execution thread pool
     */
    public static ExecutorService statusCheckExecutor;

    /**
     * Engine worker status synchronization thread pool
     */
    public static ExecutorService engineSyncExecutor;

    protected final ModelMetaConfig modelMetaConfig;

    protected final FlexlbConfig flexlbConfig;

    public AbstractEngineStatusSynchronizer(WorkerAddressService workerAddressService,
                                            EngineHealthReporter engineHealthReporter,
                                            EngineWorkerStatus engineWorkerStatus,
                                            ModelMetaConfig modelMetaConfig,
                                            ConfigService configService) {
        this.workerAddressService = workerAddressService;
        this.engineHealthReporter = engineHealthReporter;
        this.engineWorkerStatus = engineWorkerStatus;
        this.modelMetaConfig = modelMetaConfig;
        this.flexlbConfig = configService.loadBalanceConfig();
        int engineSyncThreads = positive("engineSyncExecutorThreads", flexlbConfig.getEngineSyncExecutorThreads());
        int engineSyncQueueCapacity = positive("engineSyncExecutorQueueCapacity",
                flexlbConfig.getEngineSyncExecutorQueueCapacity());
        int statusCheckThreads = positive("statusCheckExecutorThreads", flexlbConfig.getStatusCheckExecutorThreads());
        int statusCheckQueueCapacity = positive("statusCheckExecutorQueueCapacity",
                flexlbConfig.getStatusCheckExecutorQueueCapacity());

        engineSyncExecutor = new RejectionCountingThreadPoolExecutor(
                engineSyncThreads, engineSyncThreads, 0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(engineSyncQueueCapacity), new NamedThreadFactory("engine-sync-executor"),
                new ThreadPoolExecutor.AbortPolicy());

        statusCheckExecutor = new RejectionCountingThreadPoolExecutor(
                statusCheckThreads, statusCheckThreads, 0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(statusCheckQueueCapacity), new NamedThreadFactory("status-checker-executor"),
                new ThreadPoolExecutor.AbortPolicy());
    }

    protected abstract void syncEngineStatus();

    @PreDestroy
    public void destroy() {
        Optional.ofNullable(scheduler).ifPresent(s -> scheduler.shutdown());
        Optional.ofNullable(engineSyncExecutor).ifPresent(s -> engineSyncExecutor.shutdown());
        Optional.ofNullable(statusCheckExecutor).ifPresent(s -> statusCheckExecutor.shutdown());
    }

    private int positive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }
}
