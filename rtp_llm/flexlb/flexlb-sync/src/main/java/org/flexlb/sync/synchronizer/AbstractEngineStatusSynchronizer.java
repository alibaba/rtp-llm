package org.flexlb.sync.synchronizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.JsonUtils;
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
                                            ModelMetaConfig modelMetaConfig) {
        this.workerAddressService = workerAddressService;
        this.engineHealthReporter = engineHealthReporter;
        this.engineWorkerStatus = engineWorkerStatus;
        this.modelMetaConfig = modelMetaConfig;
        int corePoolSize = 500;
        int maximumPoolSize = 1000;

        engineSyncExecutor = new ThreadPoolExecutor(corePoolSize, maximumPoolSize, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(15000), new NamedThreadFactory("engine-sync-executor"),
                new ThreadPoolExecutor.AbortPolicy());

        statusCheckExecutor = new ThreadPoolExecutor(corePoolSize, maximumPoolSize, 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(15000), new NamedThreadFactory("status-checker-executor"),
                new ThreadPoolExecutor.AbortPolicy());

        String masterConfigStr = System.getenv("FLEXLB_CONFIG");
        logger.warn("FLEXLB_CONFIG = {}", masterConfigStr);
        FlexlbConfig masterConfig;
        if (masterConfigStr != null) {
            masterConfig = JsonUtils.toObject(masterConfigStr, FlexlbConfig.class);
        } else {
            masterConfig = new FlexlbConfig();
        }
        this.flexlbConfig = masterConfig;
    }

    protected abstract void syncEngineStatus();

    @PreDestroy
    public void destroy() {
        Optional.ofNullable(scheduler).ifPresent(s -> scheduler.shutdown());
        Optional.ofNullable(engineSyncExecutor).ifPresent(s -> engineSyncExecutor.shutdown());
        Optional.ofNullable(statusCheckExecutor).ifPresent(s -> statusCheckExecutor.shutdown());
    }
}
