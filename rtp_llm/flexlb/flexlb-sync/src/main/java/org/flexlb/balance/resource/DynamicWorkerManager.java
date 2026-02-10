package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Dynamic worker manager that manages the maximum number of concurrent workers
 * based on resource water levels. Periodically recalculates capacity based on
 * resource utilization.
 *
 * @author saichen.sm
 * @since 2026/02/01
 */
@Component
public class DynamicWorkerManager {

    private final ConfigService configService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final ReducibleSemaphore workerPermitSemaphore;

    private ScheduledExecutorService capacityScheduler;

    private volatile int allowedWorkers = 0;
    private final int maxTotalWorkers;
    private final AtomicInteger totalPermits;
    private static final int ADJUSTMENT_STEP = 1;

    public DynamicWorkerManager(ConfigService configService, ResourceMeasureFactory resourceMeasureFactory) {
        WhaleMasterConfig config = configService.loadBalanceConfig();
        this.maxTotalWorkers = config.getScheduleWorkerSize();
        this.configService = configService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.workerPermitSemaphore = new ReducibleSemaphore(maxTotalWorkers);
        this.totalPermits = new AtomicInteger(maxTotalWorkers);
    }

    @PostConstruct
    public void startScheduler() {
        WhaleMasterConfig config = configService.loadBalanceConfig();

        this.capacityScheduler = Executors.newScheduledThreadPool(1, r -> {
            Thread t = new Thread(r, "worker-capacity-scheduler");
            t.setDaemon(true);
            return t;
        });
        capacityScheduler.scheduleWithFixedDelay(
                this::recalculateWorkerCapacity,
                config.getResourceCheckIntervalMs(),
                config.getResourceCheckIntervalMs(),
                TimeUnit.MILLISECONDS
        );
        Logger.info("Worker capacity scheduler started, interval: {}ms", config.getResourceCheckIntervalMs());
    }

    @PreDestroy
    public void shutdown() {
        if (capacityScheduler != null && !capacityScheduler.isShutdown()) {
            capacityScheduler.shutdown();
            try {
                if (!capacityScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    capacityScheduler.shutdownNow();
                }
                Logger.info("Worker capacity scheduler stopped");
            } catch (InterruptedException e) {
                capacityScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public void recalculateWorkerCapacity() {

        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypeList = modelWorkerStatus.getRoleTypeList();
        double maxWaterLevel = 0.0;

        for (RoleType roleType : roleTypeList) {
            Map<String, WorkerStatus> workerStatusMap = modelWorkerStatus.getRoleStatusMap(roleType);
            ResourceMeasure measure = resourceMeasureFactory.getMeasure(roleType.getResourceMeasureIndicator());
            if (measure != null) {
                double waterLevel = measure.calculateAverageWaterLevel(workerStatusMap);
                maxWaterLevel = Math.max(maxWaterLevel, waterLevel);
                Logger.debug("Role: {}, water level: {}%", roleType, waterLevel);
            }
        }

        int newAllowedWorkers = calculateAllowedWorkers(maxWaterLevel);
        int oldAllowedWorkers = allowedWorkers;
        allowedWorkers = newAllowedWorkers;
        Logger.debug("Final water level: {}%, allowedWorkers: {} -> {}", maxWaterLevel, oldAllowedWorkers, newAllowedWorkers);

        adjustPermitCapacity(allowedWorkers);
    }

    private int calculateAllowedWorkers(double waterLevel) {
        double capacity = 1.0 - (waterLevel / 100.0);
        int maxWorkers = (int) (maxTotalWorkers * capacity);
        return Math.max(0, maxWorkers);
    }

    private void adjustPermitCapacity(int desiredCapacity) {
        int currentPermits = totalPermits.get();
        int capacityDelta = desiredCapacity - currentPermits;

        if (capacityDelta == 0) {
            return;
        }

        if (capacityDelta > 0) {
            workerPermitSemaphore.release(ADJUSTMENT_STEP);
            totalPermits.addAndGet(ADJUSTMENT_STEP);
            Logger.debug("Released {} worker permits, current capacity: {}, target: {}", ADJUSTMENT_STEP, totalPermits.get(), desiredCapacity);
        } else {
            workerPermitSemaphore.reducePermits(ADJUSTMENT_STEP);
            totalPermits.addAndGet(-ADJUSTMENT_STEP);
            Logger.debug("Reduced {} worker permits, current capacity: {}, target: {}", ADJUSTMENT_STEP, totalPermits.get(), desiredCapacity);
        }
    }

    public void acquirePermit() {
        workerPermitSemaphore.acquireUninterruptibly();
    }

    public void releasePermit() {
        workerPermitSemaphore.release();
    }

    public int getTotalPermits() {
        return totalPermits.get();
    }
}