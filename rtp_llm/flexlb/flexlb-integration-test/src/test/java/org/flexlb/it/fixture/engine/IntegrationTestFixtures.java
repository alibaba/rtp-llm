package org.flexlb.it.fixture.engine;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

/**
 * Shared lifecycle and scripting facade for scripted engine-worker integration-test fakes.
 *
 * <p>Context initializers start the shared fakes before Spring creates production beans. Test methods
 * then mutate a fake by {@code (RoleType, index)} and wait for the real synchronizer to publish that
 * state. A Failsafe fork uses one immutable {@link WorkerTopology}; do not try to replace a topology
 * after an application context has started.
 */
public final class IntegrationTestFixtures {

    /** Loopback address used by every fake service; ports provide distinct worker identities. */
    public static final String WORKER_IP = "127.0.0.1";

    /** Default topology used by single-role routing and cache-affinity scenarios. */
    public static final WorkerTopology PDFUSION_TWO_WORKERS = WorkerTopology.of(RoleType.PDFUSION, 2);

    private static final ScriptedWorkerCluster WORKERS = new ScriptedWorkerCluster();
    private static final AtomicLong STATUS_VERSION = new AtomicLong(1);

    private IntegrationTestFixtures() {}

    /** Returns all {@code GetWorkerStatus} calls observed by the fake cluster. */
    public static int workerStatusCalls() {
        return WORKERS.allWorkers().stream()
                .mapToInt(worker -> worker.service().workerStatusCalls())
                .sum();
    }

    /** Returns {@code GetWorkerStatus} calls for one role. */
    public static int workerStatusCalls(RoleType roleType) {
        return IntStream.range(0, workerCount(roleType))
                .map(index -> worker(roleType, index).service().workerStatusCalls())
                .sum();
    }

    /** Returns delayed {@code GetWorkerStatus} responses observed for one role/index fake. */
    public static int delayedWorkerStatusResponses(RoleType roleType, int index) {
        return worker(roleType, index).service().delayedWorkerStatusResponses();
    }

    /** Returns {@code GetCacheStatus} calls for one role. */
    public static int cacheStatusCalls(RoleType roleType) {
        return IntStream.range(0, workerCount(roleType))
                .map(index -> worker(roleType, index).service().cacheStatusCalls())
                .sum();
    }

    /** Returns the configured worker count for a role in the active topology. */
    public static int workerCount(RoleType roleType) {
        return WORKERS.topology().workerCount(roleType);
    }

    /** Returns the HTTP identity port corresponding to a scripted worker's status port. */
    public static int workerHttpPort(RoleType roleType, int index) {
        return worker(roleType, index).httpPort();
    }

    /** Returns the {@code ip:httpPort} identity used by routing and KVCM assertions. */
    public static String workerIpPort(RoleType roleType, int index) {
        return WORKER_IP + ":" + workerHttpPort(roleType, index);
    }

    /**
     * Clears FlexLB's locally scheduled work and resets scripted workers to one healthy index-zero
     * worker per configured role.
     */
    public static void resetWorkers() {
        clearScheduledTasks();
        for (RoleType roleType : RoleType.values()) {
            for (int index = 0; index < workerCount(roleType); index++) {
                worker(roleType, index).service().setCacheKeys();
                worker(roleType, index).service().clearWorkerStatusLatency();
                setWorkerStatus(roleType, index, index == 0, 0, 0, 1.0);
            }
        }
    }

    /**
     * Publishes the next status snapshot served by one fake worker.
     *
     * @param roleType target role
     * @param index zero-based worker index within that role
     * @param alive engine liveness exposed to FlexLB
     * @param waitingQueryLen engine-reported waiting request count
     * @param runningQueryLen engine-reported running request count
     * @param stepLatencyMs engine-reported step latency
     */
    public static void setWorkerStatus(
            RoleType roleType,
            int index,
            boolean alive,
            int waitingQueryLen,
            int runningQueryLen,
            double stepLatencyMs) {
        ScriptedWorker worker = worker(roleType, index);
        worker.service().setWorkerStatus(ScriptedWorkerStatusService.status(
                roleType,
                alive,
                STATUS_VERSION.incrementAndGet(),
                waitingQueryLen,
                runningQueryLen,
                stepLatencyMs));
    }

    /**
     * Publishes a 1M-token prefill snapshot with 16K/32K chunks to model a long-bucket worker.
     */
    public static void setWorkerLongPrefillStatus(RoleType roleType, int index) {
        worker(roleType, index).service().setWorkerStatus(ScriptedWorkerStatusService.status(
                roleType,
                true,
                STATUS_VERSION.incrementAndGet(),
                0,
                1,
                20.0,
                ScriptedWorkerStatusService.longRunningPrefillTask("long-prefill-" + roleType + "-" + index)));
    }

    /**
     * Publishes one running prefill task whose uncached input tokens contribute to queueing TTFT.
     *
     * <p>The cache snapshot is deliberately omitted from {@code GetWorkerStatus}; a separate real
     * {@code GetCacheStatus} response remains authoritative for cache keys.
     */
    public static void setWorkerRunningPrefillStatus(RoleType roleType, int index, long inputLength) {
        worker(roleType, index).service().setWorkerStatus(ScriptedWorkerStatusService.statusWithoutCacheStatus(
                roleType,
                true,
                STATUS_VERSION.incrementAndGet(),
                0,
                1,
                1.0,
                ScriptedWorkerStatusService.runningPrefillTask(
                        "running-prefill-" + roleType + "-" + index, inputLength)));
    }

    /** Sets cache keys returned by the worker's real {@code GetCacheStatus} RPC. */
    public static void setWorkerCacheKeys(RoleType roleType, int index, long... cacheKeys) {
        worker(roleType, index).service().setCacheKeys(cacheKeys);
    }

    /** Causes a worker's next and subsequent status RPCs to fail until cleared. */
    public static void failWorkerStatus(RoleType roleType, int index, Throwable failure) {
        worker(roleType, index).service().setWorkerStatusFailure(failure);
    }

    /** Restores successful worker-status RPCs across the active fake cluster. */
    public static void clearWorkerStatusFailures() {
        WORKERS.allWorkers().forEach(worker -> worker.service().setWorkerStatusFailure(null));
    }

    /**
     * Delays every Nth status response from one worker without blocking the fake gRPC handler.
     *
     * @param roleType target role
     * @param index zero-based worker index within that role
     * @param delay response latency to inject
     * @param everyCalls delay every Nth status RPC after this script is installed
     */
    public static void delayEveryWorkerStatusResponse(
            RoleType roleType, int index, Duration delay, int everyCalls) {
        worker(roleType, index).service().setWorkerStatusLatency(delay, everyCalls);
    }

    /** Removes all scripted worker-status latency from the active fake cluster. */
    public static void clearWorkerStatusLatencies() {
        WORKERS.allWorkers().forEach(worker -> worker.service().clearWorkerStatusLatency());
    }

    /** Starts the shared role-aware fake cluster for the current Failsafe JVM. */
    public static synchronized void startWorkers(WorkerTopology topology) {
        try {
            WORKERS.start(topology);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to start scripted workers", e);
        }
    }

    private static ScriptedWorker worker(RoleType roleType, int index) {
        return WORKERS.worker(roleType, index);
    }

    private static void clearScheduledTasks() {
        for (RoleType roleType : RoleType.values()) {
            for (WorkerStatus workerStatus : EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                    .getRoleStatusMap(roleType)
                    .values()) {
                workerStatus.getLock().lock();
                try {
                    workerStatus.getLocalTaskMap().clear();
                    workerStatus.getRunningQueueTime().set(0);
                    workerStatus.setInTransitAndWaitingTaskCount(0);
                    workerStatus.setInTransitAndWaitingUncachedTokens(0);
                    workerStatus.setRunningRemainingPrefillTokens(0);
                } finally {
                    workerStatus.getLock().unlock();
                }
            }
        }
    }

}
