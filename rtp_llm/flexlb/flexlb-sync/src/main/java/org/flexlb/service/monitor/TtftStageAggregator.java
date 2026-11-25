package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMonitor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Consumes finished P tasks after WorkerStatus commits its cursor. Keep sorting and metric publication
 * off the status polling threads. An overloaded collector may drop a batch without delaying the engine.
 */
@Component
public class TtftStageAggregator {
    private final ConfigService config;
    private final LBStatusConsistencyService leadership;
    private final TtftStageReporter reporter;
    private final TtftStageQuantiles windows;
    private final ThreadPoolExecutor executor = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(128), task -> {
                Thread thread = new Thread(task, "p-engine-ttft");
                thread.setDaemon(true);
                return thread;
            }, new ThreadPoolExecutor.DiscardPolicy());
    private long windowTerm = -1;
    private boolean remoteKvBeforeScheduler;

    public TtftStageAggregator(ConfigService config, LBStatusConsistencyService leadership, FlexMonitor monitor) {
        this.config = config;
        this.leadership = leadership;
        reporter = new TtftStageReporter(monitor);
        windows = new TtftStageQuantiles(reporter, 65_536, System.currentTimeMillis());
    }

    public void addWorkerStatus(WorkerStatus.StatusObservation observation) {
        if (observation.finishedTasks().isEmpty()
                || observation.role() != RoleType.PREFILL && observation.role() != RoleType.PDFUSION) {
            return;
        }
        long term = activeTerm();
        if (term < 0) {
            return;
        }
        executor.execute(() -> {
            long now = System.currentTimeMillis();
            if (prepareWindow(now) != term) {
                return;
            }
            for (var task : observation.finishedTasks().values()) {
                var timing = task.telemetry();
                if (timing == null) {
                    continue;
                }
                Long total = timing.engineAddToFirstTokenUs();
                Long ple = timing.schedulerWaitForPleReadyUs();
                if (ple != null && ple >= 0 && (total != null ? total >= ple : timing.firstTokenTimeMs() > 0)) {
                    reporter.reportPle(ple);
                }
                long[] stages = stages(timing, remoteKvBeforeScheduler);
                if (stages != null) {
                    windows.add(task.requestId(), stages, now);
                }
            }
        });
    }

    @Scheduled(fixedRate = 1000)
    public void flushClosedWindow() {
        executor.execute(() -> {
            long now = System.currentTimeMillis();
            if (prepareWindow(now) >= 0) {
                windows.advance(now);
            }
        });
    }

    private long activeTerm() {
        if (!config.loadBalanceConfig().getObservability().getTtftStages().isEnabled()
                || leadership.isNeedConsistency() && !leadership.isMaster()) {
            return -1;
        }
        return leadership.getLeadershipGeneration();
    }

    private long prepareWindow(long now) {
        long term = activeTerm();
        boolean includeRemoteKv = config.loadBalanceConfig().getObservability().getTtftStages().isRemoteKvBeforeScheduler();
        if (term != windowTerm || includeRemoteKv != remoteKvBeforeScheduler) {
            windows.reset(now);
            windowTerm = term;
            remoteKvBeforeScheduler = includeRemoteKv;
        }
        return term;
    }

    static long[] stages(WorkerStatus.TaskTelemetry timing, boolean includeRemoteKv) {
        Long total = timing.engineAddToFirstTokenUs();
        if (total == null || total < 0) {
            return null;
        }
        long remoteMs = includeRemoteKv ? timing.remoteKvWaitMs() : 0;
        if (remoteMs < 0 || remoteMs > total / 1000) {
            return null;
        }
        long[] values = {total, orZero(timing.hybridKvAllocWaitUs()), remoteMs * 1000,
                orZero(timing.schedulerWaitUs()), orZero(timing.prefillInflightWallUs()), 0};
        long other = total;
        for (int index = 1; index < values.length - 1; index++) {
            if (values[index] < 0 || values[index] > other) {
                return null;
            }
            other -= values[index];
        }
        values[values.length - 1] = other;
        return values;
    }

    private static long orZero(Long value) {
        return value == null ? 0 : value;
    }

    @PreDestroy
    public void close() {
        executor.shutdownNow();
    }
}
