package org.flexlb.service.lease;

import org.flexlb.config.ConfigService;
import org.flexlb.config.PrefillLeaseConfig;
import org.flexlb.service.lease.PrefillLeaseManager.WorkerView;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * One bounded node-count adjustment after fresh measurements from a stable serving set.
 */
@Component
public final class PrefillTtftLeaseController {
    private static final Logger LOG = LoggerFactory.getLogger(PrefillTtftLeaseController.class);
    private final ConfigService configs;
    private final PrefillLeaseManager leases;
    private final PrefillTtftStatusReader metrics;
    private final Map<String, GroupWindow> windows = new ConcurrentHashMap<>();
    private final ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(task -> {
        Thread thread = new Thread(task, "prefill-ttft-lease");
        thread.setDaemon(true);
        return thread;
    });
    private long lastEvaluationMs;

    public PrefillTtftLeaseController(ConfigService configs,
                                      PrefillLeaseManager leases,
                                      PrefillTtftStatusReader metrics) {
        this.configs = configs;
        this.leases = leases;
        this.metrics = metrics;
    }

    private static final class GroupWindow {
        private String members = "";
        private PrefillLeaseConfig config;
        private long stableSinceMs;
        private int direction;
        private int violations;
        private volatile String result = "waiting";
        private volatile PrefillTtftStatusReader.Measurement measurement;
    }

    @PostConstruct
    public void start() {
        executor.scheduleWithFixedDelay(this::evaluateSafely, 1, 1, TimeUnit.SECONDS);
    }

    @PreDestroy
    public void close() {
        executor.shutdownNow();
    }

    public Map<String, Object> status(String group) {
        GroupWindow window = windows.get(group);
        return window == null
                ? Map.of("result", "not evaluated")
                : Map.of("result", window.result, "measurement",
                        window.measurement == null ? "none" : window.measurement);
    }

    private void evaluateSafely() {
        try {
            PrefillLeaseConfig config = configs.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease();
            long now = System.currentTimeMillis();
            if (!config.isEnabled() || !config.isAutomatic()) {
                windows.clear();
                return;
            }
            if (now - lastEvaluationMs < config.getEvaluationIntervalMs()) {
                return;
            }
            lastEvaluationMs = now;
            windows.keySet().retainAll(config.getGroups());
            for (String group : config.getGroups()) {
                evaluate(group, config, now);
            }
        } catch (Exception failure) {
            LOG.warn("Prefill lease evaluation failed; keeping current leases: {}", failure.getClass().getSimpleName());
        }
    }

    void evaluate(String group, PrefillLeaseConfig config, long now) {
        GroupWindow window = windows.computeIfAbsent(group, ignored -> new GroupWindow());
        List<WorkerView> workers = leases.status(group);
        String members = members(workers);
        if (window.config != config || !window.members.equals(members)) {
            window.config = config;
            window.members = members;
            window.stableSinceMs = now;
            window.violations = 0;
            window.direction = 0;
            window.measurement = null;
        }
        boolean changing = workers.isEmpty() || workers.stream()
                .anyMatch(worker -> !worker.healthy() || !worker.confirmed() || "DRAINING".equals(worker.state()));
        if (changing) {
            window.stableSinceMs = now;
            window.violations = 0;
            window.result = "waiting for healthy workers and completed lease changes";
            return;
        }
        List<WorkerView> serving = workers.stream().filter(WorkerView::enabled).toList();
        if (serving.size() < config.getMinServingWorkers()) {
            List<String> targets = workers.stream()
                    .filter(worker -> !worker.enabled())
                    .limit(Math.min(config.getScaleUpStep(), config.getMinServingWorkers() - serving.size()))
                    .map(WorkerView::address)
                    .toList();
            List<String> changed = List.of();
            if (configs.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease() == config) {
                changed = leases.changeIfCurrent(group, members, targets, true, config);
            }
            if (!changed.isEmpty()) {
                window.result = "enabled " + changed + " to restore minServingWorkers";
                window.stableSinceMs = now;
            } else {
                window.result = "no current standby worker available to restore minServingWorkers";
            }
            window.violations = 0;
            return;
        }
        long waitMs = (config.getWindowSeconds() + config.getSettleSeconds() + config.getMaxMetricAgeSeconds()) * 1000;
        if (now - window.stableSinceMs < waitMs) {
            window.result = "waiting for a fresh window after the last lease change";
            return;
        }
        try {
            var measurement = metrics.read(config, serving.stream().map(WorkerView::address).toList(), now,
                    window.stableSinceMs + config.getSettleSeconds() * 1000);
            window.measurement = measurement;
            if (measurement.samples() < config.getMinSamples()) {
                window.violations = 0;
                window.result = "insufficient TTFT samples";
                return;
            }
            int direction = measurement.meanMs() > config.getTtftUpperMs()
                    ? 1
                    : measurement.meanMs() < config.getTtftLowerMs() ? -1 : 0;
            window.violations = direction == 0 ? 0 : direction == window.direction ? window.violations + 1 : 1;
            window.direction = direction;
            if (window.violations < config.getConsecutiveViolations()) {
                window.result = direction == 0 ? "TTFT inside target interval" : "waiting for consecutive violations";
                return;
            }
            List<String> targets = List.of();
            if (direction > 0 && serving.size() < config.getMaxServingWorkers()) {
                targets = workers.stream()
                        .filter(worker -> !worker.enabled())
                        .limit(Math.min(config.getScaleUpStep(), config.getMaxServingWorkers() - serving.size()))
                        .map(WorkerView::address)
                        .toList();
            } else if (direction < 0 && serving.size() > config.getMinServingWorkers()) {
                targets = serving.stream()
                        .sorted(Comparator.comparingInt(WorkerView::pendingRequests).thenComparing(WorkerView::address))
                        .limit(Math.min(config.getScaleDownStep(), serving.size() - config.getMinServingWorkers()))
                        .map(WorkerView::address)
                        .toList();
            }
            if (targets.isEmpty()) {
                window.result = "target outside available worker count bounds";
                window.violations = 0;
                return;
            }
            List<String> changed = List.of();
            if (configs.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease() == config) {
                changed = leases.changeIfCurrent(group, members, targets, direction > 0, config);
            }
            if (changed.isEmpty()) {
                window.result = "discarded measurement after configuration or worker set changed";
                window.violations = 0;
                return;
            }
            window.result = (direction > 0 ? "enabled " : "disabled ") + changed;
            window.stableSinceMs = now;
            window.violations = 0;
            LOG.info("Prefill lease group={} action={} meanTtftMs={} samples={}", group, window.result,
                    measurement.meanMs(), measurement.samples());
        } catch (Exception failure) {
            window.violations = 0;
            window.result = "Worker status TTFT unavailable: " + failure.getClass().getSimpleName();
            if (failure instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static String members(List<WorkerView> workers) {
        return workers.stream()
                .map(worker -> worker.address() + ":" + worker.generation() + ":" + worker.version() + ":"
                        + worker.enabled() + ":" + worker.confirmed() + ":" + worker.healthy() + ":" + worker.state())
                .collect(java.util.stream.Collectors.joining(";"));
    }
}
