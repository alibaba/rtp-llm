package org.flexlb.constraint;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** Wakes the existing build/publish machinery while Carbon has not yet registered a Worker. */
@Slf4j
@Component
public class ConstraintTreeBootstrapService {
    private final ConstraintTreeBootstrapRegistry registry;
    private final ConstraintTreeBuildService builds;
    private final ObjectProvider<IgraphConstraintTreePoller> poller;
    private final LBStatusConsistencyService consistency;
    private final ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread thread = new Thread(runnable, "constraint-tree-bootstrap");
        thread.setDaemon(true);
        return thread;
    });
    private long nextColdBuildNanos;

    public ConstraintTreeBootstrapService(ConstraintTreeBootstrapRegistry registry, ConstraintTreeBuildService builds,
                                         ObjectProvider<IgraphConstraintTreePoller> poller,
                                         LBStatusConsistencyService consistency) {
        this.registry = registry;
        this.builds = builds;
        this.poller = poller;
        this.consistency = consistency;
    }

    @PostConstruct
    public void start() {
        executor.scheduleWithFixedDelay(this::tick, 0, 10, TimeUnit.SECONDS);
    }

    void tick() {
        try {
            var models = registry.pendingModels();
            if (models.isEmpty() || !consistency.isMaster()) { return; }
            var current = builds.getCurrentArtifact();
            if (current.isPresent()) {
                if (models.contains(current.get().metadata().model())) { builds.reconcileCurrent(); }
                return;
            }
            // Repeated registration must not produce a tight loop of 4000-bucket scans.
            long now = System.nanoTime();
            if (now - nextColdBuildNanos < 0) { return; }
            var source = poller.getIfAvailable();
            if (source != null && models.contains(source.getModel()) && source.trigger()) {
                nextColdBuildNanos = now + TimeUnit.SECONDS.toNanos(60);
            }
        } catch (Exception e) {
            log.warn("constraint tree bootstrap will retry; Worker readiness remains gated", e);
        }
    }

    @PreDestroy
    public void close() { executor.shutdownNow(); }
}
