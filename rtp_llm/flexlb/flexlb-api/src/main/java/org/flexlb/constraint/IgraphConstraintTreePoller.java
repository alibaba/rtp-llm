package org.flexlb.constraint;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.source.SidBucketClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.time.Clock;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

/** Disabled by default. A successful read denotes a complete keyspace traversal, not a source epoch. */
@Slf4j
@Component
@ConditionalOnProperty(name = "constraint.tree.igraph.enabled", havingValue = "true")
public class IgraphConstraintTreePoller {
    private final BucketSidReader reader;
    private final ConstraintTreeBuildService builds;
    private final BooleanSupplier leader;
    private final String model;
    private final boolean sourceReady;
    private final boolean allowNonAtomicRead;
    private final long intervalSeconds;
    private final Clock clock;
    private final AtomicBoolean running = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread thread = new Thread(runnable, "constraint-tree-igraph-reader");
        thread.setDaemon(true);
        return thread;
    });
    private volatile Status status = new Status("IDLE", 0, 0, 0, 0, 0, 0, "not started");

    public record Status(String state, long startedAtMillis, long finishedAtMillis, long submittedVersion,
                         int buckets, long items, long uniqueSids, String message) { }

    @Autowired
    public IgraphConstraintTreePoller(SidBucketClient client, ConstraintTreeBuildService builds,
                                     LBStatusConsistencyService consistency, Environment env) {
        this(new BucketSidReader(client, new BucketSidReader.Settings(
                        required(env, "key.prefix"), number(env, "bucket.count", 4096),
                        number(env, "concurrency", 16), number(env, "max.rows.per.bucket", 2000),
                        Duration.ofMillis(number(env, "query.timeout.ms", 5000)),
                        Duration.ofSeconds(number(env, "round.timeout.seconds", 300)),
                        number(env, "retries", 1))),
                builds, consistency::isMaster, required(env, "model"),
                Boolean.parseBoolean(env.getProperty("constraint.tree.igraph.source.ready", "false")),
                Boolean.parseBoolean(env.getProperty("constraint.tree.igraph.allow.non.atomic.read", "false")),
                number(env, "interval.seconds", 600), Clock.systemUTC());
    }

    IgraphConstraintTreePoller(BucketSidReader reader, ConstraintTreeBuildService builds, BooleanSupplier leader,
                               String model, boolean sourceReady, boolean allowNonAtomicRead,
                               long intervalSeconds, Clock clock) {
        if (model == null || model.isBlank() || intervalSeconds < 1) {
            throw new IllegalArgumentException("model and positive polling interval are required");
        }
        this.reader = reader;
        this.builds = builds;
        this.leader = leader;
        this.model = model;
        this.sourceReady = sourceReady;
        this.allowNonAtomicRead = allowNonAtomicRead;
        this.intervalSeconds = intervalSeconds;
        this.clock = clock;
    }

    @PostConstruct
    public void start() {
        scheduler.scheduleWithFixedDelay(this::pollOnce, intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
    }

    public Status getStatus() { return status; }

    /** The HTTP handler only enqueues work; it never waits for the source or CSR build. */
    public boolean trigger() {
        if (!eligible() || !running.compareAndSet(false, true)) { return false; }
        try {
            scheduler.execute(this::readAndSubmit);
            return true;
        } catch (RuntimeException e) {
            running.set(false);
            throw e;
        }
    }

    public void pollOnce() {
        try {
            if (!eligible() || !running.compareAndSet(false, true)) { return; }
            readAndSubmit();
        } catch (RuntimeException e) {
            // ScheduledExecutor suppresses all later executions if a scheduled callback throws.
            // In particular, a transient failure of the leader/status lookup must not stop polling.
            status = new Status("FAILED", 0, clock.millis(), 0, 0, 0, 0,
                    "refresh eligibility check failed; existing tree retained: " + e.getMessage());
            log.warn("iGraph refresh eligibility check failed; will retry next interval", e);
        }
    }

    private boolean eligible() {
        if (closed.get() || !leader.getAsBoolean()) { return false; }
        if (!sourceReady || !allowNonAtomicRead) {
            status = new Status("NOT_READY", 0, clock.millis(), 0, 0, 0, 0,
                    "confirm source initialization and explicitly accept non-atomic bucket reads before enabling publication");
            return false;
        }
        var state = builds.getStatus().state();
        return state != ConstraintTreeModels.BuildState.BUILDING
                && state != ConstraintTreeModels.BuildState.QUEUED
                && state != ConstraintTreeModels.BuildState.PUBLISHING;
    }

    private void readAndSubmit() {
        long started = clock.millis();
        status = new Status("READING", started, 0, 0, 0, 0, 0, "reading all configured buckets");
        try {
            var result = reader.read(() -> !closed.get() && leader.getAsBoolean());
            if (!eligible()) { throw new IllegalStateException("no longer eligible to submit tree"); }
            long version = Math.max(clock.millis(), Math.addExact(builds.getStatus().requestedVersion(), 1));
            var request = new ConstraintTreeModels.BuildRequest(version, model, null, null, null, null, result.sids());
            var submission = builds.submit(request);
            if (submission.state() != ConstraintTreeModels.SubmissionState.ACCEPTED) {
                throw new IllegalStateException("tree submission rejected: " + submission.state());
            }
            status = new Status("SUBMITTED", started, clock.millis(), version,
                    result.bucketCount(), result.itemCount(), result.sids().size(),
                    "read completed; check constraint_tree/status for actual build and Worker activation");
            log.info("iGraph tree input submitted: version={}, buckets={}, items={}, sids={}, readMs={}",
                    version, result.bucketCount(), result.itemCount(), result.sids().size(), result.elapsedMillis());
        } catch (Exception e) {
            status = new Status("FAILED", started, clock.millis(), 0, 0, 0, 0,
                    "source read/submission failed; existing tree retained: " + e.getMessage());
            log.warn("iGraph tree refresh failed; existing tree retained", e);
        } finally {
            running.set(false);
        }
    }

    @PreDestroy
    public void close() {
        closed.set(true);
        scheduler.shutdownNow();
    }

    private static int number(Environment env, String key, int fallback) {
        return Integer.parseInt(env.getProperty("constraint.tree.igraph." + key, Integer.toString(fallback)));
    }

    private static String required(Environment env, String key) {
        String value = env.getProperty("constraint.tree.igraph." + key);
        if (value == null || value.isBlank()) { throw new IllegalArgumentException("missing iGraph " + key); }
        return value;
    }
}
