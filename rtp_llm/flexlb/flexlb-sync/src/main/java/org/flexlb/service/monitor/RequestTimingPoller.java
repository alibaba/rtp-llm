package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.RequestTimingClient;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.timing.grpc.PullRequestTimingsRequest;
import org.flexlb.timing.grpc.PullRequestTimingsResponse;
import org.flexlb.timing.grpc.RequestTimingRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Collects daemon summaries locally in standalone mode or on the elected ZooKeeper leader.
 * Cursors and callbacks run on one thread; engine polling keeps its existing cadence.
 */
@Component
public class RequestTimingPoller {
    private static final Logger log = LoggerFactory.getLogger(RequestTimingPoller.class);
    private final ConfigService configService;
    private final LBStatusConsistencyService leadership;
    private final RequestTimingClient client;
    private final FlexMonitor monitor;
    private final RequestTimingDiscovery discovery;
    private final ScheduledThreadPoolExecutor executor;
    private final Semaphore pendingWorkerBatches = new Semaphore(128);
    private final Map<RequestTimingSettings.Source, RequestTimingCursor> cursors = new HashMap<>();
    private final LinkedHashMap<String, String> workerSources = new LinkedHashMap<>(16, 0.75f, true);
    private RequestTimingSettings settings;
    private volatile TtftStageAggregator aggregator;
    private long leadershipGeneration = -1;
    private volatile long collectionGeneration;
    private volatile boolean closed;

    public RequestTimingPoller(ConfigService configService,
                               LBStatusConsistencyService leadership,
                               RequestTimingClient client,
                               FlexMonitor monitor,
                               RequestTimingDiscovery discovery) {
        this.configService = configService;
        this.leadership = leadership;
        this.client = client;
        this.monitor = monitor;
        this.discovery = discovery;
        executor = new ScheduledThreadPoolExecutor(1, task -> {
            Thread thread = new Thread(task, "ttft-timing-collector");
            thread.setDaemon(true);
            return thread;
        });
        executor.setRemoveOnCancelPolicy(true);
    }

    @PostConstruct
    public void start() {
        executor.scheduleAtFixedRate(this::tickSafely, 0, 1000, TimeUnit.MILLISECONDS);
    }

    private void tickSafely() {
        try {
            tick();
        } catch (RuntimeException error) {
            log.warn("Request timing collection failed", error);
        }
    }

    void tick() {
        RequestTimingSettings current =
                RequestTimingSettings.from(configService.loadBalanceConfig().getObservability().getTtftStages());
        long currentLeadership = leadership.getLeadershipGeneration();
        if (closed || !current.enabled() || !shouldCollectLocally()) {
            if (settings != null) {
                clear();
            }
            return;
        }
        if (!current.equals(settings) || currentLeadership != leadershipGeneration) {
            clear();
            settings = current;
            leadershipGeneration = currentLeadership;
            var nextAggregator = new TtftStageAggregator(monitor, current.maximumEntries(), current.retentionMs(),
                    current.graceMs(), current.remoteKvBeforeScheduler());
            // A new leader starts a new observation window; replayed retained records must not be counted again.
            nextAggregator.resetAll();
            aggregator = nextAggregator;
        }
        updateSources(discovery.sources());
        for (var entry : cursors.entrySet()) {
            if (!entry.getValue().inFlight && active(collectionGeneration)) {
                entry.getValue().pagesThisTick = 0;
                pull(entry.getKey(), entry.getValue());
            }
        }
        if (active(collectionGeneration)) {
            aggregator.publishReady();
        }
    }

    private void updateSources(Set<RequestTimingSettings.Source> sources) {
        Set<GrpcTarget> activeTargets = new HashSet<>();
        for (var source : sources) {
            activeTargets.add(source.target());
            cursors.computeIfAbsent(source, ignored -> new RequestTimingCursor());
        }
        cursors.keySet().removeIf(source -> {
            if (sources.contains(source)) {
                return false;
            }
            if (!activeTargets.contains(source.target())) {
                client.remove(source.target());
            }
            return true;
        });
    }

    private void pull(RequestTimingSettings.Source source, RequestTimingCursor cursor) {
        long generation = collectionGeneration;
        cursor.inFlight = true;
        cursor.pagesThisTick++;
        var request = PullRequestTimingsRequest.newBuilder()
                .setEpoch(cursor.epoch)
                .setAfterSequence(cursor.sequence)
                .setMaxRecords(settings.maxRecords())
                .build();
        try {
            client.pull(source.target(), request, settings.timeoutMs())
                    .whenComplete((response, failure) -> execute(
                            () -> finishPull(source, cursor, generation, response, failure)));
        } catch (RuntimeException failure) {
            cursor.inFlight = false;
            logFailure(source, cursor, failure);
        }
    }

    private void finishPull(RequestTimingSettings.Source source,
                            RequestTimingCursor cursor,
                            long generation,
                            PullRequestTimingsResponse response,
                            Throwable failure) {
        if (!active(generation) || cursors.get(source) != cursor) {
            return;
        }
        cursor.inFlight = false;
        if (failure != null) {
            logFailure(source, cursor, failure);
            return;
        }
        if (!validPage(cursor, response)) {
            logFailure(source, cursor, new IllegalArgumentException("invalid incremental timing response"));
            return;
        }
        String sourceKey = source.key();
        if (response.getGap() || (!cursor.epoch.isEmpty() && !cursor.epoch.equals(response.getEpoch()))) {
            aggregator.clearSource(sourceKey);
        }
        for (var record : response.getRecordsList()) {
            if (!validRecord(source.role(), record)) {
                continue;
            }
            aggregator.addTiming(sourceKey, record.getRequestId(), source.role(), record.getDeploymentName(),
                    "prefill".equals(source.role()) ? record.getReceivedTimeUs() : record.getDispatchStartUs(),
                    record.getFailedAtUs() == 0 ? null : record.getFailedAtUs());
        }
        cursor.epoch = response.getEpoch();
        cursor.sequence = response.getNextSequence();
        if (response.getHasMore() && cursor.pagesThisTick < 8 && active(generation)) {
            pull(source, cursor);
        }
    }

    private boolean validPage(RequestTimingCursor cursor, PullRequestTimingsResponse response) {
        if (response == null || response.getEpoch().isBlank() || response.getNextSequence() < 0
                || response.getRecordsCount() > settings.maxRecords()) {
            return false;
        }
        boolean reset = response.getGap() || !cursor.epoch.equals(response.getEpoch());
        long previous = reset ? 0 : cursor.sequence;
        boolean firstRecord = true;
        for (var record : response.getRecordsList()) {
            if (record.getSequence() <= previous || record.getSequence() > response.getNextSequence()
                    || ((!reset || !firstRecord) && record.getSequence() != previous + 1)) {
                return false;
            }
            previous = record.getSequence();
            firstRecord = false;
        }
        if (response.getRecordsCount() > 0) {
            return response.getNextSequence() == previous;
        }
        return !response.getHasMore() && (reset || response.getNextSequence() == cursor.sequence);
    }

    private static boolean validRecord(String role, RequestTimingRecord record) {
        if (!role.equals(record.getRole()) || !TtftStageAggregator.validTimingField(record.getRequestId())
                || !TtftStageAggregator.validTimingField(record.getDeploymentName())) {
            return false;
        }
        return switch (record.getEventKind()) {
            case PREFILL_RECEIVED -> "prefill".equals(role) && record.getReceivedTimeUs() > 0
                    && record.getDispatchStartUs() == 0 && record.getFailedAtUs() == 0;
            case DISPATCH_COMPLETED -> "frontend".equals(role) && record.getDispatchStartUs() > 0
                    && record.getReceivedTimeUs() == 0 && record.getFailedAtUs() == 0;
            case DISPATCH_FAILED -> "frontend".equals(role) && record.getDispatchStartUs() > 0
                    && record.getReceivedTimeUs() == 0 && record.getFailedAtUs() >= record.getDispatchStartUs();
            default -> false;
        };
    }

    /**
     * Adds committed finished tasks without changing the engine polling or reporting paths.
     */
    public void addWorkerStatus(WorkerStatus worker, WorkerStatus.StatusObservation observation) {
        if (closed || !configService.loadBalanceConfig().getObservability().getTtftStages().isEnabled()
                || !shouldCollectLocally()
                || (observation.role() != RoleType.PREFILL && observation.role() != RoleType.PDFUSION)
                || observation.finishedTasks().isEmpty()) {
            return;
        }
        long term = leadership.getLeadershipGeneration();
        long generation = collectionGeneration;
        TtftStageAggregator collector = aggregator;
        if (collector == null) {
            return;
        }
        String address = worker.getMetricIpPort();
        String source = "worker/" + address + "/" + worker.getGenerationId();
        if (!pendingWorkerBatches.tryAcquire()) {
            if (generation == collectionGeneration && term == leadership.getLeadershipGeneration()) {
                collector.discardWorkerStatus(source, observation.finishedTasks().values());
            }
            return;
        }
        if (!execute(() -> {
            try {
                if (active(generation) && term == leadershipGeneration && collector == aggregator) {
                    String previous = workerSources.put(address, source);
                    if (previous != null && !previous.equals(source)) {
                        collector.clearSource(previous);
                    }
                    if (workerSources.size() > settings.maximumEntries()) {
                        collector.clearSource(workerSources.pollFirstEntry().getValue());
                    }
                    for (var task : observation.finishedTasks().values()) {
                        collector.addWorkerStatus(source, task);
                    }
                }
            } finally {
                pendingWorkerBatches.release();
            }
        })) {
            pendingWorkerBatches.release();
        }
    }

    private boolean shouldCollectLocally() {
        return !leadership.isNeedConsistency() || leadership.isMaster();
    }

    private boolean active(long generation) {
        return !closed && settings != null && generation == collectionGeneration && shouldCollectLocally()
                && leadershipGeneration == leadership.getLeadershipGeneration()
                && configService.loadBalanceConfig().getObservability().getTtftStages().isEnabled();
    }

    private boolean execute(Runnable action) {
        try {
            executor.execute(action);
            return true;
        } catch (RejectedExecutionException ignored) {
            return false;
        }
    }

    private void logFailure(RequestTimingSettings.Source source, RequestTimingCursor cursor, Throwable failure) {
        long now = System.currentTimeMillis();
        if (now - cursor.lastFailureLogMs >= 60_000) {
            cursor.lastFailureLogMs = now;
            log.warn("Request timing pull failed at {}: {}", source.key(), failure.toString());
        }
    }

    private void clear() {
        collectionGeneration++;
        discovery.reset();
        for (var source : cursors.keySet()) {
            client.remove(source.target());
        }
        cursors.clear();
        workerSources.clear();
        if (aggregator != null) {
            aggregator.resetAll();
        }
        aggregator = null;
        settings = null;
        leadershipGeneration = -1;
    }

    @PreDestroy
    public void close() {
        closed = true;
        execute(this::clear);
        executor.shutdown();
    }
}
