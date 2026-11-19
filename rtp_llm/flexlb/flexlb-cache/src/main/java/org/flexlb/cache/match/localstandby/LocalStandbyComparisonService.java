package org.flexlb.cache.match.localstandby;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Compares selected-worker cache predictions with one valid Engine status sample per dispatch.
 */
final class LocalStandbyComparisonService {
    private static final Logger PV = LoggerFactory.getLogger("pvLogger");

    private final Cache<Dispatch, Prediction> pending = Caffeine.newBuilder()
            .maximumSize(100_000)
            .expireAfterWrite(Duration.ofMinutes(5))
            .build();
    private final Consumer<Comparison> reporter;

    LocalStandbyComparisonService(CacheMetricsReporter metrics) {
        this(comparison -> {
            PV.info(JsonUtils.toStringOrEmpty(comparison));
            Long standbyTokens = comparison.localStandbyPrediction() == null ? null
                    : comparison.localStandbyPrediction().predictedHitTokens();
            if (comparison.actualHitTokens() == null) {
                metrics.reportLocalStandbyPrediction(comparison.role(), comparison.worker(),
                        standbyTokens, comparison.inputTokens());
            } else {
                metrics.reportCacheHitComparison(comparison.role(), comparison.worker(), comparison.group(),
                        comparison.state(), comparison.inputTokens(), comparison.actualHitTokens(),
                        comparison.kvcmPrediction().localPredictionTokens(),
                        comparison.kvcmPrediction().globalPredictionTokens(), standbyTokens);
            }
        });
    }

    LocalStandbyComparisonService(Consumer<Comparison> reporter) {
        this.reporter = reporter;
    }

    Prediction register(Dispatch dispatch, String group, long inputTokens, long blockSize, int kvcmBlocks,
                        Long kvcmGlobalTokens) {
        Prediction prediction = new Prediction(group, inputTokens, blockSize, tokens(inputTokens, blockSize, kvcmBlocks),
                kvcmGlobalTokens == null ? -1 : kvcmGlobalTokens);
        return pending.asMap().putIfAbsent(dispatch, prediction) == null ? prediction : null;
    }

    void complete(Dispatch dispatch, Prediction prediction, Integer standbyBlocks) {
        Long standbyTokens = standbyBlocks == null ? null
                : tokens(prediction.inputTokens(), prediction.blockSize(), standbyBlocks);
        if (standbyTokens != null) {
            report("local_standby_prediction", dispatch, prediction, standbyTokens, null, null);
        }
        prediction.standbyTokens().complete(standbyTokens);
    }

    void discard(Dispatch dispatch, Prediction prediction) {
        pending.asMap().remove(dispatch, prediction);
        prediction.standbyTokens().cancel(false);
    }

    void observe(WorkerStatus worker, WorkerStatus.StatusObservation observation) {
        RoleType role = observation.role();
        if (observation.owner() != worker || !worker.isActiveGeneration()
                || (role != RoleType.PREFILL && role != RoleType.PDFUSION)) {
            return;
        }
        observation.finishedTasks().values().forEach(task -> observeTask(worker, role, task, "FINISHED"));
        observation.runningTasks().values().forEach(task -> observeTask(worker, role, task, "RUNNING"));
    }

    private void observeTask(WorkerStatus worker, RoleType role, WorkerStatus.TaskObservation task, String state) {
        if (!task.prefixLengthValid() || task.prefixLength() < 0) {
            return;
        }
        Dispatch dispatch = new Dispatch(task.requestId(), role, worker.getIpPort(),
                worker.getGenerationId(), Math.max(0, task.batchId()));
        Prediction prediction = pending.getIfPresent(dispatch);
        if (prediction != null && task.prefixLength() <= prediction.inputTokens()
                && prediction.observed().compareAndSet(false, true)) {
            prediction.standbyTokens().thenAccept(standbyTokens ->
                    report("cache_hit_comparison", dispatch, prediction, standbyTokens, task.prefixLength(), state));
        }
    }

    private void report(String event, Dispatch dispatch, Prediction prediction,
                        Long standbyTokens, Long actualTokens, String state) {
        reporter.accept(new Comparison(event, Long.toString(dispatch.requestId()), "KVCM", dispatch.role(),
                prediction.group(), dispatch.workerAddress(), dispatch.workerGeneration(), dispatch.batchId(),
                state, prediction.inputTokens(), prediction.blockSize(), actualTokens,
                new CachePrediction(prediction.kvcmLocalTokens(), prediction.kvcmLocalTokens(), prediction.kvcmGlobalTokens()),
                standbyTokens == null ? null : new CachePrediction(standbyTokens, -1, -1)));
    }

    private static long tokens(long inputTokens, long blockSize, int matchedBlocks) {
        return matchedBlocks > inputTokens / blockSize ? inputTokens : matchedBlocks * blockSize;
    }

    /**
     * Identity shared by a committed route and the selected worker's status.
     */
    record Dispatch(long requestId, RoleType role, String workerAddress, long workerGeneration, long batchId) {}

    /**
     * Holds the routing and global predictions while the shadow query and first valid status race.
     */
    record Prediction(String group, long inputTokens, long blockSize, long kvcmLocalTokens, long kvcmGlobalTokens,
                      CompletableFuture<Long> standbyTokens, AtomicBoolean observed) {
        Prediction(String group, long inputTokens, long blockSize, long kvcmLocalTokens, long kvcmGlobalTokens) {
            this(group, inputTokens, blockSize, kvcmLocalTokens, kvcmGlobalTokens,
                    new CompletableFuture<>(), new AtomicBoolean());
        }
    }

    /**
     * Records local KVCM and standby predictions separately from the Engine's actual prefix.
     */
    record Comparison(String event, String requestId, String source, RoleType role, String group, String worker,
                      long workerGeneration, long batchId, String state, long inputTokens, long blockSize,
                      Long actualHitTokens, CachePrediction kvcmPrediction, CachePrediction localStandbyPrediction) {}

    /**
     * A prediction in tokens; negative source-specific values are unavailable.
     */
    record CachePrediction(long predictedHitTokens, long localPredictionTokens, long globalPredictionTokens) {}
}
