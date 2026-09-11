package org.flexlb.cache.match;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/** Bounded request/role/generation correlation, independent of scheduling ownership. */
final class CacheHitFeedbackTracker {
    private static final Logger PV = LoggerFactory.getLogger("pvLogger");
    private final Cache<Key, Prediction> predictions = Caffeine.newBuilder()
            .maximumSize(100_000).expireAfterWrite(Duration.ofHours(1)).build();
    private final LocalStandbyComparisonService comparisonService;

    CacheHitFeedbackTracker(LocalStandbyComparisonService comparisonService) {
        this.comparisonService = comparisonService;
    }

    void track(String requestId,
               RoleType role,
               String group,
               WorkerStatus worker,
               long inputTokens,
               long predictedHitTokens,
               CacheMatchResult result) {
        HostCacheMatch match = result.hostMatch(worker);
        long blockSize = result.blockSize();
        boolean kvcm = result.source() == CacheMatchSource.KVCM && blockSize > 0;
        CacheHitFeedback seed = new CacheHitFeedback("cache_hit_comparison", requestId, result.source().name(),
                role.name(), group, worker.getWorkerIdentity(), null, inputTokens, blockSize, predictedHitTokens,
                kvcm, match == null ? 0 : CacheMatchResult.matchedTokens(match.localMatchBlocks(), blockSize, inputTokens),
                match == null ? 0 : CacheMatchResult.matchedTokens(match.p2pFetchBlocks(), blockSize, inputTokens),
                match == null ? 0 : CacheMatchResult.matchedTokens(match.p2pTotalMatchBlocks(), blockSize, inputTokens), 0, 0);
        predictions.put(new Key(worker.getGenerationId(), role, requestId), new Prediction(seed,
                comparisonService.captureComparison(requestId, role), blockSize > 0));
    }

    List<CompletableFuture<CacheHitComparisonResult>> observe(WorkerStatus worker,
                                                             WorkerStatus.StatusObservation status) {
        List<CompletableFuture<CacheHitComparisonResult>> results = new ArrayList<>();
        // Prefer terminal telemetry if the Engine includes the task in both lists.
        status.finishedTasks().values().forEach(task -> observeTask(worker, status.role(), task, true, results));
        status.runningTasks().values().forEach(task -> observeTask(worker, status.role(), task, false, results));
        return results;
    }

    private void observeTask(WorkerStatus worker,
                             RoleType role,
                             WorkerStatus.TaskObservation task,
                             boolean finished,
                             List<CompletableFuture<CacheHitComparisonResult>> results) {
        Prediction prediction = predictions.getIfPresent(new Key(worker.getGenerationId(), role, task.requestId()));
        if (prediction == null || task.telemetry() == null) {
            return;
        }
        WorkerStatus.TaskTelemetry telemetry = task.telemetry();
        CacheHitFeedback seed = prediction.seed;
        String state = finished ? "FINISHED" : task.phase() == null ? "UNKNOWN" : task.phase().name();
        if (telemetry.prefixLengthValid() && (finished || task.phase() == TaskPhase.RUNNING)
                && task.prefixLength() >= 0 && task.prefixLength() <= seed.inputTokens()) {
            Function<CacheHitFeedback, CompletableFuture<CacheHitComparisonResult>> comparison = prediction.comparison;
            prediction.comparison = null;
            if (comparison != null && prediction.predictionAvailable) {
                CacheHitFeedback feedback = new CacheHitFeedback(seed.eventType(), seed.requestId(),
                        seed.cacheMatchSource(), seed.role(), seed.group(), seed.workerIdentity(), state,
                        seed.inputTokens(), seed.blockSize(), seed.predictedHitTokens(), seed.kvcmMatchAvailable(),
                        seed.kvcmLocalMatchTokens(), seed.kvcmP2pFetchTokens(), seed.kvcmP2pTotalMatchTokens(),
                        task.prefixLength(), task.prefixLength() - seed.predictedHitTokens());
                results.add(comparison.apply(feedback));
            }
        }
        if ((finished || telemetry.firstTokenTimeMs() > 0) && !prediction.statusLogged) {
            Map<String, Object> pv = new LinkedHashMap<>(32);
            pv.put("event", "prefill_worker_status");
            pv.put("requestId", task.requestId());
            pv.put("role", role.name());
            pv.put("group", seed.group());
            pv.put("workerIp", seed.workerIdentity().getIp());
            pv.put("worker", seed.logicalWorkerId());
            pv.put("engineIndex", seed.engineIndex());
            pv.put("state", state);
            pv.put("inputTokens", seed.inputTokens());
            pv.put("engineInputTokens", positive(task.inputLength()));
            pv.put("inputTokensDelta", task.inputLength() > 0 ? task.inputLength() - seed.inputTokens() : null);
            pv.put("batchId", task.batchId());
            pv.put("errorCode", task.errorCode());
            pv.put("prefixLengthValid", telemetry.prefixLengthValid());
            pv.put("actualHitTokens", telemetry.prefixLengthValid() ? task.prefixLength() : null);
            pv.put("requestReceivedTimeMs", positive(telemetry.requestReceivedTimeMs()));
            pv.put("inputQueueEnqueueTimeMs", positive(telemetry.inputQueueEnqueueTimeMs()));
            pv.put("inputQueueDrainTimeMs", positive(telemetry.inputQueueDrainTimeMs()));
            pv.put("firstTokenTimeMs", positive(telemetry.firstTokenTimeMs()));
            pv.put("inputQueueWaitMs", elapsed(telemetry.inputQueueDrainTimeMs(), telemetry.inputQueueEnqueueTimeMs()));
            Long schedulerToRunning = elapsed(telemetry.runningEnteredTimeMs(), telemetry.waitingEnteredTimeMs());
            pv.put("schedulerToRunningMs", schedulerToRunning);
            pv.put("remoteKvWaitMs", schedulerToRunning == null ? null : telemetry.remoteKvWaitMs());
            pv.put("schedulerWaitMs", schedulerToRunning == null || schedulerToRunning < telemetry.remoteKvWaitMs()
                    ? null : schedulerToRunning - telemetry.remoteKvWaitMs());
            pv.put("runningToFirstTokenMs", elapsed(telemetry.firstTokenTimeMs(), telemetry.runningEnteredTimeMs()));
            if (telemetry.prefixLengthValid()
                    && telemetry.hbmLocalMatchTokens() + telemetry.remoteKvAddedMatchTokens() == task.prefixLength()) {
                pv.put("hbmLocalMatchTokens", telemetry.hbmLocalMatchTokens());
                pv.put("remoteKvAddedMatchTokens", telemetry.remoteKvAddedMatchTokens());
            }
            pv.put("firstPrefillStepId", positive(telemetry.firstPrefillStepId()));
            pv.put("lastPrefillStepId", positive(telemetry.lastPrefillStepId()));
            pv.put("prefillStepCount", positive(telemetry.prefillStepCount()));
            pv.put("prefillNonfinalChunkTokensMin", positive(telemetry.prefillNonfinalChunkTokensMin()));
            pv.put("prefillNonfinalChunkTokensMax", positive(telemetry.prefillNonfinalChunkTokensMax()));
            PV.info(JsonUtils.toStringOrEmpty(pv));

            prediction.statusLogged = true;
        }
    }

    private static Long positive(long value) {
        return value > 0 ? value : null;
    }

    private static Long elapsed(long end, long start) {
        return start > 0 && end >= start ? end - start : null;
    }

    private record Key(long generation, RoleType role, String requestId) {
    }

    private static final class Prediction {
        private final CacheHitFeedback seed;
        private final boolean predictionAvailable;
        private Function<CacheHitFeedback, CompletableFuture<CacheHitComparisonResult>> comparison;
        private boolean statusLogged = false;

        private Prediction(CacheHitFeedback seed,
                           Function<CacheHitFeedback, CompletableFuture<CacheHitComparisonResult>> comparison,
                           boolean predictionAvailable) {
            this.seed = seed;
            this.comparison = comparison;
            this.predictionAvailable = predictionAvailable;
        }
    }
}
