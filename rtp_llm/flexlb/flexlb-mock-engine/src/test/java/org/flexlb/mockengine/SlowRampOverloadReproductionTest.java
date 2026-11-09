package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillQueueSheddingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression for global shedding under gradually increasing cold load.
 * Real routing/ownership/queue control with simulated Engine compute and in-process transport.
 * Cold-only traffic isolates overload detection; this does not reproduce cache-hit collapse.
 * Opt in with -Dflexlb.overload.reproduction=true for this wall-clock load test.
 */
@EnabledIfSystemProperty(named = "flexlb.overload.reproduction", matches = "true")
class SlowRampOverloadReproductionTest {
    @Test
    @Timeout(120)
    void gradualRampShedsOverdueWorkDespiteDispatchWatermarkOscillation() throws Exception {
        exercise(false);
    }

    @Test
    @Timeout(70)
    void calibratedDeadlineBudgetKeepsColdWorkBoundedAndShedsBeforeSixtySeconds() throws Exception {
        exercise(true);
    }

    private void exercise(boolean enableBudget) throws Exception {
        var config = new FlexlbConfig();
        config.queueScheduler().getDecision().setType(DecisionPolicyConfig.Type.SINGLE);
        config.queueScheduler().setQueueTimeoutMs(120_000L);
        config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
        config.getDispatcher().setMaxInflightPerPrefillWorker(256);
        var shedding = new PrefillQueueSheddingConfig();
        shedding.setExpectedWaitThresholdMs(60_000L);
        shedding.setMaxWaitMs(120_000L);
        if (enableBudget) {
            var budget = new PrefillQueueSheddingConfig.DeadlineBudgetConfig();
            budget.setTokensPerSecond(70_000); // Known Mock Engine capacity, not a production recommendation.
            shedding.setDeadlineBudget(budget);
        }
        config.queueScheduler().setPrefillQueueShedding(shedding);
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        balance.setEnableWindow(true);
        balance.setEngineQueueAdmissionThreshold(30L);
        balance.setMaxPendingComputeTokens(1_000_000L);
        var rescue = new RoutingConfig.HotspotRescueConfig();
        rescue.setMaxInputTokens(64_768L);
        rescue.setMaxIdlePendingTokens(128_768L);
        rescue.setMigrationIntervalMs(500L);
        balance.setHotspotRescue(rescue);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        var performance = Map.<String, Object>of("test_pool_blocks", 65_536, "jitter_pct", 0,
                "prefill", Map.of("max_batch_requests", 1, "max_batch_tokens", 100_000),
                "decode", Map.of("tokens_per_step", 2, "step_base_ms", 0, "step_per_running_ms", 1));
        try (var h = new AutoTpmE2EHarness(64600, 1, 1, "sum(computeTokens) / 70", 1,
                false, false, config.queueScheduler().getDecision(), true, config, 20, performance)) {
            h.startAutoPump(20);
            var results = new ArrayList<CompletableFuture<Response>>();
            var arrivals = new ArrayList<Long>();
            var errors = new AtomicReference<Throwable>();
            var completed = new AtomicInteger();
            int offered = 0, maxWaiting = 0, congestedSamples = 0;
            long maxPending = 0, oldestUnplaced = 0;
            long start = System.nanoTime(), nextSample = 0;
            double nextArrival = 0;
            while (System.nanoTime() - start < TimeUnit.SECONDS.toNanos(enableBudget ? 45 : 95)) {
                double elapsed = (System.nanoTime() - start) / 1e9;
                while (nextArrival <= elapsed) {
                    long id = 9_800_000L + offered++;
                    var context = h.context(id, 50, 100_000, 1);
                    context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50,
                            System.currentTimeMillis() + 100_000L));
                    arrivals.add(System.nanoTime());
                    var result = h.scheduler.submit(context);
                    results.add(result);
                    result.thenAccept(response -> {
                        if (!response.isSuccess()) { return; }
                        var input = MockEngineTestSupport.inputWithDecode(id, 100_000,
                                h.decodeEngines.getFirst().getGrpcPort(), 1);
                        h.engineScheduler.schedule(() -> h.prefillEngines.getFirst().generateStreamCall(input,
                                new io.grpc.stub.StreamObserver<>() {
                                    public void onNext(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB value) { }
                                    public void onError(Throwable error) { errors.compareAndSet(null, error); }
                                    public void onCompleted() { completed.incrementAndGet(); }
                                }), 20, TimeUnit.MILLISECONDS);
                    }).exceptionally(error -> { errors.compareAndSet(null, error); return null; });
                    nextArrival += 1.0 / (0.4 + Math.min(6.0, nextArrival * 0.25));
                }
                var endpoint = h.prefillEndpoint(0);
                maxWaiting = Math.max(maxWaiting, ((Number) h.prefillEngines.getFirst().getSnapshot().get("waiting")).intValue());
                maxPending = Math.max(maxPending, endpoint.ownership.tokenWorkload().pendingTokens());
                if (endpoint.engineQueueCongested()) { congestedSamples++; }
                oldestUnplaced = 0;
                for (int i = 0; i < results.size(); i++) {
                    if (!results.get(i).isDone()) {
                        oldestUnplaced = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - arrivals.get(i));
                        break;
                    }
                }
                if (elapsed >= nextSample) {
                    System.out.printf("RAMP_SAMPLE budget=" + enableBudget + " seconds=%.1f offered=%d completed=%d waiting=%s pending=%d oldest_unplaced_ms=%d rejected=%d congested=%s%n",
                            elapsed, offered, completed.get(), h.prefillEngines.getFirst().getSnapshot().get("waiting"),
                            endpoint.ownership.tokenWorkload().pendingTokens(), oldestUnplaced,
                            results.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null)).filter(r -> !r.isSuccess()).count(),
                            endpoint.engineQueueCongested());
                    nextSample += 10;
                }
                assertNull(errors.get(), "mock frontend/Engine transport failed");
                Thread.sleep(10L);
            }
            h.assertAutoPumpHealthy();
            assertTrue(results.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null))
                    .allMatch(r -> r.isSuccess() || r.getCode() == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()),
                    "unexpected scheduler error must not masquerade as overload shedding");
            long rejected = results.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null))
                    .filter(r -> r.getCode() == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()).count();
            System.out.printf("RAMP_RESULT budget=" + enableBudget + " offered=%d completed=%d max_waiting=%d max_pending=%d oldest_unplaced_ms=%d rejected=%d congested_samples=%d%n",
                    offered, completed.get(), maxWaiting, maxPending, oldestUnplaced, rejected, congestedSamples);
            if (enableBudget) {
                assertTrue(maxPending <= 1_000_000L, "cold work must respect the configured token cap");
                assertTrue(rejected > 0, "budget must reject excess work before sixty seconds");
                assertTrue(completed.get() >= 25, "cold backpressure must preserve worker throughput");
                return;
            }
            assertTrue(oldestUnplaced > 60_000, "must reach actual-age shedding threshold");
            assertTrue(maxPending > 1_000_000, "Engine queue override bypasses configured token cap");
            assertEquals(0, congestedSamples, "30/31 feedback keeps resetting sustained pressure");
            assertTrue(rejected > 0, "global waiting majority must shed overdue work despite the 30/31 dispatch oscillation");
            assertTrue(completed.get() >= 55, "Engine must keep serving, not hang");
        }
    }
}
