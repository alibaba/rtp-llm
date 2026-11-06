package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillQueueSheddingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/** Real selector, scheduler, endpoint ownership and mock Engine status; delayed in-process transport. */
class PrefillDeadlineBudgetE2ETest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    @Timeout(60)
    void saturatedColdBurstIsRelievedBeforeTheLegacySixtySecondThreshold(boolean batch) throws Exception {
        long legacy = exercise(false, batch);
        long budget = exercise(true, batch);
        assertEquals(0, legacy, "legacy must remain the actual-age baseline");
        assertTrue(budget > 0 && budget < 160, "deadline policy removes only part of the overload");
    }

    private long exercise(boolean enableBudget, boolean batch) throws Exception {
        var config = new FlexlbConfig();
        config.getDispatcher().setType(batch ? DispatcherConfig.Type.BATCH : DispatcherConfig.Type.NON_BATCH);
        config.getDispatcher().setMaxInflightPerPrefillWorker(256);
        config.queueScheduler().getDecision().setType(DecisionPolicyConfig.Type.SINGLE);
        config.queueScheduler().setQueueTimeoutMs(120_000);
        config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
        var shedding = new PrefillQueueSheddingConfig();
        shedding.setExpectedWaitThresholdMs(60_000);
        shedding.setMaxWaitMs(120_000);
        if (enableBudget) {
            var budget = new PrefillQueueSheddingConfig.DeadlineBudgetConfig();
            budget.setTokensPerSecond(70_000);
            shedding.setDeadlineBudget(budget);
        }
        config.queueScheduler().setPrefillQueueShedding(shedding);
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        balance.setEngineQueueAdmissionThreshold(30L);
        balance.setMaxPendingComputeTokens(200_000);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        var performance = Map.<String, Object>of(
                "test_pool_blocks", 65_536, "jitter_pct", 0.1,
                "prefill", Map.of("max_batch_requests", 1, "max_batch_tokens", 100_000),
                "decode", Map.of("tokens_per_step", 2, "step_base_ms", 0, "step_per_running_ms", 1));
        try (var h = new AutoTpmE2EHarness(64100, 1, 1, "sum(computeTokens) / 70", 1,
                false, false, config.queueScheduler().getDecision(), true, config, 20, performance)) {
            h.startAutoPump(20);
            var futures = new ArrayList<CompletableFuture<Response>>();
            var transportFailure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
            for (int i = 0; i < 160; i++) {
                var context = h.context(900_000 + i, 50, 100_000, 32);
                context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 100_000));
                var result = h.scheduler.submit(context);
                futures.add(result);
                if (!batch) {
                    result.thenAccept(response -> {
                        if (!response.isSuccess()) { return; }
                        long id = context.data().getRequestId();
                        var input = MockEngineTestSupport.inputWithDecode(id, 100_000,
                                h.decodeEngines.getFirst().getGrpcPort(), 32);
                        h.engineScheduler.schedule(() -> {
                            h.engineArrivalOrder.add(id);
                            h.prefillEngines.getFirst().generateStreamCall(input, new io.grpc.stub.StreamObserver<>() {
                                @Override public void onNext(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB output) { }
                                @Override public void onError(Throwable failure) { transportFailure.compareAndSet(null, failure); }
                                @Override public void onCompleted() { }
                            });
                        }, 20, TimeUnit.MILLISECONDS);
                    });
                }
            }
            try { AutoTpmE2EHarness.await(() -> enableBudget
                            ? h.prefillEndpoint(0).ownership.tokenWorkload().pendingTokens() >= 200_000
                            : h.prefillEndpoint(0).engineQueueDispatchBlocked(), 10_000,
                    "real delayed WorkerStatus must report the overloaded Engine queue"); }
            catch (AssertionError failure) {
                h.assertAutoPumpHealthy();
                throw new AssertionError("arrivals=" + h.engineArrivalOrder.size() + ", pending="
                        + h.prefillEndpoint(0).ownership.tokenWorkload() + ", status="
                        + h.prefillEndpoint(0).getStatus().committedWorkerStatus().fields()
                        + ", responses=" + futures.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null))
                        .collect(java.util.stream.Collectors.groupingBy(r -> r.getCode() + ":" + r.getErrorMessage(), java.util.stream.Collectors.counting())), failure);
            }
            if (enableBudget) {
                AutoTpmE2EHarness.await(() -> shedCount(futures) > 0, 10_000,
                        "cold deadline-budget shedding must occur before 60 seconds of real queue age");
                AutoTpmE2EHarness.await(() -> ((Number) h.prefillEngines.getFirst().getSnapshot().get("prefill_ms_count")).longValue() >= 4,
                        9_000, "cold backpressure must keep refilling the real Engine after completion");
                assertTrue(h.prefillEndpoint(0).ownership.tokenWorkload().pendingTokens() <= 200_000,
                        "delayed status must not relax the cold work limit");
            } else {
                TimeUnit.SECONDS.sleep(4);
            }
            h.assertAutoPumpHealthy();
            assertNull(transportFailure.get(), "mock frontend transport must succeed");
            long shed = shedCount(futures);
            assertTrue(h.engineArrivalOrder.size() > 0, "keep the worker serving accepted requests");
            assertTrue(futures.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null))
                    .allMatch(r -> r.isSuccess() || r.getCode() == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()),
                    "no unexpected protocol, KV or timeout errors");
            return shed;
        }
    }

    private static long shedCount(ArrayList<CompletableFuture<Response>> futures) {
        return futures.stream().filter(CompletableFuture::isDone).map(f -> f.getNow(null))
                .filter(r -> r.getCode() == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()).count();
    }
}
