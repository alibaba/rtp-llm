package org.flexlb.mock.grpc;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 铁律 4 guard — the engine never consumes priority for scheduling. On the
 * Java side this pins the two halves of the revised iron rule:
 *
 * <ul>
 *   <li>priority appears on the wire ONLY as the {@code GenerateConfigPB.priority}
 *       field (metrics tagging): two requests with identical payloads and
 *       different priorities produce byte-identical {@code GenerateInputPB}s
 *       once {@code request_id} and {@code generate_config.priority} are
 *       cleared — no other field carries priority information</li>
 *   <li>priority does not change how the engine (mock) receives or processes
 *       requests: the enqueue sequence observed by the engine equals the
 *       arrival order in both priority permutations (low→high and high→low),
 *       every request is accepted identically, and no Cancel is ever issued</li>
 * </ul>
 *
 * <p>All AUTO_TPM master-side switches are ON — master-side semantics (yield,
 * preempt) must never leak into the engine protocol beyond the metrics field.
 */
class EnginePriorityNeutralityGuardTest extends FlexLBMockTestBase {

    private static final int P_LOW = 30;
    private static final int P_HIGH = 70;

    @Override
    protected FlexlbConfig createConfig() {
        FlexlbConfig cfg = super.createConfig();
        cfg.setAutoTpmEnabled(true);
        cfg.setAutoTpmQueueYieldEnabled(true);
        return cfg;
    }

    // ---- half 1: priority lives only in GenerateConfigPB.priority ----

    @Test
    void identicalPayloads_differentPriorities_differOnlyInPriorityField() throws Exception {
        assertTrue(submitWithPriority(8301, P_LOW).get(5, TimeUnit.SECONDS).isSuccess());
        assertTrue(submitWithPriority(8302, P_HIGH).get(5, TimeUnit.SECONDS).isSuccess());

        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 2, 3_000,
                "both requests must reach the engine");

        List<EngineRpcService.GenerateInputPB> inputs = enqueuedInputs();
        assertEquals(2, inputs.size());
        EngineRpcService.GenerateInputPB low = inputs.get(0);
        EngineRpcService.GenerateInputPB high = inputs.get(1);

        // Priority is delivered in the dedicated metrics-only field...
        assertEquals(P_LOW, low.getGenerateConfig().getPriority());
        assertEquals(P_HIGH, high.getGenerateConfig().getPriority());

        // ...and NOWHERE else: with request_id and generate_config.priority
        // cleared the two inputs are byte-identical.
        assertArrayEquals(neutralized(low), neutralized(high),
                "priority must not leak into any engine field other than "
                        + "GenerateConfigPB.priority");

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ---- half 2: engine-side processing order == arrival order ----

    @Test
    void engineProcessingSequence_followsArrivalOrder_notPriority() throws Exception {
        // Low priority arrives first, high priority second — the engine must
        // see them in exactly that order (master never reorders on the wire,
        // engine never reorders on priority).
        CompletableFuture<Response> first = submitWithPriority(8311, P_LOW);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 1, 3_000,
                "first (low-priority) request must reach the engine first");
        CompletableFuture<Response> second = submitWithPriority(8312, P_HIGH);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 2, 3_000,
                "second (high-priority) request must reach the engine");

        assertTrue(first.get(5, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(5, TimeUnit.SECONDS).isSuccess());

        // Reversed permutation: high first, low second.
        CompletableFuture<Response> third = submitWithPriority(8313, P_HIGH);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 3, 3_000,
                "third (high-priority) request must reach the engine");
        CompletableFuture<Response> fourth = submitWithPriority(8314, P_LOW);
        awaitTrue(() -> mockPrefillWorker.getEnqueueCount() == 4, 3_000,
                "fourth (low-priority) request must reach the engine");

        assertTrue(third.get(5, TimeUnit.SECONDS).isSuccess());
        assertTrue(fourth.get(5, TimeUnit.SECONDS).isSuccess());

        // The engine-observed sequence equals the arrival order in both
        // permutations — same payload, priorities swapped, same behavior.
        List<Long> engineSequence = enqueuedInputs().stream()
                .map(EngineRpcService.GenerateInputPB::getRequestId)
                .toList();
        assertEquals(List.of(8311L, 8312L, 8313L, 8314L), engineSequence,
                "the engine must receive requests in arrival order regardless of priority");

        // The engine's handling of prioritized traffic involves no Cancel.
        assertEquals(0, mockPrefillWorker.getCancelCount(),
                "priority must never cause an engine-side prefill Cancel");
        assertEquals(0, mockDecodeWorker.getCancelCount(),
                "priority must never cause an engine-side decode Cancel");

        simulatePrefillFinishedReport();
        assertEquals(0, inflightStore.activeCount());
    }

    // ==================== helpers ====================

    private CompletableFuture<Response> submitWithPriority(long requestId, int priority) {
        BalanceContext ctx = createBalanceContext(requestId);
        ctx.setPriority(priority);
        ctx.getRequest().setPriority(priority);
        return scheduler.submit(ctx);
    }

    /** All GenerateInputPBs received by the mock prefill engine, in arrival order. */
    private List<EngineRpcService.GenerateInputPB> enqueuedInputs() {
        List<EngineRpcService.GenerateInputPB> inputs = new ArrayList<>();
        for (EngineRpcService.EnqueueBatchRequestPB batch
                : mockPrefillWorker.getRpcService().getEnqueuedRequests()) {
            for (EngineRpcService.EnqueueBatchDpSlotPB slot : batch.getDpSlotsList()) {
                for (EngineRpcService.EnqueueBatchExternalInputPB ext : slot.getRequestsList()) {
                    inputs.add(ext.getInput());
                }
            }
        }
        return inputs;
    }

    /** Serialized input with request_id and generate_config.priority cleared. */
    private static byte[] neutralized(EngineRpcService.GenerateInputPB input) {
        return input.toBuilder()
                .clearRequestId()
                .setGenerateConfig(input.getGenerateConfig().toBuilder().clearPriority())
                .build()
                .toByteArray();
    }

    private static void awaitTrue(java.util.function.BooleanSupplier condition,
                                  long timeoutMs, String message) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(20);
        }
        assertTrue(condition.getAsBoolean(), message);
    }
}
