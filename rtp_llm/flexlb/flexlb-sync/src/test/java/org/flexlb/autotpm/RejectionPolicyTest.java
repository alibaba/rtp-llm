package org.flexlb.autotpm;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link RejectionPolicy}.
 */
class RejectionPolicyTest {

    @Test
    void rejectYielded_completesWithCode8400AndPriorityMessage() {
        BatchItem item = makeItem(1);

        boolean result = RejectionPolicy.rejectYielded(item, 70);

        assertTrue(result);
        assertTrue(item.future().isDone());
        Response response = item.future().join();
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("auto_tpm: yielded for priority=70"));
    }

    @Test
    void rejectYielded_idempotent_secondCallReturnsFalse() {
        BatchItem item = makeItem(2);

        assertTrue(RejectionPolicy.rejectYielded(item, 60));
        assertFalse(RejectionPolicy.rejectYielded(item, 60));
    }

    @Test
    void rejectYielded_alreadyCompleted_returnsFalse() {
        BatchItem item = makeItem(3);
        item.future().complete(new Response()); // pre-complete

        assertFalse(RejectionPolicy.rejectYielded(item, 50));
    }

    // ---- D10: both queue-deadline clearing paths mark deadline miss ----

    @Test
    void rejectYielded_marksDeadlineMiss() {
        BatchItem item = makeItem(4);
        assertFalse(item.deadlineMissed());

        RejectionPolicy.rejectYielded(item, 70);

        assertTrue(item.deadlineMissed(),
                "yielded-queue-deadline clearing must mark the D10 deadline-miss flag");
    }

    @Test
    void rejectYielded_alreadyCompleted_doesNotMarkDeadlineMiss() {
        BatchItem item = makeItem(5);
        item.future().complete(new Response()); // pre-complete

        RejectionPolicy.rejectYielded(item, 70);

        assertFalse(item.deadlineMissed(),
                "an already-settled item must not be attributed a deadline miss");
    }

    @Test
    void failExpired_marksDeadlineMiss() {
        BatchItem item = makeItem(6);
        assertFalse(item.deadlineMissed());

        item.failExpired();

        assertTrue(item.deadlineMissed(),
                "legacy queue-deadline expiry must mark the D10 deadline-miss flag");
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                item.future().join().getCode());
    }

    // ---- helpers ----

    private static BatchItem makeItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return new BatchItem(ctx, new CompletableFuture<>(),
                null, null, null, null, null, System.currentTimeMillis());
    }
}
