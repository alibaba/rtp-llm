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
