package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@link BatchItemOrder#PRIORITY_FIRST}: priority desc, then
 * enqueuedAtMs asc, then requestId asc.
 */
class BatchItemOrderTest {

    @Test
    void higherPriorityComesFirst() {
        BatchItem p70 = item(1, 50, 300L, 70);
        BatchItem p50 = item(2, 40, 100L, 50);
        BatchItem p30 = item(3, 30, 200L, 30);

        List<BatchItem> items = new ArrayList<>(List.of(p50, p30, p70));
        items.sort(BatchItemOrder.PRIORITY_FIRST);

        assertEquals(List.of(70, 50, 30),
                items.stream().map(BatchItem::priority).toList());
    }

    @Test
    void samePriorityKeepsArrivalOrder() {
        BatchItem early = item(11, 50, 100L, 50);
        BatchItem late = item(12, 50, 200L, 50);

        List<BatchItem> items = new ArrayList<>(List.of(late, early));
        items.sort(BatchItemOrder.PRIORITY_FIRST);

        assertEquals(List.of(11L, 12L),
                items.stream().map(BatchItem::requestId).toList());
    }

    @Test
    void samePriorityAndArrivalBreaksTieByRequestId() {
        BatchItem lowId = item(21, 50, 100L, 50);
        BatchItem highId = item(22, 50, 100L, 50);

        List<BatchItem> items = new ArrayList<>(List.of(highId, lowId));
        items.sort(BatchItemOrder.PRIORITY_FIRST);

        assertEquals(List.of(21L, 22L),
                items.stream().map(BatchItem::requestId).toList());
    }

    // ---- helpers ----

    private static BatchItem item(long requestId, long seqLen, long enqueuedAtMs, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        return new BatchItem(
                balanceContext, new CompletableFuture<>(),
                null, null, null, null, null, enqueuedAtMs);
    }
}
