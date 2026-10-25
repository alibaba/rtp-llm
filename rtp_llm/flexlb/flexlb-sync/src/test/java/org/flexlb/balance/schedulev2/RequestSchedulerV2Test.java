package org.flexlb.balance.schedulev2;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class RequestSchedulerV2Test {

    @Test
    void submitFailsUntilMainQueueIsConnectedAndLeavesNoTrackedRequest() {
        RequestSchedulerV2 scheduler = new RequestSchedulerV2();

        Response response = scheduler.submit(context("1", 60_000L)).join();

        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(0, scheduler.trackedRequestCount());
        assertNull(scheduler.getRequestState("1", 0L));
    }

    @Test
    void expiredRequestIsRejectedBeforeTracking() {
        RequestSchedulerV2 scheduler = new RequestSchedulerV2();

        Response response = scheduler.submit(context("2", -1L)).join();

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(0, scheduler.trackedRequestCount());
    }

    @Test
    void closedSchedulerRejectsNewRequests() {
        RequestSchedulerV2 scheduler = new RequestSchedulerV2();
        scheduler.close();

        Response response = scheduler.submit(context("3", 60_000L)).join();

        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(RequestSchedulerV2.SHUTTING_DOWN_MESSAGE, response.getErrorMessage());
    }

    private static BalanceContext context(String requestId, long expiresInMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + expiresInMs));
        return context;
    }
}
