package org.flexlb.service;

import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RouteServiceTest {
    private final RequestScheduler scheduler = mock(RequestScheduler.class);
    private final RecentCacheKeyTraceReporter reporter = mock(RecentCacheKeyTraceReporter.class);
    private final RouteService service = new RouteService(scheduler, reporter);

    @Test
    void vitOnlyBypassesBatchPayloadValidationAndCacheReporting() {
        BalanceContext context = context(true);
        Response response = new Response();
        response.setSuccess(true);
        CompletableFuture<Response> future = CompletableFuture.completedFuture(response);
        when(scheduler.submit(context)).thenReturn(future);

        assertSame(future, service.route(context));
        assertSame(future, context.getFuture());
        assertSame(response, context.getResponse());
        verify(scheduler).submit(context);
        verifyNoInteractions(reporter);
    }

    @Test
    void pdStillRequiresBatchPayload() {
        Response response = service.route(context(false)).join();
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        verifyNoInteractions(scheduler, reporter);
    }

    @Test
    void nonBatchPreservesSchedulerFutureAndReportsSuccessfulPd() {
        BalanceContext context = context(false);
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(scheduler.submit(context)).thenReturn(future);

        assertSame(future, service.route(context));
        Response response = new Response();
        response.setSuccess(true);
        future.complete(response);

        assertSame(response, context.getResponse());
        verify(reporter).report(context);
    }

    private static BalanceContext context(boolean vitOnly) {
        BalanceContext context = new BalanceContext(SchedulingTestConfig.batchConfig());
        Request request = new Request();
        request.setRequestId(918L);
        request.setVitOnly(vitOnly);
        context.setRequest(request);
        return context;
    }
}
