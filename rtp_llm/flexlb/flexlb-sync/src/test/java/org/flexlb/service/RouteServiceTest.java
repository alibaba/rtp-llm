package org.flexlb.service;

import com.google.protobuf.ByteString;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RouteServiceTest {
    private final RequestScheduler scheduler = mock(RequestScheduler.class);
    private final TheoryCacheHitReporter reporter = mock(TheoryCacheHitReporter.class);
    private final RouteService service = new RouteService(scheduler, reporter);

    @Test
    void vitOnlyBypassesBatchPayloadValidationAndCacheReporting() {
        BalanceContext context = context(true);
        Response response = new Response();
        response.setSuccess(true);
        DefaultRouter router = mock(DefaultRouter.class);
        ReflectionTestUtils.setField(service, "router", router);
        when(router.routeVit(context)).thenReturn(response);

        assertSame(response, service.route(context).join());
        verify(router).routeVit(context);
        verifyNoInteractions(scheduler, reporter);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void batchRejectsMissingInputBeforeAdmission(boolean empty) {
        BalanceContext context = context(false);
        if (empty) {
            context.setGenerateInputPb(ByteString.EMPTY);
        }
        Response response = service.route(context).join();
        assertEquals(StrategyErrorType.GENERATE_INPUT_REQUIRED.getErrorCode(), response.getCode());
        assertFalse(response.isSuccess());
        assertFalse(response.isEnqueuedByMaster());
        assertNull(context.getFuture());
        verifyNoInteractions(scheduler, reporter);
    }

    @Test
    void nonBatchPreservesSchedulerFutureAndReportsSuccessfulPd() {
        BalanceContext context = context(false);
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        assertNull(context.getGenerateInputPb());
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(scheduler.submit(context)).thenReturn(future);

        assertSame(future, service.route(context));
        verify(scheduler).submit(context);
        Response response = new Response();
        response.setSuccess(true);
        future.complete(response);

        assertSame(response, context.getResponse());
        verify(reporter).report(context);
    }

    private static BalanceContext context(boolean vitOnly) {
        BalanceContext context = new BalanceContext(SchedulingTestConfig.batchConfig());
        Request request = new Request();
        request.setRequestId("918");
        request.setVitRouteOnly(vitOnly);
        context.setRequest(request);
        return context;
    }
}
