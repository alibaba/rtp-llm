package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.ScheduleModeEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;

    @Mock
    private FlexlbConfig flexlbConfig;

    @Mock
    private DefaultRouter defaultRouter;

    @Mock
    private QueueManager queueManager;

    @Mock
    private org.flexlb.balance.scheduler.FlexlbBatchScheduler flexlbBatchScheduler;

    @Mock
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter, queueManager,
                flexlbBatchScheduler, recentCacheKeyTraceReporter);
    }

    @Test
    void should_report_recent_cache_key_once_after_queued_route_success() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(queueManager.tryRouteAsync(balanceContext)).thenReturn(Mono.just(response));

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(queueManager).tryRouteAsync(balanceContext);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void should_report_recent_cache_key_once_after_direct_route_success() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void should_not_report_recent_cache_key_after_route_failure() {
        Response response = new Response();
        response.setSuccess(false);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    // ---- Prefill seq_len admission gate ----

    @Test
    void should_not_reject_any_request_when_max_prefill_seq_len_is_default_zero() {
        Response response = successResponse();
        when(flexlbConfig.getMaxPrefillSeqLen()).thenReturn(0L);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        BalanceContext ctx = contextWithSeqLen(1L, 850_000L);
        when(defaultRouter.route(ctx)).thenReturn(response);

        Response actual = routeService.route(ctx).join();

        assertSame(response, actual);
        verify(defaultRouter).route(ctx);
    }

    @Test
    void should_reject_request_exceeding_max_prefill_seq_len_with_seq_len_exceeded_message() {
        when(flexlbConfig.getMaxPrefillSeqLen()).thenReturn(262_144L);
        BalanceContext ctx = contextWithSeqLen(2L, 300_000L);

        Response actual = routeService.route(ctx).join();

        assertFalse(actual.isSuccess());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), actual.getCode());
        assertTrue(actual.getErrorMessage().contains("SEQ_LEN_EXCEEDED"),
                "error message should carry the SEQ_LEN_EXCEEDED tag: " + actual.getErrorMessage());
        assertTrue(actual.getErrorMessage().contains("seq_len=300000"));
        assertTrue(actual.getErrorMessage().contains("max_prefill_seq_len=262144"));
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(flexlbBatchScheduler, never()).submit(any(BalanceContext.class));
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    @Test
    void should_admit_request_below_max_prefill_seq_len() {
        Response response = successResponse();
        when(flexlbConfig.getMaxPrefillSeqLen()).thenReturn(262_144L);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        BalanceContext ctx = contextWithSeqLen(3L, 100_000L);
        when(defaultRouter.route(ctx)).thenReturn(response);

        Response actual = routeService.route(ctx).join();

        assertSame(response, actual);
        verify(defaultRouter).route(ctx);
    }

    private static BalanceContext contextWithSeqLen(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
