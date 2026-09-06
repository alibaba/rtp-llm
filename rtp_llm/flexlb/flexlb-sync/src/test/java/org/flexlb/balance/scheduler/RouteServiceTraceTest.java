package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.RouteService;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RouteServiceTraceTest {
    @BeforeEach
    void setUp() {
        FlexlbTrace.configureEnabled(true);
    }

    @AfterEach
    void tearDown() {
        FlexlbTrace.configureEnabled(false);
    }

    @Test
    void modesKeepOriginalFutureOwnershipAndCompatibleTraceLabels() {
        for (String mode : new String[] {"DIRECT", "QUEUE", "BATCH"}) {
            FlexlbConfig config = new FlexlbConfig();
            if (mode.equals("DIRECT")) {
                config.setScheduler(SchedulerConfig.direct());
                SchedulingTestConfig.useNonBatchDispatcher(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
                if (mode.equals("QUEUE")) {
                    SchedulingTestConfig.useNonBatchDispatcher(config);
                } else {
                    SchedulingTestConfig.useBatchDispatcher(config);
                }
            }
            ConfigService configs = mock(ConfigService.class);
            when(configs.loadBalanceConfig()).thenReturn(config);
            RequestScheduler scheduler = mock(RequestScheduler.class);
            DefaultRouter router = mock(DefaultRouter.class);
            CompletableFuture<Response> pending = new CompletableFuture<>();
            when(scheduler.submit(any())).thenReturn(pending);
            Response response = new Response();
            response.setSuccess(true);
            when(router.routeDirect(any())).thenReturn(response);
            RouteService service = new RouteService(configs, router, scheduler,
                    mock(RecentCacheKeyTraceReporter.class));
            BalanceContext ctx = new BalanceContext();
            Request request = new Request();
            request.setRequestId(700L);
            ctx.setRequest(request);
            ctx.setGenerateInputPb(ByteString.copyFromUtf8("input"));
            Span span = mock(Span.class);
            when(span.storeInContext(any(Context.class))).thenCallRealMethod();
            ctx.setTraceContext(Context.root().with(span));
            CompletableFuture<Response> result = service.route(ctx);
            verify(span).setAttribute(FlexlbTrace.SCHEDULE_MODE, mode);
            if (mode.equals("DIRECT")) {
                assertSame(response, result.join());
                verifyNoInteractions(scheduler);
            } else {
                assertSame(pending, result);
                result.cancel(true);
                assertTrue(pending.isCancelled());
                verifyNoInteractions(router);
            }
        }
    }

    @Test
    void missingBatchInputStillRejectsWithoutDirectFallback() {
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        DefaultRouter router = mock(DefaultRouter.class);
        RequestScheduler scheduler = mock(RequestScheduler.class);
        RouteService service = new RouteService(configs, router, scheduler,
                mock(RecentCacheKeyTraceReporter.class));
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(701L);
        ctx.setRequest(request);
        Span span = mock(Span.class);
        when(span.storeInContext(any(Context.class))).thenCallRealMethod();
        ctx.setTraceContext(Context.root().with(span));
        Response response = service.route(ctx).join();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_BUILD_FAILED.getErrorCode(), response.getCode());
        verify(span).setAttribute(FlexlbTrace.SCHEDULE_MODE, "BATCH");
        verifyNoInteractions(router, scheduler);
    }
}
