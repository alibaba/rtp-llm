package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
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
        FlexlbTrace.configure(io.opentelemetry.api.OpenTelemetry.noop(), "");
    }

    @AfterEach
    void tearDown() {
        FlexlbTrace.configure(null, "");
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
            RequestScheduler scheduler = mock(RequestScheduler.class);
            CompletableFuture<Response> pending = new CompletableFuture<>();
            when(scheduler.submit(any())).thenReturn(pending);
            RouteService service = new RouteService(scheduler,
                    mock(RecentCacheKeyTraceReporter.class));
            BalanceContext ctx = new BalanceContext(config);
            Request request = new Request();
            request.setRequestId(700L);
            ctx.setRequest(request);
            ctx.setGenerateInputPb(ByteString.copyFromUtf8("input"));
            Span span = mock(Span.class);
            when(span.storeInContext(any(Context.class))).thenCallRealMethod();
            ctx.setTraceContext(Context.root().with(span));
            CompletableFuture<Response> result = service.route(ctx);
            verify(span).setAttribute(FlexlbTrace.SCHEDULE_MODE, mode);
            assertSame(pending, result);
            result.cancel(true);
            assertTrue(pending.isCancelled());
            verify(scheduler).submit(ctx);
        }
    }

    @Test
    void missingBatchInputStillRejectsWithoutDirectFallback() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        RequestScheduler scheduler = mock(RequestScheduler.class);
        RouteService service = new RouteService(scheduler,
                mock(RecentCacheKeyTraceReporter.class));
        BalanceContext ctx = new BalanceContext(config);
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
        verifyNoInteractions(scheduler);
    }
}
