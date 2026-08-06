package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Auto-TPM priority normalization tests for {@link FlexlbServiceImpl}:
 * valid values {30,40,50,60,70} are preserved, everything else (including
 * the proto3 default 0 from old clients) becomes 50.
 */
class FlexlbServiceImplPriorityTest {

    private RouteService routeService;
    private FlexlbServiceImpl service;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        LBStatusConsistencyService lbStatusConsistencyService = mock(LBStatusConsistencyService.class);
        when(lbStatusConsistencyService.isNeedConsistency()).thenReturn(false);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());

        ActiveRequestCounter activeRequestCounter = mock(ActiveRequestCounter.class);
        when(activeRequestCounter.acquire()).thenReturn(mock(ActiveRequestCounter.RequestToken.class));

        service = new FlexlbServiceImpl(
                routeService,
                lbStatusConsistencyService,
                mock(EngineHealthReporter.class),
                activeRequestCounter,
                mock(FlexlbGrpcForwarder.class),
                configService,
                mock(BatchSchedulerReporter.class),
                mock(ServerScheduleLatencyRecorder.class)
        );
    }

    @Test
    void missingPriorityIsNormalizedToDefault50() {
        assertEquals(50, scheduledPriority(0));
    }

    @Test
    void invalidPrioritiesAreNormalizedTo50() {
        assertEquals(50, scheduledPriority(45));
        assertEquals(50, scheduledPriority(99));
        assertEquals(50, scheduledPriority(-1));
    }

    @Test
    void validPrioritiesArePreserved() {
        assertEquals(30, scheduledPriority(30));
        assertEquals(40, scheduledPriority(40));
        assertEquals(50, scheduledPriority(50));
        assertEquals(60, scheduledPriority(60));
        assertEquals(70, scheduledPriority(70));
    }

    // ---- helpers ----

    /** Runs schedule() with the given proto priority and returns the priority set on the Request. */
    @SuppressWarnings("unchecked")
    private int scheduledPriority(int protoPriority) {
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);

        ArgumentCaptor<BalanceContext> ctxCaptor = ArgumentCaptor.forClass(BalanceContext.class);
        when(routeService.route(ctxCaptor.capture())).thenReturn(CompletableFuture.completedFuture(response));

        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request = FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(12345L)
                .setSeqLen(100)
                .setPriority(protoPriority)
                .build();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer = mock(StreamObserver.class);
        service.schedule(request, observer);

        return ctxCaptor.getValue().getRequest().getPriority();
    }
}
