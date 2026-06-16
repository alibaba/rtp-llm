package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.grace.strategy.HealthCheckHooker;
import org.flexlb.service.grace.strategy.QueryWarmerHooker;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.concurrent.LinkedBlockingDeque;

import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verifyNoInteractions;

@ExtendWith(MockitoExtension.class)
class HttpLoadBalanceServerTest {

    @Mock
    private GeneralHttpNettyService generalHttpNettyService;

    @Mock
    private RouteService routeService;

    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private QueueManager queueManager;

    @Mock
    private ActiveRequestCounter activeRequestCounter;

    @Mock
    private ConfigService configService;

    @Mock
    private ServerRequest request;

    private HttpLoadBalanceServer server;

    @BeforeEach
    void setUp() {
        HealthCheckHooker.isShutDownSignalReceived = false;
        QueryWarmerHooker.warmUpFinished = true;
        server = new HttpLoadBalanceServer(
                generalHttpNettyService,
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                queueManager,
                activeRequestCounter,
                configService
        );
    }

    @AfterEach
    void tearDown() {
        HealthCheckHooker.isShutDownSignalReceived = false;
        QueryWarmerHooker.warmUpFinished = false;
    }

    @Test
    void should_reject_schedule_request_after_shutdown_signal_without_counting_active_request() {
        HealthCheckHooker.isShutDownSignalReceived = true;

        Mono<ServerResponse> result = server.scheduleRequest(request);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
        verifyNoInteractions(activeRequestCounter, routeService);
    }

    @Test
    void should_reject_schedule_request_before_warmup_without_counting_active_request() {
        QueryWarmerHooker.warmUpFinished = false;

        Mono<ServerResponse> result = server.scheduleRequest(request);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
        verifyNoInteractions(activeRequestCounter, routeService);
    }

    @Test
    void should_reject_master_info_before_warmup() {
        QueryWarmerHooker.warmUpFinished = false;

        Mono<ServerResponse> result = server.responseMasterInfo(request);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
        verifyNoInteractions(queueManager);
    }

    @Test
    void should_reject_master_info_after_shutdown_signal() {
        HealthCheckHooker.isShutDownSignalReceived = true;

        Mono<ServerResponse> result = server.responseMasterInfo(request);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
        verifyNoInteractions(queueManager);
    }

    @Test
    void should_return_master_info_after_warmup() {
        when(request.bodyToMono(org.flexlb.dao.loadbalance.Request.class))
                .thenReturn(Mono.just(new org.flexlb.dao.loadbalance.Request()));
        when(lbStatusConsistencyService.getMasterHostIpPort()).thenReturn("127.0.0.1:8080");
        when(queueManager.getQueue()).thenReturn(new LinkedBlockingDeque<>());

        Mono<ServerResponse> result = server.responseMasterInfo(request);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }
}
