package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;
    @Mock
    private DefaultRouter defaultRouter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private RoutingQueueReporter routingQueueReporter;

    @Test
    void cancelsQueuedRouteOnceWhenSubscriberCancels() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(true);
        BalanceContext balanceContext = balanceContext("request-1");
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(queueManager.tryRouteAsync(balanceContext)).thenReturn(Mono.never());
        RouteService routeService = new RouteService(
                configService, defaultRouter, queueManager, routingQueueReporter);

        Disposable subscription = routeService.route(balanceContext).subscribe();
        subscription.dispose();
        subscription.dispose();

        assertTrue(balanceContext.isCancelled());
        verify(routingQueueReporter).reportRoutingCancelled();
        verify(defaultRouter).rollBack(balanceContext, null);
    }

    @Test
    void cancelsActualQueuedRouteOnlyOnceWhenSubscriberCancels() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(10);
        BalanceContext balanceContext = balanceContext("request-1");
        when(configService.loadBalanceConfig()).thenReturn(config);
        QueueManager actualQueueManager = new QueueManager(routingQueueReporter, configService);
        RouteService routeService = new RouteService(
                configService, defaultRouter, actualQueueManager, routingQueueReporter);

        Disposable subscription = routeService.route(balanceContext).subscribe();
        subscription.dispose();

        assertTrue(balanceContext.isCancelled());
        verify(routingQueueReporter).reportRoutingCancelled();
        verify(defaultRouter).rollBack(balanceContext, null);
    }

    @Test
    void releasesSuccessfulDirectRouteWhenCancelledAfterResponseIsStored() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(false);
        BalanceContext balanceContext = balanceContext("request-1");
        Response response = new Response();
        response.setSuccess(true);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(defaultRouter.route(balanceContext)).thenReturn(Mono.just(response));
        RouteService routeService = new RouteService(
                configService, defaultRouter, queueManager, routingQueueReporter);

        routeService.route(balanceContext).block();
        routeService.cancel(balanceContext);

        assertTrue(balanceContext.isCancelled());
        verify(defaultRouter).rollBack(balanceContext, response);
        verify(routingQueueReporter).reportRoutingCancelled();
    }

    @Test
    void releasesSuccessfulRouteWhenSubscriberCancelsDuringResponseHandling() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(false);
        BalanceContext balanceContext = balanceContext("request-1");
        Response response = new Response();
        response.setSuccess(true);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(defaultRouter.route(balanceContext)).thenReturn(Mono.just(response));
        RouteService routeService = new RouteService(
                configService, defaultRouter, queueManager, routingQueueReporter);

        Disposable subscription = routeService
                .route(balanceContext, ignored -> Mono.<String>never())
                .subscribe();
        subscription.dispose();

        assertTrue(balanceContext.isCancelled());
        verify(defaultRouter).rollBack(balanceContext, response);
        verify(routingQueueReporter).reportRoutingCancelled();
    }

    @Test
    void reportsCancellationOnlyOnceWhenCancelIsCalledMoreThanOnce() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(false);
        BalanceContext balanceContext = balanceContext("request-1");
        when(configService.loadBalanceConfig()).thenReturn(config);
        RouteService routeService = new RouteService(
                configService, defaultRouter, queueManager, routingQueueReporter);

        routeService.cancel(balanceContext);
        routeService.cancel(balanceContext);

        verify(routingQueueReporter).reportRoutingCancelled();
        verify(defaultRouter).rollBack(balanceContext, null);
    }

    private BalanceContext balanceContext(String requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(request);
        return balanceContext;
    }
}
