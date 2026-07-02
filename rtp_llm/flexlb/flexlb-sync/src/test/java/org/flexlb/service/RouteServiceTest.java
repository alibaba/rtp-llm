package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;
    @Mock
    private DefaultRouter router;
    @Mock
    private QueueManager queueManager;

    /**
     * Direct (non-queue) routing must run inline on the subscribing thread. {@code router.route}
     * commits a worker reservation (increments the selected worker's local task count); offloading
     * it to a scheduler would let a client cancel drop the produced result after the reservation is
     * taken, leaking it with no rollback path. This pins the no-scheduler-hop invariant so the
     * offload cannot be reintroduced unnoticed.
     */
    @Test
    void directRouteRunsInlineOnSubscriberThread() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(false);
        when(configService.loadBalanceConfig()).thenReturn(config);

        AtomicReference<Thread> routeThread = new AtomicReference<>();
        Response response = mock(Response.class);
        when(router.route(any())).thenAnswer(invocation -> {
            routeThread.set(Thread.currentThread());
            return response;
        });

        RouteService routeService = new RouteService(configService, router, queueManager);
        Thread caller = Thread.currentThread();

        routeService.route(new BalanceContext()).block();

        assertSame(caller, routeThread.get(),
                "direct routing must execute on the subscribing thread (no Schedulers hop), so a "
                        + "client cancel cannot race the worker reservation taken in router.route()");
    }
}
