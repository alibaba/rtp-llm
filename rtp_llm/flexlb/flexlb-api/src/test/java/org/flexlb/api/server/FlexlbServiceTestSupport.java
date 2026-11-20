package org.flexlb.api.server;

import org.flexlb.api.ServerScheduleLatencyRecorder;
import org.flexlb.api.metrics.ApiMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.ConfigService;
import org.flexlb.master.MasterStatusView;

/**
 * Assembles API fixtures through the same scheduler and configuration as production.
 */
public final class FlexlbServiceTestSupport {
    public static FlexlbServiceImpl create(
            AbstractRequestScheduler scheduler,
            RequestRepository requests,
            MasterStatusView leadership,
            ApiMetricsReporter health,
            FlexlbGrpcForwarder forwarder,
            ConfigService config,
            ServerScheduleLatencyRecorder latency,
            RequestSchedulerReporter reporter) {
        if (org.mockito.Mockito.mockingDetails(leadership).isMock()) {
            org.mockito.Mockito.doCallRealMethod().when(leadership).shouldForwardToMaster();
        }
        if (org.mockito.Mockito.mockingDetails(scheduler).isMock()
                && !org.mockito.Mockito.mockingDetails(scheduler).isSpy()) {
            org.mockito.Mockito.lenient()
                    .doAnswer(
                            call -> {
                                var result =
                                        scheduler.submit(call.getArgument(0, RequestContext.class));
                                call.getArgument(1, Runnable.class).run();
                                return result;
                            })
                    .when(scheduler)
                    .submit(
                            org.mockito.ArgumentMatchers.any(RequestContext.class),
                            org.mockito.ArgumentMatchers.any(Runnable.class));
        }
        var service =
                new FlexlbServiceImpl(
                        new org.flexlb.balance.request.RequestRoutingService(
                                scheduler,
                                requests,
                                selector(scheduler),
                                org.mockito.Mockito.mock(
                                        org.flexlb.cache.match.CacheAwareService.class)),
                        config.loadBalanceConfig(),
                        requests,
                        leadership,
                        health,
                        forwarder,
                        latency,
                        reporter);
        return service;
    }

    static void configureRouting(
            FlexlbServiceImpl service, org.flexlb.balance.strategy.RequestRouteSelector selector) {
        Object entry =
                org.springframework.test.util.ReflectionTestUtils.getField(service, "requestEntry");
        org.springframework.test.util.ReflectionTestUtils.setField(entry, "selector", selector);
    }

    static void configureCache(
            FlexlbServiceImpl service, org.flexlb.cache.match.CacheAwareService cache) {
        Object entry =
                org.springframework.test.util.ReflectionTestUtils.getField(service, "requestEntry");
        org.springframework.test.util.ReflectionTestUtils.setField(entry, "cacheService", cache);
    }

    private static org.flexlb.balance.strategy.RequestRouteSelector selector(
            AbstractRequestScheduler scheduler) {
        if (scheduler instanceof org.flexlb.balance.schedule.DirectRequestScheduler
                || scheduler instanceof org.flexlb.balance.schedule.QueuedRequestScheduler)
            return (org.flexlb.balance.strategy.RequestRouteSelector)
                    org.springframework.test.util.ReflectionTestUtils.getField(scheduler, "router");
        return org.mockito.Mockito.mock(org.flexlb.balance.strategy.RequestRouteSelector.class);
    }
}
