package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.config.MeterFilterReply;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;

import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CriticalMetricsFilterConfigTest {

    private static final String CRITICAL_LIST =
            "app.request.network.delay.ms,app.grpc.server.process.ms,"
            + "app.flexlb.route.submit.time.ms,app.routing.queue.wait.time.ms,"
            + "app.flexlb.dispatch.ack.time.ms,app.engine.balancing.master.dispatch.reason";

    private ConfigService newConfigService(String criticalMetrics) {
        ConfigService configService = new ConfigService(new HashMap<>());
        configService.loadBalanceConfig().setFlexlbMonitorCriticalMetrics(criticalMetrics);
        return configService;
    }

    @Test
    void deniesNonCriticalFlexlbMetricsWhenWhitelistConfigured() {
        ConfigService configService = newConfigService(CRITICAL_LIST);
        CriticalMetricsFilterConfig config = new CriticalMetricsFilterConfig(configService);
        config.init();
        MeterFilter filter = config.criticalMetricsOnlyFilter();
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id critical = registry.counter("flexlb.app.request.network.delay.ms").getId();
        Meter.Id dispatchReason = registry.counter(
                "flexlb.app.engine.balancing.master.dispatch.reason").getId();
        Meter.Id nonCritical = registry.counter("flexlb.grpc.server.executor.queue.size").getId();
        Meter.Id jvm = registry.counter("jvm.test.metric").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(critical));
        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(dispatchReason));
        assertEquals(MeterFilterReply.DENY, filter.accept(nonCritical));
        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(jvm));
    }

    @Test
    void allowsAllMetricsWhenConfigIsEmpty() {
        ConfigService configService = newConfigService("");
        CriticalMetricsFilterConfig config = new CriticalMetricsFilterConfig(configService);
        config.init();
        MeterFilter filter = config.criticalMetricsOnlyFilter();
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id nonCritical = registry.counter("flexlb.grpc.server.executor.queue.size").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(nonCritical));
    }

    @Test
    void allowsAllMetricsWhenConfigIsStar() {
        ConfigService configService = newConfigService("*");
        CriticalMetricsFilterConfig config = new CriticalMetricsFilterConfig(configService);
        config.init();
        MeterFilter filter = config.criticalMetricsOnlyFilter();
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id nonCritical = registry.counter("flexlb.grpc.server.executor.queue.size").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(nonCritical));
    }
}
