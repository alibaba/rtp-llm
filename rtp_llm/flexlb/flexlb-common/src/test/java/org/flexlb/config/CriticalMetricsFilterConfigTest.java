package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.config.MeterFilterReply;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CriticalMetricsFilterConfigTest {

    @Test
    void deniesNonCriticalFlexlbMetricsWithoutReturningNullIds() {
        CriticalMetricsFilterConfig config = new CriticalMetricsFilterConfig();
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
}
