package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.config.MeterFilterReply;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WhitelistMetricsFilterConfigTest {

    // The G3-collector-shaped whitelist (prometheus-form prefixes/full
    // names) — entries are copied verbatim from MASTER_PROMETHEUS_PREFIXES.
    private static final String G3_SHAPED_WHITELIST =
            "flexlb_app_cache_,"
                    + "flexlb_app_engine_balancing_master_dispatch_reason_total,"
                    + "flexlb_auto_tpm_request_count,"
                    + "flexlb_app_flexlb_inflight_batch_count";

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    private static MeterFilter filterFor(String property) {
        return new WhitelistMetricsFilterConfig(property).whitelistMetricsFilter();
    }

    @Test
    void allowsPrefixWhitelistedFlexlbMetrics() {
        MeterFilter filter = filterFor(G3_SHAPED_WHITELIST);
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id cacheHit = registry.counter("flexlb.app.cache.hit.ratio").getId();
        Meter.Id inflightBatch = registry
                .counter("flexlb.app.flexlb.inflight.batch.count").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(cacheHit));
        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(inflightBatch));
    }

    @Test
    void allowsCounterTotalFormEntryAgainstUnsuffixedMeterName() {
        // The dispatch-reason whitelist entry carries the prometheus counter
        // suffix (_total), but the micrometer meter name does not — the match
        // must try the counter exposition form.
        MeterFilter filter = filterFor(G3_SHAPED_WHITELIST);
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id dispatchReason = registry
                .counter("flexlb.app.engine.balancing.master.dispatch.reason").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(dispatchReason));
    }

    @Test
    void deniesFlexlbMetricsOutsideWhitelist() {
        MeterFilter filter = filterFor(G3_SHAPED_WHITELIST);
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id nonWhitelisted = registry
                .counter("flexlb.grpc.server.executor.queue.size").getId();
        Meter.Id nearMiss = registry
                .counter("flexlb.app.cachex.should.not.match").getId();

        assertEquals(MeterFilterReply.DENY, filter.accept(nonWhitelisted));
        // "flexlb_app_cachex..." is NOT under the "flexlb_app_cache_" prefix.
        assertEquals(MeterFilterReply.DENY, filter.accept(nearMiss));
    }

    @Test
    void alwaysAllowsNonFlexlbMetrics() {
        MeterFilter filter = filterFor(G3_SHAPED_WHITELIST);
        MeterRegistry registry = new SimpleMeterRegistry();
        Meter.Id jvm = registry.counter("jvm.test.metric").getId();
        Meter.Id process = registry.counter("process.cpu.usage").getId();

        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(jvm));
        assertEquals(MeterFilterReply.NEUTRAL, filter.accept(process));
    }

    @Test
    void emptyWhitelistFailsSafeDenyingEveryFlexlbMetric() {
        // Missing property, blank entries, whitespace-only — all parse to an
        // empty whitelist which must deny everything flexlb.* (fail closed)
        // while still allowing non-flexlb metrics.
        for (String blank : new String[] {null, "", "  ", " , ,, "}) {
            MeterFilter filter = filterFor(blank);
            MeterRegistry registry = new SimpleMeterRegistry();
            Meter.Id flexlb = registry.counter("flexlb.app.cache.hit.ratio").getId();
            Meter.Id jvm = registry.counter("jvm.test.metric").getId();

            assertEquals(MeterFilterReply.DENY, filter.accept(flexlb), "property=" + blank);
            assertEquals(MeterFilterReply.NEUTRAL, filter.accept(jvm), "property=" + blank);
        }
    }

    @Test
    void parseWhitelistTrimsAndDropsBlankEntries() {
        assertEquals(List.of(), WhitelistMetricsFilterConfig.parseWhitelist(null));
        assertEquals(List.of(), WhitelistMetricsFilterConfig.parseWhitelist(" , , "));
        assertEquals(
                List.of("flexlb_app_cache_", "flexlb_auto_tpm_request_count"),
                WhitelistMetricsFilterConfig.parseWhitelist(
                        " flexlb_app_cache_ ,,, flexlb_auto_tpm_request_count  "));
    }

    @Test
    void initMakesMicrometerFlexMonitorSkipNonWhitelistedMetrics() {
        // Early-return layer: after init(), register()/report() for a metric
        // outside the whitelist must not reach the registry at all, while a
        // whitelisted one still lands.
        WhitelistMetricsFilterConfig config =
                new WhitelistMetricsFilterConfig(G3_SHAPED_WHITELIST);
        config.init();

        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            monitor.register("app.flexlb.inflight.batch.count", FlexMetricType.GAUGE);
            monitor.report("app.flexlb.inflight.batch.count", null, 3.0D);
            assertNotNull(registry.find("flexlb.app.flexlb.inflight.batch.count").gauge());

            monitor.register("grpc.server.executor.queue.size", FlexMetricType.GAUGE);
            monitor.report("grpc.server.executor.queue.size", null, 7.0D);
            assertNull(registry.find("flexlb.grpc.server.executor.queue.size").gauge());
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    void prefixAllowSetContainsUsesPrefixSemantics() {
        WhitelistMetricsFilterConfig.PrefixAllowSet set =
                new WhitelistMetricsFilterConfig.PrefixAllowSet(
                        WhitelistMetricsFilterConfig.parseWhitelist(G3_SHAPED_WHITELIST));

        assertTrue(set.contains("app.cache.hit.ratio"));
        assertTrue(set.contains("app.engine.balancing.master.dispatch.reason"));
        assertFalse(set.contains("grpc.server.executor.queue.size"));
        assertFalse(set.contains("app.cachex.near.miss"));
        assertFalse(set.contains(42));
        assertEquals(4, set.size());
    }
}
