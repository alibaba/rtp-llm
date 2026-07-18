package org.flexlb.config;

import io.micrometer.core.instrument.config.MeterFilter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Reduces Prometheus metrics cardinality by removing per-engine tags (engineIp, engineIpPort).
 * With 750+ mock engines, per-engine tags generate ~139K time series (23.5MB per scrape),
 * causing Prometheus scrape timeouts. This filter aggregates metrics across engines.
 * Set flexlb.monitor.per-engine-tags=true to restore per-engine granularity for debugging.
 */
@Configuration
@ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
@ConditionalOnProperty(name = "flexlb.monitor.per-engine-tags", havingValue = "false", matchIfMissing = false)
public class MetricsCardinalityConfig {

    @Bean
    public MeterFilter removeEngineTagsFilter() {
        return MeterFilter.ignoreTags("engineIp", "engineIpPort");
    }
}
