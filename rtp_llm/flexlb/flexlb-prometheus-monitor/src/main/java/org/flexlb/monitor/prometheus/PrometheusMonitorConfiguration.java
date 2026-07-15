package org.flexlb.monitor.prometheus;

import io.micrometer.prometheus.PrometheusMeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Slf4j
@Configuration
public class PrometheusMonitorConfiguration {

    @Bean
    @Primary
    @ConditionalOnProperty(prefix = "flexlb.monitor", name = "provider", havingValue = "prometheus")
    public FlexMonitor prometheusFlexMonitor(
            ObjectProvider<PrometheusMeterRegistry> registryProvider) {
        PrometheusMeterRegistry registry = registryProvider.getIfAvailable();
        if (registry == null) {
            log.warn("Prometheus monitor selected but PrometheusMeterRegistry is unavailable; using NoOpFlexMonitor");
            return NoOpFlexMonitor.getInstance();
        }
        return new PrometheusFlexMonitor(registry.getPrometheusRegistry());
    }
}
