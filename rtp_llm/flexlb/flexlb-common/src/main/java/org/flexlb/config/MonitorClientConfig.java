
package org.flexlb.config;

import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MonitorClientConfig {

    /**
     * Create a Micrometer monitor when a registry is available, otherwise
     * retain the open-source NoOp fallback. Registry lookup is deferred until
     * bean creation because Metrics auto-configuration is parsed after
     * component-scanned application configuration.
     *
     * <p>This bridges FlexLB business metrics to micrometer's MeterRegistry so they
     * are exposed via the {@code /prometheus} actuator endpoint without kmonitor.
     */
    @Bean
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor micrometerFlexMonitor(ObjectProvider<MeterRegistry> registryProvider) {
        MeterRegistry meterRegistry = registryProvider.getIfAvailable();
        if (meterRegistry != null) {
            log.info("Creating MicrometerFlexMonitor - bridging FlexMonitor to micrometer/Prometheus");
            return new MicrometerFlexMonitor(meterRegistry);
        }
        log.info("Creating default NoOpFlexMonitor - monitoring disabled");
        return NoOpFlexMonitor.getInstance();
    }
}
