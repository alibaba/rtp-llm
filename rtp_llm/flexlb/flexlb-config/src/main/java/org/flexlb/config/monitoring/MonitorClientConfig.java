package org.flexlb.config.monitoring;

import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.MicrometerFlexMonitor;
import org.flexlb.util.metrics.SafeFlexMonitor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Creates the selected monitoring client while respecting an existing provider bean.
 */
@Slf4j
@Configuration
public class MonitorClientConfig {

    @Bean
    public static BeanPostProcessor safeFlexMonitorPostProcessor(
            ObjectProvider<MeterRegistry> registries) {
        return new BeanPostProcessor() {
            @Override
            public Object postProcessAfterInitialization(Object bean, String beanName) {
                if (!(bean instanceof FlexMonitor monitor)) {
                    return bean;
                }
                MeterRegistry registry = registries.getIfAvailable();
                return registry == null
                        ? SafeFlexMonitor.wrap(monitor)
                        : SafeFlexMonitor.wrap(monitor, new MicrometerFlexMonitor(registry));
            }
        };
    }

    /**
     * Create MicrometerFlexMonitor when the micrometer provider is selected and
     * kmonitor is not available (e.g. a test environment built with -P '!internal').
     *
     * <p>This bridges FlexLB business metrics to micrometer's MeterRegistry so they
     * are exposed via the {@code /prometheus} actuator endpoint without kmonitor.
     */
    @Bean
    @org.springframework.boot.autoconfigure.condition.ConditionalOnExpression(
            "'${flexlb.monitor.provider:}' == 'micrometer' && ${flexlb.monitor.enabled:true}")
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor micrometerFlexMonitor(MeterRegistry meterRegistry) {
        log.info("Creating MicrometerFlexMonitor - bridging FlexMonitor to micrometer/Prometheus");
        return SafeFlexMonitor.wrap(new MicrometerFlexMonitor(meterRegistry));
    }
}
