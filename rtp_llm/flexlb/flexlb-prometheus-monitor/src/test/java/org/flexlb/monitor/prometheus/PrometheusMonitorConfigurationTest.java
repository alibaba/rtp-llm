package org.flexlb.monitor.prometheus;

import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.config.MonitorClientConfig;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class PrometheusMonitorConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(MonitorClientConfig.class, PrometheusMonitorConfiguration.class);

    @Test
    void missingProviderFallsBackToNoOp() {
        contextRunner.run(context -> assertNoOp(context.getBean(FlexMonitor.class)));
    }

    @Test
    void unsupportedProviderFallsBackToNoOp() {
        contextRunner.withPropertyValues("flexlb.monitor.provider=unsupported")
                .run(context -> assertNoOp(context.getBean(FlexMonitor.class)));
    }

    @Test
    void uppercasePrometheusProviderUsesPrometheusMonitor() {
        contextRunner.withBean(PrometheusMeterRegistry.class,
                        () -> new PrometheusMeterRegistry(PrometheusConfig.DEFAULT))
                .withPropertyValues("flexlb.monitor.provider=PROMETHEUS")
                .run(context -> assertThat(context.getBean(FlexMonitor.class))
                        .isInstanceOf(PrometheusFlexMonitor.class));
    }

    @Test
    void kmonitorWithoutInternalProviderFallsBackToNoOp() {
        contextRunner.withPropertyValues("flexlb.monitor.provider=kmonitor")
                .run(context -> assertNoOp(context.getBean(FlexMonitor.class)));
    }

    @Test
    void prometheusProviderUsesMeterRegistryCollectorRegistry() {
        PrometheusMeterRegistry meterRegistry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);

        contextRunner.withBean(PrometheusMeterRegistry.class, () -> meterRegistry)
                .withPropertyValues("flexlb.monitor.provider=prometheus")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    FlexMonitor monitor = context.getBean(FlexMonitor.class);
                    assertThat(monitor).isInstanceOf(PrometheusFlexMonitor.class);

                    monitor.register("configuration_shared_registry", FlexMetricType.GAUGE, FlexMetricTags.of());
                    monitor.report("configuration_shared_registry", 7.0);

                    assertThat(meterRegistry.scrape())
                            .contains("flexlb_configuration_shared_registry 7.0");
                });
    }

    @Test
    void prometheusWithoutMeterRegistryFallsBackToNoOp() {
        contextRunner.withPropertyValues("flexlb.monitor.provider=prometheus")
                .run(context -> assertNoOp(context.getBean(FlexMonitor.class)));
    }

    private static void assertNoOp(FlexMonitor monitor) {
        assertThat(monitor).isSameAs(NoOpFlexMonitor.getInstance());
    }
}
