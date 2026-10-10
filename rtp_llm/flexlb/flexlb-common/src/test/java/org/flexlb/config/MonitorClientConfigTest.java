package org.flexlb.config;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.junit.jupiter.api.Test;
import org.springframework.boot.actuate.autoconfigure.metrics.CompositeMeterRegistryAutoConfiguration;
import org.springframework.boot.actuate.autoconfigure.metrics.MetricsAutoConfiguration;
import org.springframework.boot.actuate.autoconfigure.metrics.export.prometheus.PrometheusMetricsExportAutoConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class MonitorClientConfigTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(MonitorClientConfig.class, MonitorDisableConfig.class);

    @Test
    void providesMicrometerMonitorWhenBootAutoConfiguresRegistry() {
        contextRunner.withConfiguration(AutoConfigurations.of(
                        MetricsAutoConfiguration.class,
                        CompositeMeterRegistryAutoConfiguration.class,
                        PrometheusMetricsExportAutoConfiguration.class))
                .withPropertyValues("flexlb.monitor.provider=micrometer")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    FlexMonitor monitor = context.getBean(FlexMonitor.class);
                    assertThat(monitor).isInstanceOf(MicrometerFlexMonitor.class);
                    monitor.register(MetricConstant.CACHE_AFFINITY_DECISION, FlexMetricType.QPS);
                    monitor.report(MetricConstant.CACHE_AFFINITY_DECISION, FlexMetricTags.of(
                            "role", "PREFILL", "engineIp", "127.0.0.1:13215", "decision", "NO_CACHE_LEAD"), 1.0);
                    assertThat(context.getBean(PrometheusMeterRegistry.class).scrape())
                            .contains("flexlb_app_cache_affinity_decision_qps_total{")
                            .contains("engineIp=\"127.0.0.1:13215\"")
                            .contains("decision=\"NO_CACHE_LEAD\"");
                });
    }

    @Test
    void defaultsToNoOpEvenWhenRegistryIsAvailable() {
        contextRunner.withUserConfiguration(RegistryConfiguration.class).run(context -> {
            assertThat(context).hasNotFailed();
            assertThat(context).hasSingleBean(FlexMonitor.class);
            assertThat(context.getBean(FlexMonitor.class))
                    .isSameAs(NoOpFlexMonitor.getInstance());
        });
    }

    @Test
    void backsOffWhenProviderMonitorAlreadyExists() {
        contextRunner.withPropertyValues("flexlb.monitor.provider=micrometer")
                .withUserConfiguration(RegistryConfiguration.class)
                .withBean(
                        "providerMonitor",
                        FlexMonitor.class,
                        NoOpFlexMonitor::getInstance)
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    assertThat(context).hasBean("providerMonitor");
                    assertThat(context).doesNotHaveBean("micrometerFlexMonitor");
                    assertThat(context).doesNotHaveBean("flexMonitor");
                });
    }

    @Test
    void explicitMicrometerRequiresRegistry() {
        contextRunner.withPropertyValues("flexlb.monitor.provider=micrometer")
                .run(context -> assertThat(context).hasFailed());
    }

    @Test
    void providesNoOpMonitorWhenProviderIsExplicitlyNoOp() {
        contextRunner.withUserConfiguration(RegistryConfiguration.class)
                .withPropertyValues("flexlb.monitor.provider=noop")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    assertThat(context.getBean(FlexMonitor.class))
                            .isSameAs(NoOpFlexMonitor.getInstance());
                });
    }

    @Test
    void disabledMonitoringOverridesMicrometerSelection() {
        contextRunner.withUserConfiguration(RegistryConfiguration.class)
                .withPropertyValues("flexlb.monitor.provider=micrometer", "flexlb.monitor.enabled=false")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(FlexMonitor.class);
                    assertThat(context.getBean(FlexMonitor.class))
                            .isSameAs(NoOpFlexMonitor.getInstance());
                });
    }

    @Configuration(proxyBeanMethods = false)
    static class RegistryConfiguration {
        @Bean
        MeterRegistry meterRegistry() {
            return new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        }
    }
}
