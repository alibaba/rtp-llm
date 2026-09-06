package org.flexlb;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;

class KMonitorPrometheusPropertiesTemplateTest {

    @Test
    void packagesTurboOnlyKMonitorTemplateInApplicationClasspath() throws IOException {
        Properties properties = new Properties();
        InputStream resource = getClass().getResourceAsStream("/kmonitor-prometheus.properties.example");
        assertThat(resource).isNotNull();
        try (InputStream stream = resource) {
            properties.load(stream);
        }

        assertThat(properties.stringPropertyNames()).noneMatch(name -> name.startsWith("kmonitor.sink.flume"));
        assertThat(properties.getProperty("kmonitor.instance.flexlb.sinks"))
                .isEqualTo("org.flexlb.monitor.prometheus.KMonitorPrometheusExporter");
        assertThat(properties.getProperty("kmonitor.instance.system_kmonitor.sinks"))
                .isEqualTo("org.flexlb.monitor.prometheus.KMonitorPrometheusExporter");
    }
}
