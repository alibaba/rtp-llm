package org.flexlb.telemetry;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

import java.util.HashMap;
import java.util.Map;

@Data
@ConfigurationProperties(prefix = "tpp.otel.exporter")
public class OtelExporterProperties {

    @NestedConfigurationProperty
    private Otlp otlp = new Otlp();

    @NestedConfigurationProperty
    private Logging logging = new Logging();

    @Data
    public static class Otlp {

        /**
         * Enables OTLP exporter.
         */
        private boolean enabled;

        /**
         * Timeout in millis.
         */
        private Long timeout;

        /**
         * Sets the OTLP endpoint to connect to.
         */
        private String endpoint;

        /**
         * Map of headers to be added.
         */
        private Map<String, String> headers = new HashMap<>();

    }

    @Data
    public static class Logging {
        /**
         * Enables Logging exporter.
         */
        private boolean enabled = true;
    }
}
