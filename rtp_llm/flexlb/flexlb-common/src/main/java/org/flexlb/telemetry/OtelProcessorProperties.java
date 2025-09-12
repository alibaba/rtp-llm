package org.flexlb.telemetry;

import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

@Data
@ConfigurationProperties(prefix = "tpp.otel.processor")
public class OtelProcessorProperties {

    @NestedConfigurationProperty
    private Batch batch = new Batch();

    @NestedConfigurationProperty
    private Logging logging = new Logging();

    /**
     * Configuration of the {@link BatchSpanProcessor}.
     */
    @Data
    public static class Batch {

        /**
         * Schedule delay in millis.
         */
        private Long scheduleDelay;

        /**
         * Max queue size.
         */
        private Integer maxQueueSize;

        /**
         * Max export batch size.
         */
        private Integer maxExportBatchSize;

        /**
         * Exporter timeout in millis.
         */
        private Long exporterTimeout;

    }

    /**
     * Configuration of the {@link com.alibaba.whale.shard.tpp.apm.telemetry.processor.LoggingSpanProcessor}.
     */
    @Data
    public static class Logging {
        /**
         * enable the logging span processor
         */
        private Boolean enabled;
    }
}
