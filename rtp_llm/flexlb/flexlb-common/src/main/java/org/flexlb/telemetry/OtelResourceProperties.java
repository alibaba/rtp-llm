package org.flexlb.telemetry;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Data
@ConfigurationProperties("trace.otel.resource")
public class OtelResourceProperties {

    private String roleEnv = "role";
    private String env;
}
